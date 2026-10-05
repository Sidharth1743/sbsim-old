"""Exact batched SB1 engine (phase 1 of port_plan.md).

Holds the state of B independent SB1 environments in arrays. It steps them
with the same operations, in the same order, as the original Environment,
SimulatorBuilding, and TFSimulator stack. It is built from a freshly
constructed reference gin Environment. Every constant, ordering, and
per-step-index table is read from the original objects through their own
methods.

Exact mode: each environment has its own copies of the original random
streams:

* the Python `random` state left by the convection simulator's seed,
* the numpy RandomState of the occupancy model.

So, given the same actions, every environment reproduces an independent
original run bit for bit.

In this phase the grid lives on the host. The FD timestep runs on the GPU
(fd_kernel.cu) and the convection shuffle in C (convection_shuffle.c).

Semantics follow TF-Agents: a step after the terminal step resets the
environment and returns the first observation without stepping it.
"""

import random
from typing import Any

import numpy as np
import pandas as pd

from smart_control.environment import environment as env_lib
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import cuda_fd
from smart_control.simulator import randomized_arrival_departure_occupancy as occ_lib
from smart_control.simulator import stochastic_convection_simulator as conv_lib
from smart_control.simulator import thermostat as thermostat_lib
from smart_control.utils import constants
from smart_control.utils import conversion_utils
from smart_control.utils import histogram_reducer as histogram_lib
from smart_control.utils import regression_building_utils

_MODE_OFF = thermostat_lib.Thermostat.Mode.OFF.value
_MODE_HEAT = thermostat_lib.Thermostat.Mode.HEAT.value
_MODE_COOL = thermostat_lib.Thermostat.Mode.COOL.value
_MODE_PASSIVE_COOL = thermostat_lib.Thermostat.Mode.PASSIVE_COOL.value

STEP_FIRST = 0
STEP_MID = 1
STEP_LAST = 2

_BOILER_ACTIONS = {"supply_water_setpoint": "reheat_water_setpoint"}
_AHU_ACTIONS = {
    "supply_air_heating_temperature_setpoint": "heating_setpoint",
    "supply_air_cooling_temperature_setpoint": "cooling_setpoint",
}


def _f32(x) -> float:
  """Value after storage in a proto float field (P6 in port_plan.md)."""
  return float(np.float32(x))


class _StepTables:
  """Per-step-index values computed with the original functions.

  Every episode starts at the same timestamp and advances by the same step,
  so time-dependent values depend only on the step index k.
  """

  def __init__(self, ref_env: env_lib.Environment):
    sim = ref_env.building.simulator
    self._start = sim._start_timestamp  # pylint: disable=protected-access
    self._step = pd.Timedelta(sim.time_step_sec, unit="s")
    self._weather = sim._weather_controller  # pylint: disable=protected-access
    self._schedule = sim.hvac._schedule  # pylint: disable=protected-access
    self._hvac = sim.hvac
    self._occupancy = ref_env.building._occupancy  # pylint: disable=protected-access
    self._num_hod = ref_env._num_hod_features  # pylint: disable=protected-access
    self._num_dow = ref_env._num_dow_features  # pylint: disable=protected-access
    self._cache = {}

  def ts(self, k: int) -> pd.Timestamp:
    return self._start + k * self._step

  def get(self, k: int) -> dict[str, Any]:
    if k in self._cache:
      return self._cache[k]
    ts = self.ts(k)
    window = self._schedule.get_temperature_window(ts)
    time_features = {}
    hod_rad = conversion_utils.get_radian_time(
        ts, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
    )
    for name, value in regression_building_utils.expand_time_features(
        self._num_hod, hod_rad, env_lib.HOD_LABEL
    ).items():
      time_features[f"{name[0]}_{name[1]}"] = np.float32(value)
    dow_rad = conversion_utils.get_radian_time(
        ts, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK
    )
    for name, value in regression_building_utils.expand_time_features(
        self._num_dow, dow_rad, env_lib.DOW_LABEL
    ).items():
      time_features[f"{name[0]}_{name[1]}"] = np.float32(value)
    # Occupancy: ZoneOccupant._to_local_time and the day used by peek().
    tz = self._occupancy._time_zone  # pylint: disable=protected-access
    local = ts if ts.tz is None else ts.tz_convert(tz)
    day = pd.Timestamp(year=local.year, month=local.month, day=local.day)
    entry = {
        "ts": ts,
        "ambient": self._weather.get_current_temp(ts),
        "conv": self._weather.get_air_convection_coefficient(ts),
        "comfort": bool(self._schedule.is_comfort_mode(ts)),
        "heating_setpoint": window[0],
        "cooling_setpoint": window[1],
        "comfort_soon": bool(
            self._hvac.is_comfort_mode(ts + pd.Timedelta(60, unit="minute"))
        ),
        "time_features": time_features,
        "occ_local_hour": local.hour,
        "occ_work_day": bool(conversion_utils.is_work_day(day)),
    }
    self._cache[k] = entry
    return entry


class SB1ExactBatch:
  """B independent SB1 environments with exact original semantics."""

  def __init__(self, ref_env: env_lib.Environment, batch: int):
    if ref_env._episode_count != 0:  # pylint: disable=protected-access
      raise ValueError("ref_env must be freshly constructed (never reset).")
    self._ref = ref_env
    self.batch = batch
    sim = ref_env.building.simulator
    bld = sim.building
    hvac = sim.hvac
    self._sim = sim
    self._bld = bld
    self._tables = _StepTables(ref_env)
    self._num_timesteps = ref_env._num_timesteps_in_episode  # pylint: disable=protected-access
    self._step_sec = float(sim.time_step_sec)

    # Zones in room-dict order, as get_zone_average_temps() iterates them.
    self.zones = list(bld.get_zone_average_temps().keys())
    if self.zones != list(hvac.vavs.keys()):
      raise ValueError("VAV order differs from the zone order.")
    if [hvac.zone_infos[z].zone_id for z in self.zones] != [
        z.zone_id for z in ref_env.building.zones
    ]:
      raise ValueError("Zone info order differs from the zone order.")
    self.zone_ids = [
        conversion_utils.floor_plan_based_zone_identifier_to_id(z)
        for z in self.zones
    ]
    self._zone_rc = []
    self._zone_diffusers = []
    for z in self.zones:
      rc = np.asarray(bld._room_dict[z], dtype=np.int64).reshape(-1, 2)  # pylint: disable=protected-access
      self._zone_rc.append((rc[:, 0], rc[:, 1]))
      sel = bld.diffusers[rc[:, 0], rc[:, 1]] > 0.0
      rows, cols = rc[sel, 0], rc[sel, 1]
      self._zone_diffusers.append(
          (rows, cols, bld.diffusers[rows, cols].astype(np.float64))
      )
    self._reset_temp = (
        np.copy(bld._reset_temp_values)  # pylint: disable=protected-access
        if bld._reset_temp_values is not None  # pylint: disable=protected-access
        else np.full(bld._exterior_walls.shape, bld._initial_temp)  # pylint: disable=protected-access
    )
    self._shape = self._reset_temp.shape

    # Device parameters (initial values, as the devices' reset() uses).
    ahu, boiler = hvac.air_handler, hvac.boiler
    self._ahu_id, self._boiler_id = ahu.device_id(), boiler.device_id()
    self._ahu = {
        "recirculation": ahu._init_recirculation,  # pylint: disable=protected-access
        "heating_setpoint": ahu._init_heating_air_temp_setpoint,  # pylint: disable=protected-access
        "cooling_setpoint": ahu._init_cooling_air_temp_setpoint,  # pylint: disable=protected-access
        "dp": ahu._init_fan_differential_pressure,  # pylint: disable=protected-access
        "eff": ahu._init_fan_efficiency,  # pylint: disable=protected-access
        "max_flow": ahu._init_max_air_flow_rate,  # pylint: disable=protected-access
    }
    self._boiler = {
        "setpoint": boiler._init_reheat_water_setpoint,  # pylint: disable=protected-access
        "head": boiler._init_water_pump_differential_head,  # pylint: disable=protected-access
        "eff": boiler._init_water_pump_efficiency,  # pylint: disable=protected-access
        "heating_rate": boiler._heating_rate,  # pylint: disable=protected-access
        "cooling_rate": boiler._cooling_rate,  # pylint: disable=protected-access
    }
    self._boiler_obj = boiler
    vavs = [hvac.vavs[z] for z in self.zones]
    self._vav_max_flow = np.array(
        [v._init_max_air_flow_rate for v in vavs], dtype=np.float64  # pylint: disable=protected-access
    )
    self._vav_reheat_max = np.array(
        [v._init_reheat_max_water_flow_rate for v in vavs], dtype=np.float64  # pylint: disable=protected-access
    )
    self._vav_init_damper = np.array(
        [v._init_damper_setting for v in vavs], dtype=np.float64  # pylint: disable=protected-access
    )
    self._vav_init_reheat = np.array(
        [v._init_reheat_valve_setting for v in vavs], dtype=np.float64  # pylint: disable=protected-access
    )
    self._vav_init_zone_temp = np.array(
        [v._init_zone_air_temperature for v in vavs], dtype=np.float64  # pylint: disable=protected-access
    )

    # Occupancy parameters.
    occ = ref_env.building._occupancy  # pylint: disable=protected-access
    if not isinstance(occ, occ_lib.RandomizedArrivalDepartureOccupancy):
      raise NotImplementedError("Only RandomizedArrivalDepartureOccupancy.")
    probe = occ_lib.ZoneOccupant(
        occ._earliest_expected_arrival_hour,  # pylint: disable=protected-access
        occ._latest_expected_arrival_hour,  # pylint: disable=protected-access
        occ._earliest_expected_departure_hour,  # pylint: disable=protected-access
        occ._latest_expected_departure_hour,  # pylint: disable=protected-access
        occ._step_size,  # pylint: disable=protected-access
        np.random.RandomState(0),
        occ._time_zone,  # pylint: disable=protected-access
    )
    self._occ = {
        "per_zone": occ._zone_assignment,  # pylint: disable=protected-access
        "arrive_lo": occ._earliest_expected_arrival_hour,  # pylint: disable=protected-access
        "arrive_hi": occ._latest_expected_arrival_hour,  # pylint: disable=protected-access
        "depart_lo": occ._earliest_expected_departure_hour,  # pylint: disable=protected-access
        "p_arrival": probe._p_arrival,  # pylint: disable=protected-access
        "p_departure": probe._p_departure,  # pylint: disable=protected-access
        "step": occ._step_size,  # pylint: disable=protected-access
    }
    if self._occ["step"] != pd.Timedelta(self._step_sec, unit="s"):
      raise NotImplementedError("Occupancy step must equal the sim step.")
    occ_state = occ._random_state.get_state()  # pylint: disable=protected-access

    # Convection shuffle: plan and initial Python random state.
    conv = bld._convection_simulator  # pylint: disable=protected-access
    if not isinstance(conv, conv_lib.StochasticConvectionSimulator):
      raise NotImplementedError("Only StochasticConvectionSimulator.")
    self._conv_p = float(conv._p)  # pylint: disable=protected-access
    self._conv_plan = conv_lib._ShufflePlan(  # pylint: disable=protected-access
        bld._room_dict, conv._distance, self._shape  # pylint: disable=protected-access
    )
    self._conv_kernel = conv_lib._load_kernel()  # pylint: disable=protected-access
    if self._conv_kernel is None or not self._conv_plan.valid:
      raise RuntimeError("Compiled convection shuffle unavailable.")
    py_state = random.getstate()

    # Observation layout from the reference environment.
    self.field_names = list(ref_env.field_names)
    self.action_names = list(ref_env._action_names)  # pylint: disable=protected-access
    self._obs_requests = [
        (r.device_id, r.measurement_name)
        for r in ref_env._observation_request.single_observation_requests  # pylint: disable=protected-access
    ]
    normalizer = ref_env._observation_normalizer  # pylint: disable=protected-access
    self._norm = {}
    for _, m in self._obs_requests:
      c = normalizer._get_normalization_constants(m)  # pylint: disable=protected-access
      self._norm[m] = (c.sample_mean, c.sample_variance)
    self._normalize_one = normalizer._normalize_one  # pylint: disable=protected-access
    self._hist = ref_env._observation_histogram_reducer  # pylint: disable=protected-access
    self._id_inv = ref_env._id_map.inv  # pylint: disable=protected-access
    self._occ_norm = ref_env.occupancy_normalization_constant
    self._action_normalizers = [
        ref_env.action_normalizers[f] for f in self.action_names
    ]
    self._action_targets = [self._id_inv[f] for f in self.action_names]
    self._reward_fn = ref_env.reward_function

    # FD solver for the batch.
    if sim._cuda_fd is None:  # pylint: disable=protected-access
      raise RuntimeError("The CUDA FD kernel is required.")
    flags = np.zeros(self._shape, dtype=np.uint8)
    flags |= sim._t_exerior_temps_mask.numpy().astype(np.uint8)  # pylint: disable=protected-access
    for bit, mask in zip((2, 4, 8, 16), sim._t_convection_masks):  # pylint: disable=protected-access
      flags |= mask.numpy().astype(np.uint8) * np.uint8(bit)
    self._fd_kwargs = dict(
        batch=batch,
        shape=self._shape,
        iteration_limit=sim._iteration_limit,  # pylint: disable=protected-access
        convergence_threshold=sim._convergence_threshold,  # pylint: disable=protected-access
        floor_height_m=bld.floor_height_cm / 100.0,
        time_step_sec=sim._time_step_sec,  # pylint: disable=protected-access
        conductivity_left=sim._t_conductivity_left_edge.numpy(),  # pylint: disable=protected-access
        conductivity_right=sim._t_conductivity_right_edge.numpy(),  # pylint: disable=protected-access
        conductivity_top=sim._t_conductivity_top_edge.numpy(),  # pylint: disable=protected-access
        conductivity_bottom=sim._t_conductivity_bottom_edge.numpy(),  # pylint: disable=protected-access
        u=sim._t_u.numpy(),  # pylint: disable=protected-access
        v=sim._t_v.numpy(),  # pylint: disable=protected-access
        density=bld.density,
        heat_capacity=bld.heat_capacity,
        flags=flags,
        fuse_k=sim._cuda_fd_fuse_k,  # pylint: disable=protected-access
    )
    self._fd = self._make_fd()

    # Per environment state that persists across episodes.
    B, Z = batch, len(self.zones)
    self._thermo_mode = np.full((B, Z), _MODE_OFF, dtype=np.int64)
    self._thermo_prev_k = np.full(B, -10**9, dtype=np.int64)  # None
    self._occ_state = np.zeros((B, Z, self._occ["per_zone"]), dtype=bool)
    self._occ_rng = []
    self._conv_mt = np.empty((B, 624), dtype=np.uint32)
    self._conv_index = np.empty(B, dtype=np.int64)
    self._conv_gauss = []
    for b in range(B):
      rs = np.random.RandomState()
      rs.set_state(occ_state)
      self._occ_rng.append(rs)
      self._conv_mt[b] = np.array(py_state[1][:-1], dtype=np.uint32)
      self._conv_index[b] = py_state[1][-1]
      self._conv_gauss.append(py_state[2])
    # Device timestamps (step index; None when never set), not reset.
    self._boiler_action_k = [None] * B
    self._boiler_obs_k = [None] * B
    self._episode_ended = np.zeros(B, dtype=bool)
    self._step_count = np.zeros(B, dtype=np.int64)
    self._k = np.zeros(B, dtype=np.int64)
    self._temp = [None] * B
    self._input_q = [None] * B
    self._reset_device_state(np.ones(B, dtype=bool))

  def _make_fd(self):
    return cuda_fd.CudaFdSolver(cuda_fd.load_library(), **self._fd_kwargs)

  def _reset_grid(self, b: int) -> None:
    self._temp[b] = np.copy(self._reset_temp)
    self._input_q[b] = np.zeros(self._shape)

  # --------------------------------------------------------------- reset
  def _reset_device_state(self, mask: np.ndarray) -> None:
    """Building, HVAC and simulator reset (state that does reset)."""
    for b in np.flatnonzero(mask):
      self._reset_grid(int(b))
      self._k[b] = 0
    Z = len(self.zones)
    if not hasattr(self, "_damper"):
      B = self.batch
      self._damper = np.zeros((B, Z))
      self._reheat = np.zeros((B, Z))
      self._zone_air_temp = np.zeros((B, Z))
      self._ahu_heat_sp = np.zeros(B)
      self._ahu_cool_sp = np.zeros(B)
      self._ahu_flow = np.zeros(B)
      self._ahu_count = np.zeros(B, dtype=np.int64)
      self._boiler_sp = np.zeros(B)
      self._boiler_flow = np.zeros(B)
      self._boiler_count = np.zeros(B, dtype=np.int64)
      self._boiler_rwt = np.zeros(B)
      self._boiler_current = np.zeros(B)
      self._boiler_change = np.zeros(B)
      self._boiler_last_dur = np.zeros(B)
    self._damper[mask] = self._vav_init_damper
    self._reheat[mask] = self._vav_init_reheat
    self._zone_air_temp[mask] = self._vav_init_zone_temp
    self._ahu_heat_sp[mask] = self._ahu["heating_setpoint"]
    self._ahu_cool_sp[mask] = self._ahu["cooling_setpoint"]
    self._ahu_flow[mask] = 0.0
    self._ahu_count[mask] = 0
    self._boiler_sp[mask] = self._boiler["setpoint"]
    self._boiler_flow[mask] = 0.0
    self._boiler_count[mask] = 0
    self._boiler_rwt[mask] = 0.0
    self._boiler_current[mask] = self._boiler["setpoint"]
    self._boiler_change[mask] = 0.0
    self._boiler_last_dur[mask] = 0.0

  def _reset_env(self, b: int) -> np.ndarray:
    mask = np.zeros(self.batch, dtype=bool)
    mask[b] = True
    self._reset_device_state(mask)
    self._episode_ended[b] = False
    obs = self._observe(b)
    self._step_count[b] = 0
    return obs

  def reset(self) -> np.ndarray:
    return np.stack([self._reset_env(b) for b in range(self.batch)])

  # ----------------------------------------------------------- numerics
  def _zone_means(self, b: int) -> list[Any]:
    temp = self._temp[b]
    return [np.mean(temp[r, c]) for r, c in self._zone_rc]

  # ---------------------------------------------------------- occupancy
  def _peek(self, b: int, z: int, k: int) -> int:
    """ZoneOccupant.peek for every occupant of zone z; returns # at WORK."""
    t = self._tables.get(k)
    occ = self._occ
    rng = self._occ_rng[b]
    working = 0
    for o in range(occ["per_zone"]):
      state = self._occ_state[b, z, o]
      if not t["occ_work_day"]:
        state = False
      elif not state:
        hour = t["occ_local_hour"]
        if not (hour < occ["arrive_lo"] or hour > occ["arrive_hi"]):
          if rng.rand() < occ["p_arrival"]:
            state = True
      else:
        if not t["occ_local_hour"] < occ["depart_lo"]:
          if rng.rand() < occ["p_departure"]:
            state = False
      self._occ_state[b, z, o] = state
      working += int(state)
    return working

  def _average_zone_occupancy(self, b: int, z: int, k0: int) -> float:
    """average_zone_occupancy over [ts(k0), ts(k0) + step): one sub-step."""
    total = 0.0
    total += float(self._peek(b, z, k0))
    return total / 1

  # ------------------------------------------------------------- boiler
  def _boiler_supply_water_temperature(self, b: int) -> float:
    """Boiler.supply_water_temperature_sensor, including its side effects."""
    obs_k = self._boiler_obs_k[b]
    if self._boiler_action_k[b] is not None:
      self._boiler_last_dur[b] = (obs_k - self._boiler_action_k[b]) * (
          self._step_sec
      )
    else:
      self._boiler_action_k[b] = obs_k
    heating_rate = self._boiler["heating_rate"]
    cooling_rate = self._boiler["cooling_rate"]
    if cooling_rate > 0.0 and heating_rate > 0.0:
      begin = self._boiler_current[b]
      sp = self._boiler_sp[b]
      dur = self._boiler_last_dur[b]
      if sp > begin:
        new = min(begin + heating_rate * dur / 60.0, sp)
      elif sp < begin:
        new = max(begin - cooling_rate * dur / 60.0, sp)
      else:
        new = sp
      self._boiler_current[b] = new
      self._boiler_change[b] = new - begin
    else:
      self._boiler_current[b] = self._boiler_sp[b]
    return self._boiler_current[b]

  def _boiler_thermal_energy_rate(self, b: int, rwt: float, outside: float):
    boiler = self._boiler_obj
    sp = self._boiler_sp[b]
    supply = sp if sp > rwt else rwt
    flow_heating = (
        constants.WATER_HEAT_CAPACITY * self._boiler_flow[b] * (supply - rwt)
    )
    dissipation = boiler.compute_thermal_dissipation_rate(supply, outside)
    if self._boiler_last_dur[b] > 0:
      tank = (
          constants.WATER_HEAT_CAPACITY
          * boiler._water_capacity  # pylint: disable=protected-access
          * self._boiler_change[b]
          / self._boiler_last_dur[b]
      )
    else:
      tank = 0
    return flow_heating + dissipation + tank

  def _boiler_pump_power(self, b: int) -> float:
    return (
        self._boiler_flow[b]
        * constants.WATER_DENSITY
        * constants.GRAVITY
        * self._boiler["head"]
        / self._boiler["eff"]
    )

  # ---------------------------------------------------------------- AHU
  def _mixed_air_temp(self, b: int, recirc, ambient):
    rec = self._ahu["recirculation"]
    return rec * recirc + (1 - rec) * ambient

  def _supply_air_temp(self, b: int, recirc, ambient):
    mixed = self._mixed_air_temp(b, recirc, ambient)
    if mixed > self._ahu_cool_sp[b]:
      return self._ahu_cool_sp[b]
    elif mixed < self._ahu_heat_sp[b]:
      return self._ahu_heat_sp[b]
    return mixed

  def _ahu_fan_power(self, flow):
    return flow * self._ahu["dp"] / self._ahu["eff"]

  # -------------------------------------------------------- observation
  def _device_value(self, b: int, device_id: str, field: str, k: int):
    ahu = self._ahu
    if device_id == self._boiler_id:
      self._boiler_obs_k[b] = k
      if field == "heating_request_count":
        return self._boiler_count[b]
      if field == "supply_water_setpoint":
        return self._boiler_sp[b]
      if field == "supply_water_temperature_sensor":
        return self._boiler_supply_water_temperature(b)
    elif device_id == self._ahu_id:
      flow = self._ahu_flow[b]
      if field == "cooling_request_count":
        return self._ahu_count[b]
      if field == "differential_pressure_setpoint":
        return ahu["dp"]
      if field in (
          "discharge_fan_speed_percentage_command",
          "supply_fan_speed_percentage_command",
      ):
        return flow / ahu["max_flow"]
      if field == "outside_air_flowrate_sensor":
        return (1.0 - ahu["recirculation"]) * flow
      if field == "outside_air_temperature_sensor":
        return self._tables.get(k)["ambient"]
      if field == "supply_air_cooling_temperature_setpoint":
        return self._ahu_cool_sp[b]
      if field == "supply_air_flowrate_sensor":
        return flow
      if field == "supply_air_heating_temperature_setpoint":
        return self._ahu_heat_sp[b]
    else:
      z = self._vav_index[device_id]
      if field == "supply_air_damper_percentage_command":
        return self._damper[b, z]
      if field == "supply_air_flowrate_setpoint":
        return self._vav_max_flow[z]
      if field == "zone_air_temperature_sensor":
        return self._zone_air_temp[b, z]
    raise NotImplementedError(f"Observation {device_id} {field}.")

  @property
  def _vav_index(self) -> dict[str, int]:
    if not hasattr(self, "_vav_index_cache"):
      self._vav_index_cache = {
          f"vav_{z}": i for i, z in enumerate(self.zones)
      }
    return self._vav_index_cache

  def _observe(self, b: int) -> np.ndarray:
    """Environment._get_observation for environment b at its step k."""
    k = int(self._k[b])
    t = self._tables.get(k)
    # request_observations, then the proto float32, then normalization.
    normalized = {}
    for device_id, field in self._obs_requests:
      value = _f32(self._device_value(b, device_id, field, k))
      normalized[(device_id, field)] = _f32(self._normalize_one(field, value))
    observation = {}
    hist_params = self._hist.histogram_parameters
    for measurement, bins in hist_params.items():
      values = np.array(
          [v for (d, m), v in normalized.items() if m == measurement]
      )
      if values.size == 0:
        continue
      chist = histogram_lib.get_clipped_histogram(
          measurements=values, bins=bins, clip=True
      )
      total = np.sum(chist)
      for v, c in zip(bins, chist):
        observation[f"{measurement}_h_{v:.2f}"] = np.float32(c / total)
    for (device_id, field), value in normalized.items():
      if field not in hist_params:
        observation[self._ref._id_map[(device_id, field)]] = np.float32(value)  # pylint: disable=protected-access
    observation.update(t["time_features"])
    observation[env_lib.COMFORT_MODE_NOW] = np.float32(t["comfort"])
    observation[env_lib.COMFORT_MODE_SOON] = np.float32(t["comfort_soon"])
    occupants = 0.0
    for z in range(len(self.zones)):
      occupants += self._average_zone_occupancy(b, z, k - 1)
    observation[env_lib.NUM_OCCUPANTS] = np.float32(
        (int(occupants) - self._occ_norm) / (self._occ_norm + 1)
    )
    return np.array(
        [observation[f] for f in self.field_names], dtype=np.float32
    )

  # ------------------------------------------------------------- reward
  def _reward(self, b: int) -> float:
    """Environment._get_reward for environment b at its step k."""
    k = int(self._k[b])
    t = self._tables.get(k)
    means = self._post_means(b)
    zone_infos = {}
    for z, zone_id in enumerate(self.zone_ids):
      zone_infos[zone_id] = smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
          heating_setpoint_temperature=t["heating_setpoint"],
          cooling_setpoint_temperature=t["cooling_setpoint"],
          zone_air_temperature=means[z],
          air_flow_rate_setpoint=self._vav_max_flow[z],
          air_flow_rate=self._ahu_flow[b],
          average_occupancy=self._average_zone_occupancy(b, z, k),
      )
    flow = self._ahu_flow[b]
    blower = self._ahu_fan_power(flow) + self._ahu_fan_power(
        flow * (1.0 - self._ahu["recirculation"])
    )
    recirc = self._post_recirc(b)
    ambient = t["ambient"]
    mixed = self._mixed_air_temp(b, recirc, ambient)
    supply = self._supply_air_temp(b, recirc, ambient)
    ac = flow * constants.AIR_HEAT_CAPACITY * (supply - mixed)
    rwt = self._boiler_rwt[b]
    reward_info = smart_control_reward_pb2.RewardInfo(
        start_timestamp=conversion_utils.pandas_to_proto_timestamp(t["ts"]),
        end_timestamp=conversion_utils.pandas_to_proto_timestamp(
            t["ts"] + pd.Timedelta(self._step_sec, unit="s")
        ),
        zone_reward_infos=zone_infos,
        air_handler_reward_infos={
            self._ahu_id: smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
                blower_electrical_energy_rate=blower,
                air_conditioning_electrical_energy_rate=ac,
            )
        },
        boiler_reward_infos={
            self._boiler_id: smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
                natural_gas_heating_energy_rate=self._boiler_thermal_energy_rate(
                    b, rwt, ambient
                ),
                pump_electrical_energy_rate=self._boiler_pump_power(b),
            )
        },
    )
    return self._reward_fn.compute_reward(reward_info).agent_reward_value

  # --------------------------------------------------------------- step
  def _setup_step(self, b: int, means: list[Any]) -> None:
    """Simulator.setup_step_sim: thermostats and VAV settings."""
    k = int(self._k[b])
    t = self._tables.get(k)
    heat, cool = t["heating_setpoint"], t["cooling_setpoint"]
    prev_k = self._thermo_prev_k[b]
    prev_valid = prev_k > -10**9
    prev_comfort = prev_valid and self._tables.get(int(prev_k))["comfort"]
    mid = 0.5 * (cool - heat) + heat
    for z, zone_temp in enumerate(means):
      self._zone_air_temp[b, z] = zone_temp
      mode = self._thermo_mode[b, z]
      if t["comfort"]:
        mode = self._default_control(zone_temp, heat, cool, mid, mode)
      elif prev_valid and prev_comfort:
        mode = _MODE_PASSIVE_COOL
      else:
        if mode == _MODE_PASSIVE_COOL and zone_temp > heat:
          mode = _MODE_PASSIVE_COOL
        else:
          mode = self._default_control(zone_temp, heat, cool, mid, mode)
      self._thermo_mode[b, z] = mode
      if mode == _MODE_HEAT:
        self._damper[b, z], self._reheat[b, z] = 1.0, 1.0
      elif mode == _MODE_COOL:
        self._damper[b, z], self._reheat[b, z] = 1.0, 0.0
      else:  # OFF and PASSIVE_COOL
        self._damper[b, z], self._reheat[b, z] = 0.1, 0.0
    self._thermo_prev_k[b] = k

  @staticmethod
  def _default_control(zone_temp, heat, cool, mid, mode):
    if zone_temp < heat:
      return _MODE_HEAT
    elif zone_temp > cool:
      return _MODE_COOL
    elif zone_temp < mid and mode == _MODE_HEAT:
      return _MODE_HEAT
    elif zone_temp > mid and mode == _MODE_COOL:
      return _MODE_COOL
    return _MODE_OFF

  def _apply_actions(self, b: int, action: np.ndarray) -> None:
    k = int(self._k[b])
    for i, (normalizer, (device_id, setpoint)) in enumerate(
        zip(self._action_normalizers, self._action_targets)
    ):
      value = _f32(normalizer.setpoint_value(action[i]))
      if device_id == self._boiler_id and setpoint in _BOILER_ACTIONS:
        self._boiler_sp[b] = value
        self._boiler_action_k[b] = k
      elif device_id == self._ahu_id and setpoint in _AHU_ACTIONS:
        if _AHU_ACTIONS[setpoint] == "heating_setpoint":
          self._ahu_heat_sp[b] = value
        else:
          self._ahu_cool_sp[b] = value
      else:
        raise NotImplementedError(f"Action {device_id} {setpoint}.")

  def _pre_recirc(self, b: int):
    """Global mean of the grid before the FD timestep."""
    return self._temp[b].mean()

  def _post_means(self, b: int) -> list[Any]:
    """Zone means of the grid after the timestep."""
    return self._zone_means(b)

  def _post_recirc(self, b: int):
    """Global mean of the grid after the timestep."""
    return self._temp[b].mean()

  def _execute_step(self, b: int, means: list[Any], new_temp: np.ndarray):
    """Simulator.execute_step_sim after the FD timestep (new_temp)."""
    k = int(self._k[b])
    t = self._tables.get(k)
    supply_air_temp = self._supply_air_temp(
        b, self._pre_recirc(b), t["ambient"]
    )
    # Convection shuffle on the new grid with this env's random state.
    self._temp[b] = new_temp
    self._shuffle(b)
    q = self._hvac_outputs(b, means, supply_air_temp)
    for z, q_zone in enumerate(q):
      rows, cols, values = self._zone_diffusers[z]
      self._input_q[b][rows, cols] = np.float64(q_zone) * values
    self._k[b] = k + 1

  def _hvac_outputs(self, b: int, means: list[Any], supply_air_temp):
    """VAV outputs, AHU/boiler demands, return water; returns q per zone.

    Only scalar device state is touched, so it may run before or after the
    grid work of the step.
    """
    self._ahu_flow[b] = 0.0
    self._ahu_count[b] = 0
    self._boiler_flow[b] = 0.0
    self._boiler_count[b] = 0
    water_temp = self._boiler_sp[b]
    max_flow = self._ahu["max_flow"]
    zone_supply = []
    q = []
    for z, zone_temp in enumerate(means):
      damper, reheat = self._damper[b, z], self._reheat[b, z]
      self._zone_air_temp[b, z] = zone_temp
      if damper <= 0:
        raise ValueError("Damper setting must be greater than 0.")
      reheat_flow = reheat * self._vav_reheat_max[z]
      air_flow = damper * self._vav_max_flow[z]
      heat_difference = (
          constants.AIR_HEAT_CAPACITY * air_flow
          - constants.WATER_HEAT_CAPACITY * reheat_flow
      )
      input_water_heat = (
          water_temp * constants.WATER_HEAT_CAPACITY * reheat_flow
      )
      zst = (
          (supply_air_temp * heat_difference + input_water_heat)
          / air_flow
          / constants.AIR_HEAT_CAPACITY
      )
      q_zone = air_flow * constants.AIR_HEAT_CAPACITY * (zst - zone_temp)
      q.append(q_zone)
      zone_supply.append(zst)
      flow_demand = damper * self._vav_max_flow[z]
      if flow_demand > 0:
        self._ahu_flow[b] += flow_demand
        if self._ahu_flow[b] > max_flow:
          self._ahu_flow[b] = max_flow
        self._ahu_count[b] += 1
      reheat_demand = reheat * self._vav_reheat_max[z]
      if reheat_demand > 0:
        self._boiler_flow[b] += reheat_demand
        self._boiler_count[b] += 1
    numerator = 0.0
    denominator = 0.0
    for z in range(len(self.zones)):
      numerator += self._reheat[b, z] * zone_supply[z]
      denominator += self._reheat[b, z]
    self._boiler_rwt[b] = numerator / (denominator + 1e-6)
    return q

  def _shuffle(self, b: int) -> None:
    plan = self._conv_plan
    mt = np.ascontiguousarray(self._conv_mt[b])
    index = np.array([self._conv_index[b]], dtype=np.int64)
    temp = self._temp[b]
    if temp.dtype != np.float32 or not temp.flags.c_contiguous:
      raise ValueError("Convection expects a float32 C-contiguous grid.")
    self._conv_kernel.shuffle_rooms(
        mt,
        index,
        self._conv_p,
        plan.n_rooms,
        plan.room_offsets,
        plan.cvs,
        plan.cand_offsets,
        plan.cands,
        temp.reshape(-1),
        plan.swap_a,
        plan.swap_b,
    )
    self._conv_mt[b] = mt
    self._conv_index[b] = index[0]

  def step(
      self, actions: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps every environment; returns observations, rewards, step types."""
    B = self.batch
    obs = np.zeros((B, len(self.field_names)), dtype=np.float32)
    rewards = np.zeros(B, dtype=np.float32)
    step_types = np.zeros(B, dtype=np.int32)
    stepping = []
    for b in range(B):
      if self._episode_ended[b]:
        obs[b] = self._reset_env(b)
        step_types[b] = STEP_FIRST
      else:
        stepping.append(b)
    if not stepping:
      return obs, rewards, step_types
    pre_means = {}
    for b in stepping:
      pre_means[b] = self._zone_means(b)
      self._setup_step(b, pre_means[b])
      self._apply_actions(b, np.asarray(actions[b], dtype=np.float32))
    # FD for every env of the batch; results of non-stepping envs unused.
    temp_minus = np.stack(
        [np.asarray(self._temp[b], dtype=np.float32) for b in range(B)]
    )
    input_q = np.stack(
        [np.asarray(self._input_q[b], dtype=np.float32) for b in range(B)]
    )
    ambient = np.array(
        [self._tables.get(int(self._k[b]))["ambient"] for b in range(B)],
        dtype=np.float32,
    )
    conv = np.array(
        [self._tables.get(int(self._k[b]))["conv"] for b in range(B)],
        dtype=np.float32,
    )
    out, _, _ = self._fd.step(temp_minus, input_q, ambient, conv)
    for b in stepping:
      self._execute_step(b, pre_means[b], out[b].copy())
      obs[b] = self._observe(b)
      rewards[b] = np.float32(self._reward(b))
      self._episode_ended[b] = self._step_count[b] >= self._num_timesteps
      if self._episode_ended[b]:
        step_types[b] = STEP_LAST
      else:
        step_types[b] = STEP_MID
        self._step_count[b] += 1
    return obs, rewards, step_types

  # ------------------------------------------------------------- access
  def temp(self, b: int) -> np.ndarray:
    return self._temp[b]

  def python_random_state(self, b: int) -> np.ndarray:
    return np.concatenate(
        [self._conv_mt[b], np.array([self._conv_index[b]], dtype=np.uint32)]
    )
