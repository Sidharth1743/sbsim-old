"""Vectorized batched SB1 engine (phase 4a of port_plan.md).

Same semantics as SB1ExactBatch / SB1GpuBatch, with the per-env Python loops
replaced by numpy operations over the batch:

* Every quantity is a float64 array holding exactly the value the original
  Python code holds. In numpy 1.x a float32 array times a Python float stays
  float32, while a float32 scalar times a Python float is float64. So values
  that are numpy float32 scalars in the original are stored as their exact
  float64 values before any arithmetic.
* Elementwise +, -, *, / and sqrt are IEEE operations and match Python's
  scalar math. Sums that are sequential in the original stay sequential loops
  over zones, in the original order: AHU and boiler demand, return water, and
  the reward sums in the RewardInfo map order (P4).
* np.exp on arrays equals np.exp on scalars (P2b).

Occupancy draws come from each env's copy of the numpy MT19937 stream
(sb1_occupancy.c). Convection swaps come from each env's copy of the Python
`random` stream (exact mode), generated on CPU threads and applied on the GPU.
"""

import concurrent.futures
import ctypes
import hashlib
import math
import os
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd

from smart_control.batched import sb1_exact
from smart_control.batched import sb1_gpu_engine
from smart_control.environment import environment as env_lib
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import constants
from smart_control.utils import conversion_utils

_OCC_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sb1_occupancy.c"
)
_occ_lib = None


def _load_occupancy_lib() -> ctypes.CDLL:
  global _occ_lib
  if _occ_lib is not None:
    return _occ_lib
  with open(_OCC_SOURCE, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()[:16]
  cache_dir = os.path.join(
      os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
      "smart_control",
  )
  os.makedirs(cache_dir, exist_ok=True)
  path = os.path.join(cache_dir, f"sb1_occupancy_{digest}.so")
  if not os.path.exists(path):
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache_dir)
    os.close(fd)
    subprocess.run(
        ["gcc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", _OCC_SOURCE,
         "-o", tmp],
        check=True,
        capture_output=True,
    )
    os.replace(tmp, path)
  lib = ctypes.CDLL(path)
  u32p = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
  i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
  u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
  f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
  i64 = ctypes.c_int64
  lib.occupancy_query.restype = None
  lib.occupancy_query.argtypes = [
      u32p, i64p, u8p, u8p, i64p, u8p, i64, i64, i64, i64, i64, i64,
      ctypes.c_double, ctypes.c_double, f64p,
  ]
  _occ_lib = lib
  return lib


def _splitmix64(x: int) -> int:
  """SplitMix64 finalizer, for deriving per-env seeds and keys."""
  x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
  x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
  x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
  return x ^ (x >> 31)


def _f32(x: np.ndarray) -> np.ndarray:
  """Values after storage in proto float fields, as float64 (P6)."""
  return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


class SB1VecBatch(sb1_gpu_engine.SB1GpuBatch):
  """B SB1 environments: vectorized scalar logic, GPU grids, exact mode."""

  def __init__(self, ref_env: env_lib.Environment, batch: int,
               rng: str = "exact", seed: int = 0,
               swap_threads: int | None = None, gpu_parts: int = 1,
               reward: str = "sbsim"):
    """Creates the batch.

    Args:
      ref_env: freshly constructed gin Environment to read everything from.
      batch: number of environments.
      rng: "exact": the original random streams in every env (convection
        swaps generated on CPU threads from Python's MT stream). "feed": the
        same streams, but swaps generated on the GPU from the recorded MT
        words (proves the GPU generator). "fast": swaps generated on the GPU
        from per-env Philox streams keyed by (seed, env), and per-env
        occupancy streams seeded from (seed, env). Same rules, independent
        draws.
      seed: base seed for rng="fast".
      swap_threads: CPU threads for rng="exact".
      gpu_parts: GPU contexts (streams) the batch is split over, so swap
        generation of one part overlaps the FD of another. Results do not
        depend on it.
      reward: "sbsim": the gin reward function (SetpointEnergyCarbonRegret,
        bit-exact with the original). "competition": the Smart Buildings
        competition reward, r = 0.6 comfort - 0.2 energy - 0.2 carbon (see
        _reward_vec). Only the returned reward differs; the simulation is the
        same.
    """
    if rng not in ("exact", "feed", "fast"):
      raise ValueError(f"Unknown rng mode {rng}.")
    if reward not in ("sbsim", "competition"):
      raise ValueError(f"Unknown reward {reward}.")
    self.reward = reward
    self._gpu_parts = gpu_parts
    super().__init__(ref_env, batch)
    B, Z = self.batch, len(self.zones)
    self.rng = rng
    self._occ_lib = _load_occupancy_lib()
    self._occ_mt = np.zeros((B, 624), dtype=np.uint32)
    self._occ_idx = np.zeros(B, dtype=np.int64)
    for b in range(B):
      st = self._occ_rng[b].get_state()
      self._occ_mt[b] = st[1]
      self._occ_idx[b] = st[2]
    self._gpu.set_plan(self._conv_plan, self._conv_p)
    if rng == "fast":
      self._gpu.set_fast_plan(self._conv_plan)
    self._philox_keys = np.zeros((B, 2), dtype=np.uint32)
    self._philox_step = np.zeros(B, dtype=np.uint64)
    self._seed_streams(seed, occupancy=rng == "fast")
    self._feed_words = None
    self._occ_u8 = self._occ_state.astype(np.uint8)
    self._boiler_action_arr = np.full(B, -(10**9), dtype=np.int64)
    self._pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=swap_threads or min(B, os.cpu_count() or 1)
    )
    self._build_tables()
    self._build_observation_plan()
    self._build_reward_plan()

  def _seed_streams(self, seed: int, occupancy: bool) -> None:
    """Sets the fast-mode Philox keys (and occupancy streams) from seed."""
    for b in range(self.batch):
      if occupancy:
        rs = np.random.RandomState(_splitmix64(seed * 1_000_003 + b) & 0xFFFFFFFF)
        st = rs.get_state()
        self._occ_mt[b] = st[1]
        self._occ_idx[b] = st[2]
      key = _splitmix64(seed * 2_000_003 + b)
      self._philox_keys[b] = (key & 0xFFFFFFFF, key >> 32)
    self._philox_step[:] = 0

  def reseed(self, seed: int) -> None:
    """Fast mode: gives every env new occupancy and convection streams.

    Env b gets the streams that rng="fast", seed=seed gives it at
    construction. Call before reset(); state that survives reset() (occupant
    presence, thermostat modes) is kept.
    """
    if self.rng != "fast":
      raise ValueError("reseed needs rng='fast'.")
    self._seed_streams(seed, occupancy=True)

  # ------------------------------------------------------------- tables
  def set_original_seeds(self, convection_seeds, occupancy_seeds) -> None:
    """Gives each env the streams the original code would have with seeds.

    Env b then matches an original run after random.seed(convection_seeds[b])
    and an occupancy RandomState(occupancy_seeds[b]). The original gin seeds
    are 5 (convection) and 17321 (occupancy). Only for rng "exact" or "feed",
    and only before reset(), since reset() already draws occupancy.

    Args:
      convection_seeds: one Python random seed per env.
      occupancy_seeds: one numpy RandomState seed per env.
    """
    if self.rng == "fast":
      raise ValueError("Original seeds apply to rng 'exact' or 'feed'.")
    if getattr(self, "_started", False):
      raise RuntimeError("Call set_original_seeds before reset() or step().")
    if len(convection_seeds) != self.batch or len(occupancy_seeds) != self.batch:
      raise ValueError("Need one convection and one occupancy seed per env.")
    for b, (cs, os_) in enumerate(zip(convection_seeds, occupancy_seeds)):
      state = random.Random(cs).getstate()
      self._conv_mt[b] = np.array(state[1][:-1], dtype=np.uint32)
      self._conv_index[b] = state[1][-1]
      occ_state = np.random.RandomState(os_).get_state()
      self._occ_mt[b] = occ_state[1]
      self._occ_idx[b] = occ_state[2]

  def _build_tables(self) -> None:
    """Arrays over k in [-1, kmax] (index k + 1), from the original code."""
    kmax = self._num_timesteps + 3
    ks = range(-1, kmax + 1)
    rows = [self._tables.get(k) for k in ks]
    self._tab_ambient = np.array([r["ambient"] for r in rows], dtype=np.float64)
    self._tab_conv = np.array([r["conv"] for r in rows], dtype=np.float64)
    self._tab_comfort = np.array([r["comfort"] for r in rows], dtype=bool)
    self._tab_heat = np.array([r["heating_setpoint"] for r in rows], dtype=np.float64)
    self._tab_cool = np.array([r["cooling_setpoint"] for r in rows], dtype=np.float64)
    self._tab_soon = np.array([r["comfort_soon"] for r in rows], dtype=bool)
    self._tab_hour = np.array([r["occ_local_hour"] for r in rows], dtype=np.int64)
    self._tab_workday = np.array([r["occ_work_day"] for r in rows], dtype=np.uint8)
    names = list(rows[0]["time_features"].keys())
    self._tab_time = {
        n: np.array([r["time_features"][n] for r in rows], dtype=np.float32)
        for n in names
    }
    # Reward prices: start_time is the proto timestamp read back (UTC).
    rf = self._reward_fn
    ec = rf._electricity_energy_cost  # pylint: disable=protected-access
    gc = rf._natural_gas_energy_cost  # pylint: disable=protected-access
    wd = getattr(ec._weekday_energy_prices, "magnitude", ec._weekday_energy_prices)  # pylint: disable=protected-access
    we = getattr(ec._weekend_energy_prices, "magnitude", ec._weekend_energy_prices)  # pylint: disable=protected-access
    carbon = getattr(ec._carbon_emission_rates, "magnitude", ec._carbon_emission_rates)  # pylint: disable=protected-access
    price, ecarbon, gprice = [], [], []
    for r in rows:
      st = conversion_utils.proto_to_pandas_timestamp(
          conversion_utils.pandas_to_proto_timestamp(r["ts"])
      )
      price.append((wd if conversion_utils.is_work_day(st) else we)[st.hour])
      ecarbon.append(carbon[st.hour])
      gprice.append(gc._month_gas_price[st.month - 1])  # pylint: disable=protected-access
    self._tab_eprice = np.array(price, dtype=np.float64)
    self._tab_ecarbon = np.array(ecarbon, dtype=np.float64)
    self._tab_gprice = np.array(gprice, dtype=np.float64)
    self._gas_carbon_rate = float(gc._carbon_rate)  # pylint: disable=protected-access
    self._kmax = kmax

  def _tab(self, table: np.ndarray, k: np.ndarray) -> np.ndarray:
    if np.any(k < -1) or np.any(k > self._kmax):
      raise IndexError("Step index outside the precomputed tables.")
    return table[k + 1]

  # -------------------------------------------------------- observation
  def _build_observation_plan(self) -> None:
    self._vav_fields = []
    self._boiler_fields = []
    self._ahu_fields = []
    for device_id, field in self._obs_requests:
      if device_id == self._boiler_id:
        self._boiler_fields.append(field)
      elif device_id == self._ahu_id:
        self._ahu_fields.append(field)
      else:
        if field not in self._vav_fields:
          self._vav_fields.append(field)
    vav_ids = [d for d, _ in self._obs_requests if d.startswith("vav_")]
    expected = [f"vav_{z}" for z in self.zones]
    if sorted(set(vav_ids)) != sorted(expected):
      raise ValueError("Unexpected VAV observation devices.")
    self._sqrt_var = {
        m: (math.sqrt(v) if v > 0.0 else None)
        for m, (_, v) in self._norm.items()
    }
    hist = self._hist.histogram_parameters
    self._hist_bins = {m: np.asarray(b, dtype=np.float64) for m, b in hist.items()}
    for m in hist:
      if m in self._boiler_fields or m in self._ahu_fields:
        raise NotImplementedError("Histograms over boiler or AHU fields.")

  def _normalize(self, field: str, values: np.ndarray) -> np.ndarray:
    mean, _ = self._norm[field]
    sqrt_var = self._sqrt_var[field]
    if sqrt_var is None:
      return np.zeros_like(values)
    return _f32((values - mean) / sqrt_var)

  def _boiler_sensor(self, ix: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Boiler.supply_water_temperature_sensor with its side effects."""
    act = self._boiler_action_arr[ix]
    has = act > -(10**9)
    dur = self._boiler_last_dur[ix].copy()
    dur[has] = (k[has] - act[has]) * self._step_sec
    act[~has] = k[~has]
    self._boiler_action_arr[ix] = act
    self._boiler_last_dur[ix] = dur
    hr, cr = self._boiler["heating_rate"], self._boiler["cooling_rate"]
    sp = self._boiler_sp[ix]
    if cr > 0.0 and hr > 0.0:
      begin = self._boiler_current[ix]
      up = np.minimum(begin + hr * dur / 60.0, sp)
      down = np.maximum(begin - cr * dur / 60.0, sp)
      new = np.where(sp > begin, up, np.where(sp < begin, down, sp))
      self._boiler_current[ix] = new
      self._boiler_change[ix] = new - begin
    else:
      self._boiler_current[ix] = sp
    return self._boiler_current[ix]

  def _histogram(self, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """get_clipped_histogram per row, divided by its sum, as float32."""
    cbins = np.append(bins, max(bins))
    x = np.clip(values, min(cbins), max(cbins))
    lefts = np.stack([np.sum(x < e, axis=1) for e in cbins[:-1]], axis=1)
    right_last = np.sum(x <= cbins[-1], axis=1)
    counts = np.empty((x.shape[0], len(bins)), dtype=np.int64)
    counts[:, :-1] = lefts[:, 1:] - lefts[:, :-1]
    counts[:, -1] = right_last - lefts[:, -1]
    c32 = counts.astype(np.float32)
    total = np.sum(c32, axis=1, keepdims=True)
    return c32 / total

  def _occupancy(self, ix: np.ndarray, k_query: np.ndarray) -> np.ndarray:
    """Occupants at WORK per zone for envs ix at step k_query (one peek)."""
    B, Z = self.batch, len(self.zones)
    active = np.zeros(B, dtype=np.uint8)
    active[ix] = 1
    hour = np.zeros(B, dtype=np.int64)
    work = np.zeros(B, dtype=np.uint8)
    hour[ix] = self._tab(self._tab_hour, k_query)
    work[ix] = self._tab(self._tab_workday, k_query)
    out = np.zeros((B, Z), dtype=np.float64)
    occ = self._occ
    self._occ_lib.occupancy_query(
        self._occ_mt, self._occ_idx, self._occ_u8, active, hour, work, B, Z,
        occ["per_zone"], occ["arrive_lo"], occ["arrive_hi"], occ["depart_lo"],
        float(occ["p_arrival"]), float(occ["p_departure"]), out,
    )
    return out[ix]

  def _observe_vec(self, ix: np.ndarray) -> np.ndarray:
    k = self._k[ix]
    n = len(ix)
    obs = {}
    # Boiler and AHU passthrough fields, in request order.
    for field in self._boiler_fields:
      if field == "heating_request_count":
        v = self._boiler_count[ix].astype(np.float64)
      elif field == "supply_water_setpoint":
        v = self._boiler_sp[ix]
      elif field == "supply_water_temperature_sensor":
        v = self._boiler_sensor(ix, k)
      else:
        raise NotImplementedError(field)
      obs[self._ref._id_map[(self._boiler_id, field)]] = self._normalize(field, _f32(v))  # pylint: disable=protected-access
    ahu = self._ahu
    flow = self._ahu_flow[ix]
    for field in self._ahu_fields:
      if field == "cooling_request_count":
        v = self._ahu_count[ix].astype(np.float64)
      elif field == "differential_pressure_setpoint":
        v = np.full(n, ahu["dp"], dtype=np.float64)
      elif field in ("discharge_fan_speed_percentage_command",
                     "supply_fan_speed_percentage_command"):
        v = flow / ahu["max_flow"]
      elif field == "outside_air_flowrate_sensor":
        v = (1.0 - ahu["recirculation"]) * flow
      elif field == "outside_air_temperature_sensor":
        v = self._tab(self._tab_ambient, k)
      elif field == "supply_air_cooling_temperature_setpoint":
        v = self._ahu_cool_sp[ix]
      elif field == "supply_air_flowrate_sensor":
        v = flow
      elif field == "supply_air_heating_temperature_setpoint":
        v = self._ahu_heat_sp[ix]
      else:
        raise NotImplementedError(field)
      obs[self._ref._id_map[(self._ahu_id, field)]] = self._normalize(field, _f32(v))  # pylint: disable=protected-access
    # VAV fields: histograms (or passthrough per VAV).
    for field in self._vav_fields:
      if field == "supply_air_damper_percentage_command":
        v = self._damper[ix]
      elif field == "supply_air_flowrate_setpoint":
        v = np.broadcast_to(self._vav_max_flow, (n, len(self.zones)))
      elif field == "zone_air_temperature_sensor":
        v = self._zone_air_temp[ix]
      else:
        raise NotImplementedError(field)
      norm = self._normalize(field, _f32(v))
      if field in self._hist_bins:
        bins = self._hist_bins[field]
        h = self._histogram(norm, bins)
        for j, e in enumerate(bins):
          obs[f"{field}_h_{e:.2f}"] = h[:, j]
      else:
        for z, zone in enumerate(self.zones):
          obs[self._ref._id_map[(f"vav_{zone}", field)]] = norm[:, z]  # pylint: disable=protected-access
    for name, table in self._tab_time.items():
      obs[name] = self._tab(table, k)
    obs[env_lib.COMFORT_MODE_NOW] = self._tab(self._tab_comfort, k).astype(np.float32)
    obs[env_lib.COMFORT_MODE_SOON] = self._tab(self._tab_soon, k).astype(np.float32)
    occ = self._occupancy(ix, k - 1)
    occupants = np.zeros(n, dtype=np.float64)
    for z in range(len(self.zones)):
      occupants = occupants + occ[:, z]
    norm_c = self._occ_norm
    obs[env_lib.NUM_OCCUPANTS] = (
        (np.trunc(occupants).astype(np.int64) - norm_c) / (norm_c + 1)
    ).astype(np.float32)
    return np.stack(
        [np.asarray(obs[f], dtype=np.float32) for f in self.field_names],
        axis=1,
    )

  # ------------------------------------------------------------- reward
  def _build_reward_plan(self) -> None:
    rf = self._reward_fn
    probe = smart_control_reward_pb2.RewardInfo(
        start_timestamp=conversion_utils.pandas_to_proto_timestamp(
            self._tables.ts(0)
        ),
        end_timestamp=conversion_utils.pandas_to_proto_timestamp(
            self._tables.ts(1)
        ),
        zone_reward_infos={
            z: smart_control_reward_pb2.RewardInfo.ZoneRewardInfo()
            for z in self.zone_ids
        },
    )
    order = list(probe.zone_reward_infos)
    self._reward_order = [self.zone_ids.index(z) for z in order]
    start = conversion_utils.proto_to_pandas_timestamp(probe.start_timestamp)
    end = conversion_utils.proto_to_pandas_timestamp(probe.end_timestamp)
    self._reward_dt = (end - start).total_seconds()
    self._rw = {
        "max_p": rf._max_productivity_personhour_usd,  # pylint: disable=protected-access
        "min_p": rf._min_productivity_personhour_usd,  # pylint: disable=protected-access
        "delta": rf._productivity_midpoint_delta,  # pylint: disable=protected-access
        "stiff": rf._productivity_decay_stiffness,  # pylint: disable=protected-access
        "max_e": rf._max_electricity_rate,  # pylint: disable=protected-access
        "max_g": rf._max_natural_gas_rate,  # pylint: disable=protected-access
        "w_p": rf._productivity_weight,  # pylint: disable=protected-access
        "w_e": rf._energy_cost_weight,  # pylint: disable=protected-access
        "w_c": rf._carbon_emission_weight,  # pylint: disable=protected-access
    }
    boiler = self._boiler_obj
    interior = boiler._tank_radius  # pylint: disable=protected-access
    exterior = interior + boiler._insulation_thickness  # pylint: disable=protected-access
    self._diss = {
        "length": boiler._tank_length,  # pylint: disable=protected-access
        "conduction": np.log(exterior / interior) / boiler._insulation_conductivity,  # pylint: disable=protected-access
        "convection": 1.0 / boiler._convection_coefficient / exterior,  # pylint: disable=protected-access
        "capacity": boiler._water_capacity,  # pylint: disable=protected-access
    }

  def _reward_vec(self, ix: np.ndarray) -> np.ndarray:
    k = self._k[ix]
    n = len(ix)
    rw = self._rw
    dt = self._reward_dt
    occ = self._occupancy(ix, k)
    heat = _f32(self._tab(self._tab_heat, k))
    cool = _f32(self._tab(self._tab_cool, k))
    zone_t = _f32(self._cur_means_arr[ix])
    occ32 = _f32(occ)
    total_occ = np.zeros(n)
    productivity = np.zeros(n)
    for z in self._reward_order:
      zt = zone_t[:, z]
      o = occ32[:, z]
      total_occ = total_occ + o
      x0low = heat - rw["delta"]
      x0high = cool + rw["delta"]
      with np.errstate(over="ignore"):
        p_low = rw["max_p"] / (1.0 + np.exp(-rw["stiff"] * (zt - x0low)))
        p_high = rw["max_p"] * (
            1.0 - 1.0 / (1.0 + np.exp(-rw["stiff"] * (zt - x0high)))
        )
      p = np.where(zt < heat, p_low, np.where(zt > cool, p_high, rw["max_p"]))
      productivity = productivity + p * o * dt / 3600.0
    max_prod = rw["max_p"] * total_occ * dt / 3600.0
    min_prod = rw["min_p"] * total_occ * dt / 3600.0
    actual = np.where(min_prod > productivity, min_prod, productivity)
    with np.errstate(divide="ignore", invalid="ignore"):
      regret = np.where(
          total_occ > 0.0,
          (actual - min_prod) / (max_prod - min_prod) - 1.0,
          0.0,
      )
    # Air handler and boiler reward infos (proto float32).
    flow = self._ahu_flow[ix]
    ahu = self._ahu
    blower = _f32(
        flow * ahu["dp"] / ahu["eff"]
        + flow * (1.0 - ahu["recirculation"]) * ahu["dp"] / ahu["eff"]
    )
    ambient = self._tab(self._tab_ambient, k)
    recirc = self._cur_recirc_arr[ix]
    mixed = ahu["recirculation"] * recirc + (1 - ahu["recirculation"]) * ambient
    supply = np.where(
        mixed > self._ahu_cool_sp[ix],
        self._ahu_cool_sp[ix],
        np.where(mixed < self._ahu_heat_sp[ix], self._ahu_heat_sp[ix], mixed),
    )
    ac = _f32(flow * constants.AIR_HEAT_CAPACITY * (supply - mixed))
    rwt = self._boiler_rwt[ix]
    sp = self._boiler_sp[ix]
    water = np.where(sp > rwt, sp, rwt)
    if np.any(water < ambient):
      raise ValueError("Water temperature must be >= outside temperature.")
    d = self._diss
    dissipation = (d["length"] * 2.0 * np.pi * (water - ambient)) / (
        d["conduction"] + d["convection"]
    )
    bflow = self._boiler_flow[ix]
    flow_heating = constants.WATER_HEAT_CAPACITY * bflow * (water - rwt)
    last = self._boiler_last_dur[ix]
    with np.errstate(divide="ignore", invalid="ignore"):
      tank = np.where(
          last > 0,
          constants.WATER_HEAT_CAPACITY * d["capacity"]
          * self._boiler_change[ix] / last,
          0.0,
      )
    gas = _f32(flow_heating + dissipation + tank)
    pump = _f32(
        bflow * constants.WATER_DENSITY * constants.GRAVITY
        * self._boiler["head"] / self._boiler["eff"]
    )
    electricity = 0.0 + (blower + np.abs(ac))
    electricity = electricity + pump
    capped_e = np.where(rw["max_e"] < electricity, rw["max_e"], electricity)
    eprice = self._tab(self._tab_eprice, k)
    ecarbon = self._tab(self._tab_ecarbon, k)
    actual_e_cost = eprice * np.abs(capped_e) * dt
    max_e_cost = eprice * np.abs(rw["max_e"]) * dt
    actual_e_carbon = ecarbon * np.abs(capped_e) * dt
    max_e_carbon = ecarbon * np.abs(rw["max_e"]) * dt
    gas_sum = 0.0 + gas
    capped_g = np.where(rw["max_g"] < gas_sum, rw["max_g"], gas_sum)
    gprice = self._tab(self._tab_gprice, k)
    g_rate = np.where(capped_g < 0.0, 0.0, capped_g)
    max_g_rate = 0.0 if rw["max_g"] < 0.0 else rw["max_g"]
    actual_g_cost = gprice * (g_rate * dt)
    max_g_cost = gprice * (max_g_rate * dt)
    actual_g_carbon = self._gas_carbon_rate * (g_rate * dt)
    max_g_carbon = self._gas_carbon_rate * (max_g_rate * dt)
    energy = (actual_e_cost + actual_g_cost) / (max_e_cost + max_g_cost)
    carbon = (actual_e_carbon + actual_g_carbon) / (max_e_carbon + max_g_carbon)
    raw = regret * rw["w_p"] - energy * rw["w_e"] - carbon * rw["w_c"]
    value = raw / (rw["w_p"] + rw["w_e"] + rw["w_c"])
    # Competition comfort penalty (competition.html): 0 inside the zone's
    # heating/cooling band, d kelvin outside it up to 5 K, then quadratic with
    # no lower limit. The quadratic is not specified there; (d^2 + 25) / 10
    # joins d at 5 K with the same value and slope. Averaged over occupied
    # zones; 0 when no zone is occupied.
    zt64 = zone_t.astype(np.float64)
    d = (np.maximum(heat[:, None] - zt64, 0.0)
         + np.maximum(zt64 - cool[:, None], 0.0))
    penalty = np.where(d <= 5.0, d, (d * d + 25.0) / 10.0)
    occupied = occ32 > 0.0
    n_occupied = occupied.sum(axis=1)
    comfort_penalty = np.where(
        n_occupied > 0,
        np.where(occupied, penalty, 0.0).sum(axis=1) / np.maximum(n_occupied, 1),
        0.0,
    )
    if self.reward == "competition":
      value = -0.6 * comfort_penalty - 0.2 * energy - 0.2 * carbon
    # Monitoring copies of the reward terms (RewardResponse fields); they do
    # not feed back into the simulation or the reward.
    terms = {
        "normalized_productivity_regret": regret,
        "normalized_energy_cost": energy,
        "normalized_carbon_emission": carbon,
        "electricity_energy_cost_usd": actual_e_cost,
        "natural_gas_energy_cost_usd": actual_g_cost,
        "carbon_emitted_kg": actual_e_carbon + actual_g_carbon,
        "total_occupancy": total_occ,
        "electrical_power_w": electricity,
        "natural_gas_power_w": gas_sum,
        "comfort_penalty_k": comfort_penalty,
    }
    if not hasattr(self, "last_reward_terms"):
      self.last_reward_terms = {
          name: np.zeros(self.batch) for name in terms
      }
    for name, values in terms.items():
      self.last_reward_terms[name][ix] = values
    return value.astype(np.float32)


  def _swaps_feed(self, ix: np.ndarray, active: np.ndarray) -> None:
    """Records each env's MT words on the CPU; the GPU generates the swaps."""
    plan = self._conv_plan
    B, R = self.batch, int(plan.n_rooms)
    if self._feed_words is None:
      self._feed_words = np.zeros((B, 6 * int(plan.cvs.size)), dtype=np.uint32)
    offsets = np.zeros((B, R + 1), dtype=np.int64)
    for b in ix:
      mt = np.ascontiguousarray(self._conv_mt[b])
      index = np.array([self._conv_index[b]], dtype=np.int64)
      n = self._conv_kernel.record_swap_words(
          mt, index, self._conv_p, R, plan.room_offsets, plan.cand_offsets,
          self._feed_words[b], self._feed_words.shape[1], offsets[b],
      )
      if n < 0:
        raise RuntimeError("Feed buffer too small for the recorded words.")
      self._conv_mt[b] = mt
      self._conv_index[b] = index[0]
    self._gpu.swaps_feed(self._feed_words, offsets, active)

  # --------------------------------------------------------------- step
  def _make_fd(self):
    result = super()._make_fd()
    B, Z = self.batch, len(self.zones)
    self._cur_means_arr = np.zeros((B, Z), dtype=np.float64)
    self._cur_recirc_arr = np.zeros(B, dtype=np.float64)
    return result

  def _reset_grid(self, b: int) -> None:
    super()._reset_grid(b)
    self._cur_means_arr[b] = np.array(self._reset_means, dtype=np.float64)
    self._cur_recirc_arr[b] = float(self._reset_recirc)

  def _setup_vec(self, ix: np.ndarray) -> None:
    k = self._k[ix]
    heat = self._tab(self._tab_heat, k)[:, None]
    cool = self._tab(self._tab_cool, k)[:, None]
    comfort = self._tab(self._tab_comfort, k)[:, None]
    prev = self._thermo_prev_k[ix]
    prev_valid = prev > -(10**9)
    prev_comfort = np.zeros(len(ix), dtype=bool)
    prev_comfort[prev_valid] = self._tab(self._tab_comfort, prev[prev_valid])
    prev_flag = (prev_valid & prev_comfort)[:, None]
    zt = self._cur_means_arr[ix]
    mode = self._thermo_mode[ix]
    mid = 0.5 * (cool - heat) + heat
    default = np.full(mode.shape, sb1_exact._MODE_OFF)  # pylint: disable=protected-access
    default = np.where((zt > mid) & (mode == sb1_exact._MODE_COOL), sb1_exact._MODE_COOL, default)  # pylint: disable=protected-access
    default = np.where((zt < mid) & (mode == sb1_exact._MODE_HEAT), sb1_exact._MODE_HEAT, default)  # pylint: disable=protected-access
    default = np.where(zt > cool, sb1_exact._MODE_COOL, default)  # pylint: disable=protected-access
    default = np.where(zt < heat, sb1_exact._MODE_HEAT, default)  # pylint: disable=protected-access
    passive = sb1_exact._MODE_PASSIVE_COOL  # pylint: disable=protected-access
    eco = np.where((mode == passive) & (zt > heat), passive, default)
    new = np.where(comfort, default, np.where(prev_flag, passive, eco))
    self._thermo_mode[ix] = new
    heat_m = new == sb1_exact._MODE_HEAT  # pylint: disable=protected-access
    cool_m = new == sb1_exact._MODE_COOL  # pylint: disable=protected-access
    self._damper[ix] = np.where(heat_m | cool_m, 1.0, 0.1)
    self._reheat[ix] = np.where(heat_m, 1.0, 0.0)
    self._zone_air_temp[ix] = zt
    self._thermo_prev_k[ix] = k

  def _actions_vec(self, ix: np.ndarray, actions: np.ndarray) -> None:
    k = self._k[ix]
    for i, (normalizer, (device_id, setpoint)) in enumerate(
        zip(self._action_normalizers, self._action_targets)
    ):
      a = actions[:, i].astype(np.float32).astype(np.float64)
      lo, hi = normalizer._min_normalized_value, normalizer._max_normalized_value  # pylint: disable=protected-access
      tol = normalizer._tolerance  # pylint: disable=protected-access
      if np.any((a < (lo - tol)) | (a > (hi + tol))):
        raise ValueError(f"agent_action not within bounds [{lo}, {hi}]")
      ratio = (a - lo) / (hi - lo)
      out_range = normalizer._max_native_value - normalizer._min_native_value  # pylint: disable=protected-access
      value = _f32(ratio * out_range + normalizer._min_native_value)  # pylint: disable=protected-access
      if device_id == self._boiler_id and setpoint in sb1_exact._BOILER_ACTIONS:  # pylint: disable=protected-access
        self._boiler_sp[ix] = value
        self._boiler_action_arr[ix] = k
      elif device_id == self._ahu_id and setpoint in sb1_exact._AHU_ACTIONS:  # pylint: disable=protected-access
        if sb1_exact._AHU_ACTIONS[setpoint] == "heating_setpoint":  # pylint: disable=protected-access
          self._ahu_heat_sp[ix] = value
        else:
          self._ahu_cool_sp[ix] = value
      else:
        raise NotImplementedError(f"Action {device_id} {setpoint}.")

  def _hvac_vec(self, ix: np.ndarray, supply: np.ndarray) -> np.ndarray:
    zt = self._cur_means_arr[ix]
    damper, reheat = self._damper[ix], self._reheat[ix]
    if np.any(damper <= 0):
      raise ValueError("Damper setting must be greater than 0.")
    self._zone_air_temp[ix] = zt
    reheat_flow = reheat * self._vav_reheat_max
    air_flow = damper * self._vav_max_flow
    heat_difference = (
        constants.AIR_HEAT_CAPACITY * air_flow
        - constants.WATER_HEAT_CAPACITY * reheat_flow
    )
    water = self._boiler_sp[ix][:, None]
    input_water_heat = water * constants.WATER_HEAT_CAPACITY * reheat_flow
    zst = (
        (supply[:, None] * heat_difference + input_water_heat)
        / air_flow
        / constants.AIR_HEAT_CAPACITY
    )
    q = air_flow * constants.AIR_HEAT_CAPACITY * (zst - zt)
    n = len(ix)
    flow = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    bflow = np.zeros(n)
    bcount = np.zeros(n, dtype=np.int64)
    max_flow = self._ahu["max_flow"]
    num = np.zeros(n)
    den = np.zeros(n)
    for z in range(len(self.zones)):
      fd = damper[:, z] * self._vav_max_flow[z]
      m = fd > 0
      flow = np.where(m, flow + fd, flow)
      flow = np.where(m & (flow > max_flow), max_flow, flow)
      count += m
      rd = reheat[:, z] * self._vav_reheat_max[z]
      m = rd > 0
      bflow = np.where(m, bflow + rd, bflow)
      bcount += m
    for z in range(len(self.zones)):
      num = num + reheat[:, z] * zst[:, z]
      den = den + reheat[:, z]
    self._ahu_flow[ix] = flow
    self._ahu_count[ix] = count
    self._boiler_flow[ix] = bflow
    self._boiler_count[ix] = bcount
    self._boiler_rwt[ix] = num / (den + 1e-6)
    return q

  def _reset_vec(self, mask: np.ndarray) -> np.ndarray:
    self._started = True
    self._reset_device_state(mask)
    ix = np.flatnonzero(mask)
    self._episode_ended[ix] = False
    obs = self._observe_vec(ix)
    self._step_count[ix] = 0
    return obs

  def reset(self) -> np.ndarray:
    return self._reset_vec(np.ones(self.batch, dtype=bool))

  def step(
      self, actions: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, Z = self.batch, len(self.zones)
    actions = np.asarray(actions)
    obs = np.zeros((B, len(self.field_names)), dtype=np.float32)
    rewards = np.zeros(B, dtype=np.float32)
    step_types = np.full(B, sb1_exact.STEP_FIRST, dtype=np.int32)
    ended = self._episode_ended.copy()
    if ended.any():
      obs[ended] = self._reset_vec(ended)
    active = ~ended
    ix = np.flatnonzero(active)
    if ix.size == 0:
      return obs, rewards, step_types
    k = self._k[ix]
    self._setup_vec(ix)
    self._actions_vec(ix, actions[ix])
    ambient = self._tab(self._tab_ambient, k)
    recirc = self._cur_recirc_arr[ix]
    rec = self._ahu["recirculation"]
    mixed = rec * recirc + (1 - rec) * ambient
    supply = np.where(
        mixed > self._ahu_cool_sp[ix],
        self._ahu_cool_sp[ix],
        np.where(mixed < self._ahu_heat_sp[ix], self._ahu_heat_sp[ix], mixed),
    )
    q = np.zeros((B, Z))
    q[ix] = self._hvac_vec(ix, supply)
    amb_all = np.zeros(B, dtype=np.float32)
    conv_all = np.zeros(B, dtype=np.float32)
    amb_all[ix] = ambient
    conv_all[ix] = self._tab(self._tab_conv, k)
    self._gpu.fd(amb_all, conv_all, active)
    if self.rng == "exact":
      list(self._pool.map(self._generate_swaps, ix))
      self._gpu.swaps(self._swap_a, self._swap_b, self._swap_off, active)
    elif self.rng == "feed":
      self._swaps_feed(ix, active)
    else:
      self._gpu.swaps_fast(self._philox_keys, self._philox_step, active)
      self._philox_step[ix] += 1
    self._gpu.thermal_power(q, active)
    means = self._gpu.means()
    self._cur_means_arr[ix] = means[ix, :Z].astype(np.float64)
    self._cur_recirc_arr[ix] = means[ix, Z].astype(np.float64)
    self._fresh[ix] = False
    self._k[ix] = k + 1
    obs[ix] = self._observe_vec(ix)
    rewards[ix] = self._reward_vec(ix)
    done = self._step_count[ix] >= self._num_timesteps
    self._episode_ended[ix] = done
    step_types[ix] = np.where(done, sb1_exact.STEP_LAST, sb1_exact.STEP_MID)
    self._step_count[ix[~done]] += 1
    return obs, rewards, step_types
