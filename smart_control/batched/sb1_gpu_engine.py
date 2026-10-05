"""Batched SB1 engine with GPU-resident grids (phase 3 of port_plan.md).

Same step semantics as SB1ExactBatch, which provides all scalar logic. The
temperature grid and the external heat of every environment live on the GPU
(sb1_gpu.cu). Only per-env vectors cross the bus each step:

* to the GPU: ambient temperature and convection coefficient, swap lists,
  and per-zone thermal power;
* to the host: zone means and the global mean.

Pre-step means are the post-step means of the previous step: nothing changes
the grid in between. After a reset they are the means of the float64 reset
grid, computed once on the host.

Exact mode: swap lists come from each env's copy of the Python `random`
stream (convection_shuffle.c generate_swaps) and the GPU applies them.
"""

from typing import Any

import numpy as np

from smart_control.batched import sb1_exact
from smart_control.batched import sb1_gpu


class SB1GpuBatch(sb1_exact.SB1ExactBatch):
  """B SB1 environments with GPU grids and exact original semantics."""

  def _make_fd(self):
    cols = self._shape[1]
    zone_cells = [r * cols + c for r, c in self._zone_rc]
    diffusers = [(r * cols + c, v) for r, c, v in self._zone_diffusers]
    plan = self._conv_plan
    self._max_swaps = int(plan.room_offsets[-1])
    parts = getattr(self, "_gpu_parts", 1)
    if parts > 1:
      self._gpu = sb1_gpu.Sb1GpuSplit(
          self._fd_kwargs, zone_cells, diffusers,
          n_rooms=int(plan.n_rooms), max_swaps=self._max_swaps, parts=parts,
      )
    else:
      self._gpu = sb1_gpu.Sb1Gpu(
          self._fd_kwargs,
          zone_cells,
          diffusers,
          n_rooms=int(plan.n_rooms),
          max_swaps=self._max_swaps,
      )
    B = self.batch
    self._swap_a = np.zeros((B, self._max_swaps), dtype=np.int32)
    self._swap_b = np.zeros((B, self._max_swaps), dtype=np.int32)
    self._swap_off = np.zeros((B, plan.n_rooms + 1), dtype=np.int64)
    self._reset_grid_f32 = self._reset_temp.astype(np.float32)
    self._reset_means = [np.mean(self._reset_temp[r, c]) for r, c in self._zone_rc]
    self._reset_recirc = self._reset_temp.mean()
    self._cur_means = [None] * B
    self._cur_recirc = [None] * B
    self._fresh = np.zeros(B, dtype=bool)
    return None

  def _reset_grid(self, b: int) -> None:
    # The FD reads the float64 reset grid as float32; the means before the
    # first FD come from the float64 grid itself.
    self._gpu.set_env(b, self._reset_grid_f32)
    self._cur_means[b] = list(self._reset_means)
    self._cur_recirc[b] = self._reset_recirc
    self._fresh[b] = True

  def _zone_means(self, b: int) -> list[Any]:
    return self._cur_means[b]

  def _pre_recirc(self, b: int):
    return self._cur_recirc[b]

  def _post_means(self, b: int) -> list[Any]:
    return self._cur_means[b]

  def _post_recirc(self, b: int):
    return self._cur_recirc[b]

  def temp(self, b: int) -> np.ndarray:
    if self._fresh[b]:
      return self._reset_temp
    return self._gpu.get_env(b)

  def _generate_swaps(self, b: int) -> None:
    plan = self._conv_plan
    mt = np.ascontiguousarray(self._conv_mt[b])
    index = np.array([self._conv_index[b]], dtype=np.int64)
    self._conv_kernel.generate_swaps(
        mt,
        index,
        self._conv_p,
        plan.n_rooms,
        plan.room_offsets,
        plan.cvs,
        plan.cand_offsets,
        plan.cands,
        self._swap_a[b],
        self._swap_b[b],
        self._swap_off[b],
    )
    self._conv_mt[b] = mt
    self._conv_index[b] = index[0]

  def step(
      self, actions: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, Z = self.batch, len(self.zones)
    obs = np.zeros((B, len(self.field_names)), dtype=np.float32)
    rewards = np.zeros(B, dtype=np.float32)
    step_types = np.zeros(B, dtype=np.int32)
    active = np.zeros(B, dtype=bool)
    for b in range(B):
      if self._episode_ended[b]:
        obs[b] = self._reset_env(b)
        step_types[b] = sb1_exact.STEP_FIRST
      else:
        active[b] = True
    stepping = np.flatnonzero(active)
    if stepping.size == 0:
      return obs, rewards, step_types
    q = np.zeros((B, Z), dtype=np.float64)
    ambient = np.zeros(B, dtype=np.float32)
    conv = np.zeros(B, dtype=np.float32)
    for b in stepping:
      t = self._tables.get(int(self._k[b]))
      means = self._cur_means[b]
      self._setup_step(b, means)
      self._apply_actions(b, np.asarray(actions[b], dtype=np.float32))
      supply_air_temp = self._supply_air_temp(
          b, self._cur_recirc[b], t["ambient"]
      )
      # Scalar device state only; the grid work below is independent of it.
      q[b] = self._hvac_outputs(b, means, supply_air_temp)
      ambient[b] = t["ambient"]
      conv[b] = t["conv"]
    self._gpu.fd(ambient, conv, active)
    for b in stepping:
      self._generate_swaps(b)
    self._gpu.swaps(self._swap_a, self._swap_b, self._swap_off, active)
    # input_q for the next FD; the FD above read the previous one.
    self._gpu.thermal_power(q, active)
    means = self._gpu.means()
    for b in stepping:
      self._cur_means[b] = [np.float32(m) for m in means[b, :Z]]
      self._cur_recirc[b] = np.float32(means[b, Z])
      self._fresh[b] = False
      self._k[b] += 1
      obs[b] = self._observe(b)
      rewards[b] = np.float32(self._reward(b))
      self._episode_ended[b] = self._step_count[b] >= self._num_timesteps
      if self._episode_ended[b]:
        step_types[b] = sb1_exact.STEP_LAST
      else:
        step_types[b] = sb1_exact.STEP_MID
        self._step_count[b] += 1
    return obs, rewards, step_types
