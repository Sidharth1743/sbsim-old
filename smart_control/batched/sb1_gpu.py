"""Loader for sb1_gpu.cu: GPU-resident grids of a batch of SB1 environments."""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from typing import Optional

from absl import logging
import numpy as np

from smart_control.simulator import cuda_fd

_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_DIR, "sb1_gpu.cu")
_FD_SOURCE = cuda_fd._SOURCE  # pylint: disable=protected-access
_lib = None
_lib_failed = False

MEAN_CHUNK = 8192


def load_library() -> Optional[ctypes.CDLL]:
  """Compiles sb1_gpu.cu (with fd_kernel.cu) once; None if unable."""
  global _lib, _lib_failed
  if _lib is not None or _lib_failed:
    return _lib
  try:
    h = hashlib.sha256()
    for path in (_SOURCE, _FD_SOURCE):
      with open(path, "rb") as f:
        h.update(f.read())
    h.update(" ".join(cuda_fd._NVCC_FLAGS).encode())  # pylint: disable=protected-access
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "smart_control",
    )
    os.makedirs(cache_dir, exist_ok=True)
    lib_path = os.path.join(cache_dir, f"sb1_gpu_{h.hexdigest()[:16]}.so")
    if not os.path.exists(lib_path):
      fd, tmp_path = tempfile.mkstemp(suffix=".so", dir=cache_dir)
      os.close(fd)
      subprocess.run(
          ["nvcc", *cuda_fd._NVCC_FLAGS, _SOURCE, "-o", tmp_path],  # pylint: disable=protected-access
          check=True,
          capture_output=True,
      )
      os.replace(tmp_path, lib_path)
    lib = ctypes.CDLL(lib_path)
    f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    vp = ctypes.c_void_p
    # The FD entry points come from the same library.
    lib.fd_create.restype = vp
    lib.fd_create.argtypes = cuda_fd.load_library().fd_create.argtypes
    lib.fd_destroy.restype = None
    lib.fd_destroy.argtypes = [vp]
    lib.fd_error_string.restype = ctypes.c_char_p
    lib.fd_error_string.argtypes = [ctypes.c_int]
    lib.sb1_create.restype = vp
    lib.sb1_create.argtypes = [
        vp, ctypes.c_int, i32p, i64p, i64p, ctypes.c_int, i32p, i64p, i64p,
        i32p, ctypes.c_int, i32p, i32p, f64p, ctypes.c_int, ctypes.c_int64,
    ]
    lib.sb1_destroy.restype = None
    lib.sb1_destroy.argtypes = [vp]
    lib.sb1_set_env.restype = ctypes.c_int
    lib.sb1_set_env.argtypes = [vp, ctypes.c_int, f32p]
    lib.sb1_get_env.restype = ctypes.c_int
    lib.sb1_get_env.argtypes = [vp, ctypes.c_int, f32p]
    lib.sb1_fd.restype = ctypes.c_int
    lib.sb1_fd.argtypes = [vp, f32p, f32p, u8p]
    lib.sb1_swaps.restype = ctypes.c_int
    lib.sb1_swaps.argtypes = [vp, i32p, i32p, i64p, u8p]
    lib.sb1_means.restype = ctypes.c_int
    lib.sb1_means.argtypes = [vp, f32p]
    lib.sb1_thermal_power.restype = ctypes.c_int
    lib.sb1_thermal_power.argtypes = [vp, f64p, u8p]
    u32p = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
    u64p = np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")
    lib.sb1_set_plan.restype = ctypes.c_int
    lib.sb1_set_plan.argtypes = [
        vp, i64p, ctypes.c_int64, i32p, i64p, ctypes.c_int64, i32p,
        ctypes.c_double,
    ]
    lib.sb1_swaps_feed.restype = ctypes.c_int
    lib.sb1_swaps_feed.argtypes = [vp, u32p, ctypes.c_int64, i64p, u8p]
    lib.sb1_swaps_philox.restype = ctypes.c_int
    lib.sb1_swaps_philox.argtypes = [vp, u32p, u64p, u8p]
    lib.sb1_philox_test.restype = ctypes.c_int
    lib.sb1_philox_test.argtypes = [u32p, u32p, u32p]
    lib.sb1_set_fast_plan.restype = ctypes.c_int
    lib.sb1_set_fast_plan.argtypes = [
        vp, i32p, ctypes.c_int64, ctypes.c_int, i32p, i32p, i32p,
    ]
    lib.sb1_swaps_fast.restype = ctypes.c_int
    lib.sb1_swaps_fast.argtypes = [vp, u32p, u64p, u8p]
    lib.sb1_set_defer_sync.restype = None
    lib.sb1_set_defer_sync.argtypes = [vp, ctypes.c_int]
    lib.sb1_sync.restype = ctypes.c_int
    lib.sb1_sync.argtypes = [vp, ctypes.c_void_p]
    _lib = lib
  except (OSError, subprocess.CalledProcessError) as e:
    detail = getattr(e, "stderr", b"") or b""
    logging.warning("sb1_gpu unavailable: %s %s", e, detail.decode()[-3000:])
    _lib_failed = True
  return _lib


def _check(lib, err: int, what: str) -> None:
  if err:
    raise RuntimeError(f"{what} failed: {lib.fd_error_string(err).decode()}")


def mean_tasks(segment_lengths: list[int]) -> tuple[np.ndarray, ...]:
  """Chunk tasks (segment, start, length) in segment then chunk order."""
  task_seg, task_start, task_len, first = [], [], [], [0]
  for seg, n in enumerate(segment_lengths):
    for start in range(0, n, MEAN_CHUNK):
      task_seg.append(seg)
      task_start.append(start)
      task_len.append(min(MEAN_CHUNK, n - start))
    first.append(len(task_seg))
  return (
      np.array(task_seg, dtype=np.int32),
      np.array(task_start, dtype=np.int64),
      np.array(task_len, dtype=np.int64),
      np.array(first, dtype=np.int32),
  )


class Sb1Gpu:
  """GPU grids of a batch of environments sharing one floor plan."""

  def __init__(
      self,
      fd_kwargs: dict,
      zone_cells: list[np.ndarray],
      diffusers: list[tuple[np.ndarray, np.ndarray]],
      n_rooms: int,
      max_swaps: int,
  ):
    lib = load_library()
    if lib is None:
      raise RuntimeError("sb1_gpu library unavailable.")
    self._lib = lib
    # fd_create through the shared library, with CudaFdSolver's arguments.
    solver_args = cuda_fd.CudaFdSolver.prepare_create_args(**fd_kwargs)
    fd = lib.fd_create(*solver_args)
    if not fd:
      raise RuntimeError("fd_create failed.")
    self._fd = ctypes.c_void_p(fd)
    self.batch = fd_kwargs["batch"]
    self.shape = tuple(fd_kwargs["shape"])
    self.cells = self.shape[0] * self.shape[1]
    self.n_zones = len(zone_cells)
    self.n_rooms = n_rooms
    self.max_swaps = max_swaps
    lengths = [len(c) for c in zone_cells] + [self.cells]
    seg_start = np.zeros(self.n_zones + 1, dtype=np.int64)
    seg_start[1:] = np.cumsum(lengths[:-1])
    seg_start[-1] = 0  # grid segment is contiguous over the env's grid
    seg_len = np.array(lengths, dtype=np.int64)
    task_seg, task_start, task_len, first = mean_tasks(lengths)
    dif_cell = np.concatenate([d[0] for d in diffusers]).astype(np.int32)
    dif_zone = np.concatenate(
        [np.full(len(d[0]), z, dtype=np.int32) for z, d in enumerate(diffusers)]
    )
    dif_val = np.concatenate([d[1] for d in diffusers]).astype(np.float64)
    handle = lib.sb1_create(
        self._fd,
        self.n_zones,
        np.concatenate(zone_cells).astype(np.int32),
        seg_start,
        seg_len,
        len(task_seg),
        task_seg,
        task_start,
        task_len,
        first,
        len(dif_cell),
        dif_cell,
        dif_zone,
        dif_val,
        n_rooms,
        max_swaps,
    )
    if not handle:
      raise RuntimeError("sb1_create failed.")
    self._h = ctypes.c_void_p(handle)
    self._means = np.empty((self.batch, self.n_zones + 1), dtype=np.float32)

  def __del__(self):
    if getattr(self, "_h", None) is not None and self._h.value:
      self._lib.sb1_destroy(self._h)
      self._h = None
    if getattr(self, "_fd", None) is not None and self._fd.value:
      self._lib.fd_destroy(self._fd)
      self._fd = None

  def set_env(self, b: int, grid: np.ndarray) -> None:
    g = np.ascontiguousarray(grid, dtype=np.float32)
    _check(self._lib, self._lib.sb1_set_env(self._h, b, g), "sb1_set_env")

  def get_env(self, b: int) -> np.ndarray:
    out = np.empty(self.shape, dtype=np.float32)
    _check(self._lib, self._lib.sb1_get_env(self._h, b, out), "sb1_get_env")
    return out

  def fd(self, temp_inf, conv, active) -> None:
    _check(
        self._lib,
        self._lib.sb1_fd(
            self._h,
            np.ascontiguousarray(temp_inf, dtype=np.float32),
            np.ascontiguousarray(conv, dtype=np.float32),
            np.ascontiguousarray(active, dtype=np.uint8),
        ),
        "sb1_fd",
    )

  def swaps(self, swap_a, swap_b, swap_off, active) -> None:
    _check(
        self._lib,
        self._lib.sb1_swaps(
            self._h, swap_a, swap_b, swap_off,
            np.ascontiguousarray(active, dtype=np.uint8),
        ),
        "sb1_swaps",
    )

  def means(self) -> np.ndarray:
    """(batch, n_zones + 1) float32: zone means, then the grid mean."""
    _check(self._lib, self._lib.sb1_means(self._h, self._means), "sb1_means")
    return self._means.copy()

  def thermal_power(self, q: np.ndarray, active) -> None:
    _check(
        self._lib,
        self._lib.sb1_thermal_power(
            self._h,
            np.ascontiguousarray(q, dtype=np.float64),
            np.ascontiguousarray(active, dtype=np.uint8),
        ),
        "sb1_thermal_power",
    )

  def set_plan(self, plan, p: float) -> None:
    """Uploads the convection shuffle plan for GPU swap generation."""
    _check(
        self._lib,
        self._lib.sb1_set_plan(
            self._h,
            np.ascontiguousarray(plan.room_offsets, dtype=np.int64),
            int(plan.cvs.size),
            np.ascontiguousarray(plan.cvs, dtype=np.int32),
            np.ascontiguousarray(plan.cand_offsets, dtype=np.int64),
            int(plan.cands.size),
            np.ascontiguousarray(plan.cands, dtype=np.int32),
            float(p),
        ),
        "sb1_set_plan",
    )

  def swaps_feed(self, words, feed_off, active) -> None:
    """GPU swap generation from fed words (batch, max) and offsets."""
    _check(
        self._lib,
        self._lib.sb1_swaps_feed(
            self._h,
            np.ascontiguousarray(words, dtype=np.uint32),
            int(words.shape[1]),
            np.ascontiguousarray(feed_off, dtype=np.int64),
            np.ascontiguousarray(active, dtype=np.uint8),
        ),
        "sb1_swaps_feed",
    )

  def swaps_philox(self, keys, steps, active) -> None:
    """GPU swap generation from Philox; keys (batch, 2), steps (batch,)."""
    _check(
        self._lib,
        self._lib.sb1_swaps_philox(
            self._h,
            np.ascontiguousarray(keys, dtype=np.uint32),
            np.ascontiguousarray(steps, dtype=np.uint64),
            np.ascontiguousarray(active, dtype=np.uint8),
        ),
        "sb1_swaps_philox",
    )


  def set_fast_plan(self, plan) -> None:
    """Uploads room-local candidate indices and room size classes."""
    room_sizes = np.diff(plan.room_offsets)
    if room_sizes.max() >= 65536:
      raise ValueError("Fast mode needs rooms with fewer than 65536 CVs.")
    local_of = np.full(self.cells, -1, dtype=np.int64)
    for r in range(int(plan.n_rooms)):
      a, e = plan.room_offsets[r], plan.room_offsets[r + 1]
      local_of[plan.cvs[a:e]] = np.arange(e - a)
    cand_local = local_of[plan.cands]
    if np.any(cand_local < 0):
      raise ValueError("Candidate outside its room.")
    limits = (2048, 8192, 16384, 65535)
    classes, counts, maxes = [], [], []
    lower = 0
    for limit in limits:
      rooms = np.flatnonzero((room_sizes > lower) & (room_sizes <= limit))
      classes.append(rooms.astype(np.int32))
      counts.append(len(rooms))
      maxes.append(int(room_sizes[rooms].max()) if len(rooms) else 0)
      lower = limit
    if sum(counts) != int(plan.n_rooms):
      raise ValueError("Rooms of size 0 are not supported in fast mode.")
    _check(
        self._lib,
        self._lib.sb1_set_fast_plan(
            self._h,
            cand_local.astype(np.int32),
            int(cand_local.size),
            len(limits),
            np.concatenate(classes).astype(np.int32),
            np.array(counts, dtype=np.int32),
            np.array(maxes, dtype=np.int32),
        ),
        "sb1_set_fast_plan",
    )

  def swaps_fast(self, keys, steps, active) -> None:
    """Fast-mode GPU swaps (shared-memory kernel)."""
    _check(
        self._lib,
        self._lib.sb1_swaps_fast(
            self._h,
            np.ascontiguousarray(keys, dtype=np.uint32),
            np.ascontiguousarray(steps, dtype=np.uint64),
            np.ascontiguousarray(active, dtype=np.uint8),
        ),
        "sb1_swaps_fast",
    )

  def set_defer_sync(self, defer: bool) -> None:
    """When set, fd/swaps_fast/thermal_power/means return without waiting."""
    self._lib.sb1_set_defer_sync(self._h, int(defer))

  def sync(self) -> np.ndarray:
    """Waits for queued work; returns the last means (batch, zones + 1)."""
    out = np.empty((self.batch, self.n_zones + 1), dtype=np.float32)
    _check(self._lib, self._lib.sb1_sync(self._h, out.ctypes.data), "sb1_sync")
    return out

def philox4x32_10(ctr, key) -> np.ndarray:
  """Runs the GPU Philox4x32-10 once (for known-answer tests)."""
  lib = load_library()
  out = np.zeros(4, dtype=np.uint32)
  err = lib.sb1_philox_test(
      np.asarray(ctr, dtype=np.uint32), np.asarray(key, dtype=np.uint32), out
  )
  _check(lib, err, "sb1_philox_test")
  return out


class Sb1GpuSplit:
  """Several Sb1Gpu contexts over contiguous env ranges, on separate streams.

  Same methods as Sb1Gpu. Step work (fd, swaps_fast, thermal_power, means) is
  enqueued on every part before any of them is waited on, so one part's
  latency-bound swaps can overlap another part's FD. Each part computes its
  envs exactly as a single context would.
  """

  def __init__(self, fd_kwargs, zone_cells, diffusers, n_rooms, max_swaps,
               parts: int = 2):
    B = fd_kwargs["batch"]
    parts = max(1, min(parts, B))
    bounds = np.linspace(0, B, parts + 1).round().astype(int)
    self._ranges = [(int(a), int(e)) for a, e in zip(bounds[:-1], bounds[1:])]
    self._parts = []
    for a, e in self._ranges:
      kw = dict(fd_kwargs)
      kw["batch"] = e - a
      part = Sb1Gpu(kw, zone_cells, diffusers, n_rooms, max_swaps)
      part.set_defer_sync(True)
      self._parts.append(part)
    first = self._parts[0]
    self.batch, self.shape, self.cells = B, first.shape, first.cells
    self.n_zones, self.n_rooms, self.max_swaps = first.n_zones, n_rooms, max_swaps

  def _route(self, b: int):
    for part, (a, e) in zip(self._parts, self._ranges):
      if a <= b < e:
        return part, b - a
    raise IndexError(b)

  def set_env(self, b, grid):
    part, i = self._route(b)
    part.sync()
    part.set_env(i, grid)

  def get_env(self, b):
    part, i = self._route(b)
    part.sync()
    return part.get_env(i)

  def set_plan(self, plan, p):
    for part in self._parts:
      part.set_plan(plan, p)

  def set_fast_plan(self, plan):
    for part in self._parts:
      part.set_fast_plan(plan)

  def fd(self, temp_inf, conv, active):
    for part, (a, e) in zip(self._parts, self._ranges):
      part.fd(temp_inf[a:e], conv[a:e], active[a:e])

  def swaps_fast(self, keys, steps, active):
    for part, (a, e) in zip(self._parts, self._ranges):
      part.swaps_fast(keys[a:e], steps[a:e], active[a:e])

  def swaps(self, swap_a, swap_b, swap_off, active):
    for part, (a, e) in zip(self._parts, self._ranges):
      part.swaps(np.ascontiguousarray(swap_a[a:e]), np.ascontiguousarray(swap_b[a:e]),
                 np.ascontiguousarray(swap_off[a:e]), active[a:e])

  def swaps_feed(self, words, feed_off, active):
    for part, (a, e) in zip(self._parts, self._ranges):
      part.swaps_feed(np.ascontiguousarray(words[a:e]),
                      np.ascontiguousarray(feed_off[a:e]), active[a:e])

  def thermal_power(self, q, active):
    for part, (a, e) in zip(self._parts, self._ranges):
      part.thermal_power(np.ascontiguousarray(q[a:e]), active[a:e])

  def means(self):
    for part in self._parts:
      part.means()
    return np.concatenate([part.sync() for part in self._parts], axis=0)
