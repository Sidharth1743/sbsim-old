"""Loader for fd_kernel.cu, the CUDA finite-difference timestep.

The kernel is compiled with nvcc on first use and cached by source hash. It
reproduces the float32 results of TFSimulator's graph loop; see fd_kernel.cu.
"""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from typing import Optional

from absl import logging
import numpy as np

_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fd_kernel.cu"
)
_NVCC_FLAGS = (
    "-O3",
    "-arch=native",
    # Exactness: no multiply-add contraction, IEEE division, no flush to zero.
    "-fmad=false",
    "-prec-div=true",
    "-ftz=false",
    "-cudart",
    "static",
    "-shared",
    "-Xcompiler",
    "-fPIC",
)
# Iterations fused per kernel launch; must be one of these (see fd_kernel.cu).
SUPPORTED_FUSE_K = (1, 2, 4, 6, 8, 12, 16)
MAX_CLASSES = 256
_lib = None
_lib_failed = False


def load_library() -> Optional[ctypes.CDLL]:
  """Compiles fd_kernel.cu once and returns it, or None if unable."""
  global _lib, _lib_failed
  if _lib is not None or _lib_failed:
    return _lib
  try:
    with open(_SOURCE, "rb") as f:
      digest = hashlib.sha256(
          f.read() + " ".join(_NVCC_FLAGS).encode()
      ).hexdigest()[:16]
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "smart_control",
    )
    os.makedirs(cache_dir, exist_ok=True)
    lib_path = os.path.join(cache_dir, f"fd_kernel_{digest}.so")
    if not os.path.exists(lib_path):
      fd, tmp_path = tempfile.mkstemp(suffix=".so", dir=cache_dir)
      os.close(fd)
      subprocess.run(
          ["nvcc", *_NVCC_FLAGS, _SOURCE, "-o", tmp_path],
          check=True,
          capture_output=True,
      )
      os.replace(tmp_path, lib_path)
    lib = ctypes.CDLL(lib_path)
    f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    lib.fd_create.restype = ctypes.c_void_p
    lib.fd_create.argtypes = [
        ctypes.c_int,  # batch
        ctypes.c_int,  # rows
        ctypes.c_int,  # cols
        ctypes.c_int,  # limit
        ctypes.c_double,  # threshold
        ctypes.c_float,  # z
        ctypes.c_float,  # delta_t
        ctypes.c_int,  # fuse_k
        ctypes.c_int,  # n_classes
        f32p,  # kl
        f32p,  # kr
        f32p,  # kt
        f32p,  # kb
        f32p,  # u
        f32p,  # v
        f32p,  # density
        f32p,  # heat_capacity
        u8p,  # flags
        u8p,  # cls
    ]
    lib.fd_destroy.restype = None
    lib.fd_destroy.argtypes = [ctypes.c_void_p]
    lib.fd_step.restype = ctypes.c_int
    lib.fd_step.argtypes = [
        ctypes.c_void_p,
        f32p,  # temp_minus
        f32p,  # input_q
        f32p,  # temp_inf
        f32p,  # conv
        f32p,  # out
        i32p,  # n_iter
        f32p,  # max_deltas
    ]
    lib.fd_last_compute_ms.restype = ctypes.c_float
    lib.fd_last_compute_ms.argtypes = [ctypes.c_void_p]
    lib.fd_error_string.restype = ctypes.c_char_p
    lib.fd_error_string.argtypes = [ctypes.c_int]
    _lib = lib
  except (OSError, subprocess.CalledProcessError) as e:
    detail = getattr(e, "stderr", b"") or b""
    logging.warning(
        "CUDA FD kernel unavailable: %s %s", e, detail.decode()[-2000:]
    )
    _lib_failed = True
  return _lib


def _f32(x: np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(x, dtype=np.float32)


def build_cell_classes(
    per_cell: list[np.ndarray], flags: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
  """Groups cells by their exact constant tuple.

  Args:
    per_cell: float32 arrays of one shape, one per constant.
    flags: uint8 array of the same shape.

  Returns:
    One float32 array of per-class values per constant, the per-class flags,
    and the uint8 class index of every cell (flattened, row major).

  Raises:
    ValueError: if there are more than MAX_CLASSES distinct tuples.
  """
  columns = [_f32(a).reshape(-1).view(np.uint32) for a in per_cell]
  columns.append(np.ascontiguousarray(flags, dtype=np.uint8).reshape(-1))
  key = np.stack([c.astype(np.uint32) for c in columns], axis=1)
  unique, inverse = np.unique(key, axis=0, return_inverse=True)
  if unique.shape[0] > MAX_CLASSES:
    raise ValueError(
        f"{unique.shape[0]} distinct cell classes; at most {MAX_CLASSES}."
    )
  values = [
      np.ascontiguousarray(unique[:, k]).view(np.float32)
      for k in range(len(per_cell))
  ]
  class_flags = np.ascontiguousarray(unique[:, -1].astype(np.uint8))
  cls = np.ascontiguousarray(inverse.reshape(-1).astype(np.uint8))
  return values, class_flags, cls


class CudaFdSolver:
  """B environments that share one floor plan and materials, on the GPU."""

  def __init__(
      self,
      lib: ctypes.CDLL,
      batch: int,
      shape: tuple[int, int],
      iteration_limit: int,
      convergence_threshold: float,
      floor_height_m: float,
      time_step_sec: float,
      conductivity_left: np.ndarray,
      conductivity_right: np.ndarray,
      conductivity_top: np.ndarray,
      conductivity_bottom: np.ndarray,
      u: np.ndarray,
      v: np.ndarray,
      density: np.ndarray,
      heat_capacity: np.ndarray,
      flags: np.ndarray,
      fuse_k: int = 8,
  ):
    self._lib = lib
    self.batch = batch
    self.shape = tuple(shape)
    self.limit = iteration_limit
    args = self.prepare_create_args(
        batch,
        shape,
        iteration_limit,
        convergence_threshold,
        floor_height_m,
        time_step_sec,
        conductivity_left,
        conductivity_right,
        conductivity_top,
        conductivity_bottom,
        u,
        v,
        density,
        heat_capacity,
        flags,
        fuse_k,
    )
    self.n_classes = args[8]
    handle = lib.fd_create(*args)
    if not handle:
      raise RuntimeError("fd_create failed (CUDA allocation or arguments).")
    self._handle = ctypes.c_void_p(handle)
    self._out = np.empty((batch,) + self.shape, dtype=np.float32)
    self._n_iter = np.empty(batch, dtype=np.int32)
    self._deltas = np.empty((batch, iteration_limit), dtype=np.float32)

  @staticmethod
  def prepare_create_args(
      batch: int,
      shape: tuple[int, int],
      iteration_limit: int,
      convergence_threshold: float,
      floor_height_m: float,
      time_step_sec: float,
      conductivity_left: np.ndarray,
      conductivity_right: np.ndarray,
      conductivity_top: np.ndarray,
      conductivity_bottom: np.ndarray,
      u: np.ndarray,
      v: np.ndarray,
      density: np.ndarray,
      heat_capacity: np.ndarray,
      flags: np.ndarray,
      fuse_k: int = 8,
  ) -> tuple:
    """Returns the fd_create arguments, including the cell class table."""
    if fuse_k not in SUPPORTED_FUSE_K:
      raise ValueError(f"fuse_k must be one of {SUPPORTED_FUSE_K}.")
    values, class_flags, cls = build_cell_classes(
        [
            conductivity_left,
            conductivity_right,
            conductivity_top,
            conductivity_bottom,
            u,
            v,
            density,
            heat_capacity,
        ],
        flags,
    )
    return (
        batch,
        shape[0],
        shape[1],
        iteration_limit,
        float(convergence_threshold),
        np.float32(floor_height_m),
        np.float32(time_step_sec),
        fuse_k,
        len(class_flags),
        *values,
        class_flags,
        cls,
    )

  def __del__(self):
    handle = getattr(self, "_handle", None)
    if handle is not None and handle.value:
      self._lib.fd_destroy(handle)
      self._handle = None

  @property
  def last_compute_ms(self) -> float:
    """GPU time of the last step, excluding host transfers of the grids."""
    return float(self._lib.fd_last_compute_ms(self._handle))

  def step(
      self,
      temp_minus: np.ndarray,
      input_q: np.ndarray,
      temp_inf: np.ndarray,
      convection_coefficient: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs one timestep.

    Args:
      temp_minus: (batch, rows, cols) float32 temperatures at the last step.
      input_q: (batch, rows, cols) float32 external heat.
      temp_inf: (batch,) float32 ambient temperatures.
      convection_coefficient: (batch,) float32 convection coefficients.

    Returns:
      New temperatures (batch, rows, cols), iterations run (batch,), and the
      max_delta of every iteration (batch, limit; zero from n_iter on). The
      arrays are reused by the next call.
    """
    err = self._lib.fd_step(
        self._handle,
        _f32(temp_minus),
        _f32(input_q),
        _f32(temp_inf),
        _f32(convection_coefficient),
        self._out,
        self._n_iter,
        self._deltas,
    )
    if err:
      raise RuntimeError(
          "fd_step failed: " + self._lib.fd_error_string(err).decode()
      )
    return self._out, self._n_iter, self._deltas
