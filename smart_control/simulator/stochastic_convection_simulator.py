"""Stochastic simulator of convection flow in bldg.

A convection simulator that randomly shuffles control volumes as a stochastic
imitation of convection.
We specify probability of a control volume getting shuffled, as well as maximum
distance that any given control volume can be moved.
"""

import collections
import copy
import ctypes
import hashlib
import os
import random
import subprocess
import tempfile
from typing import MutableSequence, Optional

from absl import logging
import gin
import numpy as np

from smart_control.simulator import base_convection_simulator

_KERNEL_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "convection_shuffle.c"
)
_EXCLUDED_ROOMS = ("exterior_space", "interior_wall")
_kernel = None
_kernel_failed = False


def _load_kernel() -> Optional[ctypes.CDLL]:
  """Compiles convection_shuffle.c once and returns it, or None if unable."""
  global _kernel, _kernel_failed
  if _kernel is not None or _kernel_failed:
    return _kernel
  try:
    with open(_KERNEL_SOURCE, "rb") as f:
      digest = hashlib.sha256(f.read()).hexdigest()[:16]
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "smart_control",
    )
    os.makedirs(cache_dir, exist_ok=True)
    lib_path = os.path.join(cache_dir, f"convection_shuffle_{digest}.so")
    if not os.path.exists(lib_path):
      fd, tmp_path = tempfile.mkstemp(suffix=".so", dir=cache_dir)
      os.close(fd)
      subprocess.run(
          [
              "gcc",
              "-O2",
              "-ffp-contract=off",
              "-shared",
              "-fPIC",
              _KERNEL_SOURCE,
              "-o",
              tmp_path,
          ],
          check=True,
          capture_output=True,
      )
      os.replace(tmp_path, lib_path)
    lib = ctypes.CDLL(lib_path)
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    lib.shuffle_rooms.restype = None
    lib.shuffle_rooms.argtypes = [
        np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        i64p,
        ctypes.c_double,
        ctypes.c_int64,
        i64p,
        i64p,
        i64p,
        i64p,
        np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        i64p,
        i64p,
    ]
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    lib.generate_swaps.restype = None
    lib.generate_swaps.argtypes = [
        np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        i64p,
        ctypes.c_double,
        ctypes.c_int64,
        i64p,
        i64p,
        i64p,
        i64p,
        i32p,
        i32p,
        i64p,
    ]
    u32p = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
    lib.record_swap_words.restype = ctypes.c_int64
    lib.record_swap_words.argtypes = [
        u32p,
        i64p,
        ctypes.c_double,
        ctypes.c_int64,
        i64p,
        i64p,
        u32p,
        ctypes.c_int64,
        i64p,
    ]
    _kernel = lib
  except (OSError, subprocess.CalledProcessError) as e:
    logging.warning(
        "Compiled convection shuffle unavailable, using Python: %s", e
    )
    _kernel_failed = True
  return _kernel


class _ShufflePlan:
  """Rooms and per-CV candidates of _shuffle_max_dist as flat arrays."""

  def __init__(self, room_dict, max_dist: int, shape: tuple[int, int]):
    rooms = [v for k, v in room_dict.items() if k not in _EXCLUDED_ROOMS]
    n_rows, n_cols = shape
    sizes = [len(v) for v in rooms]
    self.n_rooms = len(rooms)
    self.room_offsets = np.zeros(self.n_rooms + 1, dtype=np.int64)
    self.room_offsets[1:] = np.cumsum(sizes)
    coords = np.array([cv for v in rooms for cv in v], dtype=np.int64).reshape(
        -1, 2
    )
    room_ids = np.repeat(np.arange(self.n_rooms, dtype=np.int64), sizes)

    labels = np.full(shape, -1, dtype=np.int64)
    labels[coords[:, 0], coords[:, 1]] = room_ids
    # _shuffle_max_dist caches candidates per CV, so a CV in two rooms reuses
    # its first candidates. That case is left to the Python path.
    self.valid = bool(
        len(coords) == 0
        or (
            np.array_equal(labels[coords[:, 0], coords[:, 1]], room_ids)
            and len(np.unique(coords[:, 0] * n_cols + coords[:, 1]))
            == len(coords)
        )
    )

    # Offsets in the order of the nested loops in _shuffle_max_dist.
    offsets = [
        (d0, d1)
        for d0 in range(-max_dist, max_dist)
        for d1 in range(-max_dist, max_dist)
        if d0**2 + d1**2 <= max_dist
    ]
    member = np.zeros((len(coords), len(offsets)), dtype=bool)
    flat = np.zeros((len(coords), len(offsets)), dtype=np.int64)
    for o, (d0, d1) in enumerate(offsets):
      r = coords[:, 0] + d0
      c = coords[:, 1] + d1
      inside = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
      rc = np.where(inside, r, 0)
      cc = np.where(inside, c, 0)
      member[:, o] = inside & (labels[rc, cc] == room_ids)
      flat[:, o] = rc * n_cols + cc
    counts = member.sum(axis=1)
    self.cvs = coords[:, 0] * n_cols + coords[:, 1]
    self.cand_offsets = np.zeros(len(coords) + 1, dtype=np.int64)
    self.cand_offsets[1:] = np.cumsum(counts)
    self.cands = np.ascontiguousarray(flat[member])
    max_room = max(sizes) if sizes else 1
    self.swap_a = np.zeros(max_room, dtype=np.int64)
    self.swap_b = np.zeros(max_room, dtype=np.int64)


@gin.configurable
class StochasticConvectionSimulator(
    base_convection_simulator.BaseConvectionSimulator
):
  """Stochastic method of simulating air convection.

  Attributes:
    _p: shuffling probability
    _distance: max distance a CV can move
  """

  def __init__(
      self,
      p: float,
      distance: int,
      seed: Optional[int],
      use_compiled_shuffle: bool = True,
  ):
    """Initializes the stochastic convection simulator.

    Args:
      p: shuffling probability
      distance: max distance a CV can move
      seed: random seed
      use_compiled_shuffle: run _shuffle_max_dist in convection_shuffle.c when
        it compiles. Results and the `random` state match the Python path.
    """
    self._p = p
    self._distance = distance
    # cache is used to store precalculated shuffling patterns
    # so we do not need to calculate them again
    self._cache = collections.defaultdict(lambda: {})
    self._use_compiled_shuffle = use_compiled_shuffle
    self._plan = None
    self._plan_key = None

    if seed is not None:
      random.seed(seed)

  def _apply_compiled(self, room_dict, temp: np.ndarray) -> bool:
    """Runs _shuffle_max_dist for all rooms in C. Returns False if unable."""
    if not self._use_compiled_shuffle or self._distance <= 0:
      return False
    if (
        not isinstance(temp, np.ndarray)
        or temp.dtype != np.float32
        or not temp.flags.c_contiguous
        or temp.ndim != 2
    ):
      return False
    kernel = _load_kernel()
    if kernel is None:
      return False
    key = (id(room_dict), self._distance, temp.shape)
    if self._plan_key != key:
      self._plan = _ShufflePlan(room_dict, self._distance, temp.shape)
      self._plan_key = key
    plan = self._plan
    if not plan.valid:
      return False

    version, internal, gauss_next = random.getstate()
    mt = np.array(internal[:-1], dtype=np.uint32)
    index = np.array([internal[-1]], dtype=np.int64)
    kernel.shuffle_rooms(
        mt,
        index,
        float(self._p),
        plan.n_rooms,
        plan.room_offsets,
        plan.cvs,
        plan.cand_offsets,
        plan.cands,
        temp.reshape(-1),
        plan.swap_a,
        plan.swap_b,
    )
    random.setstate(
        (version, tuple(int(x) for x in mt) + (int(index[0]),), gauss_next)
    )
    return True

  def apply_convection(
      self,
      room_dict: dict[str, MutableSequence[tuple[int, int]]],
      temp: np.ndarray,
  ) -> None:
    """Applies convection to the temperature array given.

    Splits up rooms via room_dict.

    Args:
      room_dict: A dictionary mapping of room coordinates.
      temp: An array of temperatures.
    """
    p = self._p
    distance = self._distance
    if p == 0 or distance == 0:
      return

    if not (distance == -1 and p == 1) and self._apply_compiled(
        room_dict, temp
    ):
      return

    for k, v in room_dict.items():
      if k == "exterior_space":
        continue
      if k == "interior_wall":
        continue
      if distance == -1 and p == 1:  # special case, can be more efficient
        self._shuffle_no_max_dist(v, temp)
      else:
        self._shuffle_max_dist(p, v, distance, temp)

  def _shuffle_no_max_dist(self, v, temp):
    """Special case of shuffling when no max dist is specified.and p=1.

    Args:
      v: list of CVs to shuffle
      temp: temperature array
    """
    v = copy.deepcopy(v)
    vals = {}
    for cv in v:
      vals[cv] = temp[cv[0], cv[1]]
    v_shuffle = copy.deepcopy(v)
    random.shuffle(v_shuffle)
    for i, _ in enumerate(v_shuffle):
      cv = v_shuffle[i]
      val = vals[v[i]]
      temp[cv[0], cv[1]] = val

  def _shuffle_max_dist(self, p, v, max_dist, temp):
    """Special case of shuffling when max dist is specified.

    Args:
      p: shuffle probability
      v: list of CVs to shuffle
      max_dist: max distance to shuffle
      temp: temperature array
    """
    if max_dist == -1:
      max_dist = 1000
    cv_is_in_v = {}
    for val in v:
      cv_is_in_v[val] = True

    swap_list = []
    for val in v:
      if random.uniform(0, 1) > p:
        continue
      if max_dist in self._cache and val in self._cache[max_dist]:
        candidates = self._cache[max_dist][val]
      else:
        candidates = []
        cv = val
        for cv_0 in range(cv[0] - max_dist, cv[0] + max_dist):
          for cv_1 in range(cv[1] - max_dist, cv[1] + max_dist):
            cv_2 = (cv_0, cv_1)
            # check if cv_2 is in the room
            if cv_2 not in cv_is_in_v:
              continue
            dist = (cv[0] - cv_2[0]) ** 2 + (cv[1] - cv_2[1]) ** 2
            if dist <= max_dist:
              candidates.append(cv_2)
        self._cache[max_dist][val] = candidates

      swap_list.append((val, random.choice(candidates)))
    random.shuffle(swap_list)

    for i, _ in enumerate(swap_list):
      cv_1 = swap_list[i][0]
      cv_2 = swap_list[i][1]

      val = temp[cv_1[0], cv_1[1]]
      temp[cv_1[0], cv_1[1]] = temp[cv_2[0], cv_2[1]]
      temp[cv_2[0], cv_2[1]] = val
