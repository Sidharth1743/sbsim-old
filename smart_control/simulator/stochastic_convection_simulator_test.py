"""Tests for stochastic_convection_simulator."""

import random

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from smart_control.simulator import stochastic_convection_simulator


def _room_dict():
  """Two rectangular rooms, an L-shaped room, and the excluded keys."""
  rooms = {
      "exterior_space": [(0, c) for c in range(30)],
      "interior_wall": [(r, 10) for r in range(1, 20)],
      "room_1": [(r, c) for r in range(1, 20) for c in range(1, 10)],
      "room_2": [(r, c) for r in range(1, 8) for c in range(11, 29)],
      "room_3": [(r, c) for r in range(8, 20) for c in range(11, 29)] + [
          (r, c) for r in range(20, 24) for c in range(11, 15)
      ],
  }
  return rooms


class StochasticConvectionSimulatorTest(parameterized.TestCase):

  @parameterized.parameters(
      (1.0, 5, 5),
      (0.7, 3, 123),
      (0.3, 8, 9),
      (1.0, 1, 2),
  )
  def test_compiled_shuffle_matches_python(self, p, distance, seed):
    if stochastic_convection_simulator._load_kernel() is None:  # pylint: disable=protected-access
      self.skipTest("gcc is not available to build convection_shuffle.c")
    rng = np.random.default_rng(seed)
    base = (290.0 + rng.random((25, 30))).astype(np.float32)
    results = []
    for use_compiled_shuffle in (False, True):
      sim = stochastic_convection_simulator.StochasticConvectionSimulator(
          p, distance, seed=None, use_compiled_shuffle=use_compiled_shuffle
      )
      random.seed(seed)
      temp = base.copy()
      # Two calls, so the second one starts from the advanced random state.
      sim.apply_convection(_room_dict(), temp)
      sim.apply_convection(_room_dict(), temp)
      results.append((temp, random.getstate()))

    (python_temp, python_state), (compiled_temp, compiled_state) = results  # pylint: disable=unbalanced-tuple-unpacking
    self.assertFalse(np.array_equal(python_temp, base))
    np.testing.assert_array_equal(python_temp, compiled_temp)
    self.assertEqual(python_state, compiled_state)

  def test_float64_temps_use_python_path(self):
    sim = stochastic_convection_simulator.StochasticConvectionSimulator(
        1.0, 5, seed=None
    )
    temp = np.full((25, 30), 290.0, dtype=np.float64)
    self.assertFalse(sim._apply_compiled(_room_dict(), temp))  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
