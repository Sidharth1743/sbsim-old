"""The repo's baseline schedule policy, applied to a batch of observations.

Wraps schedule_policy.create_baseline_schedule_policy and calls its own
_get_action_map and _normalize_actions per environment, so the schedule, the
hour/day decoding, and the normalization are the original code.

The original policy emits [air heating, supply water] (its action_sequence),
while the SB1 action vector is [supply water, air heating]. The repo feeds the
policy output to the environment unchanged, so each value lands on the other
setpoint. Two orders are offered:

* "intended": each scheduled setpoint goes to the env action with that
  setpoint name (what the schedule describes).
* "as_run": the policy's own order, as populate_starter_buffer.py runs it.
"""

import types

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from smart_control.environment import environment as env_lib
from smart_control.reinforcement_learning.policies import schedule_policy


class BatchedSchedulePolicy:
  """Baseline schedule for (num_envs, observation_size) observations."""

  def __init__(self, policy_env: env_lib.Environment, action_names, order: str):
    if order not in ("intended", "as_run"):
      raise ValueError(f"Unknown order {order}.")
    self.order = order
    tf_env = tf_py_environment.TFPyEnvironment(policy_env)
    self.policy = schedule_policy.create_baseline_schedule_policy(tf_env)
    sequence = list(self.policy.action_sequence)
    if order == "as_run":
      self._keys = sequence
    else:
      self._keys = []
      for name in action_names:
        matches = [key for key in sequence if name.endswith(key[1])]
        if len(matches) != 1:
          raise ValueError(f"No unique schedule setpoint for action {name}.")
        self._keys.append(matches[0])
    if len(self._keys) != len(action_names):
      raise ValueError("Schedule and action vector differ in length.")

  def __call__(self, observations: np.ndarray) -> np.ndarray:
    actions = np.zeros((len(observations), len(self._keys)), dtype=np.float32)
    for b, obs in enumerate(np.asarray(observations, dtype=np.float32)):
      # A TF tensor, as in the repo: its float32 element arithmetic decides
      # the decoded hour at hour boundaries (numpy scalars would promote to
      # float64 and can flip the floor).
      time_step = types.SimpleNamespace(observation=tf.constant(obs[None]))
      action_map = self.policy._get_action_map(time_step)  # pylint: disable=protected-access
      normalized = self.policy._normalize_actions(action_map)  # pylint: disable=protected-access
      actions[b] = [normalized[key] for key in self._keys]
    return actions
