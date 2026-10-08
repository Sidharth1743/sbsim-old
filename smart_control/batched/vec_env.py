"""Vector-env interface over SB1VecBatch for training loops.

Differences from the engine's own step() (TF-Agents semantics):

* Same-step reset: when an environment's episode ends, step() returns that
  episode's last reward, marks it truncated, puts the last observation in
  info["final_observation"], and returns the first observation of the next
  episode in obs. TF-Agents instead resets on the following call and drops
  that call's action.
* The 1-day episode ends because of a time limit, so it is reported as
  truncated (bootstrap from the final observation), never terminated.

Each reward is labeled with the local time (the environment's time zone) of
the simulator timestamp right after the step: the RewardInfo start_timestamp
the reward is computed for.
"""

import dataclasses

import numpy as np

from smart_control.batched import sb1_exact
from smart_control.batched import sb1_vec


@dataclasses.dataclass
class StepInfo:
  """Per-step extras, one entry per environment."""

  final_observation: np.ndarray  # rows valid where truncated
  episode_step: np.ndarray  # step index k of the reward within its episode
  local_hour: np.ndarray
  local_minute: np.ndarray
  reward_terms: dict[str, np.ndarray]


class SB1VectorEnv:
  """num_envs SB1 environments with same-step reset."""

  def __init__(self, engine: sb1_vec.SB1VecBatch):
    self.engine = engine
    self.num_envs = engine.batch
    self.observation_size = len(engine.field_names)
    self.action_size = len(engine.action_names)
    self.field_names = list(engine.field_names)
    self.action_names = list(engine.action_names)
    tz = engine._ref._time_zone  # pylint: disable=protected-access
    self.time_zone = tz
    kmax = engine._kmax  # pylint: disable=protected-access
    local = [engine._tables.ts(k).tz_convert(tz) for k in range(kmax + 1)]  # pylint: disable=protected-access
    self._hour = np.array([t.hour for t in local], dtype=np.int64)
    self._minute = np.array([t.minute for t in local], dtype=np.int64)
    self.episode_steps = engine._num_timesteps + 1  # pylint: disable=protected-access
    self.start_local_time = local[0]

  def reset(self) -> np.ndarray:
    return self.engine.reset()

  def step(self, actions: np.ndarray):
    """Returns obs, rewards, terminated, truncated, info."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (self.num_envs, self.action_size):
      raise ValueError(f"actions must have shape {(self.num_envs, self.action_size)}")
    if self.engine._episode_ended.any():  # pylint: disable=protected-access
      raise RuntimeError("Engine has an ended episode; use this wrapper only.")
    obs, rewards, step_types = self.engine.step(actions)
    k = self.engine._k.copy()  # pylint: disable=protected-access
    terms = {
        name: values.copy()
        for name, values in self.engine.last_reward_terms.items()
    }
    truncated = step_types == sb1_exact.STEP_LAST
    terminated = np.zeros(self.num_envs, dtype=bool)
    final_obs = obs.copy()
    if truncated.any():
      obs = obs.copy()
      obs[truncated] = self.engine._reset_vec(truncated)  # pylint: disable=protected-access
    info = StepInfo(
        final_observation=final_obs,
        episode_step=k,
        local_hour=self._hour[k],
        local_minute=self._minute[k],
        reward_terms=terms,
    )
    return obs, rewards, terminated, truncated, info
