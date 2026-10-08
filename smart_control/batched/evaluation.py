"""Deterministic evaluation episodes on an SB1VectorEnv, with run logging."""

import numpy as np

from smart_control.batched import run_logging


def evaluate_policy(policy, eval_venv, logger, step, tag_suffix, baseline_return,
                    seed=None):
  """Runs one episode in every eval env and logs it.

  Args:
    policy: maps raw observations (num_envs, obs) to actions in [-1, 1].
    eval_venv: SB1VectorEnv used only for evaluation.
    logger: RunLogger.
    step: global step for the logs.
    tag_suffix: suffix of the step-profile and summary files.
    baseline_return: baseline episode return drawn next to the policy, or None.
    seed: if set, the (fast-mode) eval envs get new random streams from it
      first, so each evaluation sees different occupancy and convection.

  Returns:
    Mean episode return.
  """
  stats = run_logging.TimeOfDayStats(eval_venv.episode_steps)
  tracker = run_logging.EpisodeTracker(eval_venv.num_envs)
  records = []
  if seed is not None:
    eval_venv.engine.reseed(seed)
  obs = eval_venv.reset()
  for _ in range(eval_venv.episode_steps):
    obs, rewards, terminated, truncated, info = eval_venv.step(policy(obs))
    stats.add(rewards, info)
    records += tracker.add(rewards, truncated, terminated, info)
  logger.episodes(step, records, "eval")
  logger.time_of_day(step, stats, "eval")
  logger.step_profile(stats, f"eval_{tag_suffix}")
  returns = np.array([r["return"] for r in records])
  values = {"eval/episode_return_mean_vs_baseline/policy": returns.mean()}
  if baseline_return is not None:
    values["eval/episode_return_mean_vs_baseline/baseline"] = baseline_return
  logger.scalars(step, values)
  logger.summary(f"eval_{tag_suffix}", {
      "global_step": step,
      "episodes": len(records),
      "episode_return_mean": returns.mean(),
      "episode_return_ci95": (1.96 * returns.std(ddof=1) / np.sqrt(len(returns))
                              if len(returns) > 1 else 0.0),
      "baseline_return": baseline_return,
  })
  return float(returns.mean())
