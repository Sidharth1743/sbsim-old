"""Measures the baseline schedule policy on the batched SB1 engine.

Runs the repo's baseline schedule (see batched/baseline_policy.py for the two
action orders) in num_envs environments for whole 1-day episodes and logs, in
smart_control/reinforcement_learning/experiment_results/<name>_<timestamp>/:
TensorBoard scalars and charts, episodes.csv, hourly.csv (reward by local
hour), step_profile_<variant>.csv, and summary_<variant>.json.

  python -m smart_control.reinforcement_learning.scripts.batched_baseline \
      --num-envs 64 --episodes-per-env 1 --rng fast
"""

import argparse
import datetime
import os
import time

os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# pylint: disable=g-import-not-at-top, wrong-import-position
import numpy as np

from smart_control.batched import baseline_policy
from smart_control.batched import run_logging
from smart_control.batched import sb1_vec
from smart_control.batched import vec_env
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment
from smart_control.utils import constants

RESULTS_DIR = os.path.abspath(
    os.path.join(
        constants.ROOT_DIR, "smart_control", "reinforcement_learning",
        "experiment_results",
    )
)


def run_variant(args, gin_path, variant, logger):
  """Runs one baseline variant; returns its summary."""
  ref_env = create_and_setup_environment(gin_path, metrics_path=None)
  engine = sb1_vec.SB1VecBatch(
      ref_env, args.num_envs, rng=args.rng, seed=args.seed,
      gpu_parts=args.gpu_parts, reward=args.reward,
  )
  venv = vec_env.SB1VectorEnv(engine)
  policy_env = create_and_setup_environment(gin_path, metrics_path=None)
  policy = baseline_policy.BatchedSchedulePolicy(
      policy_env, venv.action_names, variant
  )
  normalizers = [ref_env.action_normalizers[n] for n in venv.action_names]
  stats = run_logging.TimeOfDayStats(venv.episode_steps)
  tracker = run_logging.EpisodeTracker(venv.num_envs)
  setpoint_sum = np.zeros((24, venv.action_size))
  setpoint_count = np.zeros(24, dtype=np.int64)
  records = []
  obs = venv.reset()
  total_steps = args.episodes_per_env * venv.episode_steps
  start = time.time()
  for _ in range(total_steps):
    actions = policy(obs)
    obs, rewards, terminated, truncated, info = venv.step(actions)
    stats.add(rewards, info)
    records += tracker.add(rewards, truncated, terminated, info)
    native = np.stack(
        [[normalizers[i].setpoint_value(a[i]) for i in range(len(a))]
         for a in actions]
    )
    np.add.at(setpoint_sum, info.local_hour, native)
    np.add.at(setpoint_count, info.local_hour, 1)
  elapsed = time.time() - start
  env_steps = total_steps * venv.num_envs
  logger.episodes(env_steps, records, variant)
  logger.time_of_day(env_steps, stats, variant)
  logger.step_profile(stats, variant)
  returns = np.array([r["return"] for r in records])
  with np.errstate(invalid="ignore"):
    setpoints = setpoint_sum / setpoint_count[:, None]
  summary = {
      "variant": variant,
      "episodes": len(records),
      "episode_steps": venv.episode_steps,
      "episode_return_mean": returns.mean(),
      "episode_return_std": returns.std(ddof=1) if len(returns) > 1 else 0.0,
      "episode_return_ci95": (
          1.96 * returns.std(ddof=1) / np.sqrt(len(returns))
          if len(returns) > 1 else 0.0
      ),
      "episode_return_min": returns.min(),
      "episode_return_max": returns.max(),
      "env_steps": env_steps,
      "seconds": elapsed,
      "steps_per_second": env_steps / elapsed,
      "mean_setpoint_by_local_hour_K": {
          f"{h:02d}:00": dict(zip(venv.action_names, setpoints[h].tolist()))
          for h in range(24) if setpoint_count[h]
      },
  }
  for name in run_logging._TERMS:  # pylint: disable=protected-access
    summary[f"episode_sum_{name}_mean"] = float(
        np.mean([r[f"sum_{name}"] for r in records])
    )
  logger.summary(variant, summary)
  logger.scalars(env_steps, {
      f"{variant}/summary_episode_return_mean": summary["episode_return_mean"],
      f"{variant}/steps_per_second": summary["steps_per_second"],
  })
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--episodes-per-env", type=int, default=1)
  parser.add_argument("--rng", choices=("fast", "exact"), default="fast")
  parser.add_argument("--reward", choices=("sbsim", "competition"), default="sbsim")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gpu-parts", type=int, default=1)
  parser.add_argument("--variants", default="intended,as_run")
  parser.add_argument("--experiment-name", default="baseline_schedule")
  parser.add_argument(
      "--gin", default=os.path.join(CONFIG_PATH, "sim_config_1_day.gin")
  )
  args = parser.parse_args()
  stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
  run_dir = os.path.join(RESULTS_DIR, f"{args.experiment_name}_{stamp}")
  logger = run_logging.RunLogger(run_dir, vars(args))
  for variant in args.variants.split(","):
    s = run_variant(args, args.gin, variant, logger)
    print(
        f"{variant:9s} return {s['episode_return_mean']:.4f} "
        f"± {s['episode_return_ci95']:.4f} (95% CI, {s['episodes']} episodes, "
        f"min {s['episode_return_min']:.4f}, max {s['episode_return_max']:.4f})"
        f"  {s['steps_per_second']:.0f} SPS"
    )
  print(f"run directory: {run_dir}")


if __name__ == "__main__":
  main()
