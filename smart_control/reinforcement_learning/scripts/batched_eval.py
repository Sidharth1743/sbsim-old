"""Evaluates a trained PPO or SAC policy against the baseline schedule.

Both policies run deterministically on identical random streams, so each
env's return difference measures the policy alone:

* exact mode: the original simulator. Env 0 uses the original gin seeds
  (convection 5, occupancy 17321); env s > 0 uses convection seed 1000 + s
  and occupancy seed 5000 + s (the original code with other seeds).
* fast mode: per-env Philox streams from --fast-seed.

Results go to experiment_results/<name>_<timestamp>/: TensorBoard, episodes.csv,
hourly.csv, step_profile_<mode>_<policy>.csv, and summary_comparison.json.

The algorithm is read from the run's config.json (SAC runs have
updates_per_step). --checkpoint is a step number, "best" (SAC's
checkpoints/best), or omitted for the latest.

  python -m smart_control.reinforcement_learning.scripts.batched_eval \
      --run smart_control/reinforcement_learning/experiment_results/<run> \
      --checkpoint best --reward competition
"""

import argparse
import datetime
import json
import os
import time

os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# pylint: disable=g-import-not-at-top, wrong-import-position
import numpy as np
import tensorflow as tf

from smart_control.batched import baseline_policy
from smart_control.batched import run_logging
from smart_control.batched import sb1_vec
from smart_control.batched import vec_env
from smart_control.reinforcement_learning.scripts import batched_ppo
from smart_control.reinforcement_learning.scripts import batched_sac
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

ORIGINAL_CONVECTION_SEED = 5
ORIGINAL_OCCUPANCY_SEED = 17321


def load_policy(run_dir, checkpoint, obs_size, act_size):
  """Rebuilds the deterministic policy of a PPO or SAC run.

  Returns (policy, checkpoint path, config, algorithm name).
  """
  with open(os.path.join(run_dir, "config.json")) as f:
    config = json.load(f)
  algo = "sac" if "updates_per_step" in config else "ppo"
  hidden = [int(h) for h in str(config["hidden"]).split(",") if h]
  mean = tf.Variable(np.zeros(obs_size), dtype=tf.float64)
  var = tf.Variable(np.ones(obs_size), dtype=tf.float64)
  ckpt_dir = os.path.join(run_dir, "checkpoints")
  if checkpoint == "best":
    path = tf.train.latest_checkpoint(os.path.join(ckpt_dir, "best"))
  elif checkpoint:
    path = os.path.join(ckpt_dir, f"ckpt-{int(checkpoint)}")
  else:
    path = tf.train.latest_checkpoint(ckpt_dir)
  if path is None:
    raise FileNotFoundError(f"No checkpoint {checkpoint or ''} in {ckpt_dir}.")
  if algo == "ppo":
    net = batched_ppo.Agent(obs_size, act_size, hidden)
    status = tf.train.Checkpoint(agent=net, obs_rms_mean=mean, obs_rms_var=var).restore(path)
  else:
    net = batched_sac.Actor(obs_size, act_size, hidden)
    status = tf.train.Checkpoint(actor=net, obs_rms_mean=mean, obs_rms_var=var).restore(path)
  status.expect_partial()
  status.assert_existing_objects_matched()
  norm_obs = bool(config["norm_obs"])
  rms_mean, rms_var = mean.numpy(), var.numpy()

  def normalize(obs):
    if not norm_obs:
      return np.asarray(obs, dtype=np.float32)
    return np.clip((obs - rms_mean) / np.sqrt(rms_var + 1e-8), -10.0, 10.0).astype(np.float32)

  def act(obs):
    if algo == "ppo":
      mu, _ = net.dist_params(tf.constant(normalize(obs)))
    else:
      _, _, mu = net.sample(tf.constant(normalize(obs)))
    return np.clip(mu.numpy(), -1.0, 1.0)

  return act, path, config, algo


def run(mode, policy_name, args, logger):
  """Runs one policy for --days episodes in every env of one mode."""
  num_envs = args.exact_envs if mode == "exact" else args.fast_envs
  ref_env = create_and_setup_environment(args.gin, metrics_path=None)
  if mode == "exact":
    engine = sb1_vec.SB1VecBatch(ref_env, num_envs, rng="exact", reward=args.reward)
    engine.set_original_seeds(
        [ORIGINAL_CONVECTION_SEED] + [1000 + s for s in range(1, num_envs)],
        [ORIGINAL_OCCUPANCY_SEED] + [5000 + s for s in range(1, num_envs)],
    )
  else:
    engine = sb1_vec.SB1VecBatch(ref_env, num_envs, rng="fast", seed=args.fast_seed,
                                  reward=args.reward)
  venv = vec_env.SB1VectorEnv(engine)
  if policy_name != "baseline":
    policy, ckpt_path, _, _ = load_policy(args.run, args.checkpoint,
                                          venv.observation_size, venv.action_size)
  else:
    policy_env = create_and_setup_environment(args.gin, metrics_path=None)
    policy = baseline_policy.BatchedSchedulePolicy(policy_env, venv.action_names, "intended")
    ckpt_path = None
  stats = run_logging.TimeOfDayStats(venv.episode_steps)
  tracker = run_logging.EpisodeTracker(num_envs)
  records = []
  obs = venv.reset()
  start = time.time()
  for _ in range(args.days * venv.episode_steps):
    obs, rewards, terminated, truncated, info = venv.step(policy(obs))
    stats.add(rewards, info)
    records += tracker.add(rewards, truncated, terminated, info)
  tag = f"{mode}_{policy_name}"
  logger.episodes(0, records, tag)
  logger.time_of_day(0, stats, tag)
  logger.step_profile(stats, tag)
  per_env = np.zeros(num_envs)
  for r in records:
    per_env[r["env"]] += r["return"] / args.days
  terms = {
      name: float(np.mean([r[f"sum_{name}"] for r in records]))
      for name in run_logging._TERMS  # pylint: disable=protected-access
  }
  return {"per_env_return": per_env, "terms": terms, "checkpoint": ckpt_path,
          "seconds": time.time() - start, "hourly": stats.hourly_means()["reward"]}


def ci95(x):
  return float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", "--ppo-run", dest="run", required=True, help="PPO or SAC run directory")
  p.add_argument("--checkpoint", default=None, help='step number, "best", or omitted for the latest')
  p.add_argument("--modes", default="exact,fast")
  p.add_argument("--reward", choices=("sbsim", "competition"), default="sbsim")
  p.add_argument("--exact-envs", type=int, default=8)
  p.add_argument("--fast-envs", type=int, default=32)
  p.add_argument("--fast-seed", type=int, default=2024)
  p.add_argument("--days", type=int, default=1)
  p.add_argument("--experiment-name", default=None, help="default: eval_<algo>_vs_baseline")
  p.add_argument("--gin", default=os.path.join(batched_ppo.CONFIG_PATH, "sim_config_1_day.gin"))
  args = p.parse_args()
  args.run = os.path.abspath(args.run)
  with open(os.path.join(args.run, "config.json")) as f:
    algo = "sac" if "updates_per_step" in json.load(f) else "ppo"
  args.experiment_name = args.experiment_name or f"eval_{algo}_vs_baseline"
  stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
  run_dir = os.path.join(batched_ppo.RESULTS_DIR, f"{args.experiment_name}_{stamp}")
  logger = run_logging.RunLogger(run_dir, vars(args))
  summary = {}
  for mode in args.modes.split(","):
    res = {name: run(mode, name, args, logger) for name in (algo, "baseline")}
    pol, base = res[algo]["per_env_return"], res["baseline"]["per_env_return"]
    diff = pol - base
    summary[mode] = {
        "algorithm": algo,
        "envs": len(diff),
        "policy_return_mean": float(pol.mean()), "policy_return_ci95": ci95(pol),
        "baseline_return_mean": float(base.mean()), "baseline_return_ci95": ci95(base),
        "paired_difference_mean": float(diff.mean()),
        "paired_difference_ci95": ci95(diff),
        "envs_where_policy_better": int((diff > 0).sum()),
        "relative_improvement": float(diff.mean() / abs(base.mean())),
        "policy_terms": res[algo]["terms"], "baseline_terms": res["baseline"]["terms"],
        "reward_by_hour": {
            f"{h:02d}:00": {"policy": float(res[algo]["hourly"][h]),
                            "baseline": float(res["baseline"]["hourly"][h])}
            for h in range(24)
        },
        "checkpoint": res[algo]["checkpoint"],
    }
    s = summary[mode]
    print(f"{mode:5s} {algo.upper()} {s['policy_return_mean']:.4f} ± {s['policy_return_ci95']:.4f} | "
          f"baseline {s['baseline_return_mean']:.4f} ± {s['baseline_return_ci95']:.4f} | "
          f"paired diff {s['paired_difference_mean']:+.4f} ± {s['paired_difference_ci95']:.4f} "
          f"({100 * s['relative_improvement']:+.1f}%, {algo.upper()} better in "
          f"{s['envs_where_policy_better']}/{s['envs']} envs)")
  logger.summary("comparison", summary)
  print(f"run directory: {run_dir}")


if __name__ == "__main__":
  main()
