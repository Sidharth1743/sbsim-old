"""PPO on the batched SB1 engine (CleanRL-style, TensorFlow 2).

Single-file PPO with a Gaussian policy for continuous actions, following
CleanRL's ppo_continuous_action: orthogonal init, state-independent log std,
GAE, clipped policy and value losses, advantage normalization, gradient-norm
clipping, learning-rate annealing, and running observation / reward
normalization.

The 1-day episode ends on a time limit, so the last reward of an episode is
bootstrapped with gamma * V(final observation) and GAE stops at the boundary.

Everything is written to
smart_control/reinforcement_learning/experiment_results/<name>_<timestamp>/:
config.json, tensorboard/, episodes.csv (train and eval episodes), hourly.csv
(reward by local hour), step_profile_eval_*.csv, summary_eval_*.json, and:

* checkpoints/ckpt-<update>: agent, optimizer and observation statistics every
  --checkpoint-interval updates (all kept).
* checkpoints/best/: the same, saved whenever the deterministic eval return
  improves; summary_best.json records its update, step and return.
* resume/: the full training state (also the reward-scaling statistics,
  counters and best eval; latest only). --resume <run_dir> continues that run
  with its original config (--total-timesteps may be raised; the learning
  rate anneals over the new total). The environments restart at fresh
  episodes with a new seed, so a resumed run is not step-for-step identical
  to an uninterrupted one.

  python -m smart_control.reinforcement_learning.scripts.batched_ppo \
      --num-envs 64 --num-steps 128 --total-timesteps 2000000 \
      --baseline-return -30.77
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

from smart_control.batched import evaluation
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


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--experiment-name", default="ppo")
  p.add_argument("--gin", default=os.path.join(CONFIG_PATH, "sim_config_1_day.gin"))
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--rng", choices=("fast", "exact"), default="fast")
  p.add_argument("--reward", choices=("sbsim", "competition"), default="sbsim")
  p.add_argument("--gpu-parts", type=int, default=1)
  p.add_argument("--total-timesteps", type=int, default=2_000_000)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--num-steps", type=int, default=128, help="rollout length per env")
  p.add_argument("--learning-rate", type=float, default=3e-4)
  p.add_argument("--anneal-lr", type=int, default=1)
  p.add_argument("--gamma", type=float, default=0.99)
  p.add_argument("--gae-lambda", type=float, default=0.95)
  p.add_argument("--num-minibatches", type=int, default=32)
  p.add_argument("--update-epochs", type=int, default=10)
  p.add_argument("--norm-adv", type=int, default=1)
  p.add_argument("--clip-coef", type=float, default=0.2)
  p.add_argument("--clip-vloss", type=int, default=1)
  p.add_argument("--ent-coef", type=float, default=0.0)
  p.add_argument("--vf-coef", type=float, default=0.5)
  p.add_argument("--max-grad-norm", type=float, default=0.5)
  p.add_argument("--target-kl", type=float, default=None)
  p.add_argument("--hidden", default="256,256")
  p.add_argument("--norm-obs", type=int, default=1)
  p.add_argument("--norm-reward", type=int, default=1)
  p.add_argument("--eval-interval", type=int, default=10, help="updates between evals (0: off)")
  p.add_argument("--eval-envs", type=int, default=16)
  p.add_argument("--eval-seed", type=int, default=12345, help="eval at step t uses seed eval_seed + t")
  p.add_argument("--checkpoint-interval", type=int, default=25, help="updates between checkpoints")
  p.add_argument("--log-interval", type=int, default=1, help="updates between train logs")
  p.add_argument("--resume", default=None, help="run directory to continue")
  p.add_argument("--baseline-return", type=float, default=None,
                 help="episode return of the baseline to draw next to eval returns")
  return p.parse_args()


class RunningMeanStd:
  """Parallel running mean/variance (as in CleanRL / gym wrappers)."""

  def __init__(self, shape=()):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = 1e-4

  def update(self, x):
    x = np.asarray(x, dtype=np.float64)
    batch_mean, batch_var, n = x.mean(axis=0), x.var(axis=0), x.shape[0]
    delta = batch_mean - self.mean
    total = self.count + n
    self.mean = self.mean + delta * n / total
    m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
    self.var = m2 / total
    self.count = total


class ObsNormalizer:
  def __init__(self, size, enabled, clip=10.0):
    self.rms = RunningMeanStd((size,))
    self.enabled = enabled
    self.clip = clip

  def __call__(self, obs, update):
    if not self.enabled:
      return obs.astype(np.float32)
    if update:
      self.rms.update(obs)
    out = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
    return np.clip(out, -self.clip, self.clip).astype(np.float32)


class RewardScaler:
  """Scales rewards by the running std of the discounted return."""

  def __init__(self, num_envs, gamma, enabled, clip=10.0):
    self.rms = RunningMeanStd(())
    self.returns = np.zeros(num_envs)
    self.gamma = gamma
    self.enabled = enabled
    self.clip = clip

  def __call__(self, rewards, done):
    if not self.enabled:
      return rewards.astype(np.float32)
    self.returns = self.returns * self.gamma + rewards
    self.rms.update(self.returns)
    self.returns[done] = 0.0
    out = rewards / np.sqrt(self.rms.var + 1e-8)
    return np.clip(out, -self.clip, self.clip).astype(np.float32)

  def scale(self, values):
    if not self.enabled:
      return values
    return values / np.sqrt(self.rms.var + 1e-8)


def mlp(sizes, out, out_gain, name):
  layers = []
  for h in sizes:
    layers.append(tf.keras.layers.Dense(
        h, activation="tanh",
        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
        bias_initializer="zeros"))
  layers.append(tf.keras.layers.Dense(
      out, kernel_initializer=tf.keras.initializers.Orthogonal(out_gain),
      bias_initializer="zeros"))
  return tf.keras.Sequential(layers, name=name)


class Agent(tf.Module):
  """Actor-critic with a Gaussian policy."""

  def __init__(self, obs_size, act_size, hidden):
    super().__init__()
    self.actor_mean = mlp(hidden, act_size, 0.01, "actor_mean")
    self.critic = mlp(hidden, 1, 1.0, "critic")
    self.actor_logstd = tf.Variable(tf.zeros((1, act_size)), name="actor_logstd")
    dummy = tf.zeros((1, obs_size))
    self.actor_mean(dummy)
    self.critic(dummy)

  def value(self, obs):
    return tf.squeeze(self.critic(obs), -1)

  def dist_params(self, obs):
    mean = self.actor_mean(obs)
    log_std = tf.broadcast_to(self.actor_logstd, tf.shape(mean))
    return mean, log_std

  @staticmethod
  def log_prob(actions, mean, log_std):
    z = (actions - mean) / tf.exp(log_std)
    return tf.reduce_sum(-0.5 * z * z - log_std - 0.5 * np.log(2 * np.pi), -1)

  @staticmethod
  def entropy(log_std):
    return tf.reduce_sum(log_std + 0.5 * np.log(2 * np.pi * np.e), -1)


def main():
  args = parse_args()
  resumed_update = 0
  if args.resume:
    run_dir = os.path.abspath(args.resume)
    with open(os.path.join(run_dir, "config.json")) as f:
      saved = json.load(f)
    for k, v in saved.items():
      if hasattr(args, k) and k not in ("resume", "total_timesteps"):
        setattr(args, k, v)
    latest = tf.train.latest_checkpoint(os.path.join(run_dir, "resume"))
    if latest is None:
      raise FileNotFoundError(f"No resume checkpoint in {run_dir}/resume.")
    resumed_update = int(latest.rsplit("-", 1)[1])
  hidden = [int(h) for h in args.hidden.split(",") if h]
  batch_size = args.num_envs * args.num_steps
  minibatch_size = batch_size // args.num_minibatches
  num_updates = args.total_timesteps // batch_size
  if num_updates < 1 or minibatch_size < 1:
    raise ValueError("total-timesteps too small for num-envs * num-steps.")
  np.random.seed(args.seed + resumed_update)
  tf.random.set_seed(args.seed + resumed_update)

  if not args.resume:
    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    run_dir = os.path.join(RESULTS_DIR, f"{args.experiment_name}_{stamp}")
  config = dict(vars(args), batch_size=batch_size, minibatch_size=minibatch_size,
                num_updates=num_updates)
  logger = run_logging.RunLogger(run_dir, config, resume=bool(args.resume))

  ref_env = create_and_setup_environment(args.gin, metrics_path=None)
  engine = sb1_vec.SB1VecBatch(ref_env, args.num_envs, rng=args.rng,
                               seed=args.seed + resumed_update, gpu_parts=args.gpu_parts,
                               reward=args.reward)
  venv = vec_env.SB1VectorEnv(engine)
  normalizers = [ref_env.action_normalizers[n] for n in venv.action_names]
  setpoint_names = [target[1] for target in engine._action_targets]  # pylint: disable=protected-access
  eval_venv = None
  if args.eval_interval > 0:
    eval_ref = create_and_setup_environment(args.gin, metrics_path=None)
    eval_venv = vec_env.SB1VectorEnv(sb1_vec.SB1VecBatch(
        eval_ref, args.eval_envs, rng="fast", seed=args.eval_seed,
        reward=args.reward))

  agent = Agent(venv.observation_size, venv.action_size, hidden)
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=1e-5)
  obs_norm = ObsNormalizer(venv.observation_size, bool(args.norm_obs))
  rew_scale = RewardScaler(args.num_envs, args.gamma, bool(args.norm_reward))
  obs_rms_mean = tf.Variable(obs_norm.rms.mean, trainable=False, dtype=tf.float64)
  obs_rms_var = tf.Variable(obs_norm.rms.var, trainable=False, dtype=tf.float64)
  obs_rms_count = tf.Variable(obs_norm.rms.count, trainable=False, dtype=tf.float64)
  # Reward-scaling statistics: mean, var, count.
  rew_rms = tf.Variable([0.0, 1.0, 1e-4], trainable=False, dtype=tf.float64)
  # update, global_step; best eval return.
  counters = tf.Variable([0, 0], trainable=False, dtype=tf.int64)
  best_eval = tf.Variable(-np.inf, trainable=False, dtype=tf.float64)
  policy_state = dict(agent=agent, optimizer=optimizer,
                      obs_rms_mean=obs_rms_mean, obs_rms_var=obs_rms_var)
  ckpt = tf.train.Checkpoint(**policy_state)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, os.path.join(run_dir, "checkpoints"), max_to_keep=None)
  best_manager = tf.train.CheckpointManager(
      ckpt, os.path.join(run_dir, "checkpoints", "best"), max_to_keep=1)
  resume_ckpt = tf.train.Checkpoint(**policy_state, obs_rms_count=obs_rms_count,
                                    rew_rms=rew_rms, counters=counters,
                                    best_eval=best_eval)
  resume_manager = tf.train.CheckpointManager(
      resume_ckpt, os.path.join(run_dir, "resume"), max_to_keep=1)
  # Optimizer slots must exist before a checkpoint restores them.
  optimizer.build(agent.trainable_variables)

  @tf.function
  def act(obs):
    mean, log_std = agent.dist_params(obs)
    action = mean + tf.exp(log_std) * tf.random.normal(tf.shape(mean))
    return action, Agent.log_prob(action, mean, log_std), agent.value(obs)

  @tf.function
  def value_fn(obs):
    return agent.value(obs)

  @tf.function
  def train_minibatch(obs, actions, old_logprob, advantages, returns, old_values):
    with tf.GradientTape() as tape:
      mean, log_std = agent.dist_params(obs)
      new_logprob = Agent.log_prob(actions, mean, log_std)
      entropy = tf.reduce_mean(Agent.entropy(log_std))
      log_ratio = new_logprob - old_logprob
      ratio = tf.exp(log_ratio)
      approx_kl = tf.reduce_mean((ratio - 1.0) - log_ratio)
      clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > args.clip_coef, tf.float32))
      adv = advantages
      if args.norm_adv:
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)
      pg_loss = tf.reduce_mean(tf.maximum(
          -adv * ratio, -adv * tf.clip_by_value(ratio, 1 - args.clip_coef, 1 + args.clip_coef)))
      new_value = agent.value(obs)
      if args.clip_vloss:
        v_unclipped = tf.square(new_value - returns)
        v_clipped = old_values + tf.clip_by_value(new_value - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_unclipped, tf.square(v_clipped - returns)))
      else:
        v_loss = 0.5 * tf.reduce_mean(tf.square(new_value - returns))
      loss = pg_loss - args.ent_coef * entropy + v_loss * args.vf_coef
    variables = agent.trainable_variables
    grads = tape.gradient(loss, variables)
    grads, grad_norm = tf.clip_by_global_norm(grads, args.max_grad_norm)
    optimizer.apply_gradients(zip(grads, variables))
    return pg_loss, v_loss, entropy, approx_kl, clipfrac, grad_norm

  T, N, D, A = args.num_steps, args.num_envs, venv.observation_size, venv.action_size
  buf_obs = np.zeros((T, N, D), np.float32)
  buf_act = np.zeros((T, N, A), np.float32)
  buf_logp = np.zeros((T, N), np.float32)
  buf_rew = np.zeros((T, N), np.float32)
  buf_done = np.zeros((T, N), np.float32)
  buf_val = np.zeros((T, N), np.float32)

  global_step = 0
  if args.resume:
    resume_ckpt.restore(resume_manager.latest_checkpoint).assert_existing_objects_matched()
    obs_norm.rms.mean, obs_norm.rms.var = obs_rms_mean.numpy(), obs_rms_var.numpy()
    obs_norm.rms.count = float(obs_rms_count.numpy())
    rew_scale.rms.mean, rew_scale.rms.var, rew_scale.rms.count = (
        float(v) for v in rew_rms.numpy())
    resumed_update, global_step = (int(v) for v in counters.numpy())
    print(f"resumed {run_dir} at update {resumed_update} step {global_step} "
          f"(best eval {float(best_eval.numpy()):.4f})", flush=True)
  raw_obs = venv.reset()
  next_obs = obs_norm(raw_obs, update=not args.resume)
  next_done = np.zeros(N, np.float32)
  tracker = run_logging.EpisodeTracker(N)
  start, start_step = time.time(), global_step
  for update in range(resumed_update + 1, num_updates + 1):
    if args.anneal_lr:
      optimizer.learning_rate.assign(args.learning_rate * (1.0 - (update - 1.0) / num_updates))
    stats = run_logging.TimeOfDayStats(venv.episode_steps)
    setpoint_sum = np.zeros((24, A))
    setpoint_count = np.zeros(24, np.int64)
    episode_records = []
    rollout_start = time.time()
    for t in range(T):
      global_step += N
      buf_obs[t] = next_obs
      buf_done[t] = next_done
      action, logp, value = act(tf.constant(next_obs))
      action = action.numpy()
      buf_act[t], buf_logp[t], buf_val[t] = action, logp.numpy(), value.numpy()
      env_action = np.clip(action, -1.0, 1.0)
      raw_obs, rewards, terminated, truncated, info = venv.step(env_action)
      done = np.logical_or(terminated, truncated)
      stats.add(rewards, info)
      episode_records += tracker.add(rewards, truncated, terminated, info)
      native = np.stack([[normalizers[i].setpoint_value(a[i]) for i in range(A)] for a in env_action])
      np.add.at(setpoint_sum, info.local_hour, native)
      np.add.at(setpoint_count, info.local_hour, 1)
      scaled = rew_scale(rewards, done)
      if truncated.any():
        # Time limit: bootstrap from the final observation of the episode.
        final = obs_norm(info.final_observation[truncated], update=False)
        scaled[truncated] += args.gamma * value_fn(tf.constant(final)).numpy()
      buf_rew[t] = scaled
      next_obs = obs_norm(raw_obs, update=True)
      next_done = done.astype(np.float32)
    rollout_time = time.time() - rollout_start

    next_value = value_fn(tf.constant(next_obs)).numpy()
    advantages = np.zeros((T, N), np.float32)
    last_gae = np.zeros(N, np.float32)
    for t in reversed(range(T)):
      if t == T - 1:
        next_nonterminal = 1.0 - next_done
        next_values = next_value
      else:
        next_nonterminal = 1.0 - buf_done[t + 1]
        next_values = buf_val[t + 1]
      delta = buf_rew[t] + args.gamma * next_values * next_nonterminal - buf_val[t]
      last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
      advantages[t] = last_gae
    returns = advantages + buf_val

    b_obs = buf_obs.reshape(-1, D)
    b_act = buf_act.reshape(-1, A)
    b_logp = buf_logp.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_ret = returns.reshape(-1)
    b_val = buf_val.reshape(-1)
    learn_start = time.time()
    metrics = []
    for _ in range(args.update_epochs):
      idx = np.random.permutation(batch_size)
      stop = False
      for s in range(0, batch_size, minibatch_size):
        mb = idx[s:s + minibatch_size]
        out = train_minibatch(tf.constant(b_obs[mb]), tf.constant(b_act[mb]),
                              tf.constant(b_logp[mb]), tf.constant(b_adv[mb]),
                              tf.constant(b_ret[mb]), tf.constant(b_val[mb]))
        metrics.append([float(x) for x in out])
        if args.target_kl is not None and metrics[-1][3] > args.target_kl:
          stop = True
          break
      if stop:
        break
    learn_time = time.time() - learn_start
    obs_rms_mean.assign(obs_norm.rms.mean)
    obs_rms_var.assign(obs_norm.rms.var)

    m = np.mean(metrics, axis=0)
    var_y = np.var(b_ret)
    explained_var = np.nan if var_y == 0 else 1 - np.var(b_ret - b_val) / var_y
    if update % args.log_interval == 0:
      logger.scalars(global_step, {
          "losses/policy_loss": m[0], "losses/value_loss": m[1],
          "losses/entropy": m[2], "losses/approx_kl": m[3],
          "losses/clipfrac": m[4], "losses/grad_norm": m[5],
          "losses/explained_variance": explained_var,
          "charts/learning_rate": float(optimizer.learning_rate.numpy()),
          "charts/action_std_mean": float(np.exp(agent.actor_logstd.numpy()).mean()),
          "charts/SPS_total": (global_step - start_step) / (time.time() - start),
          "charts/SPS_env_rollout": T * N / rollout_time,
          "charts/seconds_learn": learn_time,
          **{f"charts/setpoint_K/{setpoint_names[i]}/{h:02d}:00":
             setpoint_sum[h, i] / setpoint_count[h]
             for h in range(24) if setpoint_count[h] for i in range(A)},
      })
      if episode_records:
        logger.episodes(global_step, episode_records, "train")
      logger.time_of_day(global_step, stats, "train")
    print(f"update {update}/{num_updates} step {global_step} "
          f"SPS {(global_step - start_step) / (time.time() - start):.0f} "
          f"kl {m[3]:.4f} ev {explained_var:.3f}"
          + (f" train_return {np.mean([r['return'] for r in episode_records]):.3f}"
             if episode_records else ""), flush=True)

    if eval_venv is not None and (update % args.eval_interval == 0 or update == num_updates):
      def greedy(obs):
        mean, _ = agent.dist_params(tf.constant(obs_norm(obs, update=False)))
        return np.clip(mean.numpy(), -1.0, 1.0)

      ret = evaluation.evaluate_policy(greedy, eval_venv, logger, global_step,
                                       f"u{update:05d}", args.baseline_return,
                                       seed=args.eval_seed + global_step)
      print(f"  eval return {ret:.4f}"
            + (f" (baseline {args.baseline_return:.4f})" if args.baseline_return is not None else ""),
            flush=True)
      if ret > float(best_eval.numpy()):
        best_eval.assign(ret)
        path = best_manager.save(checkpoint_number=update)
        logger.summary("best", {"update": update, "global_step": global_step,
                                "eval_return": float(ret), "checkpoint": path})
    if update % args.checkpoint_interval == 0 or update == num_updates:
      ckpt_manager.save(checkpoint_number=update)
      obs_rms_count.assign(obs_norm.rms.count)
      rew_rms.assign([float(rew_scale.rms.mean), float(rew_scale.rms.var),
                      float(rew_scale.rms.count)])
      counters.assign([update, global_step])
      resume_manager.save(checkpoint_number=update)
  print(f"run directory: {run_dir}")


if __name__ == "__main__":
  main()
