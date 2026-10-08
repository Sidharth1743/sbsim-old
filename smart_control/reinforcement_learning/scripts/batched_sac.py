"""SAC on the batched SB1 engine (CleanRL-style, TensorFlow 2).

Follows CleanRL's sac_continuous_action: twin Q networks with target
networks (Polyak tau), a tanh-squashed Gaussian actor with clamped log std,
delayed policy updates, and automatic entropy tuning (target entropy
-action_dim). Actions are already in [-1, 1].

By default the Q networks have LayerNorm after each hidden layer (as in RLPD,
DroQ, BRO and SimBa) and gradients are clipped to global norm 10. Without
them the critic diverged on SB1 (600k-step ablation, competition reward: Q
-6.6 and stable with them, -18,500 without; see changes.md Round 8).

Batched collection: every vector step adds num_envs transitions;
--updates-per-step gradient steps follow each vector step, so the
update-to-data ratio is updates_per_step / num_envs.

The 1-day episode ends on a time limit, so the replay buffer stores the real
final observation as next_obs and the transition is not terminal (Q
bootstraps through it). Raw observations are stored and normalized when
sampled with the current running statistics. The replay buffer lives in GPU
variables and each vector step runs its gradient steps in one compiled loop;
loss metrics are read back only at log steps.

Checkpoints (under the run directory):
* checkpoints/ckpt-<step>: policy, critics, optimizers, alpha and observation
  statistics every --checkpoint-interval transitions (all kept).
* checkpoints/best/: the same, saved whenever the deterministic eval return
  improves; summary_best.json records its step and return.
* resume/: the full training state including the replay buffer (only the
  latest is kept). --resume <run_dir> continues that run with its original
  config (--total-timesteps may be raised). The environments restart at fresh
  episodes with a new seed, so a resumed run is not step-for-step identical
  to an uninterrupted one.

Outputs (same layout as batched_ppo.py) go to
smart_control/reinforcement_learning/experiment_results/<name>_<timestamp>/.

  python -m smart_control.reinforcement_learning.scripts.batched_sac \
      --num-envs 32 --total-timesteps 2000000 --updates-per-step 16 \
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
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--experiment-name", default="sac")
  p.add_argument("--gin", default=os.path.join(CONFIG_PATH, "sim_config_1_day.gin"))
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--rng", choices=("fast", "exact"), default="fast")
  p.add_argument("--reward", choices=("sbsim", "competition"), default="sbsim")
  p.add_argument("--gpu-parts", type=int, default=1)
  p.add_argument("--total-timesteps", type=int, default=2_000_000, help="env transitions")
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--buffer-size", type=int, default=1_000_000)
  p.add_argument("--batch-size", type=int, default=256)
  p.add_argument("--learning-starts", type=int, default=20_000, help="random-action transitions first")
  p.add_argument("--updates-per-step", type=int, default=16, help="gradient steps per vector step")
  p.add_argument("--gamma", type=float, default=0.99)
  p.add_argument("--tau", type=float, default=0.005)
  p.add_argument("--policy-lr", type=float, default=3e-4)
  p.add_argument("--q-lr", type=float, default=1e-3)
  p.add_argument("--policy-frequency", type=int, default=2)
  p.add_argument("--target-network-frequency", type=int, default=1)
  p.add_argument("--alpha", type=float, default=0.2, help="initial / fixed entropy coefficient")
  p.add_argument("--autotune", type=int, default=1)
  p.add_argument("--reward-scale", type=float, default=1.0)
  p.add_argument("--hidden", default="256,256")
  p.add_argument("--norm-obs", type=int, default=1)
  p.add_argument("--eval-interval", type=int, default=20_000, help="env transitions between evals (0: off)")
  p.add_argument("--eval-envs", type=int, default=16)
  p.add_argument("--eval-seed", type=int, default=12345, help="eval at step t uses seed eval_seed + t")
  p.add_argument("--checkpoint-interval", type=int, default=100_000, help="env transitions between checkpoints")
  p.add_argument("--log-interval", type=int, default=8_192, help="env transitions between train logs")
  p.add_argument("--baseline-return", type=float, default=None)
  p.add_argument("--target-entropy", type=float, default=None, help="default: -action_dim")
  p.add_argument("--grad-clip", type=float, default=10.0, help="global-norm clip of critic and actor gradients (0: off)")
  p.add_argument("--critic-layernorm", type=int, default=1, help="LayerNorm after each hidden layer of the Q networks")
  p.add_argument("--resume", default=None, help="run directory to continue")
  return p.parse_args()


class RunningMeanStd:
  def __init__(self, shape):
    self.mean = np.zeros(shape)
    self.var = np.ones(shape)
    self.count = 1e-4

  def update(self, x):
    x = np.asarray(x, dtype=np.float64)
    bm, bv, n = x.mean(axis=0), x.var(axis=0), x.shape[0]
    delta = bm - self.mean
    total = self.count + n
    self.mean = self.mean + delta * n / total
    self.var = (self.var * self.count + bv * n + delta**2 * self.count * n / total) / total
    self.count = total

  def normalize(self, x, enabled, clip=10.0):
    if not enabled:
      return np.asarray(x, dtype=np.float32)
    return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)


class ReplayBuffer(tf.Module):
  """Ring buffer of raw transitions, kept in GPU variables."""

  def __init__(self, size, obs_size, act_size):
    super().__init__()
    def var(*shape):
      return tf.Variable(tf.zeros(shape, tf.float32), trainable=False)
    self.obs, self.next_obs = var(size, obs_size), var(size, obs_size)
    self.actions, self.rewards, self.dones = var(size, act_size), var(size), var(size)
    self.size, self.pos, self.full = size, 0, False
    # pos and full, mirrored here for checkpoints.
    self.ring_state = tf.Variable([0, 0], dtype=tf.int64, trainable=False)

  def sync_state(self, to_variable):
    if to_variable:
      self.ring_state.assign([self.pos, int(self.full)])
    else:
      self.pos, self.full = (int(v) for v in self.ring_state.numpy())
      self.full = bool(self.full)

  def add(self, obs, next_obs, actions, rewards, dones):
    n = len(obs)
    idx = tf.constant(((self.pos + np.arange(n)) % self.size)[:, None])
    for v, x in zip((self.obs, self.next_obs, self.actions, self.rewards, self.dones),
                    (obs, next_obs, actions, rewards, dones)):
      v.scatter_nd_update(idx, tf.constant(x, tf.float32))
    self.full = self.full or self.pos + n >= self.size
    self.pos = (self.pos + n) % self.size

  def __len__(self):
    return self.size if self.full else self.pos

  def sample(self, idx):
    return tuple(tf.gather(v, idx) for v in (self.obs, self.next_obs, self.actions,
                                             self.rewards, self.dones))


def mlp(sizes, out, name, layernorm=False):
  layers = []
  for h in sizes:
    if layernorm:
      layers += [tf.keras.layers.Dense(h), tf.keras.layers.LayerNormalization(),
                 tf.keras.layers.ReLU()]
    else:
      layers.append(tf.keras.layers.Dense(h, activation="relu"))
  layers.append(tf.keras.layers.Dense(out))
  return tf.keras.Sequential(layers, name=name)


class Actor(tf.Module):
  def __init__(self, obs_size, act_size, hidden):
    super().__init__()
    self.trunk = mlp(hidden, 2 * act_size, "actor")
    self.act_size = act_size
    self.trunk(tf.zeros((1, obs_size)))

  def params(self, obs):
    out = self.trunk(obs)
    mean, log_std = out[:, :self.act_size], out[:, self.act_size:]
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    return mean, log_std

  def sample(self, obs):
    mean, log_std = self.params(obs)
    std = tf.exp(log_std)
    x = mean + std * tf.random.normal(tf.shape(mean))
    action = tf.tanh(x)
    log_prob = (-0.5 * tf.square((x - mean) / std) - log_std
                - 0.5 * np.log(2.0 * np.pi))
    log_prob -= tf.math.log(1.0 - tf.square(action) + 1e-6)
    return action, tf.reduce_sum(log_prob, -1), tf.tanh(mean)


class QNetwork(tf.Module):
  def __init__(self, obs_size, act_size, hidden, name, layernorm=False):
    super().__init__(name=name)
    self.net = mlp(hidden, 1, name, layernorm)
    self.net(tf.zeros((1, obs_size + act_size)))

  def __call__(self, obs, actions):
    return tf.squeeze(self.net(tf.concat([obs, actions], -1)), -1)


def main():
  args = parse_args()
  resumed_step = 0
  if args.resume:
    run_dir = os.path.abspath(args.resume)
    with open(os.path.join(run_dir, "config.json")) as f:
      config = json.load(f)
    for k, v in config.items():
      if hasattr(args, k) and k not in ("resume", "total_timesteps"):
        setattr(args, k, v)
    latest = tf.train.latest_checkpoint(os.path.join(run_dir, "resume"))
    if latest is None:
      raise FileNotFoundError(f"No resume checkpoint in {run_dir}/resume.")
    resumed_step = int(latest.rsplit("-", 1)[1])
  else:
    stamp = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    run_dir = os.path.join(RESULTS_DIR, f"{args.experiment_name}_{stamp}")
  hidden = [int(h) for h in args.hidden.split(",") if h]
  np_rng = np.random.default_rng(args.seed + resumed_step)
  tf.random.set_seed(args.seed + resumed_step)
  logger = run_logging.RunLogger(
      run_dir, dict(vars(args), utd_ratio=args.updates_per_step / args.num_envs),
      resume=bool(args.resume))

  ref_env = create_and_setup_environment(args.gin, metrics_path=None)
  engine = sb1_vec.SB1VecBatch(ref_env, args.num_envs, rng=args.rng,
                               seed=args.seed + resumed_step, gpu_parts=args.gpu_parts,
                               reward=args.reward)
  venv = vec_env.SB1VectorEnv(engine)
  normalizers = [ref_env.action_normalizers[n] for n in venv.action_names]
  setpoint_names = [t[1] for t in engine._action_targets]  # pylint: disable=protected-access
  eval_venv = None
  if args.eval_interval > 0:
    eval_ref = create_and_setup_environment(args.gin, metrics_path=None)
    eval_venv = vec_env.SB1VectorEnv(sb1_vec.SB1VecBatch(
        eval_ref, args.eval_envs, rng="fast", seed=args.eval_seed,
        reward=args.reward))

  D, A, N = venv.observation_size, venv.action_size, args.num_envs
  actor = Actor(D, A, hidden)
  ln = bool(args.critic_layernorm)
  qf1, qf2 = QNetwork(D, A, hidden, "qf1", ln), QNetwork(D, A, hidden, "qf2", ln)
  qf1_t = QNetwork(D, A, hidden, "qf1_target", ln)
  qf2_t = QNetwork(D, A, hidden, "qf2_target", ln)
  for src, dst in ((qf1, qf1_t), (qf2, qf2_t)):
    for s, d in zip(src.trainable_variables, dst.trainable_variables):
      d.assign(s)
  q_opt = tf.keras.optimizers.Adam(args.q_lr)
  actor_opt = tf.keras.optimizers.Adam(args.policy_lr)
  log_alpha = tf.Variable(np.log(args.alpha), dtype=tf.float32, name="log_alpha")
  alpha_opt = tf.keras.optimizers.Adam(args.q_lr)
  target_entropy = -float(A) if args.target_entropy is None else float(args.target_entropy)
  obs_rms = RunningMeanStd(D)
  rms_mean = tf.Variable(obs_rms.mean, dtype=tf.float64, trainable=False)
  rms_var = tf.Variable(obs_rms.var, dtype=tf.float64, trainable=False)
  rms_count = tf.Variable(obs_rms.count, dtype=tf.float64, trainable=False)
  # global_step, grad_steps; best eval return.
  counters = tf.Variable([0, 0], dtype=tf.int64, trainable=False)
  best_eval = tf.Variable(-np.inf, dtype=tf.float64, trainable=False)
  buffer = ReplayBuffer(args.buffer_size, D, A)
  policy_state = dict(actor=actor, qf1=qf1, qf2=qf2, qf1_target=qf1_t,
                      qf2_target=qf2_t, q_opt=q_opt, actor_opt=actor_opt,
                      log_alpha=log_alpha, alpha_opt=alpha_opt,
                      obs_rms_mean=rms_mean, obs_rms_var=rms_var)
  ckpt = tf.train.Checkpoint(**policy_state)
  ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(run_dir, "checkpoints"),
                                            max_to_keep=None)
  best_manager = tf.train.CheckpointManager(
      ckpt, os.path.join(run_dir, "checkpoints", "best"), max_to_keep=1)
  resume_ckpt = tf.train.Checkpoint(**policy_state, obs_rms_count=rms_count,
                                    counters=counters, best_eval=best_eval,
                                    buffer=buffer)
  resume_manager = tf.train.CheckpointManager(
      resume_ckpt, os.path.join(run_dir, "resume"), max_to_keep=1)
  q_vars = qf1.trainable_variables + qf2.trainable_variables
  # Slot variables must exist before the fused update loop is traced.
  q_opt.build(q_vars)
  actor_opt.build(actor.trainable_variables)
  alpha_opt.build([log_alpha])
  # Loss sums and counts stay on the GPU until a log step reads them.
  loss_sum = tf.Variable(tf.zeros(7, tf.float64), trainable=False)
  loss_count = tf.Variable(tf.zeros(7, tf.float64), trainable=False)

  @tf.function
  def update_critic(obs, next_obs, actions, rewards, dones):
    alpha = tf.exp(log_alpha)
    next_a, next_logp, _ = actor.sample(next_obs)
    min_q_next = tf.minimum(qf1_t(next_obs, next_a), qf2_t(next_obs, next_a)) - alpha * next_logp
    target = rewards + (1.0 - dones) * args.gamma * min_q_next
    with tf.GradientTape() as tape:
      q1, q2 = qf1(obs, actions), qf2(obs, actions)
      q1_loss = tf.reduce_mean(tf.square(q1 - target))
      q2_loss = tf.reduce_mean(tf.square(q2 - target))
      q_loss = q1_loss + q2_loss
    q_grads = tape.gradient(q_loss, q_vars)
    if args.grad_clip > 0:
      q_grads, _ = tf.clip_by_global_norm(q_grads, args.grad_clip)
    q_opt.apply_gradients(zip(q_grads, q_vars))
    return q1_loss, q2_loss, tf.reduce_mean(q1), tf.reduce_mean(q2)

  @tf.function
  def update_actor(obs):
    with tf.GradientTape() as tape:
      a, logp, _ = actor.sample(obs)
      q = tf.minimum(qf1(obs, a), qf2(obs, a))
      actor_loss = tf.reduce_mean(tf.exp(log_alpha) * logp - q)
    grads = tape.gradient(actor_loss, actor.trainable_variables)
    if args.grad_clip > 0:
      grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
    actor_opt.apply_gradients(zip(grads, actor.trainable_variables))
    alpha_loss = tf.constant(0.0)
    if args.autotune:
      _, logp2, _ = actor.sample(obs)
      with tf.GradientTape() as tape2:
        alpha_loss = tf.reduce_mean(-tf.exp(log_alpha) * (tf.stop_gradient(logp2) + target_entropy))
      alpha_opt.apply_gradients([(tape2.gradient(alpha_loss, log_alpha), log_alpha)])
    return actor_loss, alpha_loss, -tf.reduce_mean(logp)

  @tf.function
  def update_targets():
    for src, dst in ((qf1, qf1_t), (qf2, qf2_t)):
      for s, d in zip(src.trainable_variables, dst.trainable_variables):
        d.assign(args.tau * s + (1.0 - args.tau) * d)

  @tf.function(jit_compile=True)
  def train(buffer_len, start_step):
    """Runs updates_per_step gradient steps on uniform replay samples."""
    def normalize(x):
      if not args.norm_obs:
        return x
      x = (tf.cast(x, tf.float64) - rms_mean) / tf.sqrt(rms_var + 1e-8)
      return tf.cast(tf.clip_by_value(x, -10.0, 10.0), tf.float32)

    def actor_updates(b_obs):
      for _ in range(args.policy_frequency):
        actor_terms = update_actor(b_obs)
      loss_sum.assign_add(tf.pad(tf.cast(tf.stack(actor_terms), tf.float64), [[4, 0]]))
      loss_count.assign_add(tf.constant([0.0, 0, 0, 0, 1, 1, 1], tf.float64))

    for i in tf.range(args.updates_per_step, dtype=tf.int64):
      idx = tf.random.uniform([args.batch_size], 0, buffer_len, dtype=tf.int64)
      b_obs, b_next, b_act, b_rew, b_done = buffer.sample(idx)
      b_obs, b_next = normalize(b_obs), normalize(b_next)
      critic = update_critic(b_obs, b_next, b_act, b_rew, b_done)
      loss_sum.assign_add(tf.pad(tf.cast(tf.stack(critic), tf.float64), [[0, 3]]))
      loss_count.assign_add(tf.constant([1.0, 1, 1, 1, 0, 0, 0], tf.float64))
      step = start_step + i
      if step % args.policy_frequency == 0:
        actor_updates(b_obs)
      if step % args.target_network_frequency == 0:
        update_targets()

  @tf.function
  def act(obs):
    a, _, _ = actor.sample(obs)
    return a

  def greedy(obs):
    _, _, mean_action = actor.sample(tf.constant(obs_rms.normalize(obs, args.norm_obs)))
    return np.clip(mean_action.numpy(), -1.0, 1.0)

  tracker = run_logging.EpisodeTracker(N)
  stats = run_logging.TimeOfDayStats(venv.episode_steps)
  setpoint_sum, setpoint_count = np.zeros((24, A)), np.zeros(24, np.int64)
  records = []
  obs = venv.reset()
  if args.resume:
    resume_ckpt.restore(resume_manager.latest_checkpoint).assert_existing_objects_matched()
    obs_rms.mean, obs_rms.var = rms_mean.numpy(), rms_var.numpy()
    obs_rms.count = float(rms_count.numpy())
    buffer.sync_state(to_variable=False)
    global_step, grad_steps = (int(v) for v in counters.numpy())
    print(f"resumed {run_dir} at step {global_step} (replay {len(buffer)}, "
          f"best eval {float(best_eval.numpy()):.4f})", flush=True)
  else:
    obs_rms.update(obs)
    global_step, grad_steps = 0, 0

  def next_multiple(interval):
    return (global_step // interval + 1) * interval if interval > 0 else 0

  next_eval = next_multiple(args.eval_interval)
  next_ckpt = next_multiple(args.checkpoint_interval)
  next_log = next_multiple(args.log_interval)
  start, start_step = time.time(), global_step
  while global_step < args.total_timesteps:
    if global_step < args.learning_starts:
      actions = np_rng.uniform(-1.0, 1.0, size=(N, A)).astype(np.float32)
    else:
      actions = act(tf.constant(obs_rms.normalize(obs, args.norm_obs))).numpy()
    next_obs, rewards, terminated, truncated, info = venv.step(actions)
    global_step += N
    real_next = next_obs.copy()
    real_next[truncated] = info.final_observation[truncated]
    buffer.add(obs, real_next, actions, rewards * args.reward_scale, terminated.astype(np.float32))
    obs_rms.update(next_obs)
    rms_mean.assign(obs_rms.mean)
    rms_var.assign(obs_rms.var)
    stats.add(rewards, info)
    records += tracker.add(rewards, truncated, terminated, info)
    native = np.stack([[normalizers[i].setpoint_value(a[i]) for i in range(A)] for a in actions])
    np.add.at(setpoint_sum, info.local_hour, native)
    np.add.at(setpoint_count, info.local_hour, 1)
    obs = next_obs

    if global_step >= args.learning_starts:
      train(tf.constant(len(buffer), tf.int64), tf.constant(grad_steps, tf.int64))
      grad_steps += args.updates_per_step

    if global_step >= next_log:
      next_log += args.log_interval
      sums, counts = loss_sum.numpy(), loss_count.numpy()
      m = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
      loss_sum.assign(tf.zeros_like(loss_sum))
      loss_count.assign(tf.zeros_like(loss_count))
      elapsed = time.time() - start
      logger.scalars(global_step, {
          "losses/qf1_loss": m[0], "losses/qf2_loss": m[1],
          "losses/qf1_values": m[2], "losses/qf2_values": m[3],
          "losses/actor_loss": m[4], "losses/alpha_loss": m[5],
          "charts/policy_entropy": m[6],
          "charts/alpha": float(tf.exp(log_alpha).numpy()),
          "charts/SPS_total": (global_step - start_step) / elapsed,
          "charts/gradient_steps": grad_steps,
          "charts/replay_size": len(buffer),
          **{f"charts/setpoint_K/{setpoint_names[i]}/{h:02d}:00": setpoint_sum[h, i] / setpoint_count[h]
             for h in range(24) if setpoint_count[h] for i in range(A)},
      })
      if records:
        logger.episodes(global_step, records, "train")
      logger.time_of_day(global_step, stats, "train")
      print(f"step {global_step} SPS {(global_step - start_step) / elapsed:.0f} grad_steps {grad_steps} "
            f"alpha {float(tf.exp(log_alpha).numpy()):.4f}"
            + (f" train_return {np.mean([r['return'] for r in records]):.3f}" if records else ""),
            flush=True)
      records = []
      stats = run_logging.TimeOfDayStats(venv.episode_steps)
      setpoint_sum[:], setpoint_count[:] = 0, 0

    if eval_venv is not None and (global_step >= next_eval or global_step >= args.total_timesteps):
      next_eval += args.eval_interval
      ret = evaluation.evaluate_policy(greedy, eval_venv, logger, global_step,
                                       f"s{global_step:09d}", args.baseline_return,
                                       seed=args.eval_seed + global_step)
      print(f"  eval return {ret:.4f}"
            + (f" (baseline {args.baseline_return:.4f})" if args.baseline_return is not None else ""),
            flush=True)
      if ret > float(best_eval.numpy()):
        best_eval.assign(ret)
        path = best_manager.save(checkpoint_number=global_step)
        logger.summary("best", {"global_step": global_step, "eval_return": ret,
                                "checkpoint": path})
    if global_step >= next_ckpt or global_step >= args.total_timesteps:
      next_ckpt += args.checkpoint_interval
      ckpt_manager.save(checkpoint_number=global_step)
      rms_count.assign(obs_rms.count)
      counters.assign([global_step, grad_steps])
      buffer.sync_state(to_variable=True)
      resume_manager.save(checkpoint_number=global_step)
  print(f"run directory: {run_dir}")


if __name__ == "__main__":
  main()
