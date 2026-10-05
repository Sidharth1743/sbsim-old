"""Records or checks a bit-exact trace of the one-day SB1 environment.

The trace is the reference for simulator speedups: an optimized simulator must
reproduce every recorded array exactly, not approximately.

  python -m smart_control.reinforcement_learning.scripts.sim_golden_trace \
      record --out /tmp/golden.npz --steps 24
  python -m smart_control.reinforcement_learning.scripts.sim_golden_trace \
      check --ref /tmp/golden.npz
"""

import argparse
import os
import random
import time

os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")

# pylint: disable=g-import-not-at-top, wrong-import-position
import numpy as np

from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment


def _actions(num_steps, action_size):
  """Deterministic actions that sweep the normalized action range."""
  t = np.arange(num_steps, dtype=np.float32)[:, None]
  phase = np.arange(action_size, dtype=np.float32)[None, :]
  return np.sin(0.7 * t + 1.3 * phase).astype(np.float32)


def _python_random_digest():
  state = random.getstate()[1]
  return np.asarray(state, dtype=np.uint32)


def run_trace(num_steps):
  """Runs the environment and returns every array the trace compares."""
  env = create_and_setup_environment(
      os.path.join(CONFIG_PATH, "sim_config_1_day.gin"), metrics_path=None
  )
  first = env.reset()
  actions = _actions(num_steps, env.action_spec().shape[0])
  temps, observations, rewards, step_types = [], [], [], []
  observations.append(np.asarray(first.observation))
  start = time.time()
  for action in actions:
    step = env.step(action)
    temps.append(np.array(env.building.simulator.building.temp, copy=True))
    observations.append(np.asarray(step.observation))
    rewards.append(np.asarray(step.reward))
    step_types.append(np.asarray(step.step_type))
  elapsed = time.time() - start
  trace = {
      "actions": actions,
      "temps": np.stack(temps),
      "observations": np.stack(observations),
      "rewards": np.stack(rewards),
      "step_types": np.stack(step_types),
      "python_random_state": _python_random_digest(),
  }
  return trace, elapsed / num_steps


def main():
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="mode", required=True)
  rec = sub.add_parser("record")
  rec.add_argument("--out", required=True)
  rec.add_argument("--steps", type=int, default=24)
  chk = sub.add_parser("check")
  chk.add_argument("--ref", required=True)
  args = parser.parse_args()

  if args.mode == "record":
    trace, sec_per_step = run_trace(args.steps)
    np.savez_compressed(args.out, **trace)
    print(
        f"recorded {args.steps} steps to {args.out}, {sec_per_step:.3f} s/step"
    )
    return

  ref = dict(np.load(args.ref))
  trace, sec_per_step = run_trace(len(ref["actions"]))
  ok = True
  for key, expected in ref.items():
    got = trace[key]
    if got.dtype != expected.dtype or got.shape != expected.shape:
      print(
          f"FAIL {key}: {got.dtype}{got.shape} !="
          f" {expected.dtype}{expected.shape}"
      )
      ok = False
    elif not np.array_equal(got, expected):
      diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
      first = int(np.argwhere(diff > 0)[0][0])
      print(
          f"FAIL {key}: max abs diff {diff.max():.3e}, first at index {first}"
      )
      ok = False
    else:
      print(f"ok   {key}")
  print(f"{sec_per_step:.3f} s/step")
  print("BIT-EXACT" if ok else "MISMATCH")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
