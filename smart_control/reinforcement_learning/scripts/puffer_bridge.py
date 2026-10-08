"""Serve the SAC training environment to the PufferLib sb1 step.

One process holds one copy of the one-day SB1 environment used by train.py.
PufferLib's C step sends the two normalized setpoints and receives the same
observation, reward, and episode end that Environment.step returns.
"""

import os
import socket
import struct
import sys

import numpy as np
from tf_agents.trajectories import time_step as ts

from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sb1_cuda")))
from accelerate import accelerate  # pylint: disable=import-error,wrong-import-position

OBS_SIZE = 53
N_ACTIONS = 2
HOST = "127.0.0.1"
PORT = 8765
OP_RESET = 0
OP_STEP = 1


def _flat_obs(observation):
  obs = np.asarray(observation, dtype=np.float32).ravel()
  if obs.shape != (OBS_SIZE,):
    raise RuntimeError(
        f"observation length is {obs.shape[0]}, expected {OBS_SIZE}. "
        "The SAC startup log Field 0..52 is the source of that size."
    )
  return obs


def _pack(obs, reward, terminal):
  payload = struct.pack("<i", OBS_SIZE)
  payload += np.ascontiguousarray(obs, dtype=np.float32).tobytes()
  payload += struct.pack("<fi", float(reward), int(terminal))
  return payload


def _recv_exact(conn, n):
  buf = bytearray()
  while len(buf) < n:
    chunk = conn.recv(n - len(buf))
    if not chunk:
      return None
    buf.extend(chunk)
  return bytes(buf)


def _serve(conn, env):
  while True:
    header = _recv_exact(conn, 4)
    if header is None:
      return
    (op,) = struct.unpack("<i", header)
    if op == OP_RESET:
      step = env.reset()
      conn.sendall(_pack(_flat_obs(step.observation), 0.0, 0))
    elif op == OP_STEP:
      raw = _recv_exact(conn, N_ACTIONS * 4)
      if raw is None:
        return
      action = np.frombuffer(raw, dtype=np.float32).copy()
      step = env.step(action)
      done = int(step.step_type == ts.StepType.LAST)
      reward = float(np.asarray(step.reward, dtype=np.float32).reshape(-1)[0])
      conn.sendall(_pack(_flat_obs(step.observation), reward, done))
    else:
      raise RuntimeError(f"unknown opcode {op}")


def main():
  gin_path = os.path.join(CONFIG_PATH, "sim_config_1_day.gin")
  env = create_and_setup_environment(gin_path, metrics_path=None)
  accelerate(env.building.simulator)
  names = list(env._action_names)  # pylint: disable=protected-access
  if (
      len(names) != N_ACTIONS
      or not names[0].endswith("supply_water_setpoint")
      or not names[1].endswith("supply_air_heating_temperature_setpoint")
  ):
    raise RuntimeError(f"unexpected action order: {names}")
  first = env.reset()
  obs = _flat_obs(first.observation)
  print(
      f"bridge ready obs={obs.shape[0]} actions={names} "
      f"listening on {HOST}:{PORT}",
      flush=True,
  )
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind((HOST, PORT))
  server.listen(1)
  while True:
    conn, _addr = server.accept()
    try:
      _serve(conn, env)
    finally:
      conn.close()


if __name__ == "__main__":
  main()
