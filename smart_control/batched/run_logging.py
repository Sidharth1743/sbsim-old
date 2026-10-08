"""Run directories, TensorBoard, and time-of-day reward logs.

A run directory holds:

* config.json — the run configuration.
* tensorboard/ — TensorBoard event files.
* episodes.csv — one row per finished episode: return, length, and summed
  reward terms.
* hourly.csv — per log event, the mean reward and reward terms for each local
  hour of day (00:00–23:00).
* step_profile_<tag>.csv — mean reward and terms at every 5-minute step of an
  episode, with its local time.
* summary_<tag>.json — headline numbers.
"""

import csv
import io
import json
import os

import numpy as np

_TERMS = (
    "normalized_productivity_regret",
    "normalized_energy_cost",
    "normalized_carbon_emission",
    "electricity_energy_cost_usd",
    "natural_gas_energy_cost_usd",
    "carbon_emitted_kg",
    "total_occupancy",
    "electrical_power_w",
    "natural_gas_power_w",
    "comfort_penalty_k",
)


class TimeOfDayStats:
  """Accumulates rewards and reward terms by local hour and episode step."""

  def __init__(self, episode_steps: int):
    self.episode_steps = episode_steps
    names = ("reward",) + _TERMS
    self.hour_sum = {n: np.zeros(24) for n in names}
    self.hour_count = np.zeros(24, dtype=np.int64)
    self.step_sum = {n: np.zeros(episode_steps + 1) for n in names}
    self.step_count = np.zeros(episode_steps + 1, dtype=np.int64)
    self.step_time = [""] * (episode_steps + 1)

  def add(self, rewards, info) -> None:
    hours = info.local_hour
    steps = info.episode_step
    values = {"reward": np.asarray(rewards, dtype=np.float64)}
    values.update({n: info.reward_terms[n] for n in _TERMS})
    np.add.at(self.hour_count, hours, 1)
    np.add.at(self.step_count, steps, 1)
    for name, v in values.items():
      np.add.at(self.hour_sum[name], hours, v)
      np.add.at(self.step_sum[name], steps, v)
    for k, h, m in zip(steps, hours, info.local_minute):
      self.step_time[k] = f"{h:02d}:{m:02d}"

  def hourly_means(self) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
      return {n: s / self.hour_count for n, s in self.hour_sum.items()}

  def step_means(self) -> dict[str, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
      return {n: s / self.step_count for n, s in self.step_sum.items()}


class EpisodeTracker:
  """Running per-env episode sums; emits a record when an episode ends."""

  def __init__(self, num_envs: int):
    self.returns = np.zeros(num_envs)
    self.lengths = np.zeros(num_envs, dtype=np.int64)
    self.terms = {n: np.zeros(num_envs) for n in _TERMS}
    self.counts = np.zeros(num_envs, dtype=np.int64)

  def add(self, rewards, truncated, terminated, info) -> list[dict]:
    self.returns += rewards
    self.lengths += 1
    for n in _TERMS:
      self.terms[n] += info.reward_terms[n]
    done = np.flatnonzero(np.logical_or(truncated, terminated))
    records = []
    for b in done:
      record = {
          "env": int(b),
          "episode_index": int(self.counts[b]),
          "return": float(self.returns[b]),
          "length": int(self.lengths[b]),
      }
      record.update({f"sum_{n}": float(self.terms[n][b]) for n in _TERMS})
      records.append(record)
      self.counts[b] += 1
      self.returns[b] = 0.0
      self.lengths[b] = 0
      for n in _TERMS:
        self.terms[n][b] = 0.0
    return records


class RunLogger:
  """Owns a run directory, its TensorBoard writer, and its CSV files."""

  def __init__(self, run_dir: str, config: dict, resume: bool = False):
    """Creates run_dir, or with resume=True continues an existing one.

    A resumed run keeps its config.json and appends to its CSV files and
    TensorBoard directory.
    """
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    self._tf = tf
    if resume and not os.path.isdir(run_dir):
      raise FileNotFoundError(f"No run directory {run_dir} to resume.")
    os.makedirs(run_dir, exist_ok=resume)
    self.run_dir = run_dir
    if not resume:
      with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    self.writer = tf.summary.create_file_writer(
        os.path.join(run_dir, "tensorboard")
    )
    self._episodes_path = os.path.join(run_dir, "episodes.csv")
    self._hourly_path = os.path.join(run_dir, "hourly.csv")

  def scalars(self, step: int, values: dict[str, float]) -> None:
    with self.writer.as_default():
      for name, value in values.items():
        self._tf.summary.scalar(name, float(value), step=step)
    self.writer.flush()

  def episodes(self, step: int, records: list[dict], tag: str) -> None:
    if not records:
      return
    new = not os.path.exists(self._episodes_path)
    with open(self._episodes_path, "a", newline="") as f:
      fields = ["global_step", "tag"] + list(records[0].keys())
      w = csv.DictWriter(f, fieldnames=fields)
      if new:
        w.writeheader()
      for r in records:
        w.writerow({"global_step": step, "tag": tag, **r})
    returns = np.array([r["return"] for r in records])
    values = {
        f"{tag}/episode_return_mean": returns.mean(),
        f"{tag}/episode_return_min": returns.min(),
        f"{tag}/episode_return_max": returns.max(),
    }
    for n in _TERMS:
      values[f"{tag}/episode_{n}_mean"] = np.mean(
          [r[f"sum_{n}"] for r in records]
      )
    self.scalars(step, values)

  def time_of_day(self, step: int, stats: TimeOfDayStats, tag: str) -> None:
    """Logs reward by local hour (scalars, chart) and appends hourly.csv."""
    hourly = stats.hourly_means()
    values = {}
    for h in range(24):
      if stats.hour_count[h]:
        values[f"{tag}/reward_by_hour/{h:02d}:00"] = hourly["reward"][h]
    self.scalars(step, values)
    new = not os.path.exists(self._hourly_path)
    with open(self._hourly_path, "a", newline="") as f:
      w = csv.writer(f)
      if new:
        w.writerow(["global_step", "tag", "local_hour", "samples", "reward"]
                   + list(_TERMS))
      for h in range(24):
        if stats.hour_count[h]:
          w.writerow([step, tag, f"{h:02d}:00", int(stats.hour_count[h]),
                      hourly["reward"][h]] + [hourly[n][h] for n in _TERMS])
    self._hour_chart(step, hourly, stats.hour_count, tag)

  def step_profile(self, stats: TimeOfDayStats, tag: str) -> str:
    """Writes step_profile_<tag>.csv (mean per episode step)."""
    means = stats.step_means()
    path = os.path.join(self.run_dir, f"step_profile_{tag}.csv")
    with open(path, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["episode_step", "local_time", "samples", "reward"] + list(_TERMS))
      for k in range(len(stats.step_count)):
        if stats.step_count[k]:
          w.writerow([k, stats.step_time[k], int(stats.step_count[k]),
                      means["reward"][k]] + [means[n][k] for n in _TERMS])
    return path

  def summary(self, tag: str, values: dict) -> None:
    with open(os.path.join(self.run_dir, f"summary_{tag}.json"), "w") as f:
      json.dump(values, f, indent=2, sort_keys=True, default=float)

  def _hour_chart(self, step, hourly, counts, tag) -> None:
    import matplotlib  # pylint: disable=g-import-not-at-top

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top

    hours = [h for h in range(24) if counts[h]]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    ax.bar(hours, [hourly["reward"][h] for h in hours], color="#4c72b0")
    ax.set_xlabel("local hour")
    ax.set_ylabel("mean reward per step")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(f"{tag}: reward by hour of day")
    ax.grid(axis="y", alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    image = self._tf.image.decode_png(buf.getvalue(), channels=4)[None]
    with self.writer.as_default():
      self._tf.summary.image(f"{tag}/reward_by_hour_chart", image, step=step)
    self.writer.flush()
