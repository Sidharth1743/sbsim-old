# Simulator speedup: changes and metrics

Goal: make the SB1 simulator fast enough for long training runs without
changing the core system. Every change below reproduces the original code
**bit for bit** on a 300-step reference trace (see Verification).

Machine: RTX 3090 (sm_86, driver 595.84), 12 logical CPUs, TensorFlow 2.15.1,
numpy 1.26.4, Python 3.11.

## Results

### Standard pipeline, default CLI settings

Measured 2026-10-05 with `populate_starter_buffer --buffer-name bench`, then
`train --starter-buffer-path .../bench_seqlen2_exp500 --experiment-name bench`
(100 iterations, 50 collect steps, eval every 10, 200 learner steps).

| Stage | Original code | Now |
| --- | --- | --- |
| `populate_starter_buffer` (500 steps) | ~26 min (estimated: 500 × 3.1 s) | **30.6 s** |
| `train` (100 iterations) | > 5 h 52 min (measured; that run stopped after iteration 80) | **9 min 34 s** |

Breakdown of the 9 min 34 s training run:

| Phase | Count | Each | Total |
| --- | --- | --- | --- |
| Setup (envs, agent, buffer) | 1 | | 13.6 s |
| Eval episode (288 steps) | 10 | 15.4 s | 153.8 s |
| Collect (50 steps) | 100 | 3.05 s | 305.1 s |
| Learner (200 SAC updates) | 100 | 0.74 s | 74.3 s |
| Final checkpoint and eval | 1 | | 25.4 s |

Inside the loop an env step costs 0.053 s (eval) to 0.061 s (collect), because
the policy runs on every step.

### Environment step (one env, `sim_config_1_day.gin`)

| Version | s/step | SPS |
| --- | --- | --- |
| Original code | 3.03–3.13 | 0.3 |
| + TF graph FD loop, C convection shuffle, zone-stats indexing (CPU only) | 0.205 | 4.9 |
| + CUDA FD kernel, thermal power and weather fixes (GPU) | 0.040 | 25 |
| + fused, class-table CUDA kernel (GPU) | **0.039** | **26** |

Per env the step is now bounded by the Python outside the solver (~36 ms); the
solver itself takes ~1.2 ms of it.

### CUDA FD solver alone (GPU time per env-step, threshold 0.1, 100 iterations)

| Kernel | B=1 | B=4 | B=16 | B=64 |
| --- | --- | --- | --- | --- |
| One launch per iteration (round 2) | 3.8 ms | | | |
| Fused, K=4 | 1.56 ms | 1.04 ms | 1.04 ms | 1.01 ms |
| Fused, K=8 (default) | 1.44 ms | 1.14 ms | 1.09 ms | 1.10 ms |

Host↔GPU copies of the grids add ~0.5 ms per env-step. From B=4 on the GPU is
saturated (compute bound, mainly the IEEE division that must stay a division),
so the solver's ceiling on this card is about **1,000 env-steps/s**.

### Several env processes sharing the GPU

| Processes | Total SPS | Per env |
| --- | --- | --- |
| 1 | 25.4 | 25.4 |
| 4 | 87.4 | 21.9 |
| 8 | 136.2 | 17.0 |
| 12 | 143.6 | 12.0 |

At 12 processes the limit is CPU (the remaining per-step Python), not the
GPU.

## What changed

| File | Change |
| --- | --- |
| `simulator/fd_kernel.cu`, `simulator/cuda_fd.py` (new) | Finite-difference timestep on the GPU. Same float32 operation order as before, explicit round-to-nearest ops, `-fmad=false`, convergence test in float64 like numpy. Per-cell constants are grouped into exact classes (12 on SB1), so each cell stores a uint8 index into an on-chip table. K iterations (default `cuda_fd_fuse_k=8`) run per launch in 64×64 shared-memory tiles. Each cell's delta is counted once, by its owner tile. If an env converges inside a group, the group reruns from its unchanged input for exactly the original iteration count. All loop control runs on the GPU, with one sync per step. Supports a batch of envs. Compiled with nvcc on first use into `~/.cache/smart_control/`. |
| `simulator/simulator_flexible_floor_plan.py`, `simulator/tf_simulator.py`, `train_sim_configs/sim_config_1_day.gin` | New `record_video` flag (default True, the upstream behavior). The 1-day training config sets it False: its episodes reset before the day-4 video, so the per-step grid copies were never used. |
| `simulator/tf_simulator.py` | Uses the CUDA kernel when available (`use_cuda_fd_kernel=True`), otherwise the whole FD loop as one `tf.function` (`use_graph_fd_loop=True`) with grappler disabled for that call only. Otherwise, or for interior mass or radiation, uses the original path. |
| `simulator/convection_shuffle.c` (new), `simulator/stochastic_convection_simulator.py` | `_shuffle_max_dist` in C. It consumes Python's `random` stream exactly as `random.uniform/choice/shuffle` do and writes the state back. `use_compiled_shuffle=False` restores the Python loop. |
| `simulator/building.py` | `get_zone_temp_stats`: cached index arrays instead of a per-cell list. `apply_thermal_power_zone`: writes only diffuser cells, vectorized, for float64 or Python float power; other types keep the loop. |
| `simulator/weather_controller.py` | `min_time` / `max_time` are cached per `weather_df` instead of recomputed on every call. |
| `reinforcement_learning/scripts/sim_golden_trace.py` (new) | Records and checks bit-exact environment traces. |
| Tests | `stochastic_convection_simulator_test.py` (new). `tf_simulator_test.py` compares the graph path and the CUDA kernel with the original loop for every fused K, for limits not divisible by K, and for thresholds that converge inside a group. `simulator_flexible_floor_plan_test.py` checks that `record_video=False` stores no frames and gives identical temperatures. All 686 simulator and environment tests pass (CPU-only: the CUDA test skips). |
| venv | Added `tensorflow[and-cuda]==2.15.1` CUDA wheels. TensorFlow itself is unchanged. `pyproject.toml` and `uv.lock` are unchanged. |

All new paths fall back automatically: no GPU or nvcc → TF graph loop; no
gcc → Python shuffle.

## Verification

- Reference: `/data/smart-building/sbsim_traces/golden300.npz`, recorded from an
  untouched HEAD worktree. It is 300 steps with fixed actions, covering a full
  day, the episode end, and the auto-reset.
- It compares actions, the full temperature grid every step, observations,
  rewards, step types, and the final Python `random` state with
  `np.array_equal`.
- Final code: **BIT-EXACT** on GPU (0.039 s/step) and on CPU only (0.213
  s/step).
- The fused kernel was also compared with the TF graph loop on identical inputs
  every step: all 7 fused K values, thresholds 0.1 / 0.2 / 0.5 (iteration counts
  2–100, including convergence inside a group). That gave 525 step comparisons
  with 0 mismatches. Batches of 4, 16, and 64 envs, each with a different
  ambient temperature, matched B=1 runs per env bit for bit.

```sh
python -m smart_control.reinforcement_learning.scripts.sim_golden_trace \
    check --ref /data/smart-building/sbsim_traces/golden300.npz
```

## Known issues found during the benchmark

- **Memory grew by 3 MB every step (fixed for the 1-day config).**
  `VisualLogger.log` (called in `execute_step_sim`) kept a copy of the full
  temperature grid every step and never cleared it. Peak RSS of the
  100-iteration run was **23.8 GB**. With `record_video = False` in
  `sim_config_1_day.gin`, RSS stays at ~2,086 MB over 600 steps. Other configs
  keep the upstream behavior.
- **Eval return is always 0.0.** `train.py` logs its own `eval_metrics`, but
  the eval actor is built with a different metrics list. The eval TensorBoard
  summaries are also 0.0 for every eval in both this run and the earlier
  original-code run. Eval quality cannot be compared from these logs.
- **`train.py` overwrites the starter buffer.** The Reverb checkpointer writes
  into the `--starter-buffer-path` directory. A 1-iteration timing run replaced
  the `starter_seqlen2_exp500` checkpoint. Its original 500 transitions are
  still inside, plus 50 collected ones. Rerunning `populate_starter_buffer`
  regenerates the original data, because the env is deterministic. Use a copy
  of the buffer for test runs.

## Limits and next steps

- The remaining cost per step is Python: building the observation (~18 ms), the
  convection shuffle (~12 ms; sequential RNG), zone averages (~9 ms; exact
  vectorization would need numpy's pairwise float32 summation), and the reward
  and HVAC logic.
- The fused CUDA kernel is compute bound at ~1 ms per env-step (100 iterations
  over 744×1004 cells), so the solver tops out near 1,000 env-steps/s on the
  3090. An earlier estimate of 2,000–4,000 was too high.
- Next: a batched, fully GPU environment for PPO-style training, with an
  estimated 700–900 SPS. That needs per-env GPU random numbers for the shuffle:
  same shuffle rule, different draws. Prove it with a mode that feeds the exact
  Python draws and matches `golden300.npz`.

## Round 4: batched GPU environment (`smart_control/batched/`)

Plan and verified numerics: `port_plan.md`. All engines are built from a
freshly constructed gin `Environment` and take actions of shape (B, 2).

| Engine | What runs where | Verification |
| --- | --- | --- |
| `SB1ExactBatch` (`sb1_exact.py`) | host grids, per-env Python logic, CUDA FD, C shuffle | `golden300.npz` bit-exact (B=1 and B=4). B=4 with different actions per env matches 4 independent original runs |
| `SB1GpuBatch` (`sb1_gpu_engine.py`, `sb1_gpu.cu`) | grids resident on GPU: FD, numpy-exact zone/global means (8192-element chunked pairwise sum), thermal power, swap application | same two checks, bit-exact |
| `SB1VecBatch` (`sb1_vec.py`) | scalar logic vectorized over the batch (float64 arrays, sequential sums kept sequential), occupancy in C (`sb1_occupancy.c`) | same two checks, bit-exact |

RNG modes of `SB1VecBatch`:

- `rng="exact"`: the original streams in every env (Python MT for the shuffle,
  numpy MT for occupancy). Bit-exact with the original.
- `rng="feed"`: the GPU generates the swaps from the recorded MT words. It is
  bit-exact on `golden300.npz` and on the different-actions test, which proves
  the GPU generator implements `_shuffle_max_dist` + `random.shuffle` exactly.
- `rng="fast"`: the same rule, with per-env Philox4x32-10 streams (verified
  against 3 Random123 known-answer vectors) and per-env occupancy seeds. The
  shared-memory fast-swap kernel matches a C reference
  (`sb1_fast_ref.c`, same word mapping) bit for bit: 4 envs × 3 steps ×
  p ∈ {1.0, 0.7}. Results are identical for `gpu_parts` = 1, 2, and 4.

FD kernel additions, still bit-exact (525 step comparisons with the TF graph
loop):

- exterior CVs skip arithmetic whose result was overwritten anyway;
- tiles whose owner CVs are all exterior skip the iteration loop, recording
  the same deltas.

Batched FD went from 1.0–1.1 to 0.80–0.83 ms/env.

Throughput (RTX 3090, steps of the full SB1 env, 1-day config):

| Mode | B | SPS |
| --- | --- | --- |
| original single env | 1 | 0.3 |
| round 3 single env | 1 | 26 |
| `SB1VecBatch` exact | 64 | 309 |
| `SB1VecBatch` fast | 256 | 679 (`gpu_parts=1`), 702 (`gpu_parts=2`) |

Fast mode at B=256, per env-step:

| Stage | Time |
| --- | --- |
| FD | ~0.8–1.1 ms |
| GPU swaps | ~0.5 ms |
| Everything else | ~0.06 ms |

Splitting the batch over 2 streams overlaps these only slightly (+3%).

### Statistical check of fast mode

Fast mode cannot be bit-exact, so it was compared with the original under
other seeds. The original's own randomness is the convection (Python
`random`) and occupancy (numpy) seeds. All runs used the same fixed actions
for one full episode (289 steps). Scripts:
`/data/smart-building/sbsim_traces/stat_check*.py`.

- 32 original-distribution samples: the exact engine, with env s seeded like
  the original code with `random.seed(1000+s)` and
  `RandomState(5000+s)`. Envs 0–7 are bit-identical to 8 runs of the original
  code with those seeds.
- 32 fast-mode envs, for each of two fast seeds (7 and 99).

| Quantity | Original seeds | Fast seed 7 | Fast seed 99 |
| --- | --- | --- | --- |
| Episode return | −38.949 ± 0.217 | −39.020 ± 0.215 (t p=0.20) | −38.944 ± 0.284 (t p=0.94) |
| Reward, morning warm-up (steps 80–90) | −0.4150 ± 0.0115 | −0.4186 ± 0.0113 (p=0.23) | −0.4132 ± 0.0101 (p=0.51) |
| Heating requests (mean count) | 18.621 ± 0.076 | 18.607 ± 0.074 (p=0.46) | 18.626 ± 0.104 (p=0.83) |

No significant difference. For scale, two halves of the original-seed samples
differ by about the same amount (return t p=0.065), and so do the two fast
seeds (t p=0.24).

An earlier comparison against only 8 original seeds suggested a gap
(p=0.01). Those 8 seeds happened to have higher returns than the 32-seed
average (−38.80 vs −38.95).
