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

## Round 5: baseline measurement and PPO script

New:

- `batched/vec_env.py` (same-step-reset vector env, local-time labels)
- `batched/run_logging.py` (TensorBoard + CSV/JSON run folders)
- `batched/baseline_policy.py`
- `reinforcement_learning/scripts/batched_baseline.py`
- `reinforcement_learning/scripts/batched_ppo.py` (CleanRL-style PPO, TF2)

Runs go to `reinforcement_learning/experiment_results/<name>_<timestamp>/`.

`SB1VecBatch` now keeps a monitoring copy of the reward terms
(`last_reward_terms`). Its results are unchanged: still bit-exact on
`golden300.npz`.

Baseline schedule, fast mode, 64 envs × 1 day:

| Variant | Episode return |
| --- | --- |
| `intended`: each scheduled setpoint applied to its own action | −30.77 ± 0.05 (95% CI) |
| `as_run`: the repo's policy output fed unchanged | −43.51 ± 0.07 |

The repo's `SchedulePolicy` emits `[air heating, supply water]`, while the
SB1 action vector is `[supply water, air heating]`. So the weekday daytime
schedule (heating 292 K, water 350 K) is applied as water ≈ 331 K and heating
≈ 298 K. The batched wrapper reproduces the original policy output exactly
(`as_run`, 301 observations checked). Inputs are passed as TF tensors because
float32 versus float64 `arccos` can flip the decoded hour at hour boundaries.
Most of the gap is at 07:00–12:00 local time.

## Round 6: evaluation script and SAC

- `reinforcement_learning/scripts/batched_eval.py` evaluates a PPO checkpoint
  against the intended baseline schedule, deterministically and paired on
  identical random streams.
  - Exact mode is the original simulator: env 0 uses the original seeds, and
    env s uses convection seed 1000+s and occupancy seed 5000+s.
  - Fast mode uses Philox streams from `--fast-seed`.
  - Writes per-env paired differences with 95% CIs, the reward terms, and the
    reward by hour (`summary_comparison.json`).
- `SB1VecBatch.set_original_seeds(...)` gives exact/feed envs the streams the
  original code would have with other seeds. Env 0 with (5, 17321) and env 1
  with (1001, 5001) were bit-identical to the original code over 40 steps.
  Calling it after `reset()` raises.
- `reinforcement_learning/scripts/batched_sac.py` is CleanRL-style SAC (TF2):
  - twin Q networks with Polyak targets, tanh-squashed actor, delayed policy
    updates, automatic entropy tuning;
  - `--updates-per-step` sets the update-to-data ratio;
  - time-limit ends store the real final observation and are not terminal;
  - raw observations are normalized at sampling time.
- `batched/evaluation.py` is a shared deterministic evaluation helper.

Smoke tests of the eval script and SAC passed end to end. They are not
training runs.

### SAC update loop (same hyperparameters)

The first 2M-step SAC run (32 envs, 16 updates/step, batch 256) ran at about
300 SPS: each update synced losses to the host, sampled and normalized on the
CPU, and launched many small graphs. `batched_sac.py` now:

- keeps the replay buffer in GPU variables (uniform sampling with replacement,
  as before; normalization in float64 with the same clip);
- runs the 16 gradient steps of a vector step in one XLA-compiled
  `tf.function` (`jit_compile=True`), with the same critic / delayed actor /
  target schedule;
- accumulates losses on the GPU and reads them only at log steps.

Smoke run (61,440 steps, defaults): learning phase 351 SPS without XLA and
520 SPS with XLA, equal to the random-action warm-up rate, so updates are no
longer the bottleneck. Replay sampling now uses the TF RNG instead of numpy,
so runs are not step-for-step identical to the old script.

## Round 7: environment kernels (final)

Profiled with Nsight Systems / Nsight Compute 2026.3. At 64 envs a vector
step was ~82 ms of GPU time, split almost evenly between the FD solver
(`iterate_fused`, limited by shared-memory load instructions) and the fast-mode
convection swaps (`fast_swaps`, one warp per room with serial single-lane swap
chains; p = 1.0, so all 436k room CVs take part every step, and large rooms
fit one block per SM).

Changes (no arithmetic or swap order changed):

- `fd_kernel.cu`: `ClassStep` is 16-byte aligned with the iteration's fields
  in its first three words; the kernel loads them from shared memory as three
  `float4` loads instead of ~12 scalar loads. `storage` (used once per launch)
  is read from global memory. FD launch 4.45 → 3.18 ms (ncu, 64 envs, K=8).
- `sb1_gpu.cu` fast swaps: split into `fast_swap_list` (draws + Fisher-Yates)
  and `fast_swap_apply` (applies the swaps to a shared permutation), each with
  one room-sized shared array, so more blocks fit per SM; blocks are 256
  threads for the parallel parts (draws, gathers, in-order list compaction
  across warps); the serial chains still run on one thread in the reference
  order, prefetching the next indices before each iteration's stores. Swap
  time per 64-env step 37.8 → ~10 ms.

Verification (all pass):

- `cmp_fused.py` (CUDA FD vs the TF graph loop, every K, thresholds 0.1 and
  0.001): 0 mismatches in temperatures, iteration counts and per-iteration
  deltas.
- `sim_golden_trace check --ref golden300.npz` (single env): BIT-EXACT.
- `check_exact_engine.py` with `SB1VecBatch` B=4, `rng="exact"` and
  `rng="feed"`: BIT-EXACT against `golden300.npz`.
- `check_fast_swaps.py` (GPU vs `sb1_fast_ref.c`, p 1.0 and 0.7): BIT-EXACT.
- Fast-mode trajectory recorded before the changes (B=8, 300 steps, obs,
  rewards, full grids; `sbsim_traces/prof/fast_ref_b8.npz`): bit-identical
  after, with `gpu_parts` 1 and 2.

Throughput, environment only (fast mode, random actions):

| B | Before | After |
| --- | --- | --- |
| 32 | 653 | 1,021 |
| 64 | 705 | 1,125 |
| 128 | — | 1,191 |
| 256 | 751 | 1,207 |

SAC end to end (defaults: 32 envs, 16 updates/step): 520 → 706 SPS, with
train returns identical to the previous run. A 2M-step SAC run is now about
47 min.

Tried and dropped: consecutive rows per thread in the FD kernel (bit-exact but
79 registers, one block per SM, slower). Not pursued: replacing `__fdiv_rn`
with a reciprocal sequence, and fused multiply-adds (both change or risk
changing rounding).

## Round 8: competition reward and SAC checkpoints

### Competition reward (`SB1VecBatch(reward="competition")`, `--reward competition`)

From `competition/competition.html`: r_t = 0.6·comfort − 0.2·energy − 0.2·carbon,
comfort 0 inside the band, linear up to 5 K outside, then quadratic with no
lower limit, only in occupied zones. Choices for what the page leaves open
(confirmed with the user, 2026-10-07):

- penalty per zone p(d) = d for d ≤ 5 K, (d² + 25)/10 above (same value and
  slope at 5 K); d is the distance below the zone's heating setpoint or above
  its cooling setpoint (the thermostat schedule's band at that step);
- comfort = −mean of p over occupied zones (0 when none is occupied);
- energy and carbon = sbsim's normalized energy cost and carbon (actual over
  the cost/carbon of the capped maximum rates; the form of the paper's
  Appendix B).

Only the returned reward changes; the default `reward="sbsim"` is unchanged
(golden300 and the fast-mode trace still bit-exact). The comfort penalty is
always logged as `comfort_penalty_k`. All batched scripts take `--reward`.

Baseline (fast mode, 64 days, `batched_baseline --reward competition`):
intended −17.48 ± 0.13 (comfort −2.66, energy −11.97, carbon −2.86);
as_run −29.52 ± 0.27.

The competition evaluates other buildings with a larger action space and
30-day episodes; only the reward is matched here.

### SAC checkpoints and options (`batched_sac.py`)

- `checkpoints/best/` + `summary_best.json`: saved whenever the eval return
  improves (the 2M run's best, at 160k, had been lost).
- `resume/`: full state (networks, optimizers, alpha, observation statistics
  with count, counters, best eval, replay buffer; latest only, ~430 MB for a
  1M buffer). `--resume <run_dir>` continues with the run's config; envs
  restart at fresh episodes with seed + step. Tested: resumed at 24,576 with
  the full replay, gradient-step count and best eval carried over.
- New flags: `--grad-clip`, `--critic-layernorm`, `--target-entropy`.

Diagnosis of the 2M-step SAC run (sbsim reward): the critic diverged. Q was
≈ −10 (the right scale) at 100k–160k, when eval peaked (−27.29), then −72
(300k), −1,983 (500k), −8,096 (1M); Q loss 0.001 → 892; eval fell to −31.9.
Alpha fell to 0.001 by 40k steps. Saturated actions (2–5%) and clipped
observations (0.03%) are too rare to explain it.

### SAC ablations (competition reward, 32 envs, 16 updates/step, 600k steps)

| Variant | Q at 200k / 400k / 600k | Q loss 600k | Eval best / last 3 / final |
| --- | --- | --- | --- |
| A defaults (q-lr 1e-3, no LN) | −54 / −17,000 / −18,500 | 5,500 | −15.34 / −21.20 / −17.96 (dip to −34) |
| B q-lr 3e-4 | −7.5 / −35 / −818 | 8.3 | −15.60 / −17.46 / −17.44 |
| C LayerNorm critics + grad clip 10 | −6.9 / −7.1 / −6.6 | 0.0009 | −15.42 / −15.53 / −15.56 |
| D reward scale 10 | −61 / −174 / −997 (×10 units) | 38.5 | −15.26 / −16.22 / −16.39 |

Baseline −17.48. Only C keeps Q at its correct scale; B and D delay the
divergence to ~400k. LayerNorm in the critic is the stabilizer shared by
RLPD (Ball et al. 2023), DroQ (Hiraoka et al. 2022), BRO (Nauman et al.
2024) and SimBa (Lee et al. 2024). C is now the default
(`--critic-layernorm 1 --grad-clip 10`). Not added (untested here): weight
decay, periodic resets (Nikishin et al. 2022; D'Oro et al. 2023), HL-Gauss
critic loss (Farebrother et al. 2024).

128 envs with 64 updates/step (same UTD 0.5): 843 SPS in the learning phase
(706 at 32 envs).

10M-step run started 2026-10-07 02:15 IST:
`sac_competition_10m_2026_10_07-02:14:59` (competition reward, 128 envs, 64
updates/step, eval every 100k, checkpoints every 250k, log
`sbsim_traces/runs/sac_competition_10m.log`).

### Randomized evaluation and SAC in `batched_eval.py`

Training evals used the same 16 fast-mode days at every eval of every run.
Now `evaluate_policy(..., seed=)` calls `SB1VecBatch.reseed(seed)` first, and
PPO and SAC pass `eval_seed + global_step`, so every eval sees new occupancy
and convection draws (reproducible per step). `reseed(s)` gives exactly the
streams of a fresh engine built with seed `s` (tested over 60 steps, 4 envs).
PPO's own eval function was replaced by the shared helper. Construction is
unchanged (fast trace and golden300 still bit-exact).

`batched_eval.py` now loads PPO or SAC (from `config.json`), takes
`--run` (`--ppo-run` still accepted), `--checkpoint <step>|best` and
`--reward`. The PPO run re-evaluated with it gives the same numbers as before
(fast: −27.2070 vs baseline −30.8454).

Paired results, competition reward, identical days for every policy
(exact: 8 envs, original simulator with original/derived seeds; fast: 32
envs):

| Policy | Exact | Fast | Fast vs baseline |
| --- | --- | --- | --- |
| Baseline schedule | −17.261 | −17.686 | |
| SAC 10M, best (9.2M) | −15.165 | −15.450 | +2.236 ± 0.071 (+12.6%) |
| SAC 10M, final | −15.173 | −15.468 | +2.218 ± 0.076 (+12.5%) |
| PPO 2M (trained on the sbsim reward) | −15.063 | −15.339 | +2.347 ± 0.072 (+13.3%) |

PPO − SAC best, paired: +0.111 ± 0.032 (fast, PPO better in 30/32),
+0.103 ± 0.077 (exact, 7/8). Same comfort (penalty 3.73 K vs 4.76 K for the
baseline, fast); PPO uses less energy (−10.55 vs −10.64) and carbon. SAC's
eval was within ~1 point of its final value from 100k steps on; the extra
9.9M steps changed little.

SAC 2M (`sac_competition_2m_2026_10_07-07:39:40`, same settings as the 10M
run, randomized evals, 41 min): best (1.5M) fast −15.447, exact −15.153;
final fast −15.460, exact −15.119. Same as SAC 10M within noise. PPO 2M − SAC
2M best, paired: +0.108 ± 0.039 (fast), +0.090 ± 0.037 (exact).

### PPO checkpoints

`batched_ppo.py` now has the same checkpoint layout as SAC:
`checkpoints/ckpt-<update>`, `checkpoints/best/` + `summary_best.json` (saved
when the eval return improves) and `resume/` (agent, optimizer, observation
and reward-scaling statistics with counts, update/step counters, best eval;
latest only). `--resume <run_dir>` continues with the run's config;
`--total-timesteps` may be raised and the learning rate anneals over the new
total. Tested: 6 updates, resumed to 10 (best eval and counters carried over,
explained variance 0.96 after resume); `batched_eval --checkpoint best` loads
it.

PPO 10M on the competition reward (`ppo_competition_10m_2026_10_07-10:55:01`,
64 envs, randomized training evals; training eval −18.48 at the first eval,
−15.39 at ~1M, −15.14 at ~2M, last 10 mean −15.32). Paired, identical days:

| Policy | Exact | Fast | Fast vs PPO 2M |
| --- | --- | --- | --- |
| PPO 10M best (update 936, 7.7M) | −15.080 | −15.396 | −0.057 ± 0.033 |
| PPO 10M final | −15.055 | −15.387 | −0.048 ± 0.034 |
| PPO 2M (sbsim reward) | −15.063 | −15.339 | |
| SAC 2M / 10M best | −15.153 / −15.165 | −15.447 / −15.450 | −0.108 / −0.111 |

The 10M PPO run is no better than the 2M one; its training-eval best
(−14.88) was a favorable set of random days (paired: −15.40). All trained
policies end within ~0.1 of each other (comfort ≈ −2.24, energy −10.55 to
−10.64, carbon ≈ −2.56 on fast mode), against −17.69 for the baseline. With
every episode replaying 2023-07-06 and two setpoints to act on, the learners
reach the same plateau within 1–2M steps.
