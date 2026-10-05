# Plan: batched GPU SB1 environment that keeps the core system

Goal: run many SB1 environments per step on the GPU, fast enough for
PPO-style training, while keeping the core system (physics, HVAC logic,
reward, observation) identical to the original sbsim code.

## Verification rule

Every phase must reproduce the original bit for bit in **exact mode**. In
exact mode the random numbers are the original streams: Python `random` for
the convection shuffle and numpy `RandomState(17321)` for occupancy. The
checks are:

- `golden300.npz` (300 steps, crossing an episode end and the reset).
- A per-step intermediate trace: zone means, thermostat modes, VAV settings,
  setpoints, air handler and boiler state, reward info, and observations, so
  a mismatch can be located exactly.

Only after a phase matches in exact mode is the fast RNG mode switched on.
Fast mode keeps the same rules with different draws.

## What runs in a step (read from the code)

| Stage | Original | Exactness notes |
| --- | --- | --- |
| Action | `BoundedActionNormalizer.setpoint_value` (float32 action, float64 math) | Stored in proto `float`, so rounded to float32 |
| `setup_step_sim` | zone means → `Vav.update_settings` → `Thermostat.update` | Thermostat mode and previous timestamp persist across episode resets |
| `set_action` | sets boiler water and AHU heating setpoints, plus the action timestamp | Device action/observation timestamps persist across resets |
| `execute_step_sim` | zone means, recirculation = global mean, ambient (weather), supply air temp, FD, convection shuffle, VAV outputs, AHU/boiler demand (sequential float64 add, AHU clamp each add), thermal power to diffusers, return water temp | Room-dict order for every sequential sum |
| Observation | device getters in sorted device/field order, proto float32, normalizer `(v - mean32)/sqrt(var32)` → float32, histogram (clipped, last edge repeated, divided by count sum), time features, comfort now/soon, occupants `int(sum)` over [t-5m, t) | Boiler `supply_water_temperature_sensor` read has side effects (tank temperature) |
| Reward | zone infos (setpoints, zone mean, flows, occupancy over [t, t+5m)), AHU and boiler energy, regret function | All RewardInfo/RewardResponse fields are proto float32; zone sums follow the proto map order |
| Occupancy | one shared numpy `RandomState(17321)` for all zones, a Markov chain advanced by every query, never reset | Call order: observation occupants, then reward infos |

## Verified numerics (probes in `/data/smart-building/sbsim_traces/probes/`)

- P1 `np.mean(float32)`: start from 0 and add, in sequence, the numpy pairwise
  sum of each consecutive 8192-element chunk (float32). Then divide by n
  (float32 and float64 division agree for n < 2^24). 137/137 exact, including
  the 744×1004 grid.
- P2 `np.exp` ≠ libm `exp` (~24% of inputs), but `np.exp(array)` == `np.exp(scalar)`.
  So productivity is computed with vectorized numpy.
- P3 Pint electricity cost and carbon == `(price * |rate|) * dt`; gas likewise.
- P4 The proto map order of `zone_reward_infos` is deterministic across steps
  and processes. It is captured from a real RewardInfo.
- P5 numpy `rand()` == the MT19937 53-bit draw from the same state (same as
  Python `random()`).
- P6 Proto float storage == `np.float32` rounding.

## Architecture

- Built from a real gin `Environment`: all constants, orders, normalizer
  lookups, action ranges, field order, and per-step-index time tables come
  from the original objects. Every episode starts at the same timestamp and
  advances 300 s per step, so time-dependent values (comfort mode, setpoint
  windows, time features, ambient temperature, prices, carbon, work-day flags)
  are tabulated by step index using the original functions.
- GPU: the temperature grid stays resident. Per-cell work runs on the GPU:
  - zone means and the recirculation mean (exact chunked pairwise sum),
  - FD (`fd_kernel.cu`),
  - thermal power to diffusers,
  - convection swaps.
- Per-env scalar logic (thermostat, VAV, AHU, boiler, occupancy, observation
  assembly) runs in C over structure-of-arrays for all envs, with float64
  operations in the original order. The reward uses vectorized numpy, for
  `np.exp`.
- The convection shuffle is sequential per room. Rooms are disjoint, so on the
  GPU each (env, room) is one sequential task.
  - Exact mode: the swap lists come from the Python MT stream (existing C
    code); the GPU applies them.
  - Fast mode: each (env, room) uses its own counter-based RNG stream with the
    same generation code. The generator is proven by feeding it the exact MT
    words it would have consumed.

## Phases

1. Exact batched engine, B=1, CPU/GPU hybrid, matching `golden300.npz` and the
   intermediate trace.
2. B>1 exact, with each env having its own copies of the RNG streams. Each
   env must equal an independent original run given the same actions.
3. Move zone means, recirculation, thermal power, and convection apply to the
   GPU. Each must be bit-exact against phase 1.
4. Fast RNG mode (GPU swap-list generation, GPU occupancy). Validate the
   generators on exact MT words.
5. PufferLib-style vector env API, benchmark, then FD kernel speedups:
   - skip the arithmetic for exterior cells (result is ambient regardless),
   - exact fixed-divisor division, verified exhaustively per divisor.
