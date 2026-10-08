These instructions apply only to work inside sbsim/.

# Agent guide for sbsim

## 1. Purpose

sbsim is the Smart Buildings Control simulator, reward, and TF-Agents environment used to train HVAC setpoint policies. Poetry package name is `smart_control` 0.1.0. `METADATA` names the export `sb-sim`. Python is `>=3.10,<3.12`. Upstream repo is https://github.com/google/sbsim. Six years of real building data are separate (TFDS `smart_buildings`, or the smaller `sb1.zip`). `BuildingDataset.__init__` defaults `dataset_id="sb1"` and `download=True`. `download()` GETs `https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/{id}.zip` into `data/`, unpacks to `data/{id}`, and skips the download if that directory already exists. Timeout default is 60s. `VALID_DATASET_PARTITIONS` lists only `sb1`: `2022_a`, `2022_b`, `2023_a`, `2023_b`, `2024_a`.

## 2. Layout

Package root is `smart_control/`.

- `proto/` — `smart_control_building.proto`, `smart_control_reward.proto`, `smart_control_normalization.proto`, plus generated `*_pb2.py`.
- `models/` — abstract `BaseBuilding`, `BaseOccupancy`, `BaseRewardFunction`, `BaseEnergyCost`, `BaseObservationNormalizer`, `BaseActionNormalizer`.
- `simulator/` — finite-difference building, HVAC devices, occupancy, weather. `rejection_simulator_building.py` wraps a `BaseBuilding` and rejects the first N actions.
- `reward/` — productivity / energy / carbon reward and cost models.
- `environment/environment.py` — TF-Agents `Environment`.
- `utils/` — normalizers, proto I/O, histogram reducer, energy conversions, constants, `regression_building_utils.py` (time-feature names), `run_command_predictor.py`.
- `dataset/` — `BuildingDataset`, `BuildingDatasetPartition`.
- `configs/resources/sb1/` — SB1 gin, floor plan `double_resolution_zone_1_2.npy`, `reset_temps.npy`, Moffett Field weather CSV.
- `reinforcement_learning/` — `scripts/train.py`, `scripts/populate_starter_buffer.py`, SAC agent, schedule policy, Reverb buffer, `utils/config.py`. Also `observers/` (`Observer`, `CompositeObserver`, `PrintStatusObserver`, `RenderingObserver`), `utils/metrics.py` (`get_trajectory`, `compute_avg_return` with defaults `num_episodes=1` and `num_steps=6`), `utils/data_processing.py` (`get_latest_episode_reader`, `get_energy_timeseries`, `get_outside_air_temperature_timeseries`, `get_reward_timeseries`, `get_zone_timeseries`, `get_action_timeseries`), `utils/time_utils.py` (`to_dow`, `to_hod`).
- `batched/` — batched SB1 engines built from a fresh gin `Environment`: `SB1ExactBatch` (`sb1_exact.py`, host grids), `SB1GpuBatch` (`sb1_gpu_engine.py`, GPU-resident grids via `sb1_gpu.cu`/`sb1_gpu.py`), `SB1VecBatch` (`sb1_vec.py`, vectorized scalar logic, occupancy in `sb1_occupancy.c`). `rng="exact"` reproduces the original bit for bit; `"feed"` generates swaps on the GPU from recorded MT words (validation); `"fast"` uses per-env Philox streams (same rules, independent draws; reference in `sb1_fast_ref.c`). `reward="competition"` returns the competition reward (0.6 comfort − 0.2 energy − 0.2 carbon; see `changes.md` Round 8) instead of the gin regret reward. Design and verified numerics: `port_plan.md`; results: `changes.md`.
- `notebooks/` — demos. Notebook cells are not the library API.
- `docs/` — MkDocs sources. Built site goes to `docs_site/`.

`BuildingDataset` also exposes `floorplan` (0 inside, 1 wall, 2 outside), `device_infos`, `devices_df`, `zone_infos`, `zones_df`, `actionable_fields`, and `observable_fields`. `BuildingDatasetPartition` matrices are `action_value_matrix`, `observation_value_matrix`, `reward_value_matrix`, and `reward_info_value_matrix`, with dataframes `actions_df`, `observations_df`, `rewards_df`, and `reward_infos_df`. Column ids are `device_id@field_name`, not `{device_id}_{field}`.

Skip `.venv`, `.git`, `poetry.lock`, `uv.lock`, generated `*_pb2.py` (edit the `.proto`), large csv/npy dumps, images, and Bazel `BUILD` files.

## 3. How to run and test

From the sbsim root, with Poetry 2.1.2 (`pip install poetry==2.1.2` in `docs/setup/linux.md`, `docs/setup/mac.md`, and CI). Linux setup then runs `poetry env use 3.11`. `pyproject.toml` pins `tensorflow = "2.15.1"` and `tf-agents = "0.18.0"`. Dependency groups used by the install commands are `dev`, `docs`, and `notebooks`.

```sh
poetry install --with dev
poetry install --with notebooks   # Jupyter
poetry install --with docs        # MkDocs
```

Docker (`Dockerfile`) is `python:3.11-slim` on `linux/amd64`, `poetry install --no-root --with dev,notebooks`, then regenerates protos with the image's protoc 3.21.12. Default command is Jupyter on port 8888 with an empty token. `docs/setup/docker.md` builds with `docker build -t sbsim:latest .`.

```sh
make test          # poetry run pytest --disable-pytest-warnings
make format        # mdformat README.md docs, then pyink, isort, pylint
make docs          # poetry run mkdocs serve  (http://localhost:8000)
make docs-build    # output directory docs_site/
```

Tests sit beside the code as `*_test.py` (`absltest`, `tf.test`, `unittest`) and are collected by pytest. CI (`.github/workflows/build.yaml`) is Ubuntu 3.10.15 and 3.11, plus macOS Intel and arm on 3.11. It runs pyink, isort, pylint, `pyrefly check --config pyrefly.toml || true`, `mdformat README.md docs --check`, then pytest.

Bit-exact reference trace of the 1-day env (records or checks temps, observations, rewards, step types, and the Python `random` state):

```sh
poetry run python -m smart_control.reinforcement_learning.scripts.sim_golden_trace record --out golden.npz --steps 300
poetry run python -m smart_control.reinforcement_learning.scripts.sim_golden_trace check --ref golden.npz
```

Any change to the simulator, step logic, or batched engines should keep `check` at `BIT-EXACT` against a trace recorded from unmodified code.

GPU paths need `nvcc` (CUDA kernels are built on first use with `-arch=native`) and the `tensorflow[and-cuda]==2.15.1` wheels in the venv; without them they fall back to TF/Python. Compiled libraries are cached by source hash in `~/.cache/smart_control/`. When TensorFlow also uses the GPU, set `TF_FORCE_GPU_ALLOW_GROWTH=true`.

Training scripts (no Makefile target):

```sh
poetry run python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer --buffer-name NAME
poetry run python -m smart_control.reinforcement_learning.scripts.train --starter-buffer-path PATH --experiment-name NAME
```

`populate_starter_buffer` CLI defaults: capacity 50000, steps-per-run 100, num-runs 5, sequence-length 2, gin `train_sim_configs/sim_config_1_day.gin`. Buffer path is `smart_control/reinforcement_learning/data/starter_buffers/{name}_seqlen{seq}_exp{runs*steps}`. It refuses to overwrite an existing directory.

`train` CLI defaults: `--agent-type sac`, `--train-iterations 100`, `--collect-steps-per-training-iteration 50`, `--batch-size 256`, `--eval-interval 10`, `--num-eval-episodes 1`, `--log-interval 1`, `--checkpoint-interval 10`, `--learner-iterations 200`. The `train_agent()` function defaults differ (100000 iterations, 1 collect step, eval every 1000, 5 eval episodes, log every 100, checkpoint every 1000). The script always loads `sim_config_1_day.gin`. Results go under `smart_control/reinforcement_learning/experiment_results/{experiment_name}_{timestamp}`. `train.py` sets `WRAPT_DISABLE_EXTENSIONS=true` before importing TensorFlow.

Protos, from `smart_control/proto`, with protoc 3.21.12:

```sh
protoc --python_out=. smart_control_building.proto
protoc --python_out=. smart_control_normalization.proto
protoc --python_out=. smart_control_reward.proto
```

Checked-in `*_pb2.py` exist so contributors can skip regeneration. Do not hand-edit them.

## 4. Architecture invariants

`Environment` in `environment/environment.py` subclasses `tf_agents.environments.py_environment.PyEnvironment`. Gin is the composition root. `load_environment` in `reinforcement_learning/utils/environment.py` calls `gin.clear_config()`, `gin.parse_config_file`, then `Environment()`. `create_and_setup_environment` then assigns `metrics_path` and `occupancy_normalization_constant` (default `DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT = 125.0`).

`reinforcement_learning/utils/config.py` imports gin classes even when unused, so `@gin.configurable` registration happens before parse. Current imports: `ElectricityEnergyCost`, `NaturalGasEnergyCost`, `SetpointEnergyCarbonRegretFunction`, `AirHandler`, `Boiler`, `MaterialProperties`, `FloorPlanBasedHvac`, `RandomizedArrivalDepartureOccupancy`, `SimulatorBuilding`, `StochasticConvectionSimulator`, `TFSimulator`, `ReplayWeatherController`, `controller_reader`, `histogram_reducer`, `ProtoWriterFactory`, `to_timestamp`, `StandardScoreObservationNormalizer`. A new gin class used from a config must be imported there (or pulled in by one of those imports) before `parse_config_file`. `Environment` and `ActionConfig` live in `environment.py`, which `load_environment` imports. `FloorPlanBasedBuilding` and `SimulatorFlexibleGeometries` are registered because `tf_simulator.py` imports those modules. `SetpointSchedule` is registered because `hvac_floorplan_based.py` imports it.

Paths from `config.py` (relative to repo root via `utils/constants.py` `ROOT_DIR`):

- `DATA_PATH` = `smart_control/configs/resources/sb1`
- `CONFIG_PATH` = `.../sb1/train_sim_configs`
- zone / floor plan = `double_resolution_zone_1_2.npy`
- weather = `local_weather_moffett_field_20230701_20231122.csv`
- reset temps = `reset_temps.npy`

Only setpoints that have an entry in `ActionConfig.action_normalizers` enter the action vector. Lookup is by setpoint name (`get_action_normalizer`), not by device id. Within a device, setpoints are sorted alphabetically. The spec is one `BoundedArraySpec` of `float32` in `[-1, 1]`. Field ids are `{device_id}_{setpoint}` with a numeric suffix on collision (`generate_field_id`). When `device_id` is omitted, `air_handler.py` uses `air_handler_id_{uuid4()}` and `boiler.py` uses `boiler_id_{uuid4()}`, so those action field ids change across processes. Floor-plan VAV ids are stable: `hvac_floorplan_based.py` sets `vav_{z}`.

Observations are normalized with `StandardScoreObservationNormalizer`: `(value - mean) / sqrt(variance)`, or `0` when variance is 0. Unknown field names fall back to mean 0 and variance 1. Missing or invalid observations are filled from the previous response; if there is no previous response, `replace_missing_observations_past` raises `ValueError`.

`discount_factor` must be in `(0, 1]`. The constructor sets the episode end to `start + pd.Timedelta(num_days_in_episode, unit="days")`, then `_num_timesteps_in_episode = int((end - start) / step_interval)`. Default `step_interval` is `pd.Timedelta(5, unit="minutes")`. `_has_episode_ended` is `_step_count >= _num_timesteps_in_episode`. `_step` calls `building.wait_time()` before that check. `_step_count` increments only on the non-terminal return, so the terminal step advances the simulator once after the count has already reached the limit. `steps_per_episode` is `(end - start).total_seconds() // building.time_step_sec`, not a division by `step_interval`. SB1 gin sets simulator `time_step_sec` to 300, and the environment default step interval is also 300 s. Changing only one desynchronizes `steps_per_episode` from termination.

`Environment` stores `run_command_predictors` and never calls them. `RunCommandPredictor` and `RandomForestLearningAlgorithm` live in `utils/run_command_predictor.py`.

## 5. Simulator model

`Simulator` (`simulator/simulator.py`) is a finite-difference / finite-volume thermal model. Control volumes exchange heat by conduction with neighbors. Boundary CVs also convect with ambient air. Interior CVs can take a diffuser heat source. Interior-mass coupling and interior longwave run only when the building sets `include_interior_mass` or `include_radiative_heat_transfer`. Both default `False` on `FloorPlanBasedBuilding`, and SB1 gin does not set them, so those terms do not run. The base simulator does not apply exterior longwave radiation or shortwave solar gains. Update formulas are `_get_corner_cv_temp_estimate`, `_get_edge_cv_temp_estimate`, `_get_interior_cv_temp_estimate`, and `update_interior_mass_temperatures`. `finite_differences_timestep` stops when `max_delta <= convergence_threshold` or `iteration_limit` is exhausted. `__init__` stores `relative_convergence_threshold` (default `1e-6`) and `relative_convergence_streak` (default 20) and never reads them.

Class chain used by SB1: `TFSimulator` (`tf_simulator.py`) subclasses `SimulatorFlexibleGeometries` (`simulator_flexible_floor_plan.py`), which subclasses `Simulator`. `TFSimulator` overrides `update_temperature_estimates`, `update_interior_mass_temperatures`, and `finite_differences_timestep`. Without interior mass or radiation (SB1), its timestep runs the CUDA kernel `simulator/fd_kernel.cu` (loader `cuda_fd.py`, `use_cuda_fd_kernel=True`, `cuda_fd_fuse_k=8`) when nvcc and a GPU are available, otherwise one `tf.function` (`use_graph_fd_loop=True`) with grappler disabled for that call only, otherwise the parent loop. All three give bit-identical float32 results; the kernel uses explicit round-to-nearest ops, `-fmad=false`, and a float64 convergence test, like numpy. `execute_step_sim` is overridden on `SimulatorFlexibleGeometries` (zone temps, HVAC, weather, then the finite-difference step); `TFSimulator` inherits that method. Legacy gin selects `SimulatorFlexibleGeometries`, so the temperature update stays the Python `Simulator.update_temperature_estimates`. After the tensor update, `apply_exterior_temps` writes ambient temperature onto exterior CVs. Floor-plan file codes in `simulator/constants.py` are `0` interior, `1` wall, `2` exterior. `SIM_VIDEOS_DIRPATH` overrides the default directory `os.path.join(ROOT_DIR, "smart_control", "simulator", "videos")`.

`FloorPlanBasedBuilding` (`building.py`) subclasses `BaseSimulatorBuilding`. The constructor requires `cv_size_cm`, `floor_height_cm`, `initial_temp`, and exactly one of `floor_plan` / `floor_plan_filepath` and exactly one of `zone_map` / `zone_map_filepath`. Passing both members of a pair, or neither, raises `ValueError`. Zone ids from a floor-plan room key are `'zone_id_' + identifier.replace('room_', '')` (`conversion_utils.floor_plan_based_zone_identifier_to_id`). Thermal constants are `MaterialProperties(conductivity, heat_capacity, density)`. Gin wires `inside_air_properties` and `building_exterior_properties` in the labeled order. `inside_wall_properties` does not: `heat_capacity = %interior_wall_cv_density` and `density = %interior_wall_cv_heat_capacity`. With the current numbers that assigns heat capacity 1.0 and density 700.0. Leave that assignment as written.

`RejectionSimulatorBuilding` wraps a `BaseBuilding`. The first `initial_rejection_count` calls to `request_action` (default 0) raise `RuntimeError('PhysicalAssetService.WriteFieldValues')`. `reset` zeroes that counter. It is not wired in the SB1 gin files.

Current SB1 gin (`configs/resources/sb1/sim_config.gin`):

- `TFSimulator` + `FloorPlanBasedBuilding` + `FloorPlanBasedHvac` + `StochasticConvectionSimulator` (`p=1.0`, `distance=5`, `seed=5`) + `ReplayWeatherController` + `RandomizedArrivalDepartureOccupancy` + `SimulatorBuilding` + `SetpointEnergyCarbonRegretFunction`.
- CV 10 cm, floor height 300 cm, initial temp 294 K.
- Exterior conductivity 0.05 W/m/K, density 1.0 kg/m3, heat capacity 700 J/kg/K. Interior air and the labeled interior-wall constants use conductivity 50, density 1.0, heat capacity 700 (wall object swaps the last two; see above).
- Time step 300 s, `convergence_threshold` 0.1 (minimum iteration temperature change to finish an FDM step; the gin file does not write a unit), iteration limit 100, warning at 30. Start `2023-07-06 07:00:00+00:00`.
- Convection coefficient 100 W/m2/K. `ReplayWeatherController` reads the Moffett CSV (`Time` `%Y%m%d-%H%M` UTC, `TempF`, `Humidity`). `WeatherController` is the unused sinusoid alternative (low at midnight, high at noon).
- `num_days_in_episode=14`, `discount_factor=0.9`, `metrics_reporting_interval=10`, label `tunable_simulator_sb1`, one hour-of-day and one day-of-week sin/cos pair. Histogram reducer comes from `get_histogram_reducer()`.

`sim_config_legacy.gin` is the same stack except the simulator class is `SimulatorFlexibleGeometries` and `Environment.observation_histogram_reducer` is `@HistogramReducer()` with `normalize_reduce = True` and `reader = @controller_reader.ProtoReader()`. `train_sim_configs/sim_config_1_day.gin` matches current SB1 except `num_days_in_episode=1`.

HVAC assumes one boiler, one air handler, one VAV per zone (`Hvac` for the rectangular building, `FloorPlanBasedHvac` for the floor plan). `FloorPlanBasedHvac` builds VAVs from `building._room_dict` when `fill_zone_identifier_exogenously` is true (the default when `zone_identifier` is omitted). Both `Hvac` and `FloorPlanBasedHvac` set `ZoneInfo.building_id` to `US-SIM-001`.

Device roles:

- `AirHandler`: recirculation, heating/cooling air setpoints, fan pressure and efficiency, `max_air_flow_rate` default 8.67. Gin: recirculation 0.3, heating setpoint 285 K, cooling 298 K, fan dP 10000 Pa, efficiency 0.9. Actions: `supply_air_heating_temperature_setpoint`, `supply_air_cooling_temperature_setpoint`.
- `Boiler`: reheat water setpoint, pump head and efficiency, heating/cooling rates in K/min. Gin: setpoint 360 K, head 6.0 m, efficiency 0.98, heating 0.5 K/min, cooling 0.1 K/min. Action: `supply_water_setpoint`.
- `Vav`: damper and reheat valve driven by `Thermostat`. The only action field is `supply_air_damper_percentage_command` (attribute `damper_setting`). Observables include that command, `supply_air_flowrate_setpoint` (`max_air_flow_rate`), and `zone_air_temperature_sensor`. Reheat is not an action field. Gin max air flow 0.035, reheat water flow 0.03. The damper is not in the SB1 `action_normalizer_map`, so it is not an agent action. An omitted `device_id` becomes `vav_id_{uuid4()}`; `FloorPlanBasedHvac` passes `vav_{z}`.
- `Thermostat`: comfort mode heats below the heating setpoint and cools above the cooling setpoint until the midpoint; otherwise off. Eco mode starts in passive cool until the eco heating setpoint is crossed. `SetpointSchedule` treats weekends as night/eco. Gin comfort window 06:00–19:00 `US/Pacific`, day (294 K, 297 K), night (289 K, 298 K).
- `SmartDevice`: maps observable and action field names onto attributes and emits building protos.

Occupancy (`BaseOccupancy.average_zone_occupancy`, local timestamps; unknown zone should raise `ValueError`):

- `StepFunctionOccupancy`: one work level and one non-work level for every zone.
- `RandomizedArrivalDepartureOccupancy`: `zone_assignment` occupants arrive and leave inside hour windows; default seed 17321. SB1 gin uses 1 occupant, arrival hours 7–12, departure 13–18, `time_step_sec` 300, `US/Pacific`.
- `LIGHTSWITCHOccupancy` (`stochastic_occupancy.py`): arrival, departure, and lunch sampled from cumulative distributions. Default lunch 12–14, step 3600 s, seed 511211.
- `EnhancedOccupancy`: minute-level schedules resampled each day, worker types `WEEKDAY_ONLY`, `WEEKEND_REGULAR`, `WEEKEND_OCCASIONAL`. Defaults include 5% weekend-regular, 5% weekend-occasional, 10% daily probability for occasional weekend work.

`solar_radiation.py` computes irradiance (`clearsky`, `linear`, `campbell_norman` via pvlib), solar position, and sky temperature. Default `BuildingInfo` lat/lon is 37.4263, -122.0349, `US/Pacific`. It is not wired into `sim_config.gin`. Sensor names: `ghi_sensor`, `dni_sensor`, `dhi_sensor`.

`StochasticConvectionSimulator` shuffles CVs with probability `p` up to `distance` cells. `_shuffle_max_dist` runs in C (`convection_shuffle.c`, `use_compiled_shuffle=True`, compiled with gcc into `~/.cache/smart_control/`). It consumes Python's global `random` stream exactly like `random.uniform/choice/shuffle` and writes the state back.

## 6. RL training loop

`populate_starter_buffer.py` rolls out `create_baseline_schedule_policy` into a Reverb buffer. `train.py` builds separate train and eval `TFPyEnvironment`s, creates the agent, loads that buffer at capacity 50000 and `sequence_length=2`, samples with `batch_size` and `num_steps=2`, and loops: eval actor, collect actor (`steps_per_run` = collect steps), `Learner.run(iterations=learner_iterations)`, periodic `replay_buffer.py_client.checkpoint()`. Policies are written by `PolicySavedModelTrigger` every `eval_interval`.

`ReplayBufferManager` uses table name `uniform_table`, `reverb.selectors.Uniform` sampler, `Fifo` remover, `MinSize(1)` rate limiter, and `ReverbAddTrajectoryObserver` with stride 1. `dm-reverb` 0.14.0 is the Linux dependency. On Darwin, Poetry installs `dm-reverb-macos` 0.11.0.dev0, which is broken for replay buffers (https://github.com/google/sbsim/issues/102). Other Mac code paths can still run.

`--agent-type` choices are `sac` and `td3`. Only `sac` is implemented (`create_sac_agent`). Any other value raises `ValueError`. TD3, DDPG, and CQL exist as notebooks, not as `train.py` agents.

SAC defaults in `create_sac_agent` / `sac_networks.py`:

- Actor MLP `(256, 256)` with ReLU and `glorot_uniform` (`dense` in `sac_networks.py`). `create_sequential_actor_network` then applies one `_TanhNormalProjectionNetworkWrapper` over the whole `action_tensor_spec` via `tf.nest.map_structure`. The environment action spec is one `BoundedArraySpec` of shape `(n,)`, so that is one projection for the vector.
- Critic: observation MLP `(256, 128)`, action MLP `(256, 128)`, concatenate, joint MLP `(256, 128)`, dense 1.
- Adam learning rates `3e-4` for actor, critic, and alpha.
- `gamma=0.99`, `target_update_tau=0.005`, `target_update_period=1`, `reward_scale_factor=1.0`, TD loss `tf.math.squared_difference`.
- This `gamma` is the agent's discount. The environment gin discount is 0.9 and is what `ts.transition` stores.

`SchedulePolicy` reads `hod_sin_000`, `hod_cos_000`, `dow_sin_000`, `dow_cos_000` (names from `get_time_feature_names` with `num_hod_features=1`, `num_dow_features=1`). The branch is `weekday_schedule if dow < 5 else weekend_schedule`. There is no holiday test. The weekend list is named `weekend_holiday_schedule_events` and holds air heating at 285 K and water at 315 K at both `Timedelta(6, hour)` and `Timedelta(19, hour)`. Weekday events (Kelvin): hour-6 air heating 292 and water 350; hour-19 air heating 285 and water 315. Lookup time is `pd.Timedelta(hod, unit="hour") + local_start_time.utcoffset()`, so those events are not local 06:00 and 19:00. `SetpointSchedule` holidays are a set of day-of-year ints, default empty, and SB1 gin does not set them. Policy `DeviceType` is `AC=0`, `HWS=1`, not the proto enum. `create_baseline_schedule_policy` writes the action array as `[AC supply_air_heating_temperature_setpoint, HWS supply_water_setpoint]`. The environment walks `SimulatorBuilding.devices` (boiler, air handler, then `hvac.vavs.values()`) and sorts setpoints inside each device, so the SB1 action vector is water then heating air. Values are mapped through `BoundedActionNormalizer.agent_value`.

Notebook opening markdown is the same SBSim tutorial title. In-notebook text names the algorithm: `SAC_Demo.ipynb` Soft Actor-Critic, `TD3_Demo.ipynb` Twin Delayed DDPG, `DDPG_Demo.ipynb` Deep Deterministic Policy Gradient, `CQL_Demo.ipynb` Conservative Q-Learning. Setup docs tell you to point notebook `data_path` at `smart_control/configs/sim_config.gin`. That file is not in the tree. The gin file is `smart_control/configs/resources/sb1/sim_config.gin`.

## 7. Reward and energy

Proto comment and `SetpointEnergyCarbonRewardFunction` use `r = s(setpoint) - u * f(cost) - w * g(carbon)`. The class that SB1 gin actually instantiates is `SetpointEnergyCarbonRegretFunction`.

Zone productivity (`BaseSetpointEnergyCarbonRewardFunction._get_zone_productivity_reward`), with `k = productivity_decay_stiffness` and `d = productivity_midpoint_delta`:

- `T < T_heat`: `Pmax / (1 + exp(-k * (T - (T_heat - d))))`
- `T > T_cool`: `Pmax * (1 - 1 / (1 + exp(-k * (T - (T_cool + d)))))`
- else: `Pmax`

Multiplied by zone occupancy and `dt_sec / 3600`. Electrical power is the sum of air-handler blower power, `abs(air conditioning power)`, and boiler pump power. Gas power is the sum of boiler `natural_gas_heating_energy_rate`.

`SetpointEnergyCarbonRewardFunction.compute_reward` (not selected by current gin):

```text
raw = productivity
      - energy_cost_weight * (electricity_cost + gas_cost)
      - carbon_cost_weight * (carbon_kg * carbon_cost_factor)
agent_reward_value = (raw - reward_normalizer_shift) / reward_normalizer_scale
```

Defaults: shift 0, scale 1.

`SetpointEnergyCarbonRegretFunction.compute_reward` clamps productivity to at least `min_productivity_personhour_usd * occupancy * dt / 3600`. If occupancy is 0, `normalized_productivity_regret = 0`. Otherwise:

```text
normalized_productivity_regret =
    (actual - min_productivity) / (max_productivity - min_productivity) - 1
```

Electricity and gas rates are capped at `max_electricity_rate` and `max_natural_gas_rate` before cost and carbon. Normalized energy cost is actual combined USD divided by the cost of those two caps. Normalized carbon is the same ratio in kg. Then:

```text
raw = normalized_productivity_regret * productivity_weight
      - normalized_energy_cost * energy_cost_weight
      - normalized_carbon_emission * carbon_emission_weight
agent_reward_value = raw / (productivity_weight + energy_cost_weight + carbon_emission_weight)
```

The batched engines can return the competition reward instead (`SB1VecBatch(reward="competition")`, scripts `--reward competition`).

SB1 gin: `max_productivity_personhour_usd=300`, `min_productivity_personhour_usd=100`, `productivity_midpoint_delta=0.5`, `productivity_decay_stiffness=4.3`, `max_electricity_rate=160000`, `max_natural_gas_rate=400000`, weights productivity 0.2, energy 0.4, carbon 0.4. The module docstring writes a different regret expression, `(s - s_max) / s_max`. The implementation above is what runs. `compute_reward` on the base class raises `NotImplementedError`.

`ElectricityEnergyCost`: 24 hourly weekday prices in cents/kWh are 16 (hours 0–5), 18 (6–11), 20 (12–18), 16 (19–23). Weekend is 16 all day. Carbon is `CARBON_EMISSION_BY_HOUR` in kg/MWh (comment: Google Carbon Free Reporting Dashboard, US-SVL-BORD1212), about 88.2 at hour 0, peak 104.74 at hour 15, 89.0 at hour 23. Prices convert by `/100/1000/3600` to USD per W·s; carbon by `/1e6/3600` to kg per W·s. Cost and carbon use `abs(energy_rate)`. Intervals longer than 3600 s log a warning. `cost()` uses `WEEKDAY_PRICE_BY_HOUR` when `conversion_utils.is_work_day` is true and `WEEKEND_PRICE_BY_HOUR` otherwise. `is_work_day` is `weekday() < 5` and the date is not in `holidays.US()`. US holidays use the weekend price table.

`NaturalGasEnergyCost`: 2020 EIA monthly USD per thousand cubic feet `GAS_PRICE_BY_MONTH_SOURCE` = 9.02, 8.35, 7.77, 7.26, 6.69, 6.86, 6.77, 6.76, 6.99, 7.19, 7.96, 8.98 (Jan–Dec). Divided by `KWH_PER_KFT3_GAS` (293.07107) and `JOULES_PER_KWH` (3.6e6) to USD/J. Carbon uses `GAS_CO2` 53.12 kg per thousand cubic feet, same conversion to kg/J. Negative gas rate is set to 0. Cost is `price * energy_rate * dt`.

`utils/energy_utils.py` public helpers: `get_water_vapor_partial_pressure`, `get_humidity_ratio`, `get_air_conditioning_energy_rate`, `get_fan_power`, `get_air_volumetric_flowrate`, `get_compressor_power_thermal`, `get_compressor_power_utilization`, `get_water_pump_power`, `get_water_volumetric_flow_rate`, `get_water_heating_energy_rate`, `get_water_heating_energy_rate_primary`, `get_water_heating_energy_rate_primary_secondary`. That module reads `AIR_HEAT_CAPACITY` (1006) and `WATER_VAPOR_HEAT_CAPACITY` (1863.8). `get_fan_power` and `get_water_pump_power` multiply by the literal `0.746`. `get_water_heating_energy_rate` uses `500.0`. `Boiler.compute_pump_power` uses `WATER_DENSITY` (1000) and `GRAVITY` (9.8). `WATER_HEAT_CAPACITY`, `WATTS_PER_HORSEPOWER`, `BTU_PER_KWH`, and `JOULES_PER_BTU` are defined in `utils/constants.py` and are not referenced by `energy_utils.py`.

`ACTION_REJECTION_REWARD` is `-np.inf` (`-np.inf`). `_step` returns it when `request_action` raises `RuntimeError` or any `SingleActionResponse` is not `ACCEPTED`. TimeStep cannot carry a rejection flag, and NaN rewards are rejected by TF-Agents, so `-inf` is the signal. A caller that should drop the transition has to notice it.

## 8. Environment API

Constructor arguments that matter: `building`, `reward_function`, `observation_normalizer`, `action_config`, `discount_factor` (default 1; gin 0.9), `num_days_in_episode` (default 3; gin 14 or 1), `default_actions`, `device_action_tuples` (optional explicit action list), `observation_histogram_reducer`, `step_interval`, `writer_factory`, `time_zone` default `US/Pacific`.

`_reset` calls `building.reset()`, clears reward accumulators, writes device and zone infos when `metrics_path` and `writer_factory` are set, and returns `ts.restart`. `_step` denormalizes the action, calls `building.request_action`, `building.wait_time()`, observes, then `reward_function.compute_reward(building.reward_info)`. Reward is `RewardResponse.agent_reward_value` unless the action was rejected. Terminal steps use `ts.termination`; otherwise `ts.transition` with `discount_factor`. If `_step` is called after the episode ended, it `reset()`s.

`action_spec` and `observation_spec` return one flat `float32` vector each. Action names follow `building.devices` order, and setpoints inside a device are sorted. A setpoint is included only when `ActionConfig` has a normalizer for that name. With a histogram reducer (the SB1 gin path), the spec walks devices sorted by `device_id`, then sorted observable fields. Measurements listed in `histogram_parameters` become bin features `{measurement}_h_{v:.2f}`, inserted once at the first device that has that measurement. Other measurements stay per device as `{device_id}_{measurement}`. `get_time_feature_names` emits every cosine index, then every sine index. Auxiliary features are appended after those fields. With `num_hod_features=1` and `num_dow_features=1` they are `hod_cos_000`, `hod_sin_000`, `dow_cos_000`, `dow_sin_000`, `comfort_mode_now`, `comfort_mode_soon`, `num_occupants`. `comfort_mode_soon` is `is_comfort_mode` 60 minutes ahead. `num_occupants` is `(building.num_occupants - occupancy_normalization_constant) / (occupancy_normalization_constant + 1)`. `Environment.__init__` defaults that constant to `0.0`. `create_and_setup_environment` sets it to `125.0`. Bins in `get_histogram_reducer` (what current SB1 calls) and in the gin `histogram_parameters_tuples` (what legacy `HistogramReducer` reads): zone air temperature 285 through 303, supply-air damper percentage 0, 0.2, 0.4, 0.6, 0.8, 1.0, supply-air flow setpoint 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9. Both reducers set `normalize_reduce=True`, which divides each bin row by `np.sum(chist)`. `HistogramReducer.__init__` calls `ProtoReader.read_observation_responses` and indexes `[0]` to assign devices to bins. `get_histogram_reducer` points that reader at `DATA_PATH`. The shard in that directory is `observation_response_2024.08.29.00`. An empty read raises `IndexError`.

`Environment.render` raises `NotImplementedError` (`Rendering not supported yet.`). `BaseBuilding.render` is still abstract. `SIM_VIDEOS_DIRPATH` is defined in `simulator/constants.py` (see Simulator model).

Gin agent actions and defaults:

| setpoint | normalized | native (K) | gin default (K) |
| --- | --- | --- | --- |
| `supply_water_setpoint` | -1 to 1 | 310 to 355 | 340 |
| `supply_air_heating_temperature_setpoint` | -1 to 1 | 285 to 300 | 285 |

`default_actions` also sets `supply_air_cooling_temperature_setpoint` to 300 K. That name is not in `action_normalizer_map`, so it is not an agent action. `BoundedActionNormalizer` linearly maps the normalized interval onto the native interval. `ACTION_TOLERANCE` is `1e-5`. Out-of-range agent or setpoint values raise `ValueError`.

`BaseBuilding` methods the simulator must implement: `reward_info`, `request_observations`, `request_observations_within_time_interval`, `request_action`, `wait_time`, `reset` (RuntimeError if reset is impossible), `devices`, `zones`, `current_timestamp`, `render`, `is_comfort_mode`, `num_occupants`, `time_step_sec`. `SimulatorBuilding` exposes the boiler, air handler, and each VAV as `DeviceInfo` and forwards proto requests to those `SmartDevice`s.

## 9. Protos

Package `smart_buildings.smart_control.proto`. Edit `.proto` files and regenerate. Do not edit `*_pb2.py`.

`smart_control_building.proto` is the agent/building exchange. `ObservationRequest` / `ObservationResponse` and `ActionRequest` / `ActionResponse` carry repeated single requests. Values are `continuous_value`, `integer_value`, `categorical_value`, `binary_value`, or `string_value`. `SingleActionResponse.ActionResponseType` includes `ACCEPTED`, `PENDING`, `TIMED_OUT`, `REJECTED_INVALID_SETTING`, `REJECTED_NOT_ENABLED_OR_AVAILABLE`, `REJECTED_OVERRIDE`, `REJECTED_INVALID_DEVICE`, `REJECTED_DEVICE_OFFLINE`. `ZoneInfo.area` is square meters. `DeviceInfo.DeviceType` includes `VAV`, `AHU`, `BLR`, and the other Carson-style enums in that file.

`smart_control_reward.proto`: `RewardInfo` holds per-zone temps (K), air flow (m^3/s), occupancy, plus air-handler and boiler energy rates (W), bounded by `start_timestamp` and `end_timestamp`. `RewardResponse.agent_reward_value` is what the environment returns. Other fields (productivity, costs, carbon kg and USD, weights, normalized regret/cost/carbon) are monitoring outputs. The regret function fills those normalized fields; the non-regret function mainly fills productivity, the two energy costs, carbon emitted, carbon cost, and `agent_reward_value`.

`smart_control_normalization.proto`: `ContinuousVariableInfo` with id, sample window, size, variance, mean, median, max, min. Gin only sets id, mean, and variance via `set_observation_normalization_constants`.

Proto file prefixes in `utils/constants.py`: `observation_response`, `action_response`, `reward_info`, `reward_response`, `device_info`, `zone_info`, plus `normalization_info`. `ProtoReader` / `ProtoWriter` use hourly shards such as `action_response_2021.05.25.04`. `get_episode_data(working_dir)` returns a dataframe.

## 10. Configs

Gin files:

- `smart_control/configs/resources/sb1/sim_config.gin` — current SB1, `TFSimulator`, 14-day episodes, `@get_histogram_reducer()`.
- `smart_control/configs/resources/sb1/sim_config_legacy.gin` — `SimulatorFlexibleGeometries`, `@HistogramReducer()`.
- `smart_control/configs/resources/sb1/train_sim_configs/sim_config_1_day.gin` — current SB1 with `num_days_in_episode=1` and `sim_building/TFSimulator.record_video = False`. Training and starter-buffer scripts use this file.

`SimulatorFlexibleGeometries.record_video` (default True) keeps a copy of the full grid every step for the video written 4 days after the start, and never frees it (about 3 MB per step). 1-day episodes reset before day 4, so the 1-day config turns it off; RSS then stays flat.

Scoped gin names in the current file include `sim/FloorPlanBasedBuilding`, `sim/FloorPlanBasedHvac`, `hvac/AirHandler`, `hvac/Boiler`, `hvac/SetpointSchedule`, `sim_building/TFSimulator`, `randomized_occupancy/RandomizedArrivalDepartureOccupancy`. Macros `to_timestamp`, `set_action_normalization_constants`, and `set_observation_normalization_constants` are in `utils/environment_utils.py`.

Observation mean/variance tables in the gin file are per measurement name (real-building stats). Do not assume one temperature normalizer covers every temperature field: lookup is exact `field_name` match, then mean 0 / variance 1.

## 11. Style and lint

Google Python style. Pyink: line length 80, indent 2, `unstable = true`, majority quotes. Pylint: `.pylintrc`, line length 80, indent 2 spaces, `indent-after-paren=4`, refactor messages disabled (`disable=R`), `pylint_per_file_ignores` allows `protected-access` in `*_test.py`. `make format` pylint command is `poetry run pylint --rcfile=.pylintrc --ignore=proto smart_control`. Isort: `force_single_line`, `known_first_party = ["smart_control"]`, `skip_glob` includes `smart_control/proto/*`, line length 200. Docstrings are Google style; MkDocs uses them (`docstring_style: google` in `mkdocs.yml`). Markdown: mdformat wrap 80, `mdformat-mkdocs`. Pre-commit: `pre-commit install`, then pyink, isort, pylint, pyrefly, mdformat.

`pyrefly.toml` sets `python-version = "3.13"` while Poetry requires `<3.12`. CI runs `pyrefly check --config pyrefly.toml || true` (TODO #174). The pre-commit pyrefly hook calls `subprocess.run` without `check=True`, so a failing pyrefly does not fail the hook. Type-check failures are currently non-blocking.

## 12. Pitfalls

- `train.py` checkpoints Reverb into the `--starter-buffer-path` directory, replacing the starter buffer's checkpoint. Use a copy for test runs.
- `train.py` logs `Final Eval AverageReturn` from its own `eval_metrics` list, but the eval actor is built with `actor.eval_metrics(...)`. The logged and TensorBoard eval returns are always 0.0.
- Exactness details the batched engines depend on: `np.mean(float32)` sums 8192-element chunks pairwise and adds them in sequence; `np.exp` differs from libm `exp` (use numpy); `RewardInfo.zone_reward_infos` iterates in a protobuf-internal order (neither insertion nor sorted), which the reward sums follow; every RewardInfo/RewardResponse/observation/action value passes through proto float32. In numpy 1.x a float32 array times a Python float stays float32 while a float32 scalar promotes to float64.
- State that survives `reset()`: thermostat modes and previous timestamps, occupancy RNG and occupant states, and device action/observation timestamps. The boiler sees a negative duration at the first observation after a reset.
- `env.reset()` then 289 simulator steps per 1-day episode; the 289th returns LAST, and the next `step()` resets without stepping the simulator.
- `train.py --agent-type td3` is accepted by argparse and then raises `ValueError`. SAC is the only implemented agent. TD3, DDPG, and CQL are notebooks.
- Docs (`docs/setup/linux.md`, `docs/setup/mac.md`) tell notebooks to load `smart_control/configs/sim_config.gin`. Use `smart_control/configs/resources/sb1/sim_config.gin`.
- `inside_wall_properties` in the SB1 gin files assigns `heat_capacity = %interior_wall_cv_density` and `density = %interior_wall_cv_heat_capacity`. That swaps 1.0 and 700.0 relative to the comments above those macros. Do not swap them back as a drive-by fix.
- Mac replay buffers: `dm-reverb-macos` 0.11.0.dev0, issue #102. Linux uses `dm-reverb` 0.14.0.
- Pyrefly is configured for Python 3.13 and does not fail CI.
- `pyproject.toml` depends on `google3 = "^1.0.0"`.
- Linux setup installs the apt package `python3.12-venv`, then tells you to `pyenv install 3.11`. The Poetry constraint is `>=3.10,<3.12`. Docker and the documented Poetry env are 3.11.
- `Environment.render` raises `NotImplementedError`.
- Rejected actions yield reward `-inf`, not a finite penalty.
- `load_environment` clears the gin config. A second environment load drops bindings from the first parse.
- Calling `train_agent()` uses different numeric defaults than the CLI.
- The baseline policy action order is air heating, then water. The environment action vector is boiler water, then air-handler heating. Mixing those orders writes the wrong setpoint. `populate_starter_buffer.py` does feed the policy output unchanged; measured on SB1 (fast mode, 64 days) that `as_run` schedule returns −43.51 versus −30.77 for the intended mapping (`batched/baseline_policy.py`, order `"intended"`).
- Schedule events add `local_start_time.utcoffset()` to the hour-of-day timedelta. They are not local clock times. The policy does not consult `SetpointSchedule.holidays`.
- `relative_convergence_threshold` and `relative_convergence_streak` are stored on `Simulator` and unused. Setting them in gin does not change the FDM stop test.
- `HistogramReducer` construction reads `observation_response*` under `DATA_PATH` and indexes the first message. The checked-in shard is `observation_response_2024.08.29.00`.
- Air-handler and boiler ids include `uuid4()` unless gin sets `device_id`. Passthrough observation names that use those ids change across processes. VAV ids from `FloorPlanBasedHvac` do not.
- Jupyter setup text says the notebook UI is at localhost:8000. Port 8000 is `mkdocs serve`. The Docker Jupyter command binds 8888.

## 13. Where to change what

| Change | Where |
| --- | --- |
| SB1 physics, HVAC setpoints, reward weights, action ranges, episode length | `configs/resources/sb1/*.gin`. Import any new gin class from `reinforcement_learning/utils/config.py`. |
| Finite-difference equations | `simulator/simulator.py`. Tensor form: `simulator/tf_simulator.py`. |
| Floor-plan geometry and materials | `simulator/building.py` (`FloorPlanBasedBuilding`, `MaterialProperties`). |
| Device observations and setpoints | `air_handler.py`, `boiler.py`, `vav.py`, `smart_device.py`. |
| Zone schedule the VAVs track | `setpoint_schedule.py`, `thermostat.py`. |
| Who is in the building | occupancy modules in `simulator/`. SB1 uses `RandomizedArrivalDepartureOccupancy`. |
| Reward math | `reward/setpoint_energy_carbon_regret.py` (active) or `setpoint_energy_carbon_reward.py`. Productivity curve: `reward/base_setpoint_energy_carbon_reward.py`. Prices: `electricity_energy_cost.py`, `natural_gas_energy_cost.py`. |
| Step, observation vector, rejection reward | `environment/environment.py`. |
| Training loop and CLI | `reinforcement_learning/scripts/train.py`. Starter data: `populate_starter_buffer.py`. Baseline setpoints: `policies/schedule_policy.py`. |
| SAC networks and hyperparameters | `agents/sac_agent.py`, `agents/networks/sac_networks.py`. |
| Replay buffer | `replay_buffer/replay_buffer.py`. |
| Data paths and histogram bins used by gin macros | `reinforcement_learning/utils/config.py`. |
| Proto fields | `smart_control/proto/*.proto`, then protoc 3.21.12. |
| Real-building partition loading | `dataset/dataset.py`, `dataset/partition.py`. Column ids use `device_id@field_name`. |
| Forced action rejection | `simulator/rejection_simulator_building.py`. |
| Episode plots and return | `reinforcement_learning/utils/data_processing.py`, `utils/metrics.py`. |
| Training logs and render hooks | `reinforcement_learning/observers/`. |
| GPU finite-difference kernel | `simulator/fd_kernel.cu` (+ `cuda_fd.py`); verify with `tf_simulator_test.py` and the golden trace. |
| Batched GPU environments | `batched/` (`sb1_vec.py` is the fast one); plan in `port_plan.md`. |
| Batched baseline, PPO, SAC, evaluation | `reinforcement_learning/scripts/batched_baseline.py`, `batched_ppo.py`, `batched_sac.py`, `batched_eval.py` (paired PPO or SAC vs baseline, exact and fast mode; `--checkpoint best`, `--reward`); vector env `batched/vec_env.py`; run folders and time-of-day logs `batched/run_logging.py`; eval helper `batched/evaluation.py`; per-env original seeds `SB1VecBatch.set_original_seeds`; new fast-mode streams before each training eval `SB1VecBatch.reseed`. |

<!-- BEGIN AWS Agent Toolkit rules -->
# AWS Guidance for the new AWS experience

This user has signed up for the new AWS experience. This experience lets you sign into AWS using a social provider and requires the following additional context.

Where this guidance conflicts with the project's own instructions, the project's instructions take precedence.

## Context

### Terminology:

- Say "project" instead of "account" — a project contains an AWS account and settings for sharing with other collaborators
- Say "team member" instead of "IAM user" — users are invited by email, not created or federated in IAM
- Say "AWS Settings" when referring to management tasks at [settings.aws.com](https://settings.aws.com/) (project management, billing, team members, spend limits). Users view their actual AWS resources in the AWS Management Console.
- Say "selected Region" when referring to the user's Region — not "home Region"
- The user has a managed IAM experience. This includes a managed service control policies (SCP) and resource control policies (RCP) that govern the use of AWS. They will still need to use IAM to create policies to let services work with each other. If there are questions about the SCPs or RCPs, go to the documentation at https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html

### Constraints:

- All projects share a single AWS Region determined by the user's contact address. Resources cannot be created in other Regions
- When developing:
  - MUST create all Regional resources in the project's assigned Region
  - You CAN create AWS WAF and Cloudwatch Logs resources in us-east-1 when there are global resources (like a global WAF instance) that require a connection to dependencies in us-east-1. You should not use these for any other reason, because resources in the selected Region will provide lower cost (due to no cross-Region traffic), increased availability (due to no cross-Region traffic), and easier manageability (due to not needing to look in another Region). When you need to do an inventory of resources, you need to look in both the selected Region and us-east-1 for Cloudwatch Logs or WAF resources.
  - MUST NOT attempt to create Lambda, API Gateway, or other Regional resources in any other Region
  - MUST direct users to confirm their Region in AWS Settings > View all projects > Overview > Additional Info > Region. If the user cannot confirm their Region, check in ~/.aws/config
  - MUST NOT use Lambda@Edge — excluded from both Lambda and CloudFront
  - MUST NOT use CloudFormation StackSets — no multi-account or multi-Region deployments
  - MUST NOT attempt cross-Region actions — no cross-Region replication for DynamoDB/S3/RDS, no multi-Region KMS keys
  - MUST NOT use Route 53 cross-Region routing — geolocation, latency-based, and failover routing policies are not available
  - CloudFront is a global service and its actions ARE allowed in `us-east-1`. A user can create a CloudFront distribution pointing to their project-region Lambda function URL or API Gateway. However, Lambda and API Gateway themselves MUST NOT be created in `us-east-1` — they must be in the project Region.
  - Reduced availability in `eu-north-1` specifically: Amazon Rekognition, Amazon Textract, Amazon Personalize, AWS App Runner are not available in that Region.
- IAM permissions for human access are managed by AWS. Don't assign roles to team members unless absolutely necessary
- The user may have a spend limit if they are on the paid plan. The limit that pauses their project if it's exceeded. If resources suddenly become inaccessible, ask if they have a spend limit configured. Only project owners can modify a spend limit.
- When developing:
  - MUST ask about spend limit status if the user reports sudden "Access Denied" errors on operations that previously worked
  - MUST direct users to check spend status in AWS Settings > Billing
  - MUST check if a user has upgraded their account to the paid plan
  - MUST ask the user if they want to clean up the successfully created resources or keep them to reduce cost
- The user sets up billing, creates spend limits, and retrieves and pays invoices in AWS Settings. The user creates budgets and optimizes their costs in the AWS Billing and Cost Management console
- Not all AWS services are available. If a service isn't working, do the following:
  1. Run the command `aws freetier get-account-plan-state`
  2. If accountPlanType": "FREE", check the [Free Tier supported services list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#supported-services-free-tier) next,
  3. If accountPlanType": "PAID", check the [Paid Tier supported services list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#supported-services-paid-plan).
  4. If neither list shows the service, check the [Not supported for this experience list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#unsupported-services). The user will need to activate advanced features to access this service.
- Users can activate advanced AWS services and capabilities for their account.
- Before starting a task, check whether a relevant AWS skill is available. Load the skill with retrieve_skill and prefer its guidance over general knowledge.

### Help level

- help_level (required): LOW, MEDIUM, or HIGH. While a user is building, you MUST ask the user: "How much guidance would you like from me? Low (I only flag security risks), medium (I ask a couple of clarifying questions if something seems off), or high (I explain what I'm doing, suggest alternatives, and flag best practices)."

You CAN update this rule file to save a user's help_level.

Constraints for each level:

**LOW:**

- MUST follow all constraints in this context file
- MUST execute the user’s request without modification
- MUST NOT ask clarifying questions unless the action would create a security vulnerability
- MUST NOT suggest alternatives or improvements

**MEDIUM:**

- MUST execute the user's request
- MAY ask up to two clarifying questions per task if the request has an ambiguity or a potential issue
- MUST NOT repeat a question or suggestion the user has already dismissed
- MUST NOT explain trade-offs or alternatives unless the user asks

**HIGH:**

- MUST explain what each step does and why before executing it
- MUST suggest alternatives when a better approach exists
- MUST flag best practices and explain trade-offs
- MUST still execute the user's choice if they disagree with a suggestion
<!-- END AWS Agent Toolkit rules -->
