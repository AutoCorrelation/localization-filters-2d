# Project Guidelines

## Scope and Goal
- This repository benchmarks 2D localization filters across multiple noise levels.
- Current priority is adding paper-based particle filter variants and comparing them against `AdaptiveParticleFilter`.
- Keep implementations reproducible and directly comparable in APE/RMSE and runtime.

## Architecture
- Main entry: `src/main_sub.m`.
- Shared execution pipeline: `src/utils/runFilter.m` (factory + `parfor` over noise levels).
- Common configuration: `src/utils/initializeConfig.m`.
- Filter implementations: `src/Filters/`.
- Data loading: `src/utils/loadSimulationData.m` from `data/simulation_data.h5`.
- Hyperparameter lookup for Adaptive PF: `src/utils/getBestParams.m`.

## Build and Run
- MATLAB baseline benchmark:
  - Run `src/main_sub.m`.
- Adaptive PF per-noise grid search:
  - Run `src/optimizePerNoiseGrid.m`.
- Data generation (if needed):
  - Run `src/dataGenerate.m`.

## Filter Implementation Conventions
- New filter classes should live in `src/Filters/`.
- Reuse existing class hierarchy where possible (for PF variants, prefer inheriting from `NonlinearParticleFilter` unless the paper requires a different model).
- Maintain the common interface used by the runner:
  - `initializeState(numPoints)`
  - `initializeFirstTwo(state, iterIdx)`
  - `step(state, iterIdx, pointIdx)`
- Register every new filter in `localCreateFilter` inside `src/utils/runFilter.m`.
- If the method needs per-noise tuned parameters, add a dedicated helper (similar to `getBestParams`) and call it from the filter factory.

## Experiment and Reproducibility Rules
- Always set RNG seed before experiments (`rng(42, 'twister')` pattern).
- Keep noise-level handling consistent with `config.noiseVariance` order and index.
- Preserve tensor shapes used by evaluation:
  - Estimated position array shape: `(2, numPoints, numIterations, numNoise)`.
- Evaluate with existing utilities before changing metrics code:
  - `src/utils/evaluateFilter.m`
  - `src/utils/plotMetricComparison.m`

## Coding Style
- Follow existing MATLAB style in nearby files.
- Keep comments concise and focused on non-obvious math or algorithmic choices.
- Avoid unrelated refactors when adding a new filter; keep changes minimal and localized.
- Update README result tables only after rerunning the corresponding experiment.

## Common Pitfalls
- Do not forget to register the new class in `runFilter`; otherwise runs fail at factory dispatch.
- Do not mix linear measurement (`z_LLS`, linear H) with nonlinear ranging model unless explicitly intended by the algorithm.
- In parallel runs, avoid introducing shared mutable state across workers.

## Codebase Ground Truth (Agent-Verified)

This section captures the currently verified runtime flow and module connectivity from the code itself.

### Runtime Flow (Benchmark)
1. `src/main_sub.m`
   - Initializes pool/config and loops over `particleCounts = [10, 50, 100, 200, 500]`.
   - Sets reproducible seed per particle count (`rng(42, 'twister')`).
   - Selects H5 by motion model (`simulation_data.h5` or `simulation_data_imm.h5`).
   - Executes each filter by calling `runFilter`.
2. `src/utils/runFilter.m`
   - Runs `parfor` over noise index.
   - Uses filter factory `localCreateFilter` to instantiate each filter.
   - Calls common interface:
     - `initializeState(numPoints)`
     - `initializeFirstTwo(state, iterIdx)`
     - `step(state, iterIdx, pointIdx)`
   - Evaluates each noise slice with `evaluateFilter(estNoise, 3, truePosNoise)`.
3. Result handling in `src/main_sub.m`
   - Builds per-filter APE matrix and runtime table.
   - Plots via `plotMetricComparison`.
   - Saves per-N CSV via `saveBenchmarkResults`.
   - Appends to batch CSV `benchmark_<motion>_batch_APE_AllN.csv`.

### Filter Factory Mapping (`runFilter.m`)
- Registered classes currently include:
  - `Baseline`
  - `LinearKalmanFilter_DecayQ`
  - `NonlinearParticleFilter`
  - `CustomNonlinearParticleFilter`
  - `EKFParticleFilter`
  - `AdaptiveParticleFilter`
  - `BeliefQShrinkAdaptiveParticleFilter` (`bqspf` alias)
  - `RDiagPriorEditAdaptiveParticleFilter` (`rdpepf` alias)
  - `BeliefRougheningAdaptiveParticleFilter` (`brapf` alias)
  - `RougheningPriorEditingParticleFilter` (`rpepf` alias)
- `Adaptive*` variants consume per-noise hyperparameters from `getBestParams(noiseIdx)`.

### Inheritance/Interface Reality (`src/Filters`)
- Core chain:
  - `LinearParticleFilter` -> `NonlinearParticleFilter` -> `AdaptiveParticleFilter` -> adaptive variants.
- `EKFParticleFilter` and `RougheningPriorEditingParticleFilter` derive from `NonlinearParticleFilter`.
- `LinearKalmanFilter_DecayQ` derives from `LinearKalmanFilter`.
- Runner compatibility requires all benchmarked filters to preserve the 3-method interface above.

### Data Contract and Shapes
- `loadSimulationData` reads these datasets from H5:
  - `/ranging`, `/x_hat_LLS`, `/z_LLS`, `/R_LLS`, `/Q`, `/P0`, `/processNoise`, `/toaNoise`, `/processbias`, `/true_state`, `/mode_history`
- Evaluation path assumes:
  - Per-noise estimate: `(2, numPoints, numIterations)`
  - Full estimate tensor: `(2, numPoints, numIterations, numNoise)`
  - Ground truth used from `data.true_state(1:2, ... , noiseIdx)`

### Motion Model Behavior
- `initializeConfig.m` currently sets `config.motionModel = 'imm'` by default.
- `main_sub.m` and `optimizePerNoiseGrid.m` both switch input file based on this flag.
- `dataGenerate.m` delegates to `dataGenerateIMM(config)` when motion model is IMM.

### Result File Behavior
- `saveBenchmarkResults.m` writes one per-particle-count APE CSV with motion prefix:
  - `benchmark_<motion>_N<k>_APE.csv`
- Runtime values are embedded as a synthetic row:
  - `RowType = RuntimeSec`, inserted under `NoiseVariance = 100` when present.

### Current Non-Goal/Legacy Boundary
- `src/archives/` is legacy reference code and not part of active benchmark pipeline.
- `pytorch/` scripts are auxiliary and currently expect dataset names (`toaPos`, `realPos`) that do not match MATLAB-generated H5 keys; treat them as separate workflow unless synchronized.

### Agent Update Rule
- When runtime flow, filter registration, or data schema changes, update this section first so future code edits follow actual codebase truth.