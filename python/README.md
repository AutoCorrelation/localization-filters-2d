# Python Benchmark Pipeline (MATLAB parity)

This folder mirrors the MATLAB benchmark flow in `src/main_sub.m`.

## Files
- `config.py`: `initialize_config()` equivalent
- `data_loader.py`: `loadSimulationData()` equivalent
- `metrics.py`: `evaluateFilter()` equivalent
- `filters.py`: filter implementations and factory
- `runner.py`: `runFilter` and `main_sub`-like orchestration
- `run_experiment.py`: CLI entrypoint
- `run_experiment.ipynb`: notebook main workflow

## Install
```bash
pip install -r requirements.txt
```

## Script Run (main_sub equivalent)
```bash
python run_experiment.py --motion-model cv --path-data ../data --path-result ../result --iterations 1000
```

## Notebook Run
Open `run_experiment.ipynb` and run cells in order.

## Reproducibility notes
- Uses seed `42`.
- RNG uses `numpy.random.RandomState` for MATLAB `twister`-style behavior.
- Exact bitwise equality with MATLAB is not guaranteed because library linear algebra and random transform details differ across environments.
