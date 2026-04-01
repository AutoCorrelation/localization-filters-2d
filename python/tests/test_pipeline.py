from __future__ import annotations

from pathlib import Path

from config import initialize_config
from data_loader import load_simulation_data
from runner import run_filter


def test_h5_load_keys_and_shapes():
    cfg = initialize_config(10)
    h5_path = Path(__file__).resolve().parents[2] / "data" / "simulation_data.h5"
    data = load_simulation_data(str(h5_path))

    assert data.ranging.ndim >= 3
    assert data.x_hat_lls.ndim >= 3
    assert data.true_state.ndim >= 3


def test_single_filter_smoke():
    cfg = initialize_config(10)
    cfg.iterations = 3
    cfg.noise_variance = cfg.noise_variance[:1]

    h5_path = Path(__file__).resolve().parents[2] / "data" / "simulation_data.h5"
    data = load_simulation_data(str(h5_path))

    estimated, metric = run_filter("NonlinearParticleFilter", data, cfg)

    assert estimated.ndim == 4
    assert "RMSE" in metric
    assert metric["RMSE"].shape[0] == 1
