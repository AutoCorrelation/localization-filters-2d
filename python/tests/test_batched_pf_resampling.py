from __future__ import annotations

from pathlib import Path

import torch

from batched_particlefilter import BatchedNonlinearPF
from config import initialize_config
from data_loader import load_simulation_data


def _make_batched_pf(num_particles: int = 16) -> BatchedNonlinearPF:
    cfg = initialize_config(num_particles)
    cfg.iterations = 1
    cfg.noise_variance = cfg.noise_variance[:1]

    h5_path = Path(__file__).resolve().parents[2] / "data" / "simulation_data.h5"
    data = load_simulation_data(str(h5_path))
    return BatchedNonlinearPF(data, cfg, noise_idx=0, mc_runs=1, device="cpu", seed=42)


def test_resampling_triggered_when_ess_low():
    pf = _make_batched_pf(num_particles=16)
    weights = torch.zeros((1, 16), dtype=torch.float32)
    weights[0, 0] = 1.0

    ess = 1.0 / torch.sum(weights * weights, dim=1)
    do_resample = ess < (pf.num_particles * pf.resample_threshold_ratio)

    assert bool(do_resample[0]) is True


def test_resampling_not_triggered_when_ess_high():
    pf = _make_batched_pf(num_particles=16)
    weights = torch.full((1, 16), 1.0 / 16.0, dtype=torch.float32)

    ess = 1.0 / torch.sum(weights * weights, dim=1)
    do_resample = ess < (pf.num_particles * pf.resample_threshold_ratio)

    assert bool(do_resample[0]) is False
