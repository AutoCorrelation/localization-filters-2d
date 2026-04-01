from __future__ import annotations

from typing import Tuple


def get_best_params(noise_idx: int) -> Tuple[float, float]:
    """Per-noise tuned hyperparameters matching MATLAB getBestParams."""
    beta_table = [0.98, 0.99, 0.990, 0.90, 0.995]
    lambda_r_table = [10.0, 20.0, 200.0, 2.0, 20.0]

    if noise_idx is None:
        return 0.90, 1.0

    idx = int(round(noise_idx))
    idx = max(0, min(idx, len(beta_table) - 1))
    return beta_table[idx], lambda_r_table[idx]
