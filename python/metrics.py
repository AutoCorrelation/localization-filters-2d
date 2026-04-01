from __future__ import annotations

import numpy as np


def evaluate_filter(
    estimated_pos: np.ndarray,
    start_point: int = 3,
    true_pos: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute RMSE and APE with MATLAB-compatible indexing semantics."""
    # MATLAB startPoint=3 means Python index starts from 2.
    start_idx = max(int(start_point) - 1, 0)
    num_points = estimated_pos.shape[1]
    num_iterations = estimated_pos.shape[2]

    if true_pos is not None and true_pos.ndim == 2:
        true_pos = np.repeat(true_pos[:, :, None], num_iterations, axis=2)

    errors = []
    errors_sq = []

    for iter_idx in range(num_iterations):
        for point_idx in range(start_idx, num_points):
            if true_pos is None:
                gt = np.array([point_idx + 1, point_idx + 1], dtype=float)
            else:
                gt = true_pos[0:2, point_idx, iter_idx]

            d = estimated_pos[:, point_idx, iter_idx] - gt
            err_sq = float(d[0] ** 2 + d[1] ** 2)
            errors_sq.append(err_sq)
            errors.append(np.sqrt(err_sq))

    rmse = float(np.sqrt(np.mean(errors_sq)))
    ape = float(np.mean(errors))
    return rmse, ape
