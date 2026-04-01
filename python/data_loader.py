from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class SimulationData:
    ranging: np.ndarray
    x_hat_lls: np.ndarray
    z_lls: np.ndarray
    r_lls: np.ndarray
    q: np.ndarray
    p0: np.ndarray
    process_noise: np.ndarray
    toa_noise: np.ndarray
    process_bias: np.ndarray
    true_state: np.ndarray
    mode_history: np.ndarray


def _read(h5: h5py.File, key: str) -> np.ndarray:
    return np.asarray(h5[key])


def _swap_first_last_dims(arr: np.ndarray) -> np.ndarray:
    """Convert H5 storage order to MATLAB array order.
    
    MATLAB writes multi-dimensional arrays to HDF5 using column-major transposition.
    For any N-D array, move noise dimension (axis 0) to last position.
    
    Examples:
    - 2D (5, M) → (M, 5) using axes=[1, 0]
    - 3D (5, A, B) → (B, A, 5) using axes=[2, 1, 0]
    - 4D (5, points, iters, dims) → (dims, points, iters, 5) using axes=[3, 1, 2, 0]
    - 5D (5, poi nts, iters, dim1, dim2) → (dim1, points, iters, dim2, 5) using axes=[3, 1, 2, 4, 0]
    """
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (5, M) → (M, 5)
        return np.transpose(arr, axes=[1, 0])
    if arr.ndim == 3:
        # (5, A, B) → (B, A, 5)
        return np.transpose(arr, axes=[2, 1, 0])
    if arr.ndim == 4:
        # (5, points, iters, dims) → (dims, points, iters, 5)
        return np.transpose(arr, axes=[3, 1, 2, 0])
    elif arr.ndim == 5:
        # (5, points, iters, dim1, dim2) → (dim1, points, iters, dim2, 5)
        return np.transpose(arr, axes=[3, 1, 2, 4, 0])
    else:
        # Fallback: move first axis to last
        axes = list(range(1, arr.ndim)) + [0]
        return np.transpose(arr, axes)


def load_simulation_data(h5_file: str) -> SimulationData:
    with h5py.File(h5_file, "r") as h5:
        ranging = _swap_first_last_dims(_read(h5, "/ranging"))
        x_hat_lls = _swap_first_last_dims(_read(h5, "/x_hat_LLS"))
        z_lls = _swap_first_last_dims(_read(h5, "/z_LLS"))
        r_lls = _swap_first_last_dims(_read(h5, "/R_LLS"))
        q = _swap_first_last_dims(_read(h5, "/Q"))
        p0 = _swap_first_last_dims(_read(h5, "/P0"))
        process_noise = _swap_first_last_dims(_read(h5, "/processNoise"))
        toa_noise = _swap_first_last_dims(_read(h5, "/toaNoise"))
        process_bias = _swap_first_last_dims(_read(h5, "/processbias"))
        true_state = _swap_first_last_dims(_read(h5, "/true_state"))
        mode_history = _swap_first_last_dims(_read(h5, "/mode_history"))

        return SimulationData(
            ranging=ranging,
            x_hat_lls=x_hat_lls,
            z_lls=z_lls,
            r_lls=r_lls,
            q=q,
            p0=p0,
            process_noise=process_noise,
            toa_noise=toa_noise,
            process_bias=process_bias,
            true_state=true_state,
            mode_history=mode_history,
        )
