from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Config:
    path_data: str = "../data"
    path_result: str = "../result"
    num_samples: int = int(1e5)
    iterations: int = int(1e3)
    noise_variance: tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2)
    num_points: int = 10
    motion_model: Literal["cv", "imm"] = "cv"
    num_particles: int = 150
    resample_threshold_ratio: float = 0.2
    decay_gamma: tuple[float, ...] = (0.4, 0.5, 0.5, 0.3, 0.5)

    rdiag_prior_sigma_gate: float = 6.0
    rdiag_prior_max_retry: int = 20
    rdiag_roughening_k: float = 0.2

    roughening_k: float = 0.2
    prior_sigma_gate: float = 6.0
    prior_max_retry: int = 30


    @property
    def anchor(self) -> np.ndarray:
        return np.array([[0, 10], [0, 0], [10, 0], [10, 10]], dtype=float).T

    @property
    def h(self) -> np.ndarray:
        return np.array(
            [
                [0, -20],
                [20, -20],
                [20, 0],
                [20, 0],
                [20, 20],
                [0, 20],
            ],
            dtype=float,
        )

    @property
    def pinv_h(self) -> np.ndarray:
        return np.linalg.pinv(self.h)


def initialize_config(num_particles: int | None = None) -> Config:
    cfg = Config()
    if num_particles is not None:
        cfg.num_particles = int(num_particles)
    return cfg
