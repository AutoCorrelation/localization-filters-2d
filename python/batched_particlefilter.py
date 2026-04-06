from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import torch

from config import Config
from data_loader import SimulationData


TensorRefiner = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class BatchedPFOutput:
    estimated_pos: torch.Tensor  # [mc_runs, 2, num_points, num_iterations]


class BatchedNonlinearPF:
    def __init__(
        self,
        data: SimulationData,
        config: Config,
        noise_idx: int,
        *,
        mc_runs: int = 1,
        device: str = "cuda",
        seed: int = 42,
        measurement_refiner: TensorRefiner | None = None,
    ):
        self.mc_runs = int(mc_runs)
        self.num_particles = int(config.num_particles)
        self.resample_threshold_ratio = float(config.resample_threshold_ratio)
        self.measurement_refiner = measurement_refiner

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.uniform_log_weight = torch.tensor(-math.log(float(self.num_particles)), dtype=torch.float32, device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

        self.anchor_pos = torch.as_tensor(config.anchor.T, dtype=torch.float32, device=self.device)  # [A,2]
        self.noise_std = float(config.noise_variance[noise_idx]) ** 0.5
        self.process_bias = self._extract_process_bias(data, noise_idx)
        self.process_noise_bank = self._extract_noise_bank(data.process_noise, noise_idx)
        self.toa_noise_bank = self._extract_noise_bank(data.toa_noise, noise_idx)

        self.x_hat = torch.as_tensor(data.x_hat_lls[..., noise_idx], dtype=torch.float32, device=self.device)
        self.z = torch.as_tensor(data.ranging[..., noise_idx], dtype=torch.float32, device=self.device)

    def _extract_process_bias(self, data: SimulationData, noise_idx: int) -> torch.Tensor:
        if data.process_bias.ndim == 2:
            bias = data.process_bias[:, noise_idx]
        else:
            bias = data.process_bias.reshape(-1)
        return torch.as_tensor(bias, dtype=torch.float32, device=self.device).reshape(1, 1, 2)

    def _extract_noise_bank(self, noise_tensor, noise_idx: int) -> torch.Tensor:
        if noise_tensor.ndim >= 3:
            bank = noise_tensor[..., noise_idx]
        else:
            bank = noise_tensor
        bank_t = torch.as_tensor(bank, dtype=torch.float32, device=self.device)
        if bank_t.ndim < 2:
            return torch.empty((2, 0), dtype=torch.float32, device=self.device)
        return bank_t.reshape(bank_t.shape[0], -1)

    def _sample_noise(self, bank: torch.Tensor) -> torch.Tensor:
        if bank.numel() == 0 or bank.ndim < 2 or bank.shape[1] == 0:
            return self.noise_std * torch.randn(
                (self.mc_runs, self.num_particles, 2),
                dtype=torch.float32,
                device=self.device,
                generator=self.generator,
            )
        idx = torch.randint(
            low=0,
            high=bank.shape[1],
            size=(self.mc_runs, self.num_particles),
            device=self.device,
            generator=self.generator,
        )
        sampled = bank[:, idx]  # [2, mc, n]
        # Convert feature-first sampled bank into particle-last layout used by state propagation.
        return sampled.permute(1, 2, 0).contiguous()  # [mc, n, 2]

    def _h_nonlinear(self, particles: torch.Tensor) -> torch.Tensor:
        # particles: [mc, n, 2] -> [mc, n, num_anchors]
        diff = particles.unsqueeze(2) - self.anchor_pos.unsqueeze(0).unsqueeze(0)
        return torch.linalg.norm(diff, dim=-1)

    def _update_log_weights(self, particles_pred: torch.Tensor, log_prev_w: torch.Tensor, z_now: torch.Tensor) -> torch.Tensor:
        y_pred = self._h_nonlinear(particles_pred)
        err = z_now.unsqueeze(1) - y_pred
        log_like = -0.5 * torch.sum((err / self.noise_std) ** 2, dim=-1)
        log_w = log_prev_w + log_like
        return log_w - torch.logsumexp(log_w, dim=1, keepdim=True)

    def _vectorized_systematic_resample(self, weights: torch.Tensor) -> torch.Tensor:
        # weights: [mc, n], returns idx: [mc, n]
        mc, n = weights.shape
        cdf = torch.cumsum(weights, dim=1)
        cdf[:, -1] = 1.0

        u0 = torch.rand((mc, 1), device=self.device, generator=self.generator) / float(n)
        grid = torch.arange(n, device=self.device, dtype=torch.float32).view(1, -1) / float(n)
        u = u0 + grid
        return torch.searchsorted(cdf, u, right=False)

    def run(self) -> BatchedPFOutput:
        _, num_points, num_iterations = self.x_hat.shape
        estimated = torch.zeros((self.mc_runs, 2, num_points, num_iterations), dtype=torch.float32, device=self.device)

        for iter_idx in range(num_iterations):
            p1 = self.x_hat[:, 0, iter_idx]
            p2 = self.x_hat[:, 1, iter_idx]
            estimated[:, :, 0, iter_idx] = p1.unsqueeze(0).expand(self.mc_runs, -1)
            estimated[:, :, 1, iter_idx] = p2.unsqueeze(0).expand(self.mc_runs, -1)

            sampled_prev = p1.view(1, 1, 2) + self._sample_noise(self.toa_noise_bank)
            sampled_curr = p2.view(1, 1, 2) + self._sample_noise(self.toa_noise_bank)
            particles_prev = sampled_curr
            vel_prev = sampled_curr - sampled_prev
            log_weights = torch.full(
                (self.mc_runs, self.num_particles),
                fill_value=float(self.uniform_log_weight),
                dtype=torch.float32,
                device=self.device,
            )

            for point_idx in range(2, num_points):
                particles_pred = particles_prev + vel_prev + self.process_bias + self._sample_noise(self.process_noise_bank)

                z_now = self.z[:, point_idx, iter_idx].unsqueeze(0).expand(self.mc_runs, -1)
                if self.measurement_refiner is not None:
                    z_now = self.measurement_refiner(z_now)

                log_weights = self._update_log_weights(particles_pred, log_weights, z_now)
                weights = torch.exp(log_weights)
                est = torch.sum(particles_pred * weights.unsqueeze(-1), dim=1)
                estimated[:, :, point_idx, iter_idx] = est

                ess = 1.0 / torch.sum(weights * weights, dim=1)
                do_resample = ess < (self.num_particles * self.resample_threshold_ratio)
                idx = self._vectorized_systematic_resample(weights)
                idx_exp = idx.unsqueeze(-1).expand(-1, -1, 2)

                particles_res = torch.gather(particles_pred, dim=1, index=idx_exp)
                prev_res = torch.gather(particles_prev, dim=1, index=idx_exp)
                vel_res = particles_res - prev_res

                particles_no = particles_pred
                vel_no = particles_pred - particles_prev

                mask3 = do_resample.view(-1, 1, 1)
                particles_prev = torch.where(mask3, particles_res, particles_no)
                vel_prev = torch.where(mask3, vel_res, vel_no)

                log_weights = torch.where(
                    do_resample.view(-1, 1),
                    self.uniform_log_weight,
                    log_weights,
                )

        return BatchedPFOutput(estimated_pos=estimated)


def run_batched_nonlinear_pf(
    data: SimulationData,
    config: Config,
    noise_idx: int,
    *,
    mc_runs: int = 1,
    device: str = "cuda",
    seed: int = 42,
    measurement_refiner: TensorRefiner | None = None,
) -> BatchedPFOutput:
    return BatchedNonlinearPF(
        data,
        config,
        noise_idx,
        mc_runs=mc_runs,
        device=device,
        seed=seed,
        measurement_refiner=measurement_refiner,
    ).run()
