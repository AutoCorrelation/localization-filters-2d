from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class DPFStepConfig:
    state_dim: int = 2
    min_scale: float = 1e-4
    eps: float = 1e-8
    soft_resample_alpha: float = 0.5
    use_soft_resampling: bool = True


@dataclass
class BasicParticleFilterConfig:
    state_dim: int = 2
    process_scale: float = 1.0
    obs_scale: float = 1.0
    min_scale: float = 1e-4
    eps: float = 1e-8
    soft_resample_alpha: float = 0.5
    use_soft_resampling: bool = True


class ParticleFilterBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _predict_ranges(self, x_particles: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        diff = x_particles.unsqueeze(-2) - anchors.view(1, 1, anchors.shape[0], anchors.shape[1])
        return torch.linalg.norm(diff, dim=-1)

    def _gaussian_log_like(self, innovation: torch.Tensor, obs_scale: torch.Tensor) -> torch.Tensor:
        var = torch.clamp(obs_scale * obs_scale, min=self.cfg.min_scale)
        log_var = torch.log(var)
        sq = innovation * innovation
        return -0.5 * torch.sum(sq / var + log_var + torch.log(torch.tensor(2.0 * torch.pi, device=innovation.device)), dim=-1)

    def _soft_resample(self, x_pred: torch.Tensor, log_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_particles, _ = x_pred.shape
        alpha = float(min(max(self.cfg.soft_resample_alpha, 0.0), 1.0))

        w = torch.softmax(log_w, dim=-1)
        q = alpha * w + (1.0 - alpha) * (1.0 / num_particles)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        q_sum = torch.sum(q, dim=-1, keepdim=True)
        uniform_q = torch.full_like(q, 1.0 / num_particles)
        safe_q = torch.where(q_sum > self.cfg.eps, q / torch.clamp(q_sum, min=self.cfg.eps), uniform_q)
        safe_q = torch.clamp(safe_q, min=self.cfg.eps)
        safe_q = safe_q / torch.clamp(torch.sum(safe_q, dim=-1, keepdim=True), min=self.cfg.eps)

        ancestor_idx = torch.multinomial(safe_q, num_samples=num_particles, replacement=True)
        gather_idx = ancestor_idx.unsqueeze(-1).expand(-1, -1, x_pred.shape[-1])
        x_res = torch.gather(x_pred, dim=1, index=gather_idx)

        w_sel = torch.gather(w, dim=1, index=ancestor_idx)
        q_sel = torch.gather(safe_q, dim=1, index=ancestor_idx)
        w_corr = w_sel / torch.clamp(q_sel, min=self.cfg.eps)
        log_w_res = torch.log(torch.clamp(w_corr, min=self.cfg.eps))
        log_w_res = log_w_res - torch.logsumexp(log_w_res, dim=-1, keepdim=True)
        return x_res, log_w_res

    def _estimate_state(self, x_pred: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
        w = torch.exp(log_w)
        return torch.sum(w.unsqueeze(-1) * x_pred, dim=1)

    def step(
        self,
        x_prev: torch.Tensor,
        v_prev: torch.Tensor,
        log_w_prev: torch.Tensor,
        z_t: torch.Tensor,
        anchors: torch.Tensor,
    ):
        return self.forward(x_prev, v_prev, log_w_prev, z_t, anchors)


class BasicParticleFilter(ParticleFilterBase):
    def __init__(self, cfg: BasicParticleFilterConfig):
        super().__init__(cfg)

        process_scale = torch.as_tensor(cfg.process_scale, dtype=torch.float32)
        if process_scale.ndim == 0:
            process_scale = process_scale.repeat(cfg.state_dim)
        if process_scale.numel() != cfg.state_dim:
            raise ValueError(f"process_scale must have {cfg.state_dim} values or be a scalar")
        self.register_buffer("process_scale", torch.clamp(process_scale, min=cfg.min_scale))

        obs_scale = torch.as_tensor(cfg.obs_scale, dtype=torch.float32).reshape(1)
        self.register_buffer("obs_scale", torch.clamp(obs_scale, min=cfg.min_scale))

    def forward(
        self,
        x_prev: torch.Tensor,
        v_prev: torch.Tensor,
        log_w_prev: torch.Tensor,
        z_t: torch.Tensor,
        anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        process_scale = self.process_scale.to(device=x_prev.device, dtype=x_prev.dtype)
        obs_scale = self.obs_scale.to(device=x_prev.device, dtype=x_prev.dtype)

        noise = torch.randn_like(x_prev) * process_scale.view(1, 1, -1)
        x_pred = x_prev + v_prev + noise

        y_pred = self._predict_ranges(x_pred, anchors)
        innovation = z_t.unsqueeze(1) - y_pred

        log_like = self._gaussian_log_like(innovation, obs_scale)
        log_w_unnorm = log_w_prev + log_like
        log_w = log_w_unnorm - torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)

        x_est = self._estimate_state(x_pred, log_w)

        if self.cfg.use_soft_resampling:
            x_next, log_w_next = self._soft_resample(x_pred, log_w)
        else:
            x_next = x_pred
            log_w_next = log_w

        v_next = x_next - x_prev
        return x_next, v_next, log_w_next, x_est


class DPFStepModule(nn.Module):
    """Single differentiable PF step.

    Recommended format for training: nn.Module.
    - Keeps learnable noise scales as Parameters.
    - Easy to integrate with optimizer/checkpointing.
    """

    def __init__(self, cfg: DPFStepConfig):
        super().__init__()
        self.cfg = cfg
        self.log_process_scale = nn.Parameter(torch.zeros(cfg.state_dim))
        self.log_obs_scale = nn.Parameter(torch.zeros(1))

    def _predict_ranges(self, x_particles: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        # x_particles: (B, N, 2), anchors: (M, 2) -> ranges: (B, N, M)
        diff = x_particles.unsqueeze(-2) - anchors.view(1, 1, anchors.shape[0], anchors.shape[1])
        return torch.linalg.norm(diff, dim=-1)

    def _gaussian_log_like(self, innovation: torch.Tensor, obs_scale: torch.Tensor) -> torch.Tensor:
        # innovation: (B, N, M)
        var = torch.clamp(obs_scale * obs_scale, min=self.cfg.min_scale)
        log_var = torch.log(var)
        sq = innovation * innovation
        # Sum anchor dimension M.
        return -0.5 * torch.sum(sq / var + log_var + torch.log(torch.tensor(2.0 * torch.pi, device=innovation.device)), dim=-1)

    def _soft_resample(self, x_pred: torch.Tensor, log_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paper-style soft resampling with importance correction.

        q = alpha * w + (1 - alpha) / N
        a ~ Categorical(q)
        w_new proportional to w[a] / q[a]
        """
        bsz, num_particles, _ = x_pred.shape
        alpha = float(min(max(self.cfg.soft_resample_alpha, 0.0), 1.0))

        w = torch.softmax(log_w, dim=-1)
        q = alpha * w + (1.0 - alpha) * (1.0 / num_particles)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        q_sum = torch.sum(q, dim=-1, keepdim=True)
        uniform_q = torch.full_like(q, 1.0 / num_particles)
        safe_q = torch.where(q_sum > self.cfg.eps, q / torch.clamp(q_sum, min=self.cfg.eps), uniform_q)
        safe_q = torch.clamp(safe_q, min=self.cfg.eps)
        safe_q = safe_q / torch.clamp(torch.sum(safe_q, dim=-1, keepdim=True), min=self.cfg.eps)

        ancestor_idx = torch.multinomial(safe_q, num_samples=num_particles, replacement=True)
        gather_idx = ancestor_idx.unsqueeze(-1).expand(-1, -1, x_pred.shape[-1])
        x_res = torch.gather(x_pred, dim=1, index=gather_idx)

        w_sel = torch.gather(w, dim=1, index=ancestor_idx)
        q_sel = torch.gather(safe_q, dim=1, index=ancestor_idx)
        w_corr = w_sel / torch.clamp(q_sel, min=self.cfg.eps)
        log_w_res = torch.log(torch.clamp(w_corr, min=self.cfg.eps))
        log_w_res = log_w_res - torch.logsumexp(log_w_res, dim=-1, keepdim=True)
        return x_res, log_w_res

    def forward(
        self,
        x_prev: torch.Tensor,
        v_prev: torch.Tensor,
        log_w_prev: torch.Tensor,
        z_t: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one DPF step.

        Shapes:
        - x_prev: (B, N, 2)
        - v_prev: (B, N, 2)
        - log_w_prev: (B, N)
        - z_t: (B, M)         # ranging measurement for anchors
        - anchors: (M, 2)

        Returns:
        - x_next: (B, N, 2)
        - v_next: (B, N, 2)
        - log_w_next: (B, N)
        - x_est: (B, 2)
        """
        process_scale = torch.nn.functional.softplus(self.log_process_scale) + self.cfg.min_scale
        obs_scale = torch.nn.functional.softplus(self.log_obs_scale) + self.cfg.min_scale

        noise = torch.randn_like(x_prev) * process_scale.view(1, 1, -1)
        x_pred = x_prev + v_prev + noise

        y_pred = self._predict_ranges(x_pred, anchors)
        innovation = z_t.unsqueeze(1) - y_pred

        log_like = self._gaussian_log_like(innovation, obs_scale)
        log_w_unnorm = log_w_prev + log_like
        log_w = log_w_unnorm - torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)

        w = torch.exp(log_w)
        x_est = torch.sum(w.unsqueeze(-1) * x_pred, dim=1)

        if self.cfg.use_soft_resampling:
            x_next, log_w_next = self._soft_resample(x_pred, log_w)
        else:
            x_next = x_pred
            log_w_next = log_w

        v_next = x_next - x_prev
        return x_next, v_next, log_w_next, x_est
DPFParticleFilter = DPFStepModule


def dpf_one_step_functional(
    x_prev: torch.Tensor,
    v_prev: torch.Tensor,
    log_w_prev: torch.Tensor,
    z_t: torch.Tensor,
    anchors: torch.Tensor,
    process_scale: torch.Tensor,
    obs_scale: torch.Tensor,
    cfg: DPFStepConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional one-step variant (no nn.Module state).

    Useful for quick experiments or JAX-style stateless composition.
    Returns only (log_w, x_est) to stay minimal.
    """
    process_scale = torch.clamp(process_scale, min=cfg.min_scale)
    obs_scale = torch.clamp(obs_scale, min=cfg.min_scale)

    noise = torch.randn_like(x_prev) * process_scale.view(1, 1, -1)
    x_pred = x_prev + v_prev + noise

    diff = x_pred.unsqueeze(-2) - anchors.view(1, 1, anchors.shape[0], anchors.shape[1])
    y_pred = torch.linalg.norm(diff, dim=-1)

    innovation = z_t.unsqueeze(1) - y_pred
    var = obs_scale * obs_scale
    log_like = -0.5 * torch.sum((innovation * innovation) / var + torch.log(var), dim=-1)

    log_w_unnorm = log_w_prev + log_like
    log_w = log_w_unnorm - torch.logsumexp(log_w_unnorm, dim=-1, keepdim=True)

    w = torch.exp(log_w)
    x_est = torch.sum(w.unsqueeze(-1) * x_pred, dim=1)
    return log_w, x_est


def build_batch_from_simulator(
    ranging: torch.Tensor,
    true_state: torch.Tensor,
    noise_idx: int,
    point_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create one training batch from existing simulator tensors.

    Input tensors must follow python/data_loader.py output order:
    - ranging: (M, P, T, K)
    - true_state: (S, P, T, K)

    Returns:
    - z_t: (B=T, M)
    - target_pos: (B=T, 2)
    """
    z_t = ranging[:, point_idx, :, noise_idx].transpose(0, 1).contiguous()
    target_pos = true_state[0:2, point_idx, :, noise_idx].transpose(0, 1).contiguous()
    return z_t, target_pos


if __name__ == "__main__":
    # Minimal smoke example.
    torch.manual_seed(42)

    bsz = 8
    n = 64
    m = 4

    cfg = DPFStepConfig()
    step = DPFStepModule(cfg)

    x_prev = torch.zeros(bsz, n, 2)
    v_prev = torch.zeros(bsz, n, 2)
    log_w_prev = torch.full((bsz, n), fill_value=-torch.log(torch.tensor(float(n))))
    z_t = torch.rand(bsz, m)
    anchors = torch.tensor([[0.0, 10.0], [0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

    x_next, v_next, log_w_next, x_est = step(x_prev, v_prev, log_w_prev, z_t, anchors)
    loss = (x_est ** 2).mean()
    loss.backward()

    print("x_next:", x_next.shape)
    print("v_next:", v_next.shape)
    print("log_w_next:", log_w_next.shape)
    print("x_est:", x_est.shape)
    print("grad(log_process_scale):", step.log_process_scale.grad)
