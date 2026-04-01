from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_params import get_best_params
from config import Config
from data_loader import SimulationData


def _extract_noise_bank(noise_tensor: np.ndarray, noise_idx: int) -> np.ndarray:
    dims = noise_tensor.ndim
    if dims == 2:
        return noise_tensor
    if dims == 3:
        return np.squeeze(noise_tensor[:, :, noise_idx])
    if dims == 4:
        temp = np.squeeze(noise_tensor[:, :, :, noise_idx])
        return temp.reshape(temp.shape[0], -1)
    raise ValueError(f"Unsupported noise tensor dimensions: {dims}")


def _systematic_resample_indices(weights: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    cdf = np.cumsum(weights.ravel())
    cdf[-1] = 1.0
    u0 = rng.rand() / n
    u = u0 + np.arange(n) / n
    idx = np.zeros(n, dtype=int)
    j = 0
    for i in range(n):
        while u[i] > cdf[j]:
            j += 1
        idx[i] = j
    return idx


@dataclass
class FilterState:
    estimated_pos: np.ndarray
    particles_prev: np.ndarray
    vel_prev: np.ndarray
    weights: np.ndarray
    err_cov: np.ndarray | None = None
    vel: np.ndarray | None = None
    m_moment: np.ndarray | None = None
    s_moment: np.ndarray | None = None
    nominal_diag_r: np.ndarray | None = None
    diag_r: np.ndarray | None = None
    kf_mean: np.ndarray | None = None
    kf_cov: np.ndarray | None = None


class Baseline:
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        self.rng = rng
        z_noise = np.squeeze(data.z_lls[:, :, :, noise_idx])
        num_points = z_noise.shape[1]
        num_iterations = z_noise.shape[2]
        self.x_hat_projected = np.zeros((2, num_points, num_iterations), dtype=float)
        for iter_idx in range(num_iterations):
            self.x_hat_projected[:, :, iter_idx] = config.pinv_h @ z_noise[:, :, iter_idx]

    def initialize_state(self, num_points: int) -> FilterState:
        return FilterState(
            estimated_pos=np.zeros((2, num_points), dtype=float),
            particles_prev=np.zeros((2, 1), dtype=float),
            vel_prev=np.zeros((2, 1), dtype=float),
            weights=np.ones(1, dtype=float),
        )

    def initialize_first_two(self, state: FilterState, iter_idx: int):
        p1 = self.x_hat_projected[:, 0, iter_idx]
        p2 = self.x_hat_projected[:, 1, iter_idx]
        state.estimated_pos[:, 0] = p1
        state.estimated_pos[:, 1] = p2
        return state, p1, p2

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        est = self.x_hat_projected[:, point_idx, iter_idx]
        state.estimated_pos[:, point_idx] = est
        return state, est


class LinearKalmanFilter:
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        self.rng = rng
        self.h = config.h
        self.x_hat = np.squeeze(data.x_hat_lls[:, :, :, noise_idx])
        self.z = np.squeeze(data.z_lls[:, :, :, noise_idx])
        self.r = np.squeeze(data.r_lls[:, :, :, :, noise_idx])
        self.q = np.squeeze(data.q[:, :, noise_idx])
        self.p0 = np.squeeze(data.p0[:, :, noise_idx])
        self.process_bias = np.reshape(np.squeeze(data.process_bias[:, noise_idx]), (2, 1))

    def initialize_state(self, num_points: int) -> FilterState:
        return FilterState(
            estimated_pos=np.zeros((2, num_points), dtype=float),
            particles_prev=np.zeros((2, 1), dtype=float),
            vel_prev=np.zeros((2, 1), dtype=float),
            weights=np.ones(1, dtype=float),
            err_cov=np.zeros((2, 2, num_points), dtype=float),
            vel=np.zeros((2, num_points), dtype=float),
        )

    def initialize_first_two(self, state: FilterState, iter_idx: int):
        p1 = self.x_hat[:, 0, iter_idx]
        p2 = self.x_hat[:, 1, iter_idx]
        state.err_cov[:, :, 0] = self.p0
        state.err_cov[:, :, 1] = self.p0
        state.estimated_pos[:, 0] = p1
        state.estimated_pos[:, 1] = p2
        state.vel[:, 1] = p2 - p1
        return state, p1, p2

    def _predict_cov(self, p_prev: np.ndarray, point_idx: int) -> np.ndarray:
        return p_prev + self.q

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        x_prev = state.estimated_pos[:, point_idx - 1:point_idx]
        p_prev = state.err_cov[:, :, point_idx - 1]
        vel_prev = state.vel[:, point_idx - 1:point_idx]

        x_pred = x_prev + vel_prev + self.process_bias
        p_pred = self._predict_cov(p_prev, point_idx)

        r_mat = self.r[:, :, point_idx, iter_idx] + 1e-6 * np.eye(self.r.shape[0])
        s = self.h @ p_pred @ self.h.T + r_mat
        k = p_pred @ self.h.T @ np.linalg.inv(s)

        z_now = self.z[:, point_idx, iter_idx:iter_idx + 1]
        est = x_pred + k @ (z_now - self.h @ x_pred)
        p_now = (np.eye(2) - k @ self.h) @ p_pred

        state.estimated_pos[:, point_idx] = est.ravel()
        state.err_cov[:, :, point_idx] = p_now
        state.vel[:, point_idx] = (est - x_prev).ravel()
        return state, est.ravel()


class LinearKalmanFilterDecayQ(LinearKalmanFilter):
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        super().__init__(data, config, noise_idx, rng)
        self.decay_gamma = config.decay_gamma[noise_idx]

    def _predict_cov(self, p_prev: np.ndarray, point_idx: int) -> np.ndarray:
        return p_prev + self.q * np.exp(-self.decay_gamma * (point_idx - 2))


class LinearParticleFilter:
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        self.rng = rng
        self.h = config.h
        self.x_hat = np.squeeze(data.x_hat_lls[..., noise_idx])
        self.z = np.squeeze(data.z_lls[..., noise_idx])
        self.r = np.squeeze(data.r_lls[..., noise_idx])

        self.process_noise = data.process_noise[..., noise_idx] if data.process_noise.ndim >= 3 else data.process_noise
        if data.process_bias.ndim == 2:
            self.process_bias = data.process_bias[:, noise_idx:noise_idx + 1]
        else:
            self.process_bias = np.reshape(data.process_bias, (2, 1))
        self.toa_noise = data.toa_noise[..., noise_idx] if data.toa_noise.ndim >= 3 else data.toa_noise
        self.num_particles = config.num_particles
        self.resample_threshold_ratio = config.resample_threshold_ratio
        self.noise_std = np.sqrt(config.noise_variance[noise_idx])

    def initialize_state(self, num_points: int) -> FilterState:
        return FilterState(
            estimated_pos=np.zeros((2, num_points), dtype=float),
            particles_prev=np.zeros((2, self.num_particles), dtype=float),
            vel_prev=np.zeros((2, self.num_particles), dtype=float),
            weights=np.ones(self.num_particles, dtype=float) / self.num_particles,
        )

    def initialize_first_two(self, state: FilterState, iter_idx: int):
        p1 = self.x_hat[:, 0, iter_idx]
        p2 = self.x_hat[:, 1, iter_idx]
        state.estimated_pos[:, 0] = p1
        state.estimated_pos[:, 1] = p2

        sampled_prev = self.sample_toa(p1.reshape(2, 1))
        sampled_curr = self.sample_toa(p2.reshape(2, 1))
        state.particles_prev = sampled_curr
        state.vel_prev = sampled_curr - sampled_prev
        return state, p1, p2

    def sample_toa(self, center: np.ndarray) -> np.ndarray:
        if self.toa_noise.size == 0 or self.toa_noise.ndim < 2:
            return center + self.noise_std * self.rng.randn(2, self.num_particles)
        idx = self.rng.randint(0, self.toa_noise.shape[1], size=self.num_particles)
        return center + self.toa_noise[:, idx]

    def sample_process(self) -> np.ndarray:
        if self.process_noise.size == 0 or self.process_noise.ndim < 2:
            return self.noise_std * self.rng.randn(2, self.num_particles)
        idx = self.rng.randint(0, self.process_noise.shape[1], size=self.num_particles)
        return self.process_noise[:, idx]

    def sample_process_single(self) -> np.ndarray:
        if self.process_noise.size == 0 or self.process_noise.ndim < 2:
            return self.noise_std * self.rng.randn(2, 1)
        idx = self.rng.randint(0, self.process_noise.shape[1])
        return self.process_noise[:, idx:idx + 1]

    def update_weights_linear(self, particles: np.ndarray, prev_weights: np.ndarray, z_now: np.ndarray, r_mat: np.ndarray) -> np.ndarray:
        r_reg = r_mat + 1e-8 * np.eye(r_mat.shape[0])
        residual = z_now[:, None] - self.h @ particles
        r_inv = np.linalg.inv(r_reg)
        dist = np.sum((r_inv @ residual) * residual, axis=0)
        w = prev_weights * np.exp(-0.5 * dist)
        w = w + 1e-300
        w = w / np.sum(w)
        return w

    def resample_ess_with_indices(self, particles: np.ndarray, weights: np.ndarray):
        ess = 1.0 / np.sum(weights ** 2)
        if ess < self.num_particles * self.resample_threshold_ratio:
            idx = _systematic_resample_indices(weights, self.num_particles, self.rng)
            particles_out = particles[:, idx]
            weights_out = np.ones(self.num_particles, dtype=float) / self.num_particles
            return particles_out, weights_out, idx, True
        idx = np.arange(self.num_particles)
        return particles, weights, idx, False

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        particles_pred = state.particles_prev + state.vel_prev + self.process_bias + self.sample_process()

        r_mat = self.r[:, :, point_idx, iter_idx]
        z_now = self.z[:, point_idx, iter_idx]
        weights_upd = self.update_weights_linear(particles_pred, state.weights, z_now, r_mat)

        est = particles_pred @ weights_upd
        particles_res, weights_res, idx_resampled, did_resample = self.resample_ess_with_indices(particles_pred, weights_upd)

        if did_resample:
            state.vel_prev = particles_res - state.particles_prev[:, idx_resampled]
        else:
            state.vel_prev = particles_res - state.particles_prev
        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


class NonlinearParticleFilter(LinearParticleFilter):
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        super().__init__(data, config, noise_idx, rng)
        self.z = np.squeeze(data.ranging[:, :, :, noise_idx])
        self.r = 4.0 * config.noise_variance[noise_idx] * np.eye(data.ranging.shape[0])
        self.anchor_pos = config.anchor.T

    def h_nonlinear(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(2, 1)
        num_anchors = self.anchor_pos.shape[0]
        num_particles = x.shape[1]
        y_pred = np.zeros((num_anchors, num_particles), dtype=float)
        for i in range(num_anchors):
            dx = x[0, :] - self.anchor_pos[i, 0]
            dy = x[1, :] - self.anchor_pos[i, 1]
            y_pred[i, :] = np.sqrt(dx ** 2 + dy ** 2)
        return y_pred

    def update_weights_nonlinear(self, particles: np.ndarray, prev_weights: np.ndarray, z_now: np.ndarray) -> np.ndarray:
        y_pred = self.h_nonlinear(particles)
        num_anchors = y_pred.shape[0]
        r_inv = np.eye(num_anchors) / (self.noise_std ** 2)
        errors = z_now[:, None] - y_pred
        distances = np.sum((r_inv @ errors) * errors, axis=0)
        w = prev_weights * np.exp(-0.5 * distances)
        w = w + 1e-300
        return w / np.sum(w)

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        particles_pred = state.particles_prev + state.vel_prev + self.process_bias + self.sample_process()
        z_now = self.z[:, point_idx, iter_idx]
        weights_upd = self.update_weights_nonlinear(particles_pred, state.weights, z_now)

        est = particles_pred @ weights_upd
        particles_res, weights_res, idx_resampled, did_resample = self.resample_ess_with_indices(particles_pred, weights_upd)

        if did_resample:
            state.vel_prev = particles_res - state.particles_prev[:, idx_resampled]
        else:
            state.vel_prev = particles_res - state.particles_prev
        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


class AdaptiveParticleFilter(NonlinearParticleFilter):
    def __init__(
        self,
        data: SimulationData,
        config: Config,
        noise_idx: int,
        rng: np.random.RandomState,
        beta: float = 0.99,
        lambda_r: float = 1.0,
    ):
        super().__init__(data, config, noise_idx, rng)
        self.beta = beta
        self.lambda_r = lambda_r
        self.r_floor = 1e-6
        self.r_ceil = 1e4

    def initialize_state(self, num_points: int) -> FilterState:
        state = super().initialize_state(num_points)
        num_anchors = self.anchor_pos.shape[0]
        nominal_var = self.noise_std ** 2
        state.nominal_diag_r = nominal_var * np.ones(num_anchors, dtype=float)
        state.m_moment = np.zeros(num_anchors, dtype=float)
        state.s_moment = np.zeros(num_anchors, dtype=float)
        state.diag_r = state.nominal_diag_r.copy()
        return state

    def update_weights_with_r(
        self,
        particles: np.ndarray,
        prev_weights: np.ndarray,
        z_now: np.ndarray,
        r_mat: np.ndarray,
    ) -> np.ndarray:
        y_pred = self.h_nonlinear(particles)
        errors = z_now[:, None] - y_pred
        r_inv = np.diag(1.0 / np.diag(r_mat))
        distances = np.sum((r_inv @ errors) * errors, axis=0)
        w = prev_weights * np.exp(-0.5 * distances)
        w = w + 1e-300
        return w / np.sum(w)

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        particles_pred = state.particles_prev + state.vel_prev + self.process_bias + self.sample_process()
        z_now = self.z[:, point_idx, iter_idx]

        x_hat_weighted = particles_pred @ state.weights
        y_pred_weighted = self.h_nonlinear(x_hat_weighted).ravel()
        e = z_now - y_pred_weighted

        state.m_moment = self.beta * state.m_moment + (1 - self.beta) * e
        state.s_moment = self.beta * state.s_moment + (1 - self.beta) * ((e - state.m_moment) ** 2)

        state.diag_r = state.nominal_diag_r + self.lambda_r * state.s_moment
        state.diag_r = np.clip(state.diag_r, self.r_floor, self.r_ceil)

        r_mat = np.diag(state.diag_r)
        weights_upd = self.update_weights_with_r(particles_pred, state.weights, z_now, r_mat)

        est = particles_pred @ weights_upd
        particles_res, weights_res, idx_resampled, did_resample = self.resample_ess_with_indices(particles_pred, weights_upd)
        if did_resample:
            state.vel_prev = particles_res - state.particles_prev[:, idx_resampled]
        else:
            state.vel_prev = particles_res - state.particles_prev
        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


class RDiagPriorEditAdaptiveParticleFilter(AdaptiveParticleFilter):
    def __init__(
        self,
        data: SimulationData,
        config: Config,
        noise_idx: int,
        rng: np.random.RandomState,
        beta: float,
        lambda_r: float,
    ):
        super().__init__(data, config, noise_idx, rng, beta, lambda_r)
        self.prior_sigma_gate = max(config.rdiag_prior_sigma_gate, 0.0)
        self.prior_max_retry = max(1, int(round(config.rdiag_prior_max_retry)))
        self.roughening_k = max(config.rdiag_roughening_k, 0.0)

    def _compute_roughening_sigma(self, particles: np.ndarray) -> np.ndarray:
        n = particles.shape[0]
        n_particles = particles.shape[1]
        span = np.max(particles, axis=1, keepdims=True) - np.min(particles, axis=1, keepdims=True)
        return self.roughening_k * span * (n_particles ** (-1.0 / n))

    def _is_within_gate(self, particles: np.ndarray, z_now: np.ndarray, sigma_gate: np.ndarray) -> np.ndarray:
        y_pred = self.h_nonlinear(particles)
        residual = np.abs(z_now[:, None] - y_pred)
        return np.all(residual <= sigma_gate[:, None], axis=0)

    def _score(self, particle: np.ndarray, z_now: np.ndarray, sigma_gate: np.ndarray) -> float:
        y_pred = self.h_nonlinear(particle.reshape(2, 1)).ravel()
        residual = np.abs(z_now - y_pred)
        return float(np.sum(residual / np.maximum(sigma_gate, self.r_floor)))

    def apply_prior_editing_with_adaptive_r(
        self,
        particles_in: np.ndarray,
        prev_particles: np.ndarray,
        vel_prev: np.ndarray,
        z_now: np.ndarray,
        diag_r: np.ndarray,
    ) -> np.ndarray:
        particles_out = particles_in.copy()
        sigma_gate = self.prior_sigma_gate * np.sqrt(np.maximum(diag_r, self.r_floor))
        valid_mask = self._is_within_gate(particles_out, z_now, sigma_gate)
        reject_idx = np.where(~valid_mask)[0]
        sigma_rough = self._compute_roughening_sigma(particles_in)

        for idx in reject_idx:
            best_particle = particles_out[:, idx:idx + 1].copy()
            best_score = self._score(best_particle.ravel(), z_now, sigma_gate)
            accepted = False

            for _ in range(self.prior_max_retry):
                base = prev_particles[:, idx:idx + 1] + sigma_rough * self.rng.randn(2, 1)
                candidate = base + vel_prev[:, idx:idx + 1] + self.process_bias + self.sample_process_single()
                score = self._score(candidate.ravel(), z_now, sigma_gate)
                if score < best_score:
                    best_score = score
                    best_particle = candidate

                if self._is_within_gate(candidate, z_now, sigma_gate)[0]:
                    particles_out[:, idx:idx + 1] = candidate
                    accepted = True
                    break

            if not accepted:
                particles_out[:, idx:idx + 1] = best_particle

        return particles_out

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        particles_pred = state.particles_prev + state.vel_prev + self.process_bias + self.sample_process()
        z_now = self.z[:, point_idx, iter_idx]

        x_hat_weighted = particles_pred @ state.weights
        y_pred_weighted = self.h_nonlinear(x_hat_weighted).ravel()
        e = z_now - y_pred_weighted

        state.m_moment = self.beta * state.m_moment + (1 - self.beta) * e
        state.s_moment = self.beta * state.s_moment + (1 - self.beta) * ((e - state.m_moment) ** 2)
        state.diag_r = np.clip(state.nominal_diag_r + self.lambda_r * state.s_moment, self.r_floor, self.r_ceil)

        particles_edited = self.apply_prior_editing_with_adaptive_r(
            particles_pred,
            state.particles_prev,
            state.vel_prev,
            z_now,
            state.diag_r,
        )

        weights_upd = self.update_weights_with_r(particles_edited, state.weights, z_now, np.diag(state.diag_r))
        est = particles_edited @ weights_upd

        particles_res, weights_res, idx_resampled, did_resample = self.resample_ess_with_indices(particles_edited, weights_upd)
        if did_resample:
            state.vel_prev = particles_res - state.particles_prev[:, idx_resampled]
        else:
            state.vel_prev = particles_res - state.particles_prev
        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


class RougheningPriorEditingParticleFilter(NonlinearParticleFilter):
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        super().__init__(data, config, noise_idx, rng)
        self.roughening_k = max(config.roughening_k, 0.0)
        self.prior_sigma_gate = max(config.prior_sigma_gate, 0.0)
        self.prior_max_retry = max(1, int(round(config.prior_max_retry)))

    def _compute_roughening_sigma(self, particles: np.ndarray) -> np.ndarray:
        n = particles.shape[0]
        n_particles = particles.shape[1]
        max_diff = np.max(particles, axis=1, keepdims=True) - np.min(particles, axis=1, keepdims=True)
        return self.roughening_k * max_diff * (n_particles ** (-1.0 / n))

    def _is_within_gate(self, particles: np.ndarray, z_now: np.ndarray, gate: float) -> np.ndarray:
        y_pred = self.h_nonlinear(particles)
        residual = z_now[:, None] - y_pred
        return np.all(np.abs(residual) <= gate, axis=0)

    def _score(self, particle: np.ndarray, z_now: np.ndarray) -> float:
        y_pred = self.h_nonlinear(particle.reshape(2, 1)).ravel()
        residual = z_now - y_pred
        return float(np.max(np.abs(residual)))

    def apply_prior_editing(
        self,
        particles_in: np.ndarray,
        prev_particles: np.ndarray,
        vel_prev: np.ndarray,
        z_now: np.ndarray,
    ) -> np.ndarray:
        particles_out = particles_in.copy()
        gate = self.prior_sigma_gate * self.noise_std
        valid_mask = self._is_within_gate(particles_out, z_now, gate)
        reject_idx = np.where(~valid_mask)[0]
        if reject_idx.size == 0:
            return particles_out

        sigma_rough = self._compute_roughening_sigma(particles_in)
        for idx in reject_idx:
            best_particle = particles_out[:, idx:idx + 1].copy()
            best_score = self._score(best_particle.ravel(), z_now)
            accepted = False
            for _ in range(self.prior_max_retry):
                base = prev_particles[:, idx:idx + 1] + sigma_rough * self.rng.randn(2, 1)
                candidate = base + vel_prev[:, idx:idx + 1] + self.process_bias + self.sample_process_single()
                score = self._score(candidate.ravel(), z_now)
                if score < best_score:
                    best_score = score
                    best_particle = candidate
                if self._is_within_gate(candidate, z_now, gate)[0]:
                    particles_out[:, idx:idx + 1] = candidate
                    accepted = True
                    break
            if not accepted:
                particles_out[:, idx:idx + 1] = best_particle
        return particles_out

    def apply_roughening(self, particles_in: np.ndarray) -> np.ndarray:
        sigma = self._compute_roughening_sigma(particles_in)
        return particles_in + sigma * self.rng.randn(*particles_in.shape)

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        particles_pred = state.particles_prev + state.vel_prev + self.process_bias + self.sample_process()
        z_now = self.z[:, point_idx, iter_idx]
        particles_pred = self.apply_prior_editing(particles_pred, state.particles_prev, state.vel_prev, z_now)

        weights_upd = self.update_weights_nonlinear(particles_pred, state.weights, z_now)
        est = particles_pred @ weights_upd

        particles_res, weights_res, idx_resampled, did_resample = self.resample_ess_with_indices(particles_pred, weights_upd)
        particles_res = self.apply_roughening(particles_res)

        if did_resample:
            state.vel_prev = particles_res - state.particles_prev[:, idx_resampled]
        else:
            state.vel_prev = particles_res - state.particles_prev
        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


class RegularizedParticleFilter(NonlinearParticleFilter):
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        super().__init__(data, config, noise_idx, rng)
        self.bandwidth_floor = 1e-3

    def _robust_cholesky(self, s: np.ndarray) -> np.ndarray:
        n = s.shape[0]
        eye = np.eye(n)
        trace_scale = np.trace(s) / max(n, 1)
        base_jitter = max(1e-12, 1e-10 * max(trace_scale, 1.0))
        s_sym = 0.5 * (s + s.T)

        for k in range(8):
            jitter = base_jitter * (10 ** k)
            try:
                return np.linalg.cholesky(s_sym + jitter * eye)
            except np.linalg.LinAlgError:
                continue

        eigvals, eigvecs = np.linalg.eigh(s_sym)
        eigvals = np.clip(eigvals, 1e-12, None)
        return eigvecs @ np.diag(np.sqrt(eigvals))

    def _sample_epanechnikov_noise(self, n: int, n_particles: int) -> np.ndarray:
        noise = np.zeros((n, n_particles), dtype=float)
        for k in range(n_particles):
            accepted = False
            while not accepted:
                direction = self.rng.randn(n, 1)
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-12:
                    continue
                direction = direction / direction_norm
                r = self.rng.rand() ** (1.0 / n)
                candidate = r * direction
                if self.rng.rand() <= (1 - r ** 2):
                    noise[:, k:k + 1] = candidate
                    accepted = True
        return noise

    def _compute_cov_root_and_bandwidth(self, particles: np.ndarray):
        n = particles.shape[0]
        n_particles = particles.shape[1]
        if n_particles <= 1:
            s = (self.noise_std ** 2) * np.eye(n)
        else:
            mu = np.mean(particles, axis=1, keepdims=True)
            centered = particles - mu
            s = (centered @ centered.T) / (n_particles - 1)
        s = 0.5 * (s + s.T)
        a = self._robust_cholesky(s)

        v_n = np.pi ** (n / 2) / np.math.gamma(n / 2 + 1)
        h = 0.5 * ((8 * (v_n ** -1) * (n + 4) * (2 * np.pi) ** n) ** (1 / (n + 4))) * (n_particles ** (-1 / (n + 4)))
        h = max(h, self.bandwidth_floor)
        return a, h

    def regularized_resample_ess(self, particles: np.ndarray, weights: np.ndarray):
        ess = 1.0 / np.sum(weights ** 2)
        if ess >= self.num_particles * self.resample_threshold_ratio:
            idx = np.arange(self.num_particles)
            return particles, weights, idx, False

        idx = _systematic_resample_indices(weights, self.num_particles, self.rng)
        base_particles = particles[:, idx]
        a, h = self._compute_cov_root_and_bandwidth(base_particles)
        kernel_noise = self._sample_epanechnikov_noise(base_particles.shape[0], self.num_particles)
        particles_out = base_particles + h * (a @ kernel_noise)
        weights_out = np.ones(self.num_particles, dtype=float) / self.num_particles
        return particles_out, weights_out, idx, True

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        particles_pred = state.particles_prev + state.vel_prev + self.process_bias + self.sample_process()
        z_now = self.z[:, point_idx, iter_idx]
        weights_upd = self.update_weights_nonlinear(particles_pred, state.weights, z_now)

        est = particles_pred @ weights_upd
        particles_res, weights_res, idx_resampled, did_resample = self.regularized_resample_ess(particles_pred, weights_upd)

        if did_resample:
            state.vel_prev = particles_res - state.particles_prev[:, idx_resampled]
        else:
            state.vel_prev = particles_res - state.particles_prev
        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


class RBPF(NonlinearParticleFilter):
    def __init__(self, data: SimulationData, config: Config, noise_idx: int, rng: np.random.RandomState):
        super().__init__(data, config, noise_idx, rng)
        self.q2_scale = 1.0
        self.vel_meas_scale = 1.0
        self.init_vel_cov_scale = 1.0
        self.reg_jitter = 1e-9

    def initialize_state(self, num_points: int) -> FilterState:
        state = super().initialize_state(num_points)
        vel_dim = 2
        init_cov = self.init_vel_cov_scale * (self.noise_std ** 2) * np.eye(vel_dim)
        state.kf_mean = np.zeros((vel_dim, self.num_particles), dtype=float)
        state.kf_cov = np.repeat(init_cov[:, :, None], self.num_particles, axis=2)
        return state

    def initialize_first_two(self, state: FilterState, iter_idx: int):
        p1 = self.x_hat[:, 0, iter_idx]
        p2 = self.x_hat[:, 1, iter_idx]
        state.estimated_pos[:, 0] = p1
        state.estimated_pos[:, 1] = p2

        sampled_prev = self.sample_toa(p1.reshape(2, 1))
        sampled_curr = self.sample_toa(p2.reshape(2, 1))
        state.particles_prev = sampled_curr
        state.kf_mean = sampled_curr - sampled_prev
        return state, p1, p2

    def _compute_range_jacobian(self, x: np.ndarray) -> np.ndarray:
        num_anchors = self.anchor_pos.shape[0]
        j = np.zeros((num_anchors, 2), dtype=float)
        for a in range(num_anchors):
            d = x - self.anchor_pos[a, :].reshape(2, 1)
            r = np.linalg.norm(d)
            r = max(r, 1e-10)
            j[a, :] = (d.ravel() / r)
        return j

    def _robust_cholesky(self, s: np.ndarray) -> np.ndarray:
        s_sym = 0.5 * (s + s.T)
        eye = np.eye(s_sym.shape[0])
        for k in range(8):
            jitter = max(1e-12, self.reg_jitter) * (10 ** k)
            try:
                return np.linalg.cholesky(s_sym + jitter * eye)
            except np.linalg.LinAlgError:
                continue
        eigvals, eigvecs = np.linalg.eigh(s_sym)
        eigvals = np.clip(eigvals, 1e-12, None)
        return eigvecs @ np.diag(np.sqrt(eigvals))

    def _sample_gaussian(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        l = self._robust_cholesky(sigma)
        return mu + l @ self.rng.randn(mu.shape[0], 1)

    def _log_gaussian_zero_mean(self, e: np.ndarray, s: np.ndarray) -> float:
        l = self._robust_cholesky(s)
        y = np.linalg.solve(l, e)
        maha = float(y.T @ y)
        log_det = 2 * float(np.sum(np.log(np.abs(np.diag(l)))))
        d = e.shape[0]
        return -0.5 * (maha + log_det + d * np.log(2 * np.pi))

    def _normalize_log_weights(self, log_w: np.ndarray) -> np.ndarray:
        m = np.max(log_w)
        shifted = log_w - m
        lse = m + np.log(np.sum(np.exp(shifted)))
        w = np.exp(log_w - lse)
        w = w + 1e-300
        return w / np.sum(w)

    def step(self, state: FilterState, iter_idx: int, point_idx: int):
        n = self.num_particles
        vel_dim = 2
        num_anchors = self.anchor_pos.shape[0]

        q2 = self.q2_scale * (self.noise_std ** 2) * np.eye(vel_dim)
        r_vel = self.vel_meas_scale * (self.noise_std ** 2) * np.eye(vel_dim)
        r_range = 0.5 * (self.r + self.r.T) + self.reg_jitter * np.eye(num_anchors)
        i2 = np.eye(2)

        z_vel = self.x_hat[:, point_idx, iter_idx:iter_idx + 1] - self.x_hat[:, point_idx - 1, iter_idx:iter_idx + 1]

        particles_pred = np.zeros((2, n), dtype=float)
        kf_mean_upd = np.zeros((vel_dim, n), dtype=float)
        kf_cov_upd = np.zeros((vel_dim, vel_dim, n), dtype=float)
        log_w = np.zeros(n, dtype=float)
        log_prev_w = np.log(np.maximum(state.weights, 1e-300))

        for i in range(n):
            m_prev = state.kf_mean[:, i:i + 1]
            p_prev = state.kf_cov[:, :, i]

            m_pred = m_prev
            p_pred = 0.5 * ((p_prev + q2) + (p_prev + q2).T) + self.reg_jitter * i2

            v_sample = self._sample_gaussian(m_pred, p_pred)
            particles_pred[:, i:i + 1] = state.particles_prev[:, i:i + 1] + v_sample + self.process_bias + self.sample_process_single()

            s_vel = 0.5 * ((p_pred + r_vel) + (p_pred + r_vel).T) + self.reg_jitter * i2
            k_vel = p_pred @ np.linalg.inv(s_vel)
            innov_vel = z_vel - m_pred
            m_upd = m_pred + k_vel @ innov_vel
            p_upd = (i2 - k_vel) @ p_pred @ (i2 - k_vel).T + k_vel @ r_vel @ k_vel.T
            p_upd = 0.5 * (p_upd + p_upd.T) + self.reg_jitter * i2

            kf_mean_upd[:, i:i + 1] = m_upd
            kf_cov_upd[:, :, i] = p_upd

            y_pred = self.h_nonlinear(particles_pred[:, i:i + 1])
            j = self._compute_range_jacobian(particles_pred[:, i:i + 1])
            s_range = j @ p_pred @ j.T + r_range
            s_range = 0.5 * (s_range + s_range.T) + self.reg_jitter * np.eye(num_anchors)

            innov_range = self.z[:, point_idx, iter_idx:iter_idx + 1] - y_pred
            log_like = self._log_gaussian_zero_mean(innov_range, s_range)
            log_w[i] = log_prev_w[i] + log_like

        weights_upd = self._normalize_log_weights(log_w)
        est = particles_pred @ weights_upd

        particles_res, weights_res, idx_resampled, did_resample = self.resample_ess_with_indices(particles_pred, weights_upd)
        if did_resample:
            state.kf_mean = kf_mean_upd[:, idx_resampled]
            state.kf_cov = kf_cov_upd[:, :, idx_resampled]
        else:
            state.kf_mean = kf_mean_upd
            state.kf_cov = kf_cov_upd

        state.particles_prev = particles_res
        state.weights = weights_res
        state.estimated_pos[:, point_idx] = est
        return state, est


def create_filter(
    filter_class: str,
    data: SimulationData,
    config: Config,
    noise_idx: int,
    rng: np.random.RandomState,
):
    name = filter_class.strip().lower()
    if name == "baseline":
        return Baseline(data, config, noise_idx, rng)
    if name == "linearkalmanfilter_decayq":
        return LinearKalmanFilterDecayQ(data, config, noise_idx, rng)
    if name == "nonlinearparticlefilter":
        return NonlinearParticleFilter(data, config, noise_idx, rng)
    if name in {"rbpf", "raoblackwellizedparticlefilter"}:
        return RBPF(data, config, noise_idx, rng)
    if name in {"regularizedparticlefilter", "rpf"}:
        return RegularizedParticleFilter(data, config, noise_idx, rng)
    if name == "customnonlinearparticlefilter":
        return NonlinearParticleFilter(data, config, noise_idx, rng)
    if name == "adaptiveparticlefilter":
        beta, lambda_r = get_best_params(noise_idx)
        return AdaptiveParticleFilter(data, config, noise_idx, rng, beta=beta, lambda_r=lambda_r)
    if name in {"rdiagprioreditadaptiveparticlefilter", "rdpepf"}:
        beta, lambda_r = get_best_params(noise_idx)
        return RDiagPriorEditAdaptiveParticleFilter(data, config, noise_idx, rng, beta=beta, lambda_r=lambda_r)
    if name in {"rougheningprioreditingparticlefilter", "rpepf"}:
        return RougheningPriorEditingParticleFilter(data, config, noise_idx, rng)
    raise ValueError(f"Unsupported filterClass: {filter_class}")
