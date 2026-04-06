from __future__ import annotations

from pathlib import Path

import numpy as np

from config import initialize_config
from data_loader import load_simulation_data
from filters import NonlinearParticleFilter


def _make_filter_and_state(seed: int = 42):
    cfg = initialize_config(30)
    cfg.iterations = 1
    cfg.noise_variance = cfg.noise_variance[:1]

    h5_path = Path(__file__).resolve().parents[2] / "data" / "simulation_data.h5"
    data = load_simulation_data(str(h5_path))

    rng = np.random.RandomState(seed)
    filt = NonlinearParticleFilter(data, cfg, noise_idx=0, rng=rng)
    num_points = data.x_hat_lls.shape[1]
    state = filt.initialize_state(num_points)
    state, _, _ = filt.initialize_first_two(state, iter_idx=0)
    return filt, state, data, cfg


def test_nonlinear_pf_seed_reproducibility_single_run():
    f1, s1, _, _ = _make_filter_and_state(seed=42)
    s1, e1 = f1.step(s1, iter_idx=0, point_idx=2)

    f2, s2, _, _ = _make_filter_and_state(seed=42)
    s2, e2 = f2.step(s2, iter_idx=0, point_idx=2)

    np.testing.assert_allclose(e1, e2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(s1.particles_prev, s2.particles_prev, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(s1.weights, s2.weights, rtol=0.0, atol=0.0)


def test_nonlinear_pf_step_equation_parity():
    filt, state, _, _ = _make_filter_and_state(seed=42)

    prev_particles = state.particles_prev.copy()
    prev_vel = state.vel_prev.copy()
    prev_weights = state.weights.copy()

    particles_pred = prev_particles + prev_vel + filt.process_bias + filt.sample_process()
    z_now = filt.z[:, 2, 0]
    weights_upd = filt.update_weights_nonlinear(particles_pred, prev_weights, z_now)
    est_expected = particles_pred @ weights_upd
    particles_res, weights_res, idx_resampled, did_resample = filt.resample_ess_with_indices(particles_pred, weights_upd)

    if did_resample:
        vel_expected = particles_res - prev_particles[:, idx_resampled]
    else:
        vel_expected = particles_res - prev_particles

    state_after, est = filt.step(state, iter_idx=0, point_idx=2)

    np.testing.assert_allclose(est, est_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state_after.particles_prev, particles_res, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state_after.weights, weights_res, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state_after.vel_prev, vel_expected, rtol=0.0, atol=1e-12)
