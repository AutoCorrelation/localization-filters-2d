#!/usr/bin/env python
"""Quick smoke test to verify Python port works with H5 data."""

import sys
from pathlib import Path

# Add python dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import initialize_config
from data_loader import load_simulation_data
from runner import run_filter

def main():
    print("=" * 60)
    print("SMOKE TEST: Python MATLAB Pipeline Parity")
    print("=" * 60)

    try:
        # Step 1: Initialize config
        print("\n[1] Initialize config...")
        cfg = initialize_config(num_particles=5)  # Even smaller
        cfg.iterations = 2  # Minimal for smoke test
        cfg.noise_variance = cfg.noise_variance[:1]  # First noise only
        print(f"    Config: {cfg.num_particles} particles, {cfg.iterations} iterations")

        # Step 2: Load H5
        print("\n[2] Load H5 data...")
        h5_path = Path(__file__).resolve().parents[1] / "data" / "simulation_data.h5"
        print(f"    Loading: {h5_path}")
        data = load_simulation_data(str(h5_path))
        print(f"    true_state: {data.true_state.shape}")
        print(f"    x_hat_LLS: {data.x_hat_lls.shape}")

        # Step 3: Slice to first 100 points only
        print("\n[2.5] Slice to first 100 points...")
        data.true_state = data.true_state[:100, :, :, :]
        data.x_hat_lls = data.x_hat_lls[:100, :, :, :]
        data.z_lls = data.z_lls[:100, :, :, :]
        data.r_lls = data.r_lls[:100, :, :, :]
        data.ranging = data.ranging[:100, :, :, :]
        data.process_noise = data.process_noise[:, :100, :]
        data.toa_noise = data.toa_noise[:, :100, :]
        print(f"    Sliced true_state: {data.true_state.shape}")

        # Step 4: Run single filter
        print("\n[3] Run NonlinearParticleFilter (first noise only)...")
        est, metric = run_filter("NonlinearParticleFilter", data, cfg)
        print(f"    Estimated shape: {est.shape}")
        print(f"    RMSE: {metric['RMSE']}")
        print(f"    APE: {metric['APE']}")

        # Step 5: Run Baseline
        print("\n[4] Run Baseline...")
        est2, metric2 = run_filter("Baseline", data, cfg)
        print(f"    RMSE: {metric2['RMSE']}")

        # Step 6: Verify results have expected shapes
        print("\n[5] Validation checks...")
        assert est.ndim == 4, f"Expected 4D, got {est.ndim}"
        assert est.shape[0] == 2, f"Position dim should be 2, got {est.shape[0]}"
        assert est.shape[3] == len(cfg.noise_variance), f"Noise dim mismatch"
        assert metric["RMSE"].shape[0] == len(cfg.noise_variance), f"RMSE length mismatch"
        print("    ✓ All shape checks passed")

        print("\n" + "=" * 60)
        print("✓ SMOKE TEST PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
