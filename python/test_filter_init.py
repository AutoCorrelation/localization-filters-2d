"""Test that filters can initialize with corrected H5 data dimensions."""
import numpy as np
from data_loader import load_simulation_data
from config import initialize_config
from filters import Baseline, NonlinearParticleFilter, AdaptiveParticleFilter

def test_baseline_filter():
    """Test Baseline filter initialization with one noise level."""
    print("Loading data...")
    data = load_simulation_data('data/simulation_data.h5')
    
    print(f"Data shapes after loading:")
    print(f"  x_hat_lls: {data.x_hat_lls.shape}")
    print(f"  z_lls: {data.z_lls.shape}")
    print()
    
    config = initialize_config(num_particles=10)
    rng = np.random.RandomState(42)
    
    print("Testing Baseline filter initialization...")
    noise_idx = 0
    try:
        baseline = Baseline(data, config, noise_idx, rng)
        print(f"  Successfully initialized Baseline filter")
        print(f"  x_hat_projected shape: {baseline.x_hat_projected.shape}")
        
        # Test state initialization
        state = baseline.initialize_state(num_points=100000)
        print(f"  State initialized with estimated_pos shape: {state.estimated_pos.shape}")
        
        # Test first two points initialization
        iter_idx = 0
        state, p1, p2 = baseline.initialize_first_two(state, iter_idx)
        print(f"  First two points: p1={p1.shape}, p2={p2.shape}")
        
        # Test step
        point_idx = 2
        state, est = baseline.step(state, iter_idx, point_idx)
        print(f"  Step result: est shape={est.shape}")
        
        print("  [PASS] Baseline filter works!")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nonlinear_particle_filter():
    """Test NonlinearParticleFilter initialization."""
    print("\nTesting NonlinearParticleFilter initialization...")
    data = load_simulation_data('data/simulation_data.h5')
    config = initialize_config(num_particles=10)
    rng = np.random.RandomState(42)
    
    try:
        npf = NonlinearParticleFilter(data, config, noise_idx=0, rng=rng)
        print(f"  Successfully initialized NonlinearParticleFilter")
        
        state = npf.initialize_state(num_points=100000)
        print(f"  State shape: particles_prev={state.particles_prev.shape}")
        
        print("  [PASS] NonlinearParticleFilter works!")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = []
    results.append(test_baseline_filter())
    results.append(test_nonlinear_particle_filter())
    
    print("\n" + "="*50)
    if all(results):
        print("All tests PASSED!")
    else:
        print(f"Some tests FAILED: {sum(results)}/{len(results)} passed")
