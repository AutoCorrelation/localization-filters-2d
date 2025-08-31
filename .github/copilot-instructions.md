# 2D Localization Filters

This repository contains MATLAB/Octave implementations of 2D localization filters including Kalman Filter (KF), Enhanced Kalman Filter (KF1), and Particle Filter (PF) for Time of Arrival (TOA) localization applications.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- Install GNU Octave: `sudo apt-get update && sudo apt-get install -y octave`
- Create required data directory: `mkdir -p /home/runner/work/localization-filters-2d/localization-filters-2d/data`
- Navigate to source directory: `cd /home/runner/work/localization-filters-2d/localization-filters-2d/src`

### Key Execution Workflow

**CRITICAL TIMING**: All builds and simulations take significant time. NEVER CANCEL long-running processes.

#### First-time Setup (Data Generation)
- Run preSimulation to generate required data files:
  - `echo 'Y' | octave --no-gui main.m`
  - **NEVER CANCEL**: preSimulation takes 15+ minutes for 1e3 iterations, 45+ minutes for 1e5 iterations. Set timeout to 90+ minutes.
  - Generates ~40MB of data files in `/data` directory including z.mat, toaPos.mat, R.mat, and CSV files

#### Standard Execution (After Data Generation)
- Run main analysis: `echo 'N' | octave --no-gui main.m`
  - **NEVER CANCEL**: Main processing takes 25+ minutes. Set timeout to 60+ minutes.
  - Loads pre-generated data and runs KF, KF1, and PF comparisons
  - Outputs RMSE comparison plots across noise levels [0.01, 0.1, 1, 10, 100]

#### Parameter Optimization
- Run parameter optimization: `echo 'N' | octave --no-gui opti.m`
  - **NEVER CANCEL**: Optimization takes 60+ minutes with 1e4 iterations and 2000 particles. Set timeout to 120+ minutes.
  - Finds optimal gamma parameters for enhanced Kalman filter

### Validation Scenarios

**MANDATORY**: Always test these scenarios after making changes:

1. **Data Generation Test**: 
   - `cd /home/runner/work/localization-filters-2d/localization-filters-2d/src`
   - `echo 'Y' | timeout 5400 octave --no-gui --eval "addpath('.'); Env = Env(1e3); Env.preSimulate(); disp('Data generation successful');"`
   - Verify data files created: `ls -la ../data/ | wc -l` should show ~25+ files

2. **Full Pipeline Test**:
   - Ensure data exists, then run: `echo 'N' | timeout 3600 octave --no-gui main.m`
   - Verify completion by checking process exits successfully
   - Expected output: RMSE values and comparison plots

3. **Octave Compatibility Test**:
   - Verify writematrix compatibility function exists: `test -f src/writematrix.m`
   - Test basic functionality: `octave --no-gui --eval "addpath('.'); disp('Octave compatibility verified');"`

## Critical Dependencies and Compatibility

### MATLAB/Octave Compatibility
- Code requires `writematrix` function which is MATLAB-specific
- **ALWAYS** ensure `src/writematrix.m` compatibility function exists
- If missing, create: `src/writematrix.m` with csvwrite wrapper

### Required Directory Structure
```
/home/runner/work/localization-filters-2d/localization-filters-2d/
├── data/          # Generated simulation data (required, ~40MB when populated)
├── src/           # Source code
│   ├── main.m     # Main execution script
│   ├── opti.m     # Parameter optimization
│   ├── Env.m      # Environment/simulation class
│   ├── KalmanFilter.m, KalmanFilter1.m  # Kalman filter implementations
│   ├── ParticleFilter.m  # Particle filter implementation
│   ├── RMSE.m     # Root mean square error calculation
│   └── writematrix.m  # Octave compatibility function
```

## Common Tasks and Parameters

### Key Parameters (main.m)
- Particles: 1e3
- KF Iterations: 1e3
- PF Iterations: 1e3
- preSimulation: 1e5 iterations (Env class)

### Key Parameters (opti.m)
- Particles: 2000
- Iterations: 1e4
- Alpha range: 1-9 for parameter optimization

### Expected Results
The algorithms compare performance across noise variance levels:
- Noise levels: [0.01, 0.1, 1, 10, 100]
- Expected RMSE ranges:
  - KF: 0.0897 to 8.1255
  - KF1: 0.0852 to 7.5216  
  - PF: 0.0905 to 8.0361

## Build and Test Commands

**NO BUILD SYSTEM**: This is a pure MATLAB/Octave project with no build step.

### Testing
- **NO UNIT TESTS**: No test framework exists. Validation is through execution verification.
- Manual validation: Run the validation scenarios above
- Performance validation: Compare RMSE outputs against expected ranges

### Dependencies
- GNU Octave 8.4.0+ (or MATLAB R2019b+)
- No external packages required
- All algorithms implemented from scratch in source files

## Troubleshooting

### Common Issues
1. **"writematrix undefined"**: Ensure `src/writematrix.m` compatibility function exists
2. **"cannot load data files"**: Run preSimulation first with 'Y' option
3. **Process appears hung**: Normal - algorithms are computationally intensive, wait for completion
4. **Out of memory**: Reduce iteration parameters (1e3 → 1e2) for testing

### Performance Notes
- preSimulation: ~1 second per 100 iterations
- Main processing: ~1.5 seconds per iteration with 1e3 particles
- Memory usage: Up to 2.5GB during execution
- Disk usage: ~40MB for generated data files

## Code Structure

### Key Classes
- `Env`: Simulation environment and data generation
- `KalmanFilter`: Standard Kalman filter implementation  
- `KalmanFilter1`: Enhanced Kalman filter with gamma parameter
- `ParticleFilter`: Particle filter with resampling
- `RMSE`: Error calculation utilities

### Main Workflows
- **Data Generation**: Env.preSimulate() creates synthetic TOA measurements
- **Filter Comparison**: main.m runs all three filters and compares RMSE performance
- **Parameter Optimization**: opti.m finds optimal gamma values for KF1