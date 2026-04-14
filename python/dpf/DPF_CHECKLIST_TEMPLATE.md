# DPF Paper Reproduction Checklist Template

Use this checklist to implement one exact paper variant of DPF without being constrained by legacy architecture.

## 0) Paper Lock (must be fixed first)
- [ ] Paper title/version is fixed (arXiv version/date or conference camera-ready).
- [ ] Target algorithm block is identified (section/figure/algorithm number).
- [ ] Reproduction target metric(s) are fixed (RMSE/APE/NLL/etc).
- [ ] Allowed deviations from the paper are explicitly listed.

Notes:
- Paper ID:
- Algorithm section:
- Intended deviation list:

## 1) Math Contract
- [ ] State definition is fixed ($x_t$, optional $v_t$, latent modes).
- [ ] Transition model is fixed ($p(x_t|x_{t-1})$).
- [ ] Observation model is fixed ($p(z_t|x_t)$).
- [ ] Proposal distribution is fixed (bootstrap/proposal net/hybrid).
- [ ] Weight update is fixed (log-domain equation preferred).
- [ ] Resampling method is fixed (soft/systematic/Gumbel-Softmax/etc).

Notes:
- Forward equation summary:
- Log-weight normalization equation:
- Resampling equation:

## 2) Learnable Parameters
- [ ] Learnable parameter list is fixed.
- [ ] Non-learnable physics constants are fixed.
- [ ] Parameter constraints are encoded (softplus/clamp/cholesky).

Template:
- Learnable:
- Frozen:
- Constraints:

## 3) Existing Simulator Data Mapping (this repo)
- [ ] Data source file fixed: data/simulation_data.h5 or data/simulation_data_imm.h5.
- [ ] Noise slice policy fixed (single noise index or all).
- [ ] Input tensors mapped from loader output.

Expected mapped tensors after data_loader transpose:
- ranging: (num_anchors, num_points, num_iterations, num_noise)
- true_state: (state_dim, num_points, num_iterations, num_noise)
- process_noise: (state_like_dim, num_samples, num_noise) or equivalent
- process_bias: (state_like_dim, num_noise)

Batch build decision:
- [ ] Batch axis = iteration axis (recommended first)
- [ ] Sequence length = num_points
- [ ] Feature targets = true_state[0:2]

## 4) One-Step Unit Test Contract (before full training)
- [ ] Input/output shapes documented.
- [ ] One forward pass works on synthetic data.
- [ ] Backward pass gives non-None gradients.
- [ ] No NaN/Inf in outputs or gradients.

Template I/O contract:
- Input x_prev:
- Input log_w_prev:
- Input z_t:
- Output x_t_particles:
- Output log_w_t:
- Output x_t_est:

## 5) Training Loop Contract
- [ ] Loss is fixed (trajectory MSE, NLL, hybrid).
- [ ] Teacher forcing policy fixed.
- [ ] Truncated BPTT length fixed.
- [ ] Optimizer/lr schedule fixed.
- [ ] Gradient clipping threshold fixed.
- [ ] Seed policy fixed.

Template:
- Loss:
- Optimizer:
- LR schedule:
- Clip:
- Epochs:

## 6) Numerical Stability Checklist
- [ ] Log-sum-exp used in normalization.
- [ ] Variance/covariance floors applied.
- [ ] Epsilon policy consistent across modules.
- [ ] Resampling temperature limits defined.

Template constants:
- eps:
- min_var:
- min_log_weight:
- max_grad_norm:

## 7) Reproduction Runbook
- [ ] Full config saved to file.
- [ ] Checkpoint + best metric saved.
- [ ] Eval script uses identical preprocessing.
- [ ] Results exported to table and figure.

Template outputs:
- Output dir:
- Checkpoint naming:
- Metric csv path:
- Figure path:

## 8) Ablation Matrix
- [ ] Baseline PF vs DPF core
- [ ] Resampling variant ablation
- [ ] Particle count sweep
- [ ] Noise-level sweep

Ablation table template:
- Exp ID:
- Changed factor:
- RMSE:
- APE:
- Runtime:

## 9) Done Criteria
- [ ] One-step differentiable test passes.
- [ ] Small-scale training loss decreases consistently.
- [ ] Main metric reproduced within target tolerance.
- [ ] Re-run from scratch reproduces same trend.
