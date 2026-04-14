from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import initialize_config
from data_loader import load_simulation_data
from dpf.dpf_one_step_example import DPFStepConfig, DPFStepModule, build_batch_from_simulator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _uniform_log_weights(batch_size: int, num_particles: int, device: torch.device) -> torch.Tensor:
    return torch.full(
        (batch_size, num_particles),
        fill_value=-torch.log(torch.tensor(float(num_particles), device=device)),
        device=device,
        dtype=torch.float32,
    )


def run_minimal_training(
    data_path: str,
    motion_model: str,
    noise_idx: int,
    num_particles: int,
    epochs: int,
    max_iterations: int,
    train_points: int,
    lr: float,
    device: torch.device,
):
    cfg = initialize_config(num_particles=num_particles)
    h5_file = Path(data_path) / ("simulation_data_imm.h5" if motion_model.lower() == "imm" else "simulation_data.h5")
    data = load_simulation_data(str(h5_file))

    num_noise = data.ranging.shape[-1]
    if noise_idx < 0 or noise_idx >= num_noise:
        raise ValueError(f"noise_idx must be in [0, {num_noise - 1}], got {noise_idx}")

    max_t = min(max_iterations, data.ranging.shape[2], cfg.iterations)
    point_start = 2
    point_end = min(point_start + train_points, data.ranging.shape[1])
    if point_end <= point_start:
        raise ValueError("train_points is too small for sequence training")

    ranging_t = _to_tensor(data.ranging, device)
    true_state_t = _to_tensor(data.true_state, device)
    x_hat_t = _to_tensor(data.x_hat_lls, device)
    anchors = _to_tensor(cfg.anchor.T, device)

    step_cfg = DPFStepConfig(
        state_dim=2,
        min_scale=1e-4,
        eps=1e-8,
        soft_resample_alpha=0.5,
        use_soft_resampling=True,
    )
    model = DPFStepModule(step_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use iteration axis as batch axis for minimal reproducible training.
    x0 = x_hat_t[:, 0, :max_t, noise_idx].transpose(0, 1)
    x1 = x_hat_t[:, 1, :max_t, noise_idx].transpose(0, 1)
    batch_size = x1.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        x_prev = x1.unsqueeze(1).repeat(1, num_particles, 1)
        v_prev = (x1 - x0).unsqueeze(1).repeat(1, num_particles, 1)
        log_w_prev = _uniform_log_weights(batch_size, num_particles, device)

        loss_acc = torch.tensor(0.0, device=device)
        num_steps = 0

        for point_idx in range(point_start, point_end):
            z_t, target_pos = build_batch_from_simulator(ranging_t, true_state_t, noise_idx=noise_idx, point_idx=point_idx)
            z_t = z_t[:max_t]
            target_pos = target_pos[:max_t]

            x_prev, v_prev, log_w_prev, x_est = model(x_prev, v_prev, log_w_prev, z_t, anchors)
            step_loss = F.mse_loss(x_est, target_pos)
            loss_acc = loss_acc + step_loss
            num_steps += 1

        loss = loss_acc / max(num_steps, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        print(
            f"epoch={epoch:03d} loss={loss.item():.6f} "
            f"proc_scale={torch.nn.functional.softplus(model.log_process_scale).detach().cpu().numpy()} "
            f"obs_scale={torch.nn.functional.softplus(model.log_obs_scale).item():.6f}"
        )

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Minimal DPF training loop on simulator data")
    p.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--motion-model", type=str, default="cv", choices=["cv", "imm"])
    p.add_argument("--noise-idx", type=int, default=2)
    p.add_argument("--num-particles", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=256)
    p.add_argument("--train-points", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"device={device}")
    print(f"noise_idx={args.noise_idx}, particles={args.num_particles}")

    run_minimal_training(
        data_path=args.data_path,
        motion_model=args.motion_model,
        noise_idx=args.noise_idx,
        num_particles=args.num_particles,
        epochs=args.epochs,
        max_iterations=args.max_iterations,
        train_points=args.train_points,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
