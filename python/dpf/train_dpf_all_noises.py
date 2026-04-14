from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
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
DEFAULT_RESULT_DIR = REPO_ROOT / "result" / "dpf"


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _uniform_log_weights(batch_size: int, num_particles: int, device: torch.device) -> torch.Tensor:
    return torch.full(
        (batch_size, num_particles),
        fill_value=-torch.log(torch.tensor(float(num_particles), device=device)),
        device=device,
        dtype=torch.float32,
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, lr: float, epochs: int):
    name = scheduler_name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=lr * 0.1,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(epochs // 3, 1),
            gamma=0.5,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def _compute_metrics(est_pos: np.ndarray, true_pos: np.ndarray, start_point: int = 3) -> tuple[float, float]:
    start_idx = max(start_point - 1, 0)
    delta = est_pos[:, start_idx:, :] - true_pos[:, start_idx:, :]
    dist_sq = np.sum(delta * delta, axis=0)
    dist = np.sqrt(dist_sq)
    rmse = float(np.sqrt(np.mean(dist_sq)))
    ape = float(np.mean(dist))
    return rmse, ape


def _split_iterations(num_iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(num_iterations)
    rng.shuffle(all_idx)

    train_end = int(0.8 * num_iterations)
    val_end = int(0.9 * num_iterations)

    train_idx = np.sort(all_idx[:train_end])
    val_idx = np.sort(all_idx[train_end:val_end])
    test_idx = np.sort(all_idx[val_end:])

    return train_idx, val_idx, test_idx


def _sequence_loss_for_indices(
    model: DPFStepModule,
    ranging_t: torch.Tensor,
    true_state_t: torch.Tensor,
    x_hat_t: torch.Tensor,
    anchors_t: torch.Tensor,
    noise_idx: int,
    iter_indices: np.ndarray,
    num_particles: int,
    point_start: int,
    point_end: int,
    device: torch.device,
) -> torch.Tensor:
    x0 = x_hat_t[:, 0, iter_indices, noise_idx].transpose(0, 1)
    x1 = x_hat_t[:, 1, iter_indices, noise_idx].transpose(0, 1)
    batch_size = x1.shape[0]

    x_prev = x1.unsqueeze(1).repeat(1, num_particles, 1)
    v_prev = (x1 - x0).unsqueeze(1).repeat(1, num_particles, 1)
    log_w_prev = _uniform_log_weights(batch_size, num_particles, device)

    total = torch.tensor(0.0, device=device)
    count = 0

    for point_idx in range(point_start, point_end):
        z_t, target_pos = build_batch_from_simulator(ranging_t, true_state_t, noise_idx=noise_idx, point_idx=point_idx)
        z_t = z_t[iter_indices]
        target_pos = target_pos[iter_indices]

        x_prev, v_prev, log_w_prev, x_est = model(x_prev, v_prev, log_w_prev, z_t, anchors_t)
        total = total + F.mse_loss(x_est, target_pos)
        count += 1

    return total / max(count, 1)


@torch.no_grad()
def _rollout_estimates(
    model: DPFStepModule,
    ranging_t: torch.Tensor,
    true_state_t: torch.Tensor,
    x_hat_t: torch.Tensor,
    anchors_t: torch.Tensor,
    noise_idx: int,
    iter_indices: np.ndarray,
    num_particles: int,
    point_start: int,
    point_end: int,
    device: torch.device,
) -> np.ndarray:
    num_points = point_end
    num_iters = len(iter_indices)

    est = np.zeros((2, num_points, num_iters), dtype=np.float32)

    x0 = x_hat_t[:, 0, iter_indices, noise_idx].transpose(0, 1)
    x1 = x_hat_t[:, 1, iter_indices, noise_idx].transpose(0, 1)
    est[:, 0, :] = x0.transpose(0, 1).cpu().numpy()
    est[:, 1, :] = x1.transpose(0, 1).cpu().numpy()

    x_prev = x1.unsqueeze(1).repeat(1, num_particles, 1)
    v_prev = (x1 - x0).unsqueeze(1).repeat(1, num_particles, 1)
    log_w_prev = _uniform_log_weights(num_iters, num_particles, device)

    for point_idx in range(point_start, point_end):
        z_t, _ = build_batch_from_simulator(ranging_t, true_state_t, noise_idx=noise_idx, point_idx=point_idx)
        z_t = z_t[iter_indices]
        x_prev, v_prev, log_w_prev, x_est = model(x_prev, v_prev, log_w_prev, z_t, anchors_t)
        est[:, point_idx, :] = x_est.transpose(0, 1).cpu().numpy()

    return est


def train_single_noise(
    ranging_t: torch.Tensor,
    true_state_t: torch.Tensor,
    x_hat_t: torch.Tensor,
    anchors_t: torch.Tensor,
    noise_idx: int,
    noise_variance: float,
    num_particles: int,
    epochs: int,
    lr: float,
    scheduler_name: str,
    iter_batch_size: int,
    point_start: int,
    point_end: int,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
) -> dict:
    num_iterations = ranging_t.shape[2]
    train_idx, val_idx, test_idx = _split_iterations(num_iterations, seed=seed + noise_idx)

    step_cfg = DPFStepConfig(
        state_dim=2,
        min_scale=1e-4,
        eps=1e-8,
        soft_resample_alpha=0.5,
        use_soft_resampling=True,
    )
    model = DPFStepModule(step_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _build_scheduler(optimizer, scheduler_name=scheduler_name, lr=lr, epochs=epochs)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch_iter_idx in _iter_batches(train_idx, iter_batch_size):
            optimizer.zero_grad()
            loss = _sequence_loss_for_indices(
                model=model,
                ranging_t=ranging_t,
                true_state_t=true_state_t,
                x_hat_t=x_hat_t,
                anchors_t=anchors_t,
                noise_idx=noise_idx,
                iter_indices=batch_iter_idx,
                num_particles=num_particles,
                point_start=point_start,
                point_end=point_end,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        for batch_iter_idx in _iter_batches(val_idx, iter_batch_size):
            val_loss = _sequence_loss_for_indices(
                model=model,
                ranging_t=ranging_t,
                true_state_t=true_state_t,
                x_hat_t=x_hat_t,
                anchors_t=anchors_t,
                noise_idx=noise_idx,
                iter_indices=batch_iter_idx,
                num_particles=num_particles,
                point_start=point_start,
                point_end=point_end,
                device=device,
            )
            val_losses.append(float(val_loss.item()))

        train_mean = float(np.mean(train_losses)) if train_losses else float("inf")
        val_mean = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_mean < best_val:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"noise={noise_idx} var={noise_variance:g} "
                f"epoch={epoch:03d}/{epochs} train={train_mean:.6f} val={val_mean:.6f} lr={current_lr:.3e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    est_test = _rollout_estimates(
        model=model,
        ranging_t=ranging_t,
        true_state_t=true_state_t,
        x_hat_t=x_hat_t,
        anchors_t=anchors_t,
        noise_idx=noise_idx,
        iter_indices=test_idx,
        num_particles=num_particles,
        point_start=point_start,
        point_end=point_end,
        device=device,
    )

    true_test = true_state_t[0:2, :point_end, test_idx, noise_idx].cpu().numpy()
    rmse, ape = _compute_metrics(est_test, true_test, start_point=3)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"dpf_noise{noise_idx}_var{noise_variance:g}.pt"
    torch.save(
        {
            "noise_idx": noise_idx,
            "noise_variance": noise_variance,
            "state_dict": model.state_dict(),
            "best_val_loss": best_val,
            "rmse": rmse,
            "ape": ape,
            "num_particles": num_particles,
            "epochs": epochs,
            "scheduler": scheduler_name,
            "point_start": point_start,
            "point_end": point_end,
        },
        ckpt_path,
    )

    return {
        "noise_idx": noise_idx,
        "noise_variance": float(noise_variance),
        "best_val_loss": float(best_val),
        "test_rmse": float(rmse),
        "test_ape": float(ape),
        "checkpoint": str(ckpt_path),
    }


def parse_noise_indices(arg: str, num_noise: int) -> list[int]:
    if arg.lower() in {"all", "*"}:
        return list(range(num_noise))
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    idx = sorted({int(p) for p in parts})
    for x in idx:
        if x < 0 or x >= num_noise:
            raise ValueError(f"Invalid noise index {x}; valid range is [0, {num_noise - 1}]")
    return idx


def main():
    p = argparse.ArgumentParser(description="Train DPF models for multiple noise levels and export performance table")
    p.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--motion-model", type=str, default="cv", choices=["cv", "imm"])
    p.add_argument("--noise-indices", type=str, default="0,1,2,3,4")
    p.add_argument("--num-particles", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--iter-batch-size", type=int, default=128)
    p.add_argument("--max-iterations", type=int, default=0, help="0 means full iteration range")
    p.add_argument("--max-points", type=int, default=0, help="0 means full sequence points")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = initialize_config(num_particles=args.num_particles)

    h5_file = Path(args.data_path) / ("simulation_data_imm.h5" if args.motion_model.lower() == "imm" else "simulation_data.h5")
    data = load_simulation_data(str(h5_file))

    num_noise = data.ranging.shape[-1]
    noise_indices = parse_noise_indices(args.noise_indices, num_noise)

    max_iterations = data.ranging.shape[2] if args.max_iterations <= 0 else min(args.max_iterations, data.ranging.shape[2])
    point_start = 2
    point_end = data.ranging.shape[1] if args.max_points <= 0 else min(args.max_points, data.ranging.shape[1])

    if point_end <= point_start:
        raise ValueError("max_points is too small; need at least 3 points")

    print(f"device={device}")
    print(f"h5_file={h5_file}")
    print(f"noise_indices={noise_indices}")
    print(f"sequence_points=[{point_start}, {point_end - 1}] (full={args.max_points <= 0})")
    print(f"iterations=0..{max_iterations - 1}")
    print(f"scheduler={args.scheduler}")

    ranging_t = _to_tensor(data.ranging[:, :point_end, :max_iterations, :], device)
    true_state_t = _to_tensor(data.true_state[:, :point_end, :max_iterations, :], device)
    x_hat_t = _to_tensor(data.x_hat_lls[:, :point_end, :max_iterations, :], device)
    anchors_t = _to_tensor(cfg.anchor.T, device)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for noise_idx in noise_indices:
        row = train_single_noise(
            ranging_t=ranging_t,
            true_state_t=true_state_t,
            x_hat_t=x_hat_t,
            anchors_t=anchors_t,
            noise_idx=noise_idx,
            noise_variance=cfg.noise_variance[noise_idx],
            num_particles=args.num_particles,
            epochs=args.epochs,
            lr=args.lr,
            scheduler_name=args.scheduler,
            iter_batch_size=args.iter_batch_size,
            point_start=point_start,
            point_end=point_end,
            device=device,
            seed=args.seed,
            checkpoint_dir=ckpt_dir,
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.sort_values("noise_idx").reset_index(drop=True)

    csv_path = out_dir / "dpf_noise_performance.csv"
    table.to_csv(csv_path, index=False)

    print("\nNoise-wise performance table")
    print(table[["noise_idx", "noise_variance", "best_val_loss", "test_rmse", "test_ape"]].to_string(index=False))
    print(f"\nSaved performance csv: {csv_path}")


if __name__ == "__main__":
    main()
