from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from batched_particlefilter import run_batched_nonlinear_pf
from config import initialize_config
from data_loader import load_simulation_data
from runner import run_main_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python benchmark runner compatible with MATLAB main_sub pipeline")
    parser.add_argument("--motion-model", choices=["cv", "imm"], default="cv")
    parser.add_argument("--path-data", default="../data")
    parser.add_argument("--path-result", default="../result")
    parser.add_argument("--iterations", type=int, default=1000)

    parser.add_argument("--batched-nonlinear-pf", action="store_true")
    parser.add_argument("--noise-idx", type=int, default=0)
    parser.add_argument("--num-particles", type=int, default=150)
    parser.add_argument("--mc-runs", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dnn-weights", default="")
    parser.add_argument("--dnn-input-size", type=int, default=4)
    parser.add_argument("--dnn-hidden-size", type=int, default=32)
    parser.add_argument("--dnn-output-size", type=int, default=4)
    return parser.parse_args()


def _build_dnn_refiner(args: argparse.Namespace, device: str):
    if not args.dnn_weights:
        return None

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from DNN import DNN  # pylint: disable=import-error

    model = DNN(args.dnn_input_size, args.dnn_hidden_size, args.dnn_output_size).to(device)
    state = torch.load(args.dnn_weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    def refine(z_now: torch.Tensor) -> torch.Tensor:
        # z_now: [mc_runs, anchors]
        with torch.no_grad():
            return model(z_now)

    return refine


def _run_batched_mode(args: argparse.Namespace) -> None:
    cfg = initialize_config(args.num_particles)
    cfg.motion_model = args.motion_model
    cfg.path_data = args.path_data
    cfg.path_result = args.path_result
    cfg.iterations = args.iterations

    motion_file = "simulation_data_imm.h5" if cfg.motion_model.lower() == "imm" else "simulation_data.h5"
    h5_file = f"{cfg.path_data}/{motion_file}"
    data = load_simulation_data(h5_file)

    device = args.device
    refiner = _build_dnn_refiner(args, device)
    output = run_batched_nonlinear_pf(
        data,
        cfg,
        noise_idx=args.noise_idx,
        mc_runs=args.mc_runs,
        device=device,
        seed=args.seed,
        measurement_refiner=refiner,
    )

    est = output.estimated_pos
    true_pos = torch.as_tensor(
        data.true_state[0:2, 0 : est.shape[2], 0 : est.shape[3], args.noise_idx],
        dtype=est.dtype,
        device=est.device,
    )
    err = est - true_pos.unsqueeze(0)
    rmse_per_run = torch.sqrt(torch.mean(torch.sum(err * err, dim=1), dim=(1, 2)))

    print("Batched NonlinearPF done")
    print("estimated_pos shape:", tuple(est.shape))
    print("rmse mean:", float(torch.mean(rmse_per_run).cpu().numpy()))
    print("rmse std:", float(torch.std(rmse_per_run).cpu().numpy()))


def main() -> None:
    args = parse_args()

    if args.batched_nonlinear_pf:
        _run_batched_mode(args)
        return

    cfg = initialize_config()
    cfg.motion_model = args.motion_model
    cfg.path_data = args.path_data
    cfg.path_result = args.path_result
    cfg.iterations = args.iterations

    output = run_main_like(cfg)
    print("Saved aggregated file:", output)


if __name__ == "__main__":
    main()
