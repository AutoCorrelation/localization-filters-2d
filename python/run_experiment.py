from __future__ import annotations

import argparse

from config import initialize_config
from runner import run_main_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python benchmark runner compatible with MATLAB main_sub pipeline")
    parser.add_argument("--motion-model", choices=["cv", "imm"], default="cv")
    parser.add_argument("--path-data", default="../data")
    parser.add_argument("--path-result", default="../result")
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = initialize_config()
    cfg.motion_model = args.motion_model
    cfg.path_data = args.path_data
    cfg.path_result = args.path_result
    cfg.iterations = args.iterations

    output = run_main_like(cfg)
    print("Saved aggregated file:", output)


if __name__ == "__main__":
    main()
