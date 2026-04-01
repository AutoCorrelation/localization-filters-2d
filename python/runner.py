from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from data_loader import SimulationData, load_simulation_data
from filters import create_filter
from metrics import evaluate_filter


@dataclass
class RunOutput:
    estimated_pos: np.ndarray
    rmse: np.ndarray
    ape: np.ndarray


FILTER_NAMES = [
    "Baseline",
    "LinearKalmanFilter_DecayQ",
    "NonlinearParticleFilter",
    "RegularizedParticleFilter",
    "AdaptiveParticleFilter",
    "RDiagPriorEditAdaptiveParticleFilter",
    "RougheningPriorEditingParticleFilter",
    "RBPF",
]


def run_filter(filter_class: str, data: SimulationData, config: Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    num_noise = len(config.noise_variance)
    num_points = data.x_hat_lls.shape[1]
    num_iterations = config.iterations

    estimated_pos = np.zeros((2, num_points, num_iterations, num_noise), dtype=float)
    rmse = np.zeros(num_noise, dtype=float)
    ape = np.zeros(num_noise, dtype=float)

    for noise_idx in range(num_noise):
        rng = np.random.RandomState(42)
        filter_obj = create_filter(filter_class, data, config, noise_idx, rng)

        est_noise = np.zeros((2, num_points, num_iterations), dtype=float)
        true_pos_noise = data.true_state[0:2, 0:num_points, 0:num_iterations, noise_idx]

        for iter_idx in range(num_iterations):
            state = filter_obj.initialize_state(num_points)
            state, p1, p2 = filter_obj.initialize_first_two(state, iter_idx)
            est_noise[:, 0, iter_idx] = p1
            est_noise[:, 1, iter_idx] = p2

            for point_idx in range(2, num_points):
                state, est = filter_obj.step(state, iter_idx, point_idx)
                est_noise[:, point_idx, iter_idx] = est

        estimated_pos[:, :, :, noise_idx] = est_noise
        rmse[noise_idx], ape[noise_idx] = evaluate_filter(est_noise, start_point=3, true_pos=true_pos_noise)

    metric = {"RMSE": rmse, "APE": ape}
    return estimated_pos, metric


def _class_names_to_legend(filter_names: list[str]) -> list[str]:
    mapping = {
        "LinearKalmanFilter_DecayQ": "LinearKF_DecayQ",
        "NonlinearParticleFilter": "NonLinearPF",
        "RBPF": "RBPF",
        "RegularizedParticleFilter": "RegularizedPF",
        "EKFParticleFilter": "EKF-PF",
        "AdaptiveParticleFilter": "AdaptivePF(AdaBelief)",
        "RDiagPriorEditAdaptiveParticleFilter": "RDiagPriorEditAdaptivePF",
        "RougheningPriorEditingParticleFilter": "RougheningPriorEditingPF",
    }
    return [mapping.get(name, name) for name in filter_names]


def plot_metric_comparison(
    noise_variance: np.ndarray,
    rmse_matrix: np.ndarray,
    filter_names: list[str],
    runtime_table: pd.DataFrame,
    particle_count: int,
    result_dir: str,
    motion_model: str,
):
    labels = _class_names_to_legend(filter_names)
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    for idx in range(rmse_matrix.shape[1]):
        ax1.semilogx(noise_variance, rmse_matrix[:, idx], marker="o", linewidth=1.5)
    ax1.legend(labels, loc="upper left")
    ax1.set_xlabel("Noise Variance")
    ax1.set_ylabel("RMSE")
    ax1.set_title(f"RMSE Comparison by Noise Level (N={particle_count})")
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 1, 2)
    runtime_map = dict(zip(runtime_table["FilterName"], runtime_table["RuntimeSec"]))
    runtime_values = [runtime_map.get(name, np.nan) for name in filter_names]
    ax2.bar(np.arange(len(labels)), runtime_values)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("Runtime (s)")
    ax2.set_title(f"Runtime Comparison (N={particle_count})")
    ax2.grid(True)

    motion_prefix = f"{motion_model}_"
    fig_path = f"{result_dir}/{motion_prefix}N{int(round(particle_count))}.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    return fig, fig_path


def _insert_runtime_under_variance_100(rmse_table: pd.DataFrame, runtime_table: pd.DataFrame) -> pd.DataFrame:
    out = rmse_table.copy()
    out.insert(0, "RowType", "RMSE")

    runtime_row = {col: np.nan for col in out.columns}
    runtime_row["RowType"] = "RuntimeSec"
    runtime_row["NoiseVariance"] = 100.0

    for col in out.columns:
        if col in ("RowType", "NoiseVariance"):
            continue
        row = runtime_table[runtime_table["FilterName"] == col]
        if not row.empty:
            runtime_row[col] = float(row["RuntimeSec"].iloc[0])

    if (out["NoiseVariance"] == 100).any():
        insert_after = int(np.where(out["NoiseVariance"].values == 100)[0][0])
        top = out.iloc[: insert_after + 1]
        bottom = out.iloc[insert_after + 1 :]
        return pd.concat([top, pd.DataFrame([runtime_row]), bottom], ignore_index=True)
    return pd.concat([out, pd.DataFrame([runtime_row])], ignore_index=True)


def save_benchmark_results(
    result_dir: str,
    particle_count: int,
    rmse_table: pd.DataFrame,
    runtime_table: pd.DataFrame,
    motion_model: str,
):
    particle_count_tag = f"N{int(round(particle_count))}"
    motion_prefix = f"{motion_model}_"
    result_base_name = f"benchmark_{motion_prefix}{particle_count_tag}"

    rmse_with_runtime = _insert_runtime_under_variance_100(rmse_table, runtime_table)
    rmse_csv_path = f"{result_dir}/{result_base_name}_RMSE.csv"
    rmse_with_runtime.to_csv(rmse_csv_path, index=False)

    return {
        "rmseCsvPath": rmse_csv_path,
        "rmseTableWithRuntime": rmse_with_runtime,
    }


def run_main_like(config: Config):
    particle_counts = [10, 50, 100, 200, 500, 1000]
    filter_names = FILTER_NAMES.copy()

    motion_file = "simulation_data_imm.h5" if config.motion_model.lower() == "imm" else "simulation_data.h5"
    h5_file = f"{config.path_data}/{motion_file}"
    data = load_simulation_data(h5_file)

    all_rmse_table = []
    for n_particles in particle_counts:
        cfg = Config(**config.__dict__)
        cfg.num_particles = int(n_particles)

        np.random.seed(42)

        filter_times = np.zeros(len(filter_names), dtype=float)
        filter_metrics = []

        for f_idx, filter_name in enumerate(filter_names):
            t0 = time.perf_counter()
            _, metric_out = run_filter(filter_name, data, cfg)
            filter_times[f_idx] = time.perf_counter() - t0
            filter_metrics.append(metric_out)

        num_noise = len(cfg.noise_variance)
        rmse_matrix = np.zeros((num_noise, len(filter_names)), dtype=float)
        for f_idx in range(len(filter_names)):
            rmse_matrix[:, f_idx] = filter_metrics[f_idx]["RMSE"].ravel()

        rmse_table = pd.DataFrame(rmse_matrix, columns=filter_names)
        rmse_table.insert(0, "NoiseVariance", np.array(cfg.noise_variance, dtype=float))

        runtime_table = pd.DataFrame({"FilterName": filter_names, "RuntimeSec": filter_times})
        runtime_table = runtime_table.sort_values("RuntimeSec", ascending=True).reset_index(drop=True)

        plot_metric_comparison(
            np.array(cfg.noise_variance, dtype=float),
            rmse_matrix,
            filter_names,
            runtime_table,
            cfg.num_particles,
            cfg.path_result,
            cfg.motion_model,
        )

        saved_paths = save_benchmark_results(
            cfg.path_result,
            cfg.num_particles,
            rmse_table,
            runtime_table,
            cfg.motion_model,
        )

        rmse_table_out = saved_paths["rmseTableWithRuntime"].copy()
        rmse_table_out.insert(0, "ParticleCount", int(round(cfg.num_particles)))
        all_rmse_table.append(rmse_table_out)

    all_rmse = pd.concat(all_rmse_table, ignore_index=True)
    motion_prefix = f"{config.motion_model}_"
    all_rmse_csv_path = f"{config.path_result}/benchmark_{motion_prefix}batch_RMSE_AllN.csv"
    all_rmse.to_csv(all_rmse_csv_path, index=False)
    return all_rmse_csv_path
