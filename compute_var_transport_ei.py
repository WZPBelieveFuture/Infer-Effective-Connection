from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.autograd.functional import jacobian


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_jacobian_matrix import (  # noqa: E402
    build_state_model,
    load_origin_data,
    load_stage2_artifact,
    resolve_data_path,
    resolve_device,
)
from generate_causal_comparison_ppt import (  # noqa: E402
    build_minimal_pptx,
    compute_threshold_metrics,
    create_lorzen_comparison_figure,
    format_metric_text,
    load_square_matrix_npz,
)
from src.models_macro import _build_windows_from_series  # noqa: E402


DEFAULT_RUN_NAME = "stage2_var_macro"
DEFAULT_DATA_PATH = Path("loc_data_var") / "generated_data.npz"
DEFAULT_RESULT_DIR = Path("result") / "var"


def group_slices(group_sizes: np.ndarray) -> list[slice]:
    starts = np.concatenate(([0], np.cumsum(group_sizes[:-1])))
    return [slice(int(start), int(start + size)) for start, size in zip(starts, group_sizes)]


def load_var_groups(data_path: Path) -> np.ndarray:
    with np.load(data_path, allow_pickle=False) as archive:
        if "group" not in archive:
            raise KeyError(f"'group' was not found in {data_path}")
        group = np.asarray(archive["group"], dtype=np.int64).reshape(-1)
    if group.size == 0 or np.any(group < 1):
        raise ValueError(f"Invalid VAR group sizes in {data_path}: {group}")
    return group


def sample_source_states(origin_data: np.ndarray, time_delay: int, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    x_all, _ = _build_windows_from_series(origin_data, int(time_delay))
    source_states = x_all[:, 0] if x_all.ndim == 3 else x_all
    if source_states.shape[0] < int(sample_count):
        raise ValueError(
            f"Requested {int(sample_count)} samples, but only {source_states.shape[0]} windowed samples are available."
        )
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.choice(source_states.shape[0], size=int(sample_count), replace=False)
    return source_states[sampled_indices].astype(np.float32, copy=False), sampled_indices.astype(np.int64), int(source_states.shape[0])


def estimate_transport_noise(model, source_states: np.ndarray, target_states: np.ndarray, scale_id: int, device: torch.device) -> np.ndarray:
    source_tensor = torch.tensor(source_states, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        target_latent = model.encoding1(target_tensor, int(scale_id))[int(scale_id)]
        source_latent = model.encoding1(source_tensor, int(scale_id))[int(scale_id)]
        predicted_latent = model._apply_dynamics(source_latent, int(scale_id), inverse=False)
        sigma = torch.mean((predicted_latent - target_latent).pow(2), dim=0) + 1e-6
    return np.asarray(sigma.detach().cpu().tolist(), dtype=np.float32)


def compute_transport_jacobians(model, source_states: np.ndarray, scale_id: int, device: torch.device) -> np.ndarray:
    jacobians = []
    model.eval()
    for source_state in torch.as_tensor(source_states, dtype=torch.float32, device=device):
        source_state = source_state.detach().clone().requires_grad_(True)

        def transport_map(x: torch.Tensor) -> torch.Tensor:
            encoded = model.encoding1(x.unsqueeze(0), int(scale_id))[int(scale_id)]
            return model._apply_dynamics(encoded, int(scale_id), inverse=False).squeeze(0)

        jac = jacobian(transport_map, source_state)
        jacobians.append(jac.detach().cpu())
    if not jacobians:
        raise ValueError("No transport-map Jacobians were computed.")
    return np.asarray(torch.stack(jacobians, dim=0).tolist(), dtype=np.float32)


def aggregate_group_transport_ei(
    transport_jacobians: np.ndarray,
    group_sizes: np.ndarray,
    sigma_diag: np.ndarray,
    L: float,
    eps: float,
    noise_mode: str = "global",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_blocks = group_slices(group_sizes)
    target_dim = int(transport_jacobians.shape[1])
    if target_dim != len(group_blocks):
        raise ValueError(
            f"Transport output dimension {target_dim} does not match VAR group count {len(group_blocks)}."
        )

    group_energy = np.zeros((transport_jacobians.shape[0], target_dim, len(group_blocks)), dtype=np.float32)
    for group_index, block in enumerate(group_blocks):
        group_energy[:, :, group_index] = np.sum(np.square(transport_jacobians[:, :, block]), axis=2)

    group_gain = np.sqrt(np.maximum(group_energy, eps))
    expected_log_gain = np.mean(np.log(group_gain), axis=0)
    mean_group_gain = np.mean(group_gain, axis=0)
    mean_group_energy = np.mean(group_energy, axis=0)

    variance_scale = float(L**2) / 12.0
    signal_variance = variance_scale * mean_group_energy
    noise_mode = str(noise_mode or "global").strip().lower()
    if noise_mode == "global":
        effective_variance = np.full_like(signal_variance, max(float(np.mean(sigma_diag)), eps))
    elif noise_mode == "target":
        effective_variance = np.maximum(sigma_diag.reshape(-1, 1), eps)
    elif noise_mode == "target_excluded":
        excluded_energy = np.sum(mean_group_energy, axis=1, keepdims=True) - mean_group_energy
        effective_variance = np.maximum(sigma_diag.reshape(-1, 1) + variance_scale * excluded_energy, eps)
    else:
        raise ValueError(f"Unsupported noise_mode: {noise_mode}")
    variance_term = -0.5 * np.log(effective_variance)
    transport_ei = 0.5 * np.log1p(signal_variance / effective_variance)
    return (
        transport_ei.astype(np.float32, copy=False),
        mean_group_gain.astype(np.float32, copy=False),
        expected_log_gain.astype(np.float32, copy=False),
        variance_term.astype(np.float32, copy=False),
    )


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    labels = [f"M{i + 1}" for i in range(matrix.shape[0])]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


def write_outputs(args: argparse.Namespace) -> None:
    project_root = PROJECT_ROOT
    device = resolve_device(args.device)
    data_path = resolve_data_path(project_root, args.data_path)
    result_dir = resolve_data_path(project_root, args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_stage2_artifact(project_root, args.run_name, args.model_scale)
    origin_data = load_origin_data(project_root, data_path)
    group = load_var_groups(data_path)
    if int(group.sum()) != int(origin_data.shape[-1]):
        raise ValueError(f"group sum {int(group.sum())} does not match data dimension {origin_data.shape[-1]}")

    model = build_state_model(origin_data.shape[-1], artifact, device=device)
    source_states, sampled_indices, total_window_count = sample_source_states(
        origin_data,
        time_delay=int(artifact["time_delay"]),
        sample_count=int(args.sample_count),
        seed=int(args.seed),
    )
    target_states = origin_data[sampled_indices + 1]
    sigma_diag = estimate_transport_noise(
        model,
        source_states=source_states,
        target_states=target_states,
        scale_id=int(artifact["logical_scale_id"]),
        device=device,
    )
    transport_jacobians = compute_transport_jacobians(
        model,
        source_states=source_states,
        scale_id=int(artifact["logical_scale_id"]),
        device=device,
    )
    transport_ei, mean_group_gain, expected_log_gain, variance_term = aggregate_group_transport_ei(
        transport_jacobians,
        group_sizes=group,
        sigma_diag=sigma_diag,
        L=float(args.L),
        eps=float(args.eps),
        noise_mode=args.noise_mode,
    )
    ground_truth = load_square_matrix_npz(data_path, "causal_matrix", "var_ground_truth")
    metrics = compute_threshold_metrics(transport_ei, ground_truth)

    scale_number = int(args.model_scale)
    ei_csv = result_dir / f"var_transport_map_ei_matrix_scale{scale_number}.csv"
    gain_csv = result_dir / f"var_transport_map_mean_group_gain_scale{scale_number}.csv"
    log_gain_csv = result_dir / f"var_transport_map_expected_log_gain_scale{scale_number}.csv"
    variance_csv = result_dir / f"var_transport_map_variance_term_scale{scale_number}.csv"
    summary_csv = result_dir / f"var_transport_map_ei_summary_scale{scale_number}.csv"
    output_png = result_dir / f"var_transport_map_ei_comparison_scale{scale_number}.png"
    output_pptx = result_dir / f"var_transport_map_ei_comparison_scale{scale_number}.pptx"

    save_matrix(transport_ei, ei_csv)
    save_matrix(mean_group_gain, gain_csv)
    save_matrix(expected_log_gain, log_gain_csv)
    save_matrix(variance_term, variance_csv)
    pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "model_scale": int(args.model_scale),
                "logical_scale_id": int(artifact["logical_scale_id"]),
                "sample_count": int(args.sample_count),
                "seed": int(args.seed),
                "total_window_count": int(total_window_count),
                "L": float(args.L),
                "noise_mode": str(args.noise_mode),
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "device": str(device),
                "data_path": str(data_path.resolve()),
                "model_path": str(Path(artifact["model_path"]).resolve()),
                "ei_matrix_path": str(ei_csv.resolve()),
            }
        ]
    ).to_csv(summary_csv, index=False)

    create_lorzen_comparison_figure(
        jacobian_strength=mean_group_gain,
        ei_causal_graph=transport_ei,
        ground_truth=ground_truth,
        metrics_by_name={"jacobian": compute_threshold_metrics(mean_group_gain, ground_truth), "ei": metrics},
        output_png=output_png,
        scale_number=scale_number,
        comparison_title="VAR Transport-Map EI Comparison",
        comparison_subtitle=(
            "Computing EI from the end-to-end transport map encoder+dynamics, grouped by VAR observed variables. "
            "Edge direction: source -> target."
        ),
        ground_truth_title="Groundtruth VAR",
    )
    build_minimal_pptx(output_png.read_bytes(), output_png.name, output_pptx, "VAR Transport-Map EI Comparison")

    print(f"Saved transport-map EI matrix to {ei_csv}")
    print(f"Saved transport-map EI summary to {summary_csv}")
    print(f"Saved comparison image to {output_png}")
    print(f"Saved PPT to {output_pptx}")
    print(f"Transport-map EI metrics: {format_metric_text(metrics)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a VAR EI matrix from the learned encoder+dynamics transport map.")
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model_scale", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--sample_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--noise_mode", type=str, default="global", choices=["global", "target", "target_excluded"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    write_outputs(parse_args())
