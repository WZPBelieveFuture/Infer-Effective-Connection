from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.autograd.functional import jacobian


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import load_array_from_path
from src.models_macro import Parellel_Renorm_Dynamic, _build_windows_from_series


DEFAULT_RUN_NAME = "stage2_real_fmri_macro"
DEFAULT_DATA_PATH = Path("loc_data_real_fmri") / "generated_data.npz"
DEFAULT_SAMPLE_COUNT = 2000
DEFAULT_SEED = 0


def parse_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def find_project_root(start: Path | None = None) -> Path:
    start = (start or PROJECT_ROOT).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src" / "models_macro.py").exists() and (candidate / "loc_model_stage2").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root containing src/models_macro.py and loc_model_stage2.")


def resolve_data_path(project_root: Path, data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_stage2_artifact(project_root: Path, run_name: str, model_scale: int) -> Dict[str, Any]:
    if int(model_scale) < 1:
        raise ValueError("model_scale must be >= 1.")

    model_path = project_root / "loc_model_stage2" / run_name / f"model_scale{int(model_scale)}.pkl"
    summary_path = project_root / "loc_result_stage2" / run_name / f"summary_scale{int(model_scale)}.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary_row = pd.read_csv(summary_path).iloc[0].to_dict()
    scale_dims = parse_int_list(summary_row.get("scale_dims", ""))
    reduce_dims = parse_int_list(summary_row.get("reduce_dims", ""))
    group = parse_int_list(summary_row.get("group", ""))
    logical_scale_id = int(summary_row.get("scale_id", max(int(model_scale) - 1, 0)))
    scale_dim = None
    if scale_dims and 0 <= logical_scale_id < len(scale_dims):
        scale_dim = int(scale_dims[logical_scale_id])

    return {
        "model_scale": int(model_scale),
        "logical_scale_id": logical_scale_id,
        "scale_dim": scale_dim,
        "scale_dims": scale_dims,
        "reduce_dims": reduce_dims,
        "group": group,
        "hidden_units1": int(summary_row.get("hidden_units1", 100)),
        "hidden_units2": int(summary_row.get("hidden_units2", 100)),
        "flow_num_layers": int(summary_row.get("flow_num_layers", 3)),
        "dynamics_num_layers": int(summary_row.get("dynamics_num_layers", 4)),
        "latent_size": int(summary_row.get("latent_size", 1)),
        "time_delay": int(summary_row.get("time_delay", 1)),
        "encoder_type": str(summary_row.get("encoder_type", "mlp")),
        "model_path": model_path,
        "summary_path": summary_path,
    }


def load_origin_data(project_root: Path, data_path: str | Path) -> np.ndarray:
    resolved_path = resolve_data_path(project_root, data_path)
    origin_data = load_array_from_path(str(resolved_path))
    if origin_data is None:
        raise FileNotFoundError(f"Could not load origin data from: {resolved_path}")
    return np.asarray(origin_data, dtype=np.float32)


def build_state_model(num_nodes: int, artifact: Dict[str, Any], device: torch.device) -> Parellel_Renorm_Dynamic:
    model = Parellel_Renorm_Dynamic(
        sym_size=int(num_nodes),
        latent_size=int(artifact["latent_size"]),
        effect_size=int(num_nodes),
        cut_size=2,
        hidden_units1=int(artifact["hidden_units1"]),
        hidden_units2=int(artifact["hidden_units2"]),
        normalized_state=True,
        device=device,
        is_random=False,
        flow_num_layers=int(artifact["flow_num_layers"]),
        dynamics_num_layers=int(artifact["dynamics_num_layers"]),
        decode_noise_scale=0.0,
        reduce_dims=artifact["reduce_dims"],
        group=artifact["group"] or None,
        encoder_type=artifact["encoder_type"],
    ).to(device)
    state_dict = torch.load(artifact["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_current_inputs(x_windows: np.ndarray) -> np.ndarray:
    if x_windows.ndim == 3:
        return np.asarray(x_windows[:, 0], dtype=np.float32)
    if x_windows.ndim == 2:
        return np.asarray(x_windows, dtype=np.float32)
    raise ValueError(f"x_windows must have shape [B, T, N] or [B, N], but got {x_windows.shape}")


def sample_real_source_states(x_windows: np.ndarray, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x_windows) < int(sample_count):
        raise ValueError(
            f"Requested {int(sample_count)} samples, but only {len(x_windows)} windowed real samples are available."
        )

    source_states = extract_current_inputs(x_windows)
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.choice(source_states.shape[0], size=int(sample_count), replace=False)
    return source_states[sampled_indices].astype(np.float32, copy=False), sampled_indices.astype(np.int64, copy=False)


def encode_macro_states(
    model: Parellel_Renorm_Dynamic,
    source_states: np.ndarray,
    logical_scale_id: int,
    device: torch.device,
) -> np.ndarray:
    source_tensor = torch.tensor(source_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        macro_states = model.encoding1(source_tensor, int(logical_scale_id))[int(logical_scale_id)]
    return macro_states.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_mean_abs_jacobian_matrix(
    model: Parellel_Renorm_Dynamic,
    macro_states: np.ndarray,
    logical_scale_id: int,
    device: torch.device,
) -> np.ndarray:
    latent_states = torch.as_tensor(macro_states, dtype=torch.float32, device=device)
    jacobians = []
    model.eval()
    for macro_state in latent_states:
        macro_state = macro_state.detach().clone().requires_grad_(True)
        jac = jacobian(
            lambda macro: model._apply_dynamics(macro.unsqueeze(0), int(logical_scale_id), inverse=False).squeeze(0),
            macro_state,
        )
        jacobians.append(jac.detach().abs().cpu())

    if not jacobians:
        raise ValueError("No Jacobians were computed from the sampled macro states.")
    return torch.stack(jacobians, dim=0).mean(dim=0).numpy().astype(np.float32, copy=False)


def save_outputs(
    project_root: Path,
    run_name: str,
    artifact: Dict[str, Any],
    data_path: Path,
    device: torch.device,
    sample_count: int,
    seed: int,
    total_window_count: int,
    sampled_indices: np.ndarray,
    jacobian_mean_abs: np.ndarray,
) -> tuple[Path, Path]:
    output_dir = project_root / "loc_result_stage2" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_scale = int(artifact["model_scale"])
    matrix_path = output_dir / f"jacobian_mean_abs_realdata{int(sample_count)}_scale{model_scale}.csv"
    summary_path = output_dir / f"jacobian_mean_abs_realdata{int(sample_count)}_scale{model_scale}_summary.csv"

    pd.DataFrame(jacobian_mean_abs).to_csv(matrix_path, index=False)
    pd.DataFrame(
        [
            {
                "model_scale": model_scale,
                "logical_scale_id": int(artifact["logical_scale_id"]),
                "scale_dim": int(jacobian_mean_abs.shape[0]),
                "sample_count": int(sample_count),
                "seed": int(seed),
                "time_delay": int(artifact["time_delay"]),
                "source_mode": "all_windows_random",
                "total_window_count": int(total_window_count),
                "sampled_index_min": int(sampled_indices.min()),
                "sampled_index_max": int(sampled_indices.max()),
                "device": str(device),
                "data_path": str(data_path.resolve()),
                "model_path": str(Path(artifact["model_path"]).resolve()),
                "summary_source_path": str(Path(artifact["summary_path"]).resolve()),
                "output_matrix_path": str(matrix_path.resolve()),
            }
        ]
    ).to_csv(summary_path, index=False)

    return matrix_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a mean absolute Jacobian matrix from real-fMRI samples for a stage2 macro model."
    )
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model_scale", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--sample_count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.sample_count) < 1:
        raise ValueError("sample_count must be >= 1.")

    project_root = find_project_root()
    device = resolve_device(args.device)
    artifact = load_stage2_artifact(project_root, args.run_name, args.model_scale)
    resolved_data_path = resolve_data_path(project_root, args.data_path)
    origin_data = load_origin_data(project_root, args.data_path)
    model = build_state_model(origin_data.shape[-1], artifact, device=device)

    x_all, _ = _build_windows_from_series(origin_data, int(artifact["time_delay"]))
    total_window_count = int(len(x_all))
    source_states, sampled_indices = sample_real_source_states(x_all, sample_count=int(args.sample_count), seed=int(args.seed))
    macro_states = encode_macro_states(
        model,
        source_states=source_states,
        logical_scale_id=int(artifact["logical_scale_id"]),
        device=device,
    )
    jacobian_mean_abs = compute_mean_abs_jacobian_matrix(
        model,
        macro_states=macro_states,
        logical_scale_id=int(artifact["logical_scale_id"]),
        device=device,
    )
    matrix_path, summary_path = save_outputs(
        project_root=project_root,
        run_name=args.run_name,
        artifact=artifact,
        data_path=resolved_data_path,
        device=device,
        sample_count=int(args.sample_count),
        seed=int(args.seed),
        total_window_count=total_window_count,
        sampled_indices=sampled_indices,
        jacobian_mean_abs=jacobian_mean_abs,
    )

    print(f"project_root: {project_root}")
    print(f"device: {device}")
    print(f"model_scale: {artifact['model_scale']} | logical_scale_id: {artifact['logical_scale_id']}")
    print(f"source_mode: all_windows_random | sample_count: {args.sample_count} | total_window_count: {total_window_count}")
    print(f"macro_state_dim: {jacobian_mean_abs.shape[0]} | matrix_shape: {jacobian_mean_abs.shape}")
    print(f"saved_matrix: {matrix_path}")
    print(f"saved_summary: {summary_path}")


if __name__ == "__main__":
    main()
