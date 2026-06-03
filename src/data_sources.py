import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.kuramoto_data import KuramotoConfig, generate_kuramoto_data


def parse_coupling_strength(value):
    """Accept a scalar or a 'low,high' range string for Kuramoto coupling strengths."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if text == "":
        raise ValueError("coupling strength must not be empty")

    if "," not in text:
        return float(text)

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("coupling strength range must look like 'low,high'")
    return tuple(float(part) for part in parts)


def read_csv_if_exists(path):
    if path and os.path.exists(path):
        return pd.read_csv(path, header=None).values.astype(np.float32)
    return None


def load_array_from_path(path):
    if not path:
        return None

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return read_csv_if_exists(path)
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        archive = np.load(path)
        if "data" in archive:
            return archive["data"].astype(np.float32)
        keys = list(archive.keys())
        if not keys:
            raise ValueError(f"No arrays found in npz file: {path}")
        return archive[keys[0]].astype(np.float32)
    raise ValueError(f"Unsupported data file type: {path}")


def load_npz_archive(path):
    if not path:
        raise ValueError("path must not be empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"npz file not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.keys())
        if not keys:
            raise ValueError(f"No arrays found in npz file: {path}")
        return {key: np.asarray(archive[key]) for key in keys}


def visualize_generated_npz(
    npz_path=None,
    output_path=None,
    max_time_steps=300,
    max_features=16,
    max_lines=6,
):
    npz_path = npz_path or _default_generated_data_path(None)
    archive = load_npz_archive(npz_path)

    if "data" not in archive:
        raise ValueError(f"'data' array is required in npz file: {npz_path}")

    data = np.asarray(archive["data"], dtype=np.float32)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 'data' to have 2 or 3 dimensions, but got shape {data.shape}"
        )

    _, time_steps, feature_dim = data.shape
    show_time_steps = min(int(max_time_steps), time_steps)
    show_features = min(int(max_features), feature_dim)
    show_lines = min(int(max_lines), feature_dim)

    if output_path is None:
        output_path = Path(npz_path).with_name("generated_data_overview.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    traces_axis = figure.add_subplot(grid[0, :])
    matrix_axis = figure.add_subplot(grid[1, 0])
    omega_axis = figure.add_subplot(grid[1, 1])

    time_axis = np.arange(show_time_steps)
    for feature_index in range(show_lines):
        traces_axis.plot(
            time_axis,
            data[0, :show_time_steps, feature_index],
            linewidth=1.4,
            label=f"feature {feature_index}",
        )
    traces_axis.set_title(f"Series 0 Traces ({show_lines} features)")
    traces_axis.set_xlabel("Time Step")
    traces_axis.set_ylabel("Value")
    traces_axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    traces_axis.legend(loc="upper right", fontsize=8)

    if "obj_matrix" in archive:
        obj_matrix = np.asarray(archive["obj_matrix"], dtype=np.float32)
        obj_im = matrix_axis.imshow(obj_matrix, aspect="auto", origin="lower", cmap="viridis")
        matrix_axis.set_title("Object Coupling Matrix")
        matrix_axis.set_xlabel("Node")
        matrix_axis.set_ylabel("Node")
        figure.colorbar(obj_im, ax=matrix_axis, fraction=0.046, pad=0.04)
    else:
        matrix_axis.hist(data.reshape(-1), bins=60, color="#2563EB", alpha=0.85)
        matrix_axis.set_title("Data Value Distribution")
        matrix_axis.set_xlabel("Value")
        matrix_axis.set_ylabel("Count")

    if "omegas" in archive:
        omegas = np.asarray(archive["omegas"], dtype=np.float32)
        group_ids = None
        if "group_matrix" in archive:
            group_matrix = np.asarray(archive["group_matrix"], dtype=np.float32)
            if group_matrix.ndim == 2 and group_matrix.shape[0] == omegas.shape[0]:
                group_ids = np.argmax(group_matrix, axis=1)

        indices = np.arange(len(omegas))
        if group_ids is None:
            omega_axis.bar(indices, omegas, color="#D97706", alpha=0.9)
        else:
            cmap = plt.cm.get_cmap("tab10", int(group_ids.max()) + 1)
            colors = [cmap(int(group_id)) for group_id in group_ids]
            omega_axis.bar(indices, omegas, color=colors, alpha=0.9)
        omega_axis.set_title("Natural Frequencies (omegas)")
        omega_axis.set_xlabel("Node")
        omega_axis.set_ylabel("Omega")
    else:
        flat_means = data.mean(axis=(0, 1))
        omega_axis.bar(np.arange(len(flat_means[:show_features])), flat_means[:show_features], color="#D97706", alpha=0.9)
        omega_axis.set_title("Mean Value Per Feature")
        omega_axis.set_xlabel("Feature")
        omega_axis.set_ylabel("Mean")

    summary = ", ".join(
        f"{key}: shape={value.shape}, dtype={value.dtype}"
        for key, value in archive.items()
    )
    figure.suptitle(
        f"generated_data.npz overview\n{summary}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {
        "npz_path": str(Path(npz_path).resolve()),
        "output_path": str(output_path.resolve()),
        "keys": tuple(archive.keys()),
        "data_shape": tuple(data.shape),
    }


def _default_existing_data_path(args):
    return os.path.join(
        ".",
        "loc_data_kuramoto",
        "generated_data.npz",
    )


def _default_generated_data_path(args):
    return os.path.join(
        ".",
        "loc_data_kuramoto",
        "generated_data.npz",
    )


def load_origin_data(args):
    data_source = str(getattr(args, "data_source", "kuramoto")).strip().lower()
    if data_source == "lorzen":
        data_source = "existing"
    if data_source == "real_fmri":
        data_source = "existing"
    
    if data_source == "existing":
        data_path = args.data_path or _default_existing_data_path(args)
        origin_data = load_array_from_path(data_path)
        if origin_data is None:
            raise FileNotFoundError(f"No usable data found at: {data_path}")
        metadata = {
            "data_source": "existing",
            "data_path": data_path,
            "data_shape": tuple(origin_data.shape),
        }
        return origin_data, metadata
    
    if data_source == "kuramoto":
        target_path = args.generated_data_path or _default_generated_data_path(args)
        train_stage = getattr(args, "train_stage", 1)
        
        if train_stage >= 2:
            origin_data = load_array_from_path(target_path)
            if origin_data is None:
                raise FileNotFoundError(
                    f"Stage {train_stage} expects existing Kuramoto data at: {target_path}"
                )
            metadata = {
                "data_source": "kuramoto",
                "generated_data_path": target_path,
                "data_shape": tuple(origin_data.shape),
            }
            return origin_data, metadata

        config = KuramotoConfig(
            sz=args.kuramoto_sz,
            groups=args.kuramoto_groups,
            num_series=args.kuramoto_num_series,
            time_steps=args.kuramoto_time_steps,
            dt=args.kuramoto_dt,
            sample_interval=args.kuramoto_sample_interval,
            intra_coupling_strength=parse_coupling_strength(args.kuramoto_intra_coupling_strength),
            inter_coupling_strength=parse_coupling_strength(args.kuramoto_inter_coupling_strength),
            noise_level=args.kuramoto_noise_level,
            include_order_parameters=args.kuramoto_include_order_parameters,
        )
        origin_data, metadata = generate_kuramoto_data(config)
        metadata["data_source"] = "kuramoto"
        
        if target_path:
            target_dir = os.path.dirname(target_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            if target_path.lower().endswith(".npz"):
                np.savez(
                    target_path,
                    data=origin_data,
                    obj_matrix=metadata["obj_matrix"],
                    group_matrix=metadata["group_matrix"],
                    omegas=metadata["omegas"],
                    theta_data=metadata["theta_data"],
                    series_lengths=metadata["series_lengths"],
                )
            else:
                np.save(target_path, origin_data)
            metadata["generated_data_path"] = target_path

        metadata["data_shape"] = tuple(origin_data.shape)
        return origin_data, metadata

    raise ValueError(f"Unsupported data_source: {data_source}")


def add_data_source_args(parser):
    parser.add_argument(
        "--data_source",
        type=str,
        default="kuramoto",
        choices=["existing", "kuramoto", "lorzen", "real_fmri"],
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--generated_data_path", type=str, default=_default_generated_data_path(None))
    
    parser.add_argument("--kuramoto_sz", type=int, default=32)
    parser.add_argument("--kuramoto_groups", type=int, default=2)
    parser.add_argument("--kuramoto_num_series", type=int, default=1)
    parser.add_argument("--kuramoto_time_steps", type=int, default=100000)
    parser.add_argument("--kuramoto_dt", type=float, default=0.05)
    parser.add_argument("--kuramoto_sample_interval", type=int, default=1)
    parser.add_argument(
        "--kuramoto_intra_coupling_strength",
        type=str,
        default="4.0,5.0",
        help="Scalar coupling or range string like '4.0,5.0' for intra-group coupling.",
    )
    parser.add_argument(
        "--kuramoto_inter_coupling_strength",
        type=str,
        default="-1.0,-0.3",
        help="Scalar coupling or range string like '-0.5,0.0' for inter-group coupling.",
    )
    parser.add_argument("--kuramoto_noise_level", type=float, default=0.1)
    parser.add_argument("--kuramoto_include_order_parameters", action="store_true")
    return parser
