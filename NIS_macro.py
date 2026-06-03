import argparse
from pathlib import Path

import numpy as np
import torch

from src.models_macro import train_and_memorize
from src.data_sources import add_data_source_args, load_origin_data


def parse_int_list(text):
    if text is None or text == "":
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]

def load_group_sizes_from_npz(npz_path):
    if not npz_path:
        return None

    path = Path(npz_path)
    if not path.exists() or path.suffix.lower() != ".npz":
        return None

    with np.load(path, allow_pickle=False) as archive:
        if "group" in archive:
            values = np.asarray(archive["group"]).reshape(-1)
            return [int(value) for value in values]

        if "group_matrix" in archive:
            group_matrix = np.asarray(archive["group_matrix"])
            if group_matrix.ndim == 2:
                values = np.rint(group_matrix.sum(axis=0)).astype(int).reshape(-1)
                return [int(value) for value in values if int(value) > 0]

    return None

def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_causal_graph_mode(args):
    mode = str(args.causal_graph_mode).strip().lower()
    if mode != "auto":
        return mode
    if int(args.train_stage) != 2:
        return "none"
    if str(args.data_source).strip().lower() in {"existing", "lorzen"}:
        return "macro_lorenz96"
    if str(args.data_source).strip().lower() == "real_fmri":
        return "none"
    return "none"


def resolve_group_sizes_path(args):
    data_source = str(args.data_source).strip().lower()
    if data_source == "kuramoto":
        return args.generated_data_path
    if data_source in {"existing", "lorzen", "real_fmri"} and args.data_path:
        return args.data_path
    return ""


def resolve_group_schedule_path(args, reduce_dims):
    if reduce_dims:
        return None

    mode = str(args.group_schedule_mode).strip().lower()
    if mode == "off":
        return None
    if args.group_schedule_path:
        return args.group_schedule_path
    if mode == "on":
        if str(args.data_source).strip().lower() == "real_fmri":
            return "loc_data_real_fmri/group.csv"
        return "loc_data_kuramoto/group.csv"
    if str(args.data_source).strip().lower() == "kuramoto":
        return "loc_data_kuramoto/group.csv"
    return None


def main():
    parser = argparse.ArgumentParser(description="Causal Emergence of Mice")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage1_run_name", type=str, default="stage1_macro")
    parser.add_argument("--stage2_run_name", type=str, default="stage2_macro")
    parser.add_argument("--reduce_dims", type=str, default="")
    parser.add_argument("--scale_id", type=int, default=0)
    parser.add_argument("--ref_scale", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--train_stage", type=int, default=1)
    parser.add_argument("--time_delay", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_units1", type=int, default=64)
    parser.add_argument("--hidden_units2", type=int, default=64)
    parser.add_argument("--flow_layers", type=int, default=3)
    parser.add_argument("--dynamics_layers", type=int, default=5)
    parser.add_argument("--min_dim", type=int, default=1)
    parser.add_argument("--encoder_type", type=str, default="mlp", choices=["mlp", "invertible"])
    parser.add_argument("--group_schedule_mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--group_schedule_path", type=str, default="")
    parser.add_argument(
        "--causal_graph_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "lorenz96", "macro_lorenz96"],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--ei_samples", type=int, default=1000)
    parser.add_argument("--backward_weight", type=float, default=1.0)
    add_data_source_args(parser)
    args = parser.parse_args()
    
    device = resolve_device(args)
    origin_data, _ = load_origin_data(args)
    reduce_dims = parse_int_list(args.reduce_dims)
    group = None if reduce_dims else load_group_sizes_from_npz(resolve_group_sizes_path(args))
    causal_graph_mode = resolve_causal_graph_mode(args)
    group_schedule_path = resolve_group_schedule_path(args, reduce_dims)
    
    result = train_and_memorize(
        origin_data=origin_data,
        scale_id=args.scale_id,
        ref_scale=args.ref_scale,
        learning_rate=args.learning_rate,
        stage1_run_name=args.stage1_run_name,
        stage2_run_name=args.stage2_run_name,
        method="normal_encoder",
        device=device,
        epoches=args.epoch,
        hidden_units=args.hidden_units1,
        min_dim=args.min_dim,
        batch_size=args.batch_size,
        train_stage=args.train_stage,
        group=group,
        time_delay=args.time_delay,
        dynamics_hidden_units=args.hidden_units2,
        flow_num_layers=args.flow_layers,
        dynamics_num_layers=args.dynamics_layers,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        ei_samples=args.ei_samples,
        patience=args.patience,
        eval_interval=args.eval_interval,
        backward_weight=args.backward_weight,
        reduce_dims=reduce_dims,
        encoder_type=args.encoder_type,
        causal_graph_mode=causal_graph_mode,
        group_schedule_path=group_schedule_path,
    )
    
    summary = {
        "test_mse": result["test_mse"],
        "val_mse": result["val_mse"],
        **result["config"],
    }
    # print(summary)


if __name__ == "__main__":
    main()
