import copy
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, TensorDataset

from src.EI_calculation import approx_ei


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(input_dim, hidden_dim, output_dim, num_layers, activation_cls=nn.GELU, final_activation=None):
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers = []
    in_dim = int(input_dim)
    hidden_dim = int(hidden_dim)
    hidden_depth = max(int(num_layers) - 1, 0)
    
    for _ in range(hidden_depth):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_cls())
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, int(output_dim)))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


def normalize_encoder_type(value):
    text = str(value or "mlp").strip().lower()
    if text in {"mlp", "dense", "default", "normal_encoder"}:
        return "mlp"
    if text in {"invertible", "invertible_nn", "inn"}:
        return "invertible"
    raise ValueError(f"Unsupported encoder_type: {value}")


def _default_group_csv_path():
    return Path(__file__).resolve().parent.parent / "loc_data_kuramoto" / "group.csv"


def _parse_group_csv_value(token, group_csv_path, line_no):
    text = str(token).strip()
    if text == "":
        raise ValueError(f"{group_csv_path} line {line_no} contains an empty value")
    try:
        value = int(float(text))
    except ValueError as exc:
        raise ValueError(f"{group_csv_path} line {line_no} contains a non-integer value: {text}") from exc
    if value < 1:
        raise ValueError(f"{group_csv_path} line {line_no} contains a value < 1: {value}")
    return value


def _parse_group_csv_segment(segment_text, group_csv_path, line_no):
    values = [
        _parse_group_csv_value(token, group_csv_path, line_no)
        for token in str(segment_text).split(",")
        if str(token).strip()
    ]
    if not values:
        raise ValueError(f"{group_csv_path} line {line_no} contains an empty segment")
    return values


def load_group_schedule_from_csv(group_csv_path=None, expected_input_dim=None):
    group_csv_path = Path(group_csv_path) if group_csv_path is not None else _default_group_csv_path()
    if not group_csv_path.exists():
        return []

    schedule = []
    current_dim = None if expected_input_dim is None else int(expected_input_dim)
    raw_lines = []
    with group_csv_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if stripped:
                raw_lines.append((line_no, stripped))

    if not raw_lines:
        return []

    uses_chain_format = any(";" in line_text for _, line_text in raw_lines)
    if uses_chain_format:
        for line_no, line_text in raw_lines:
            segments = [_parse_group_csv_segment(segment, group_csv_path, line_no) for segment in line_text.split(";")]
            if len(segments) < 2:
                raise ValueError(f"{group_csv_path} line {line_no} must contain at least one coarse-graining step")

            current_groups = [int(size) for size in segments[0]]
            if current_dim is not None and sum(current_groups) != current_dim:
                raise ValueError(
                    f"{group_csv_path} line {line_no} starts from dimension {sum(current_groups)}, expected {current_dim}"
                )
            current_dim = int(sum(current_groups))

            for segment_idx, output_spec in enumerate(segments[1:], start=1):
                if len(output_spec) == len(current_groups):
                    if any(output_dim > group_size for group_size, output_dim in zip(current_groups, output_spec)):
                        raise ValueError(
                            f"{group_csv_path} line {line_no} segment {segment_idx + 1} contains an output dimension larger than its group size"
                        )
                    output_dim = int(sum(output_spec))
                    schedule.append(
                        {
                            "type": "group",
                            "macro_dim": output_dim,
                            "group_sizes": [int(size) for size in current_groups],
                            "group_output_dims": [int(dim) for dim in output_spec],
                            "source_lines": (int(line_no), int(segment_idx + 1)),
                        }
                    )
                    current_groups = [int(dim) for dim in output_spec]
                    current_dim = output_dim
                    continue

                if len(output_spec) == 1:
                    output_dim = int(output_spec[0])
                    if output_dim > current_dim:
                        raise ValueError(
                            f"{group_csv_path} line {line_no} segment {segment_idx + 1} requests DenseReducer output {output_dim} "
                            f"from current dimension {current_dim}"
                        )
                    schedule.append(
                        {
                            "type": "dense",
                            "macro_dim": output_dim,
                            "input_dim": int(current_dim),
                            "output_dim": output_dim,
                            "group_sizes": [int(size) for size in current_groups],
                            "group_output_dims": [output_dim],
                            "source_lines": (int(line_no), int(segment_idx + 1)),
                        }
                    )
                    current_groups = [output_dim]
                    current_dim = output_dim
                    continue

                raise ValueError(
                    f"{group_csv_path} line {line_no} segment {segment_idx + 1} must have length 1 "
                    f"for DenseReducer or length {len(current_groups)} for LocalGroupEncoder, but got {len(output_spec)}"
                )
        return schedule

    raw_rows = []
    for line_no, line_text in raw_lines:
        values = _parse_group_csv_segment(line_text, group_csv_path, line_no)
        raw_rows.append((line_no, values))

    if len(raw_rows) % 2 != 0:
        raise ValueError(f"{group_csv_path} must contain an even number of non-empty rows")

    for layer_idx in range(0, len(raw_rows), 2):
        group_line_no, group_sizes = raw_rows[layer_idx]
        output_line_no, group_output_dims = raw_rows[layer_idx + 1]

        if len(group_output_dims) == 1 and len(group_sizes) > 1:
            group_output_dims = group_output_dims * len(group_sizes)

        if len(group_output_dims) != len(group_sizes):
            raise ValueError(
                f"{group_csv_path} lines {group_line_no}-{output_line_no} must describe the same number of groups, "
                f"but got {len(group_sizes)} group sizes and {len(group_output_dims)} output sizes"
            )

        if current_dim is not None and sum(group_sizes) != current_dim:
            raise ValueError(
                f"{group_csv_path} line {group_line_no} must sum to the current scale dimension {current_dim}, "
                f"but got {sum(group_sizes)}"
            )
        if any(output_dim > group_size for group_size, output_dim in zip(group_sizes, group_output_dims)):
            raise ValueError(
                f"{group_csv_path} lines {group_line_no}-{output_line_no} contain an output dimension larger than its group size"
            )

        output_dim = int(sum(group_output_dims))
        schedule.append(
            {
                "type": "group",
                "macro_dim": output_dim,
                "group_sizes": [int(size) for size in group_sizes],
                "group_output_dims": [int(dim) for dim in group_output_dims],
                "source_lines": (int(group_line_no), int(output_line_no)),
            }
        )
        current_dim = output_dim

    return schedule


def serialize_group_schedule(group_schedule):
    if not group_schedule:
        return ""

    first_group_sizes = group_schedule[0].get("group_sizes", [])
    parts = [",".join(str(int(size)) for size in first_group_sizes)] if first_group_sizes else []
    for layer in group_schedule:
        if layer.get("type") == "dense":
            parts.append(str(int(layer["output_dim"])))
        else:
            parts.append(",".join(str(int(dim)) for dim in layer["group_output_dims"]))
    return ";".join(part for part in parts if part)


class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = int(mask.size(0)) // 2
        self.t = nn.ModuleList([nett() for _ in range(length)])
        self.s = nn.ModuleList([nets() for _ in range(length)])

    def g(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0])
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        z = x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J


class InvertibleFlowAdapter(nn.Module):
    def __init__(self, input_size, latent_size, hidden_units, num_layers, coupling_layers=8):
        super().__init__()
        self.input_size = int(input_size)
        self.latent_size = int(latent_size)
        if self.input_size < 1:
            raise ValueError("input_size must be >= 1")
        if self.latent_size < 1 or self.latent_size > self.input_size:
            raise ValueError(
                f"latent_size must be in [1, input_size], but got latent_size={self.latent_size}, input_size={self.input_size}"
            )

        nets = lambda: build_mlp(
            self.input_size,
            hidden_units,
            self.input_size,
            num_layers=num_layers,
            activation_cls=nn.LeakyReLU,
            final_activation=nn.Tanh,
        )
        nett = lambda: build_mlp(
            self.input_size,
            hidden_units,
            self.input_size,
            num_layers=num_layers,
            activation_cls=nn.LeakyReLU,
        )
        
        split = self.input_size // 2
        mask1 = torch.cat(
            (
                torch.zeros(1, split, dtype=torch.float32),
                torch.ones(1, self.input_size - split, dtype=torch.float32),
            ),
            dim=1,
        )
        mask2 = 1.0 - mask1
        masks = torch.cat([mask1 if idx % 2 == 0 else mask2 for idx in range(int(coupling_layers))], dim=0)
        self.flow = InvertibleNN(nets, nett, masks)

    def encode(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected input dimension {self.input_size}, but got {x.shape[1]}")
        latent, _ = self.flow.f(x)
        return latent[:, : self.latent_size]

    def decode(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[1] != self.latent_size:
            raise ValueError(f"Expected latent dimension {self.latent_size}, but got {z.shape[1]}")

        if self.input_size > self.latent_size:
            noise = torch.randn(
                z.shape[0],
                self.input_size - self.latent_size,
                device=z.device,
                dtype=z.dtype,
            ) / 3.0
            z = torch.cat((z, noise), dim=1)

        decoded, _ = self.flow.g(z)
        return decoded[:, : self.input_size]


class MLPFlow(nn.Module):
    def __init__(self, input_size, latent_size, hidden_units, num_layers):
        super().__init__()
        self.input_size = int(input_size)
        self.latent_size = int(latent_size)
        self.encoder = build_mlp(
            self.input_size,
            hidden_units,
            self.latent_size,
            num_layers=num_layers,
            activation_cls=nn.LeakyReLU,
        )
        self.decoder = build_mlp(
            self.latent_size,
            hidden_units,
            self.input_size,
            num_layers=num_layers,
            activation_cls=nn.LeakyReLU,
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class LocalGroupEncoder(nn.Module):
    def __init__(
        self,
        macro_dim,
        group_sizes,
        hidden_units,
        num_layers,
        encoder_type="mlp",
        device=None,
        group_output_dims=None,
    ):
        super().__init__()
        self.macro_dim = int(macro_dim)
        self.group_sizes = [int(size) for size in group_sizes]
        self.group_output_dims = [int(dim) for dim in (group_output_dims or [1] * len(self.group_sizes))]
        self.encoder_type = normalize_encoder_type(encoder_type)
        self.device = device
        if not self.group_sizes:
            raise ValueError("group_sizes must not be empty")
        if any(size < 1 for size in self.group_sizes):
            raise ValueError("each group size must be >= 1")
        if len(self.group_output_dims) == 1 and len(self.group_sizes) > 1:
            self.group_output_dims = self.group_output_dims * len(self.group_sizes)
        if len(self.group_output_dims) != len(self.group_sizes):
            raise ValueError(
                f"group_output_dims must align with group_sizes={len(self.group_sizes)}, but got {len(self.group_output_dims)}"
            )
        if any(dim < 1 for dim in self.group_output_dims):
            raise ValueError("each group output dimension must be >= 1")
        if any(output_dim > group_size for group_size, output_dim in zip(self.group_sizes, self.group_output_dims)):
            raise ValueError("each group output dimension must be <= its input group size")
        if sum(self.group_output_dims) != self.macro_dim:
            raise ValueError(
                f"macro_dim must equal the total output dimension {sum(self.group_output_dims)}, but got {self.macro_dim}"
            )

        self.num_groups = len(self.group_sizes)
        self.micro_dim = sum(self.group_sizes)
        self.group_slices = []
        self.latent_slices = []
        start = 0
        latent_start = 0
        for size in self.group_sizes:
            end = start + size
            self.group_slices.append((start, end))
            start = end
        for output_dim in self.group_output_dims:
            latent_end = latent_start + output_dim
            self.latent_slices.append((latent_start, latent_end))
            latent_start = latent_end

        if self.encoder_type == "invertible":
            self.group_flows = nn.ModuleList(
                [
                    InvertibleFlowAdapter(
                        input_size=group_size,
                        latent_size=group_output_dim,
                        hidden_units=hidden_units,
                        num_layers=num_layers,
                    )
                    for group_size, group_output_dim in zip(self.group_sizes, self.group_output_dims)
                ]
            )
        else:
            self.encoders = nn.ModuleList(
                [
                    build_mlp(
                        group_size,
                        hidden_units,
                        group_output_dim,
                        num_layers=num_layers,
                        activation_cls=nn.LeakyReLU,
                    )
                    for group_size, group_output_dim in zip(self.group_sizes, self.group_output_dims)
                ]
            )
            self.decoders = nn.ModuleList(
                [
                    build_mlp(
                        group_output_dim,
                        hidden_units,
                        group_size,
                        num_layers=num_layers,
                        activation_cls=nn.LeakyReLU,
                    )
                    for group_size, group_output_dim in zip(self.group_sizes, self.group_output_dims)
                ]
            )

    def encode(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.micro_dim:
            raise ValueError(f"Expected micro input dimension {self.micro_dim}, but got {x.shape[1]}")

        latents = []
        if self.encoder_type == "invertible":
            for flow, (start, end) in zip(self.group_flows, self.group_slices):
                latents.append(flow.encode(x[:, start:end]))
        else:
            for encoder, (start, end) in zip(self.encoders, self.group_slices):
                latents.append(encoder(x[:, start:end]))
        return torch.cat(latents, dim=1)

    def decode(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[1] != self.macro_dim:
            raise ValueError(f"Expected macro latent dimension {self.macro_dim}, but got {z.shape[1]}")

        recon = []
        if self.encoder_type == "invertible":
            for flow, (start, end) in zip(self.group_flows, self.latent_slices):
                recon.append(flow.decode(z[:, start:end]))
        else:
            for decoder, (start, end) in zip(self.decoders, self.latent_slices):
                recon.append(decoder(z[:, start:end]))
        return torch.cat(recon, dim=1)


class DenseReducer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, num_layers, encoder_type="mlp", device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.encoder_type = normalize_encoder_type(encoder_type)
        self.device = device
        if self.input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if self.output_dim > self.input_dim:
            raise ValueError("output_dim must be <= input_dim")

        if self.encoder_type == "invertible":
            self.flow = InvertibleFlowAdapter(
                input_size=self.input_dim,
                latent_size=self.output_dim,
                hidden_units=hidden_units,
                num_layers=num_layers,
            )
        else:
            self.flow = MLPFlow(
                input_size=self.input_dim,
                latent_size=self.output_dim,
                hidden_units=hidden_units,
                num_layers=num_layers,
            )

    def encode(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected reducer input dimension {self.input_dim}, but got {x.shape[1]}")
        return self.flow.encode(x)

    def decode(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[1] != self.output_dim:
            raise ValueError(f"Expected reducer latent dimension {self.output_dim}, but got {z.shape[1]}")
        return self.flow.decode(z)


class Parellel_Renorm_Dynamic(nn.Module):
    def __init__(
        self,
        sym_size,
        latent_size,
        effect_size,
        cut_size,
        hidden_units1,
        hidden_units2,
        normalized_state,
        device,
        is_random=False,
        flow_num_layers=3,
        dynamics_num_layers=4,
        decode_noise_scale=0.0,
        reduce_dims=None,
        group=None,
        group_schedule=None,
        encoder_type="mlp",
    ):
        super().__init__()
        del effect_size, decode_noise_scale
        if int(latent_size) < 1 or int(latent_size) > int(sym_size):
            raise ValueError(f"latent_size must be in [1, {sym_size}]")
        if int(cut_size) < 2:
            raise ValueError("cut_size must be >= 2")

        self.device = device
        self.sym_size = int(sym_size)
        self.latent_size = int(latent_size)
        self.cut_size = int(cut_size)
        self.normalized_state = normalized_state
        self.is_random = is_random
        self.encoder_type = normalize_encoder_type(encoder_type)
        self.group_schedule = self._normalize_group_schedule(group_schedule)
        self.group = self._normalize_group(group)
        if self.group_schedule:
            self.group = list(self.group_schedule[0]["group_sizes"])
        self.scale_dims, self.transition_specs, self.reduce_dims = self._build_scale_dims(reduce_dims, self.group)
        self.group_transitions = nn.ModuleList(
            [
                self._build_transition(spec, hidden_units1, flow_num_layers)
                for spec in self.transition_specs
            ]
        )
        self.dynamics_modules = nn.ModuleList(
            [self.build_dynamics(scale_dim, hidden_units2, dynamics_num_layers) for scale_dim in self.scale_dims]
        )
        self.inverse_dynamics_modules = nn.ModuleList(
            [self.build_dynamics(scale_dim, hidden_units2, dynamics_num_layers) for scale_dim in self.scale_dims]
        )

    def _resolve_reduce_dims(self, reduce_dims):
        if reduce_dims is None:
            return []
        if isinstance(reduce_dims, (int, np.integer)):
            dims = [int(reduce_dims)]
        else:
            dims = [int(dim) for dim in reduce_dims]
        if any(dim < 1 for dim in dims):
            raise ValueError("reduce_dims entries must be >= 1")
        return dims

    def _normalize_group(self, group):
        if group is None:
            return []

        if isinstance(group, np.ndarray):
            normalized = [int(size) for size in group.reshape(-1).tolist()]
        else:
            normalized = [int(size) for size in group]

        if not normalized:
            return []
        if any(size < 1 for size in normalized):
            raise ValueError("group entries must be >= 1")
        if sum(normalized) != self.sym_size:
            raise ValueError(f"group must sum to sym_size={self.sym_size}, but got {sum(normalized)}")
        return normalized

    def _normalize_group_schedule(self, group_schedule):
        if group_schedule is None:
            return []

        normalized = []
        current_dim = self.sym_size
        for layer_idx, layer in enumerate(group_schedule):
            if isinstance(layer, dict):
                layer_type = str(layer.get("type", "group")).strip().lower()
                raw_group_sizes = layer.get("group_sizes", [])
                raw_group_output_dims = layer.get("group_output_dims", layer.get("output_dims", []))
                raw_input_dim = layer.get("input_dim")
                raw_output_dim = layer.get("output_dim")
            else:
                raise ValueError(f"group_schedule layer {layer_idx + 1} must be a dict")

            if isinstance(raw_group_sizes, np.ndarray):
                group_sizes = [int(size) for size in raw_group_sizes.reshape(-1).tolist()]
            else:
                group_sizes = [int(size) for size in raw_group_sizes]

            if isinstance(raw_group_output_dims, np.ndarray):
                group_output_dims = [int(dim) for dim in raw_group_output_dims.reshape(-1).tolist()]
            else:
                group_output_dims = [int(dim) for dim in raw_group_output_dims]

            if layer_type == "dense":
                input_dim = int(raw_input_dim if raw_input_dim is not None else sum(group_sizes))
                output_dim = int(raw_output_dim if raw_output_dim is not None else (group_output_dims[0] if group_output_dims else 0))
                if input_dim != current_dim:
                    raise ValueError(
                        f"group_schedule layer {layer_idx + 1} expects DenseReducer input {input_dim}, "
                        f"but current dimension is {current_dim}"
                    )
                if output_dim < 1 or output_dim > input_dim:
                    raise ValueError(
                        f"group_schedule layer {layer_idx + 1} must have DenseReducer output in [1, {input_dim}], got {output_dim}"
                    )
                normalized.append(
                    {
                        "type": "dense",
                        "macro_dim": output_dim,
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "group_sizes": group_sizes if group_sizes else [input_dim],
                        "group_output_dims": [output_dim],
                    }
                )
                current_dim = output_dim
                continue

            if not group_sizes:
                raise ValueError(f"group_schedule layer {layer_idx + 1} must not be empty")
            if any(size < 1 for size in group_sizes):
                raise ValueError(f"group_schedule layer {layer_idx + 1} contains a group size < 1")
            if any(dim < 1 for dim in group_output_dims):
                raise ValueError(f"group_schedule layer {layer_idx + 1} contains an output dimension < 1")
            if len(group_output_dims) != len(group_sizes):
                raise ValueError(
                    f"group_schedule layer {layer_idx + 1} must align group sizes and output dims, "
                    f"but got {len(group_sizes)} and {len(group_output_dims)}"
                )
            if sum(group_sizes) != current_dim:
                raise ValueError(
                    f"group_schedule layer {layer_idx + 1} must cover the current scale dimension {current_dim}, "
                    f"but got {sum(group_sizes)}"
                )
            if any(output_dim > group_size for group_size, output_dim in zip(group_sizes, group_output_dims)):
                raise ValueError(
                    f"group_schedule layer {layer_idx + 1} contains an output dimension larger than its group size"
                )

            output_dim = int(sum(group_output_dims))
            normalized.append(
                {
                    "type": "group",
                    "macro_dim": output_dim,
                    "group_sizes": group_sizes,
                    "group_output_dims": group_output_dims,
                }
            )
            current_dim = output_dim

        return normalized
    
    def _contiguous_groups(self, input_dim, group_size):
        input_dim = int(input_dim)
        group_size = int(group_size)
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        if input_dim <= group_size:
            return [input_dim]

        groups = []
        remaining = input_dim
        while remaining > 0:
            current_size = min(group_size, remaining)
            groups.append(current_size)
            remaining -= current_size
        
        # From the second layer onward, merge a tiny tail group into the
        # previous group so the final coarse-graining block is not too small.
        if groups[-1] == 1 or (groups[-1] < 4 and group_size > 5):
            groups[-2] += groups[-1]
            groups.pop()
        return groups

    def _build_transition(self, spec, hidden_units, num_layers):
        if spec["type"] == "group":
            return LocalGroupEncoder(
                macro_dim=spec["macro_dim"],
                group_sizes=spec["group_sizes"],
                group_output_dims=spec.get("group_output_dims"),
                hidden_units=hidden_units,
                num_layers=num_layers,
                encoder_type=self.encoder_type,
                device=self.device,
            )
        if spec["type"] == "dense":
            return DenseReducer(
                input_dim=spec["input_dim"],
                output_dim=spec["output_dim"],
                hidden_units=hidden_units,
                num_layers=num_layers,
                encoder_type=self.encoder_type,
                device=self.device,
            )
        raise ValueError(f"Unsupported transition spec type: {spec['type']}")

    def _build_scale_dims(self, reduce_dims, group):
        dims = []
        transition_specs = []
        current_dim = self.sym_size

        if self.group_schedule:
            for layer in self.group_schedule:
                transition_specs.append(dict(layer))
                dims.append(int(layer.get("macro_dim", layer.get("output_dim"))))
            return dims, transition_specs, []
        
        reduce_dim_schedule = self._resolve_reduce_dims(reduce_dims)
        if reduce_dim_schedule:
            for target_dim in reduce_dim_schedule:
                target_dim = int(target_dim)
                # if target_dim == current_dim:
                #     continue
                if target_dim > current_dim:
                    raise ValueError(
                        f"reduce_dims must be non-increasing, but got target_dim={target_dim} after {current_dim}"
                    )
                transition_specs.append(
                    {
                        "type": "dense",
                        "input_dim": current_dim,
                        "output_dim": target_dim,
                    }
                )
                dims.append(target_dim)
                current_dim = target_dim
        else:
            if group:
                next_dim = len(group)
                if next_dim < current_dim:
                    transition_specs.append(
                        {
                            "type": "group",
                            "macro_dim": next_dim,
                            "group_sizes": list(group),
                        }
                    )
                    dims.append(next_dim)
                    current_dim = next_dim
            
            while current_dim > self.latent_size:
                group_size = self.cut_size

                if group_size < 2 and current_dim > self.latent_size:
                    raise ValueError("local group size must be >= 2 before reaching the latent scale")

                group_sizes = self._contiguous_groups(current_dim, group_size)
                next_dim = len(group_sizes)
                if next_dim == current_dim:
                    break

                transition_specs.append(
                    {
                        "type": "group",
                        "macro_dim": next_dim,
                        "group_sizes": group_sizes,
                    }
                )
                dims.append(next_dim)
                current_dim = next_dim

        if not dims:
            raise ValueError(
                "No macro scales were constructed. Please provide reduce_dims or allow cut_size-based coarse graining."
            )
        return dims, transition_specs, reduce_dim_schedule

    def build_dynamics(self, mid_size, hidden_units, num_layers):
        return build_mlp(mid_size, hidden_units, mid_size, num_layers=num_layers, activation_cls=nn.LeakyReLU)
    
    def _prepare_input(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] < self.sym_size:
            padding = torch.zeros(x.shape[0], self.sym_size - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat((x, padding), dim=1)
        elif x.shape[1] > self.sym_size:
            x = x[:, : self.sym_size]
        return x
    
    def _apply_dynamics(self, latent, scale_id, inverse=False):
        module = self.inverse_dynamics_modules[scale_id] if inverse else self.dynamics_modules[scale_id]
        next_latent = module(latent) + latent
        if self.normalized_state:
            next_latent = torch.tanh(next_latent)
        return next_latent

    def encoding(self, x):
        y = self._prepare_input(x)
        ys = []
        for transition in self.group_transitions:
            y = transition.encode(y)
            ys.append(y)
        return ys

    def encoding1(self, x, scale_id):
        y = self._prepare_input(x)
        ys = []
        for i, transition in enumerate(self.group_transitions):
            y = transition.encode(y)
            ys.append(y)
            if i >= int(scale_id):
                break
        return ys

    def decoding(self, s_next, level):
        y = s_next
        for i in range(int(level), -1, -1):
            y = self.group_transitions[i].decode(y)
        return y[:, : self.sym_size]

    def forward(self, x, delay=1):
        if x.dim() == 3:
            x = x[:, 0]
        ss = self.encoding(x)
        s_nexts = []
        ys = []
        for scale_id, latent in enumerate(ss):
            current = latent
            rollout = []
            preds = []
            for _ in range(int(delay)):
                current = self._apply_dynamics(current, scale_id, inverse=False)
                rollout.append(current)
                preds.append(self.decoding(current, scale_id))
            s_nexts.append(torch.stack(rollout, dim=1))
            ys.append(torch.stack(preds, dim=1))
        return ys, ss, s_nexts
    
    def train_forward(self, x, scale_id, delay=1):
        if x.dim() == 3:
            x = x[:, 0]
        ss = self.encoding1(x, scale_id)
        current = ss[int(scale_id)]
        rollout = []
        preds = []
        for _ in range(int(delay)):
            current = self._apply_dynamics(current, int(scale_id), inverse=False)
            rollout.append(current)
            preds.append(self.decoding(current, int(scale_id)))
        return [torch.stack(preds, dim=1)], ss, [torch.stack(rollout, dim=1)]

    def train_backward(self, x, scale_id, delay=1):
        if x.dim() == 3:
            x = x[:, -1]
        ss = self.encoding1(x, scale_id)
        current = ss[int(scale_id)]
        rollout = []
        for _ in range(int(delay)):
            current = self._apply_dynamics(current, int(scale_id), inverse=True)
            rollout.append(current)
        return [], ss, [torch.stack(rollout[::-1], dim=1)]
    
    def loss(self, predictions, real, loss_f):
        losses = []
        total = 0.0
        for predict in predictions:
            target = real
            if predict.dim() == 3 and real.dim() == 2:
                target = real.unsqueeze(1).expand_as(predict)
            loss = loss_f(predict, target)
            losses.append(loss.item())
            total += loss
        return losses, total / max(len(losses), 1)
    
    def loss_weights_general(self, predictions, real, scale_id, w, loss_f, forward_func):
        del scale_id, w, forward_func
        return self.loss(predictions, real, loss_f)

    def loss_weights(self, predictions, real, scale_id, w, loss_f):
        return self.loss_weights_general(predictions, real, scale_id, w, loss_f, self.train_forward)

    def loss_weights_back(self, predictions, real, scale_id, w, loss_f):
        return self.loss_weights_general(predictions, real, scale_id, w, loss_f, self.train_backward)

    def to_weights(self, log_w, temperature=10):
        logsoft = nn.LogSoftmax(dim=0)
        return torch.exp(logsoft(log_w / temperature))

    def estimate_sigmas_matrix(self, s, sp, scale_id, mse_raw):
        if s.dim() == 3:
            source = s[:, 0]
            target = sp[:, 0]
        else:
            source = s
            target = sp

        with torch.no_grad():
            encodes = self.encoding1(target, scale_id)
            predicts, _, latentp1s = self.train_forward(source, scale_id, delay=1)
            latentp1 = latentp1s[0][:, 0]
            encode = encodes[int(scale_id)]
            mse1 = mse_raw(latentp1, encode)
            sigmas = mse1.mean(dim=0) + 1e-6
            sigmas_matrix = torch.diag(sigmas)
        return sigmas, sigmas_matrix, predicts, target

    def calc_EIs_kde1(self, s, sp, samples, mae, mse_raw, L, bigL, scale_id, device, use_cuda=True, ei_samples=128):
        del samples, bigL, use_cuda
        if s.dim() == 3:
            source = s[:, 0]
            target = sp[:, 0]
        else:
            source = s
            target = sp
        
        sigmas, sigmas_matrix, predicts, target = self.estimate_sigmas_matrix(s, sp, scale_id, mse_raw)
        _, loss_test = self.loss(predicts, target, mae)
        
        dynamics = self.dynamics_modules[int(scale_id)]
        scale = sigmas_matrix.shape[0]
        ei = approx_ei(
            scale,
            scale,
            sigmas_matrix.data,
            lambda x: self._apply_dynamics(x.unsqueeze(0), int(scale_id), inverse=False),
            num_samples=ei_samples,
            L=L,
            easy=True,
            device=device,
        )
        return [ei], [sigmas], [], loss_test


def build_lorenz96_adjacency(num_nodes):
    adjacency = np.zeros((int(num_nodes), int(num_nodes)), dtype=np.float32)
    for i in range(int(num_nodes)):
        adjacency[i, i] = 1.0
        adjacency[i, (i - 1) % int(num_nodes)] = 1.0
        adjacency[i, (i - 2) % int(num_nodes)] = 1.0
        adjacency[i, (i + 1) % int(num_nodes)] = 1.0
    return adjacency


def build_macro_lorenz96_adjacency(group):
    if isinstance(group, (int, np.integer)):
        macro_size = int(group)
    else:
        macro_size = len(group)
    return build_lorenz96_adjacency(macro_size)


def resolve_causal_ground_truth(causal_graph_mode, model, scale_id):
    mode = str(causal_graph_mode or "none").strip().lower()
    if mode in {"", "none"}:
        return None
    if mode == "lorenz96":
        return build_lorenz96_adjacency(int(model.scale_dims[int(scale_id)]))
    if mode == "macro_lorenz96":
        return build_macro_lorenz96_adjacency(int(model.scale_dims[int(scale_id)]))
    raise ValueError(f"Unsupported causal_graph_mode: {causal_graph_mode}")


def _sample_uniform_latent_jacobians(model, scale_id, device, num_samples, L):
    if num_samples is None or int(num_samples) <= 0:
        raise ValueError("num_samples must be a positive integer")
    if L <= 0:
        raise ValueError("L must be positive")

    scale_id = int(scale_id)
    if scale_id < 0 or scale_id >= len(model.scale_dims):
        raise ValueError(f"scale_id must be in [0, {len(model.scale_dims) - 1}]")

    model.eval()
    jacobians = []
    latent_dim = int(model.scale_dims[scale_id])
    sampled_states = L * (torch.rand(int(num_samples), latent_dim, device=device) - 0.5)
    for macro_state in sampled_states:
        macro_state = macro_state.detach().clone().requires_grad_(True)
        jac = jacobian(
            lambda macro: model._apply_dynamics(macro.unsqueeze(0), scale_id, inverse=False).squeeze(0),
            macro_state,
        )
        jacobians.append(jac.detach())
    return torch.stack(jacobians, dim=0)


def compute_jacobian_strength(model, scale_id, device, max_samples=None, L=1.0):
    num_samples = 48 if max_samples is None else max_samples
    jacobians = _sample_uniform_latent_jacobians(model, scale_id, device, num_samples=num_samples, L=L)
    return jacobians.abs().mean(dim=0)


def compute_pairwise_ei_network(model, sigmas_matrix, scale_id, L, device, num_samples=48, eps=1e-12):
    if L <= 0:
        raise ValueError("L must be positive")

    J = _sample_uniform_latent_jacobians(model, scale_id, device, num_samples=num_samples, L=L).to(dtype=torch.float32)
    mean_J = J.mean(dim=0)

    Sigma = torch.as_tensor(sigmas_matrix, dtype=J.dtype, device=J.device)
    if J.shape[1] != J.shape[2]:
        raise ValueError("jacobians must describe a square causal network")
    if Sigma.dim() != 2 or Sigma.shape[0] != Sigma.shape[1] or Sigma.shape[0] != J.shape[1]:
        raise ValueError("sigmas_matrix must be a square matrix matching the Jacobian dimensions")

    variance_scale = float(L**2) / 12.0
    log_constant = torch.log(
        torch.tensor(2.0 * np.pi * np.e, dtype=J.dtype, device=J.device)
    )
    sigma_diag = torch.diagonal(Sigma).unsqueeze(1)

    abs_J = torch.abs(J)
    log_abs_J = torch.full_like(abs_J, -torch.inf)
    valid_mask = abs_J > eps
    log_abs_J[valid_mask] = torch.log(abs_J[valid_mask])
    expected_log_abs_J = log_abs_J.mean(dim=0)

    squared_J = J.pow(2)
    excluded_energy = squared_J.sum(dim=2, keepdim=True) - squared_J
    mean_excluded_energy = excluded_energy.mean(dim=0)
    effective_variance = torch.clamp(sigma_diag + variance_scale * mean_excluded_energy, min=eps)
    variance_term = -0.5 * torch.log(effective_variance)

    ei = (
        torch.log(torch.tensor(L, dtype=J.dtype, device=J.device))
        + expected_log_abs_J
        - 0.5 * log_constant
        + variance_term
    )
    return ei, mean_J, expected_log_abs_J, variance_term


def evaluate_causal_graph(
    model,
    scale_id,
    states,
    device,
    ground_truth=None,
    max_samples=48,
    inferred_causal_graph=None,
    jacobian_mean_abs=None,
):
    del states
    if jacobian_mean_abs is None:
        jacobian_mean_abs = compute_jacobian_strength(
            model,
            scale_id,
            device,
            max_samples=max_samples,
        ).detach().cpu().numpy()
    else:
        jacobian_mean_abs = np.asarray(jacobian_mean_abs, dtype=np.float32)

    if inferred_causal_graph is None:
        inferred_graph = jacobian_mean_abs
    else:
        inferred_graph = np.asarray(inferred_causal_graph, dtype=np.float32)

    if ground_truth is None:
        return {
            "jacobian_mean_abs": jacobian_mean_abs,
            "ei_causal_graph": inferred_graph,
            "best_threshold": None,
            "causal_accuracy": None,
            "causal_f1": None,
            "precision": None,
            "recall": None,
            "auc": None,
        }

    gt = np.asarray(ground_truth, dtype=np.float32).astype(bool)
    gt_flat = gt.reshape(-1).astype(np.int32)
    score_flat = inferred_graph.reshape(-1)
    try:
        auc = float(roc_auc_score(gt_flat, score_flat))
    except ValueError:
        auc = None

    flat_scores = inferred_graph.reshape(-1)
    thresholds = np.unique(np.quantile(flat_scores, np.linspace(0.4, 0.98, 30)))

    best_metrics = None
    for threshold in thresholds:
        pred = inferred_graph >= threshold
        tp = np.logical_and(pred, gt).sum()
        tn = np.logical_and(~pred, ~gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        candidate = {
            "jacobian_mean_abs": jacobian_mean_abs,
            "ei_causal_graph": inferred_graph,
            "best_threshold": float(threshold),
            "causal_accuracy": float(accuracy),
            "causal_f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "auc": auc,
        }
        if best_metrics is None or (candidate["causal_f1"], candidate["causal_accuracy"]) > (
            best_metrics["causal_f1"],
            best_metrics["causal_accuracy"],
        ):
            best_metrics = candidate
    return best_metrics


def _to_numpy(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    if torch.is_tensor(data):
        return data.detach().cpu().numpy().astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _trim_trailing_nan_rows(series):
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError(f"series must have shape [T, N], but got {series.shape}")

    finite_rows = np.all(np.isfinite(series), axis=1)
    if finite_rows.all():
        return series

    invalid_rows = np.flatnonzero(~finite_rows)
    if invalid_rows.size == 0:
        return series
    return series[: int(invalid_rows[0])]


def _build_windows_from_series(data, time_delay):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        all_inputs = []
        all_targets = []
        for series in data:
            series = _trim_trailing_nan_rows(series)
            if len(series) <= time_delay:
                continue
            series_inputs, series_targets = _build_windows_from_series(series, time_delay)
            if len(series_inputs) == 0:
                continue
            all_inputs.append(series_inputs)
            all_targets.append(series_targets)

        if not all_inputs:
            feature_dim = int(data.shape[-1])
            empty_shape = (0, int(time_delay), feature_dim)
            return np.zeros(empty_shape, dtype=np.float32), np.zeros(empty_shape, dtype=np.float32)
        return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)

    if data.ndim != 2:
        raise ValueError(f"origin_data must have shape [T, N] or [M, T, N], but got {data.shape}")

    data = _trim_trailing_nan_rows(data)
    inputs = []
    targets = []
    for i in range(len(data) - time_delay):
        inputs.append(data[i : i + time_delay])
        targets.append(data[i + 1 : i + time_delay + 1])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _train_val_split(x, y, val_ratio):
    if len(x) < 2 or val_ratio <= 0:
        return x, y, x[:0], y[:0]

    split_index = max(1, int(len(x) * (1 - val_ratio)))
    split_index = min(split_index, len(x) - 1)
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]


def _prediction_metrics(model, x, y, device, batch_size=128, scale_id=None):
    if len(x) == 0:
        return float("nan"), float("nan")

    model.eval()
    maes = []
    mses = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y[start : start + batch_size], dtype=torch.float32, device=device)
            if scale_id is None:
                preds = model.forward(xb, delay=yb.shape[1])[0]
            else:
                preds = model.train_forward(xb, scale_id=scale_id, delay=yb.shape[1])[0]

            batch_mae = []
            batch_mse = []
            for pred in preds:
                batch_mae.append(torch.mean(torch.abs(pred - yb)).item())
                batch_mse.append(torch.mean((pred - yb) ** 2).item())
            maes.append(float(np.mean(batch_mae)))
            mses.append(float(np.mean(batch_mse)))
    return float(np.mean(maes)), float(np.mean(mses))


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_history_files(history, ei_value, ei_term1, ei_term2, forward_scale_loss, result_dir, scale_id, train_stage):
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(result_dir, "train_history_scale.csv" if int(train_stage) == 1 else f"train_history_scale{scale_id+1}.csv"), index=False)

    train_loss_path = os.path.join(
        result_dir,
        "train_loss.csv" if int(train_stage) == 1 else f"train_loss{scale_id+1}.csv",
    )
    test_loss_path = os.path.join(
        result_dir,
        "test_loss.csv" if int(train_stage) == 1 else f"test_loss{scale_id+1}.csv",
    )
    
    history_df[["train_mae"]].to_csv(train_loss_path, index=False, header=False)
    history_df[["val_mae"]].to_csv(test_loss_path, index=False, header=False)
    if int(train_stage) == 2:
        pd.DataFrame(np.array(ei_value)).to_csv(os.path.join(result_dir, f"EI_scale{scale_id+1}.csv"), index=False, header=False)
        pd.DataFrame(np.array(ei_term1)).to_csv(os.path.join(result_dir, f"term1_scale{scale_id+1}.csv"), index=False, header=False)
        pd.DataFrame(np.array(ei_term2)).to_csv(os.path.join(result_dir, f"term2_scale{scale_id+1}.csv"), index=False, header=False)
    
    if int(train_stage) == 1:
        scale_loss_path = os.path.join(
        result_dir,
        "train_scale_loss.csv",)
        pd.DataFrame(forward_scale_loss).to_csv(scale_loss_path, index=False, header=False)
        # pd.DataFrame(np.array(ei_value)).to_csv(os.path.join(result_dir, f"EI_scale.csv"), index=False, header=False)
        # pd.DataFrame(np.array(ei_term1)).to_csv(os.path.join(result_dir, f"term1_scale.csv"), index=False, header=False)
        # pd.DataFrame(np.array(ei_term2)).to_csv(os.path.join(result_dir, f"term2_scale.csv"), index=False, header=False)


def train_and_memorize(
    scale_id=0,
    ref_scale=0,
    learning_rate=1e-3,
    stage1_run_name="stage1_macro",
    stage2_run_name="stage2_macro",
    method="mlp",
    device="cpu",
    epoches=300,
    hidden_units=64,
    min_dim=1,
    batch_size=128,
    train_stage=1,
    origin_data=None,
    group=None,
    time_delay=3,
    dynamics_hidden_units=None,
    flow_num_layers=3,
    dynamics_num_layers=4,
    weight_decay=1e-5,
    val_ratio=0.1,
    seed=42,
    ei_samples=1000,
    patience=40,
    decode_noise_scale=0.0,
    backward_weight=1.0,
    reduce_dims=None,
    encoder_type="mlp",
    causal_graph_mode="none",
    eval_interval=20,
    group_schedule_path=None,
):  
    del ref_scale
    # set_random_seed(seed)
    device = torch.device(device)
    stage1_run_name = str(stage1_run_name or "stage1_macro")
    stage2_run_name = str(stage2_run_name or "stage2_macro")
    encoder_type = normalize_encoder_type(encoder_type or method)
    causal_graph_mode = str(causal_graph_mode or "none").strip().lower()
    eval_interval = int(eval_interval)
    if eval_interval <= 0:
        raise ValueError("eval_interval must be a positive integer")

    origin_data = _to_numpy(origin_data)
    if origin_data is None:
        raise ValueError("origin_data must be provided")

    x_all, y_all = _build_windows_from_series(origin_data, int(time_delay))
    if len(x_all) < 3:
        raise ValueError("Not enough sequential samples to build multi-step windows from origin_data")

    test_start = int(len(x_all) * 0.95)
    x_train_full, y_train_full = x_all[:test_start], y_all[:test_start]
    x_test, y_test = x_all[test_start:], y_all[test_start:]
    
    x_train, y_train, x_val, y_val = _train_val_split(x_train_full, y_train_full, float(val_ratio))
    if len(x_val) == 0:
        x_val, y_val = x_test, y_test
    if len(x_test) == 0:
        x_test, y_test = x_val, y_val

    # pd.DataFrame(x_train_full[:,0]).to_csv('./loc_data_kuramoto/train_input.csv', header=None, index=None)
    # pd.DataFrame(y_train_full[:,0]).to_csv('./loc_data_kuramoto/train_target.csv', header=None, index=None)
    # pd.DataFrame(x_test[:,0]).to_csv('./loc_data_kuramoto/test_input.csv', header=None, index=None)
    # pd.DataFrame(y_test[:,0]).to_csv('./loc_data_kuramoto/test_target.csv', header=None, index=None)
    

    num_nodes = x_train.shape[-1]
    dynamics_hidden_units = int(dynamics_hidden_units or hidden_units)
    group_schedule = load_group_schedule_from_csv(
        group_csv_path=group_schedule_path,
        expected_input_dim=num_nodes,
    ) if group_schedule_path else []
    if group_schedule:
        group = None
        reduce_dims = None
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model = Parellel_Renorm_Dynamic(
        sym_size=num_nodes,
        latent_size=min_dim,
        effect_size=num_nodes,
        cut_size=2,
        hidden_units1=hidden_units,
        hidden_units2=dynamics_hidden_units,
        normalized_state=True,
        device=device,
        is_random=False,
        flow_num_layers=flow_num_layers,
        dynamics_num_layers=dynamics_num_layers,
        decode_noise_scale=decode_noise_scale,
        reduce_dims=reduce_dims,
        group=group,
        group_schedule=group_schedule,
        encoder_type=encoder_type,
    ).to(device)
    
    max_scale_id = len(model.scale_dims) - 1
    scale_id = max(0, min(int(scale_id), max_scale_id))
    
    stage1_model_dir = os.path.join(".", "loc_model_stage1", stage1_run_name)
    stage2_model_dir = os.path.join(".", "loc_model_stage2", stage2_run_name)
    if int(train_stage) == 1:
        _ensure_dir(stage1_model_dir)
    if int(train_stage) == 2:
        _ensure_dir(stage2_model_dir)
    
    stage1_ckpt = os.path.join(stage1_model_dir, "model.pkl")
    if int(train_stage) == 2 and os.path.exists(stage1_ckpt):
        try:
            model.load_state_dict(torch.load(stage1_ckpt, map_location=device), strict=False)
        except RuntimeError:
            pass
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.L1Loss()
    mae = nn.L1Loss()
    mse_raw = nn.MSELoss(reduction="none")
    
    best_state = copy.deepcopy(model.state_dict())
    best_val_mse = float("inf")
    wait = 0
    history, all_scale_loss = [], []
    ei_value, ei_term1, ei_term2 = [], [], []
    for epoch in range(int(epoches)):
        model.train()
        should_evaluate = epoch % eval_interval == 0
        batch_losses = []
        batch_backward_losses = []
        loss_forward_scale = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            if int(train_stage) == 1:
                preds, _, _ = model.forward(xb, delay=yb.shape[1])
                pred_scale, pred_loss = model.loss(preds, yb, loss_fn)
                total_loss = pred_loss
                backward_loss_value = 0.0
            else:
                preds, latents, _ = model.train_forward(xb, scale_id, delay=yb.shape[1])
                _, pred_loss = model.loss(preds, yb, loss_fn)
                _, _, latent_backward = model.train_backward(yb[:, 0], scale_id, delay=1)
                backward_pred = latent_backward[0]
                backward_target = latents[scale_id].unsqueeze(1)
                _, backward_loss = model.loss([backward_pred], backward_target, loss_fn)
                total_loss = pred_loss + float(backward_weight) * backward_loss
                backward_loss_value = float(backward_loss.item())
                # total_loss = pred_loss
                # backward_loss_value = 0
            
            
            
            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(float(pred_loss.item()))
            batch_backward_losses.append(backward_loss_value)
            if int(train_stage) == 1:
                loss_forward_scale.append(pred_scale)
        
        if should_evaluate:
            metric_scale = None if int(train_stage) == 1 else scale_id
            val_mae, val_mse = _prediction_metrics(model, x_val, y_val, device, batch_size=batch_size, scale_id=metric_scale)
            train_mae, train_mse = _prediction_metrics(
                model,
                x_train[: min(len(x_train), batch_size)],
                y_train[: min(len(y_train), batch_size)],
                device,
                batch_size=batch_size,
                scale_id=metric_scale,
            )
        
        if int(train_stage) == 2:
            if should_evaluate:
                eval_count = min(len(x_train), batch_size)
                eval_indices = np.random.choice(len(x_train), size=eval_count, replace=False)
                eval_tensor_x = torch.tensor(x_train[eval_indices], dtype=torch.float32, device=device)
                eval_tensor_y = torch.tensor(y_train[eval_indices], dtype=torch.float32, device=device)
                ei_tuple, _, _, _ = model.calc_EIs_kde1(
                    eval_tensor_x,
                    eval_tensor_y,
                    eval_count,
                    mae,
                    mse_raw,
                    L=1,
                    bigL=1,
                    scale_id=scale_id,
                    device=device,
                    ei_samples=ei_samples,
                )
                ei_value.append(float(ei_tuple[0][0]))
                ei_term1.append(float(ei_tuple[0][2]))
                ei_term2.append(float(ei_tuple[0][3]))
        if int(train_stage) == 1:
            pass
            # if epoch % 500 == 0:
            #     eval_count = min(len(x_train), batch_size)
            #     eval_indices = np.random.choice(len(x_train), size=eval_count, replace=False)
            #     eval_tensor_x = torch.tensor(x_train[eval_indices], dtype=torch.float32, device=device)
            #     eval_tensor_y = torch.tensor(y_train[eval_indices], dtype=torch.float32, device=device)
            #     temp_ei_value, temp_ei_term1, temp_ei_term2 = [], [], []
            #     for scale_id in range(0, 5):
            #         ei_tuple, _, _, _ = model.calc_EIs_kde1(
            #             eval_tensor_x,
            #             eval_tensor_y,
            #             eval_count,
            #             mae,
            #             mse_raw,
            #             L=1,
            #             bigL=1,
            #             scale_id=scale_id,
            #             device=device,
            #             ei_samples=ei_samples,
            #         )
            #         temp_ei_value.append(float(ei_tuple[0][0]))
            #         temp_ei_term1.append(float(ei_tuple[0][2]))
            #         temp_ei_term2.append(float(ei_tuple[0][3]))
            #     ei_value.append(temp_ei_value)
            #     ei_term1.append(temp_ei_term1)
            #     ei_term2.append(temp_ei_term2)

        if should_evaluate:
            if int(train_stage) == 1:
                all_scale_loss.append(np.mean(np.array(loss_forward_scale), axis=0))
                
            history.append(
                {
                    "epoch": epoch,
                    "train_mae": train_mae,
                    "train_mse": train_mse,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "pred_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                    "backward_loss": float(np.mean(batch_backward_losses)) if batch_backward_losses else 0.0,
                }
            )
        
        if should_evaluate:
            if np.isfinite(val_mse) and val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(patience):
                    break
        
        if should_evaluate:
            model.load_state_dict(best_state)
            metric_scale = None if int(train_stage) == 1 else scale_id
            test_mae, test_mse = _prediction_metrics(model, x_test, y_test, device, batch_size=batch_size, scale_id=metric_scale)
            
            ei_causal_graph = None
            mean_jacobian_tensor = None
            sigmas_tensor = None
            sigmas_matrix_tensor = None
            ei_expected_log_abs_J = None
            ei_variance_term = None
            jacobian_mean_abs = None
            causal_accuracy = None
            causal_f1 = None
            precision = None
            recall = None
            auc = None
            best_threshold = None
            if int(train_stage) == 2 and causal_graph_mode != "none":
                gt_graph = resolve_causal_ground_truth(causal_graph_mode, model, scale_id)
                causal_eval_states = x_train[:, 0] if x_train.ndim == 3 else x_train
                causal_eval_states_tensor = torch.tensor(causal_eval_states, dtype=torch.float32)
                causal_eval_count = min(len(x_train), max(int(batch_size), int(ei_samples)))
                causal_eval_x = torch.tensor(x_train[:causal_eval_count], dtype=torch.float32, device=device)
                causal_eval_y = torch.tensor(y_train[:causal_eval_count], dtype=torch.float32, device=device)
                sigmas_tensor, sigmas_matrix_tensor, _, _ = model.estimate_sigmas_matrix(
                    causal_eval_x,
                    causal_eval_y,
                    scale_id,
                    mse_raw,
                )
                (
                    ei_causal_graph,
                    mean_jacobian_tensor,
                    ei_expected_log_abs_J,
                    ei_variance_term,
                ) = compute_pairwise_ei_network(
                    model,
                    sigmas_matrix_tensor,
                    scale_id=scale_id,
                    L=1.0,
                    device=device,
                    num_samples=ei_samples,
                )
                jacobian_strength_tensor = mean_jacobian_tensor.abs()
                causal_metrics = evaluate_causal_graph(
                    model,
                    scale_id,
                    causal_eval_states_tensor,
                    device,
                    ground_truth=gt_graph,
                    max_samples=ei_samples,
                    inferred_causal_graph=ei_causal_graph.detach().cpu().numpy(),
                    jacobian_mean_abs=jacobian_strength_tensor.detach().cpu().numpy(),
                )
                jacobian_mean_abs = causal_metrics["jacobian_mean_abs"]
                causal_accuracy = causal_metrics["causal_accuracy"]
                causal_f1 = causal_metrics["causal_f1"]
                precision = causal_metrics["precision"]
                recall = causal_metrics["recall"]
                auc = causal_metrics["auc"]
                best_threshold = causal_metrics["best_threshold"]

            result = {
                "model": model,
                "history": history,
                "test_mse": test_mse,
                "val_mse": best_val_mse,
                "causal_accuracy": causal_accuracy,
                "causal_f1": causal_f1,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "best_threshold": best_threshold,
                "jacobian_mean_abs": jacobian_mean_abs,
                "sigmas": None if sigmas_tensor is None else sigmas_tensor.detach().cpu().numpy(),
                "sigmas_matrix": None if sigmas_matrix_tensor is None else sigmas_matrix_tensor.detach().cpu().numpy(),
                "mean_jacobian": None if mean_jacobian_tensor is None else mean_jacobian_tensor.detach().cpu().numpy(),
                "ei_causal_graph": None if ei_causal_graph is None else ei_causal_graph.detach().cpu().numpy(),
                "ei_expected_log_abs_J": None if ei_expected_log_abs_J is None else ei_expected_log_abs_J.detach().cpu().numpy(),
                "ei_variance_term": None if ei_variance_term is None else ei_variance_term.detach().cpu().numpy(),
                "config": {
                    "hidden_units1": int(hidden_units),
                    "hidden_units2": int(dynamics_hidden_units),
                    "flow_num_layers": int(flow_num_layers),
                    "dynamics_num_layers": int(dynamics_num_layers),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "latent_size": int(min_dim),
                    "batch_size": int(batch_size),
                    "stage1_run_name": stage1_run_name,
                    "stage2_run_name": stage2_run_name,
                    "time_delay": int(time_delay),
                    "train_stage": int(train_stage),
                    "backward_weight": float(backward_weight),
                    "scale_id": int(scale_id),
                    "num_scales": int(len(model.scale_dims)),
                    "scale_dims": ",".join(str(dim) for dim in model.scale_dims),
                    "reduce_dims": ",".join(str(dim) for dim in model.reduce_dims),
                    "group": ",".join(str(size) for size in model.group),
                    "group_schedule": serialize_group_schedule(model.group_schedule),
                    "encoder_type": str(model.encoder_type),
                    "eval_interval": int(eval_interval),
                },
            }
            
            model_dir = stage1_model_dir if int(train_stage) == 1 else stage2_model_dir
            result_dir = os.path.join(
                ".",
                "loc_result_stage1" if int(train_stage) == 1 else "loc_result_stage2",
                stage1_run_name if int(train_stage) == 1 else stage2_run_name,
            )
            _ensure_dir(result_dir)
            
            model_name = "model.pkl" if int(train_stage) == 1 else f"model_scale{scale_id+1}.pkl"
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
            _save_history_files(history, ei_value, ei_term1, ei_term2, np.array(all_scale_loss), result_dir, scale_id, train_stage)
            pd.DataFrame(
                [
                    {
                        "test_mae": test_mae,
                        "test_mse": test_mse,
                        "val_mse": best_val_mse,
                        "causal_accuracy": result["causal_accuracy"],
                        "causal_f1": result["causal_f1"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "auc": result["auc"],
                        "best_threshold": result["best_threshold"],
                        **result["config"],
                    }
                ]
            ).to_csv(os.path.join(result_dir, "summary_scale.csv" if int(train_stage) == 1 else f"summary_scale{scale_id+1}.csv"), index=False)
            if result["jacobian_mean_abs"] is not None:
                pd.DataFrame(result["jacobian_mean_abs"]).to_csv(
                    os.path.join(result_dir, f"jacobian_mean_abs_scale{scale_id+1}.csv"),
                    index=False,
                )
            if result["sigmas"] is not None:
                pd.DataFrame(result["sigmas"]).to_csv(
                    os.path.join(result_dir, f"sigmas_scale{scale_id+1}.csv"),
                    index=False,
                )
            if result["sigmas_matrix"] is not None:
                pd.DataFrame(result["sigmas_matrix"]).to_csv(
                    os.path.join(result_dir, f"sigmas_matrix_scale{scale_id+1}.csv"),
                    index=False,
                )
            if result["mean_jacobian"] is not None:
                pd.DataFrame(result["mean_jacobian"]).to_csv(
                    os.path.join(result_dir, f"mean_jacobian_scale{scale_id+1}.csv"),
                    index=False,
                )
            if result["ei_causal_graph"] is not None:
                pd.DataFrame(result["ei_causal_graph"]).to_csv(
                    os.path.join(result_dir, f"ei_causal_graph_scale{scale_id+1}.csv"),
                    index=False,
                )
            if result["ei_expected_log_abs_J"] is not None:
                pd.DataFrame(result["ei_expected_log_abs_J"]).to_csv(
                    os.path.join(result_dir, f"ei_expected_log_abs_J_scale{scale_id+1}.csv"),
                    index=False,
                )
            if result["ei_variance_term"] is not None:
                pd.DataFrame(result["ei_variance_term"]).to_csv(
                    os.path.join(result_dir, f"ei_variance_term_scale{scale_id+1}.csv"),
                    index=False,
                )

    return result
