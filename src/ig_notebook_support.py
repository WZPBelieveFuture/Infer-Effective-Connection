from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.data_sources import load_array_from_path
from src.models_macro import Parellel_Renorm_Dynamic as MacroParellelRenormDynamic


class CompatibleMacroParellelRenormDynamic(MacroParellelRenormDynamic):
    """Keep older scale-0 checkpoints loadable after the macro API changed."""

    def _build_scale_dims(self, reduce_dims, group=None):
        reduce_dim_schedule = self._resolve_reduce_dims(reduce_dims)
        if not group and len(reduce_dim_schedule) == 1 and int(reduce_dim_schedule[0]) == self.sym_size:
            return (
                [self.sym_size],
                [{"type": "dense", "input_dim": self.sym_size, "output_dim": self.sym_size}],
                reduce_dim_schedule,
            )
        return super()._build_scale_dims(reduce_dims, group)


def load_ig_origin_data(
    data_path: Union[str, Path],
    data_key: str = "data",
    max_initial_points: Optional[int] = None,
) -> np.ndarray:
    data_path = Path(data_path)
    suffix = data_path.suffix.lower()

    if suffix == ".npz":
        with np.load(data_path, allow_pickle=False) as archive:
            selected_key = str(data_key or "data")
            if selected_key not in archive:
                available_keys = ", ".join(sorted(archive.files))
                raise KeyError(
                    f"Array key '{selected_key}' was not found in {data_path}. "
                    f"Available keys: {available_keys}"
                )
            origin_data = np.asarray(archive[selected_key], dtype=np.float32)
    else:
        loaded = load_array_from_path(str(data_path))
        if loaded is None:
            raise FileNotFoundError(f"Could not load origin data from: {data_path}")
        origin_data = np.asarray(loaded, dtype=np.float32)

    if origin_data.ndim == 3 and max_initial_points is not None:
        limit = max(1, min(int(max_initial_points), int(origin_data.shape[0])))
        origin_data = origin_data[:limit]
    return origin_data


def resolve_preferred_device(preferred: Union[str, torch.device] = "cpu") -> str:
    preferred_text = str(preferred)
    if preferred_text == "cpu":
        return preferred_text

    if preferred_text == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if preferred_text.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            requested_index = int(preferred_text.split(":", 1)[1])
        except ValueError:
            return "cuda:0"
        if requested_index < torch.cuda.device_count():
            return preferred_text
        return "cuda:0"

    return preferred_text
