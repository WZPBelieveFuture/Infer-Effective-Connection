from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


CouplingStrength = Union[float, Sequence[float]]
ORDER_PARAMETER_TAIL_STEPS = 100
MIN_GROUP_ORDER_PARAMETER = 0.8
MAX_GLOBAL_ORDER_PARAMETER = 0.2
MAX_SERIES_GENERATION_ATTEMPTS = 1000


@dataclass
class KuramotoConfig:
    sz: int = 32
    groups: int = 2
    num_series: int = 64
    time_steps: int = 100
    dt: float = 0.01
    sample_interval: int = 1
    intra_coupling_strength: CouplingStrength = 2.0
    inter_coupling_strength: CouplingStrength = 0.3
    noise_level: float = 1.0
    include_order_parameters: bool = False


def _normalize_coupling_range(coupling_strength: CouplingStrength) -> Tuple[float, float]:
    """Convert a scalar or length-2 iterable into a valid sampling range."""
    if np.isscalar(coupling_strength):
        value = float(coupling_strength)
        return value, value

    if len(coupling_strength) != 2:
        raise ValueError("coupling_strength must be a scalar or a length-2 range.")

    low, high = map(float, coupling_strength)
    if low > high:
        raise ValueError("coupling_strength range must satisfy min <= max.")
    return low, high


def _sample_symmetric_matrix(size: int, coupling_range: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Sample a symmetric matrix from the given coupling range."""
    low, high = coupling_range
    if low == high:
        matrix = np.full((size, size), low, dtype=np.float32)
    else:
        upper = rng.uniform(low, high, size=(size, size)).astype(np.float32)
        matrix = np.triu(upper, k=1)
        matrix = matrix + matrix.T
    return matrix.astype(np.float32)


def build_kuramoto_matrices(
    sz,
    groups,
    intra_coupling_strength,
    inter_coupling_strength,
    rng: Optional[np.random.Generator] = None,
):
    sz = int(sz)
    groups = int(groups)
    if sz < 1:
        raise ValueError("sz must be >= 1")
    if groups < 1:
        raise ValueError("groups must be >= 1")

    rng = np.random.default_rng() if rng is None else rng
    intra_coupling_range = _normalize_coupling_range(intra_coupling_strength)
    inter_coupling_range = _normalize_coupling_range(inter_coupling_strength)

    obj_matrix = _sample_symmetric_matrix(sz, inter_coupling_range, rng)
    group_matrix = np.zeros((sz, groups), dtype=np.float32)
    base_group_size = sz // groups
    remainder = sz % groups
    start = 0

    for group_id in range(groups):
        group_size = base_group_size + (1 if group_id < remainder else 0)
        end = start + group_size
        obj_matrix[start:end, start:end] = _sample_symmetric_matrix(group_size, intra_coupling_range, rng)
        group_matrix[start:end, group_id] = 1.0
        start = end

    np.fill_diagonal(obj_matrix, 0.0)
    return obj_matrix, group_matrix


def _get_group_oscillator_indices(group_matrix: np.ndarray) -> List[np.ndarray]:
    group_matrix = np.asarray(group_matrix, dtype=np.float32)
    if group_matrix.ndim != 2:
        raise ValueError(f"group_matrix must have shape [N, G], but got {group_matrix.shape}")

    indices_per_group = []
    for group_idx in range(group_matrix.shape[1]):
        oscillator_indices = np.flatnonzero(group_matrix[:, group_idx] > 0.5)
        if oscillator_indices.size > 0:
            indices_per_group.append(oscillator_indices)
    return indices_per_group


def _compute_order_parameter_curves(
    theta_series: np.ndarray,
    group_matrix: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    theta_series = np.asarray(theta_series, dtype=np.float32)
    if theta_series.ndim != 2:
        raise ValueError(f"theta_series must have shape [T, N], but got {theta_series.shape}")

    phase = np.exp(1j * theta_series.astype(np.float64))
    global_r = np.abs(np.mean(phase, axis=1))

    group_curves = []
    for oscillator_indices in _get_group_oscillator_indices(group_matrix):
        group_phase = phase[:, oscillator_indices]
        group_r = np.abs(np.mean(group_phase, axis=1))
        group_curves.append(group_r)

    return global_r, group_curves


def _summarize_tail_order_parameters(
    theta_series: np.ndarray,
    group_matrix: np.ndarray,
    tail_steps: int = ORDER_PARAMETER_TAIL_STEPS,
) -> Tuple[float, List[float]]:
    theta_series = np.asarray(theta_series, dtype=np.float32)
    if theta_series.ndim != 2:
        raise ValueError(f"theta_series must have shape [T, N], but got {theta_series.shape}")
    if theta_series.shape[0] == 0:
        return float("nan"), []

    tail_steps = max(1, int(tail_steps))
    theta_tail = theta_series[-min(theta_series.shape[0], tail_steps) :]
    global_r, group_curves = _compute_order_parameter_curves(theta_tail, group_matrix)
    return float(np.mean(global_r)), [float(np.mean(group_r)) for group_r in group_curves]


def _satisfies_order_parameter_constraints(
    theta_series: np.ndarray,
    group_matrix: np.ndarray,
    tail_steps: int = ORDER_PARAMETER_TAIL_STEPS,
    min_group_r: float = MIN_GROUP_ORDER_PARAMETER,
    max_global_r: float = MAX_GLOBAL_ORDER_PARAMETER,
) -> Tuple[bool, float, List[float]]:
    global_r_mean, group_r_means = _summarize_tail_order_parameters(
        theta_series=theta_series,
        group_matrix=group_matrix,
        tail_steps=tail_steps,
    )
    has_valid_groups = bool(group_r_means) and all(group_r > float(min_group_r) for group_r in group_r_means)
    has_valid_global = bool(np.isfinite(global_r_mean)) and global_r_mean < float(max_global_r)
    return has_valid_groups and has_valid_global, global_r_mean, group_r_means


def _simulate_single_series(
    obj_matrix,
    group_matrix,
    omegas,
    time_steps,
    dt,
    sample_interval,
    noise_level,
    rng,
    include_order_parameters,
):  
    sz = int(obj_matrix.shape[0])
    groups = int(group_matrix.shape[1])
    feature_dim = sz + (2 * groups if include_order_parameters else 0)
    theta_min = 0.0
    theta_max = 2.0 * np.pi
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=sz).astype(np.float32)
    trajectory = []
    theta_history = []
    
    phase_obs = thetas.astype(np.float32).copy()
    if include_order_parameters:
        group_order = (np.exp(1j * thetas.astype(np.float64)) @ group_matrix) * groups / sz
        obs = np.concatenate(
            [
                phase_obs,
                group_order.real.astype(np.float32),
                group_order.imag.astype(np.float32),
            ]
        )
    else:
        obs = phase_obs
    trajectory.append(obs.astype(np.float32))
    theta_history.append(thetas.astype(np.float32).copy())
    
    for step in range(int(time_steps)):
        phase_diff = thetas[None, :] - thetas[:, None]
        coupling = np.sum(obj_matrix * np.sin(phase_diff), axis=1) / sz
        noise = rng.standard_normal(sz).astype(np.float32) * float(noise_level)
        thetas = thetas + float(dt) * (omegas + coupling + noise)

        # if np.any((thetas < theta_min) | (thetas > theta_max)):
        #     break

        if (step + 1) % int(sample_interval) == 0:
            phase_obs = thetas.astype(np.float32).copy()
            if include_order_parameters:
                group_order = (np.exp(1j * thetas.astype(np.float64)) @ group_matrix) * groups / sz
                obs = np.concatenate(
                    [
                        phase_obs,
                        group_order.real.astype(np.float32),
                        group_order.imag.astype(np.float32),
                    ]
                )
            else:
                obs = phase_obs
            trajectory.append(obs.astype(np.float32))
            theta_history.append(thetas.astype(np.float32).copy())
    
    if trajectory:
        trajectory_array = np.asarray(trajectory, dtype=np.float32)
        mean = np.mean(trajectory_array, axis=0, keepdims=True)
        std = np.std(trajectory_array, axis=0, keepdims=True)
        std = np.where(std > 0.0, std, 1.0).astype(np.float32)
        trajectory_array = ((trajectory_array - mean) / std).astype(np.float32)
    else:
        trajectory_array = np.empty((0, feature_dim), dtype=np.float32)

    if theta_history:
        theta_history_array = np.asarray(theta_history, dtype=np.float32)
    else:
        theta_history_array = np.empty((0, sz), dtype=np.float32)

    return trajectory_array, theta_history_array


def generate_kuramoto_data(config):
    if not isinstance(config, KuramotoConfig):
        config_dict = dict(config)
        config_dict.pop("seed", None)
        config = KuramotoConfig(**config_dict)

    if int(config.time_steps) < 2:
        raise ValueError("time_steps must be >= 2")
    if int(config.sample_interval) < 1:
        raise ValueError("sample_interval must be >= 1")
    if int(config.num_series) < 1:
        raise ValueError("num_series must be >= 1")
    
    trajectories = []
    theta_trajectories = []
    accepted_group_r_means = []
    accepted_global_r_means = []
    filtered_zero_length = 0
    filtered_order_parameter = 0
    total_generation_attempts = 0
    obj_matrix = None
    group_matrix = None
    omegas = None

    for series_idx in range(int(config.num_series)):
        accepted = False
        for _ in range(MAX_SERIES_GENERATION_ATTEMPTS):
            total_generation_attempts += 1
            rng = np.random.default_rng()
            obj_matrix, group_matrix = build_kuramoto_matrices(
                config.sz,
                config.groups,
                config.intra_coupling_strength,
                config.inter_coupling_strength,
                rng=rng,
            )
            omegas = rng.standard_normal(int(config.sz)).astype(np.float32)

            trajectory, theta_history = _simulate_single_series(
                obj_matrix=obj_matrix,
                group_matrix=group_matrix,
                omegas=omegas,
                time_steps=config.time_steps,
                dt=config.dt,
                sample_interval=config.sample_interval,
                noise_level=config.noise_level,
                rng=rng,
                include_order_parameters=config.include_order_parameters,
            )
            if trajectory.shape[0] == 0:
                filtered_zero_length += 1
                continue
            
            is_valid, global_r_mean, group_r_means = _satisfies_order_parameter_constraints(
                theta_series=theta_history,
                group_matrix=group_matrix,
                tail_steps=ORDER_PARAMETER_TAIL_STEPS,
                min_group_r=MIN_GROUP_ORDER_PARAMETER,
                max_global_r=MAX_GLOBAL_ORDER_PARAMETER,
            )
            if not is_valid:
                filtered_order_parameter += 1
                continue
            
            trajectories.append(trajectory[::5])
            theta_trajectories.append(theta_history[::5])
            accepted_global_r_means.append(global_r_mean)
            accepted_group_r_means.append(group_r_means)
            accepted = True
            break
        
        if not accepted:
            raise ValueError(
                f"Failed to generate Kuramoto series {series_idx + 1}/{int(config.num_series)} "
                f"that satisfies the order-parameter constraints after "
                f"{MAX_SERIES_GENERATION_ATTEMPTS} attempts."
            )
    
    valid_series = list(zip(trajectories, theta_trajectories))
    if not valid_series or obj_matrix is None or group_matrix is None or omegas is None:
        raise ValueError(
            "No valid Kuramoto series were generated. "
            "Try adjusting dt, noise_level, or coupling strengths."
        )

    series_lengths = np.asarray(
        [trajectory.shape[0] for trajectory, _ in valid_series],
        dtype=np.int32,
    )
    max_steps = int(series_lengths.max())
    feature_dim = int(valid_series[0][0].shape[1])
    theta_dim = int(valid_series[0][1].shape[1])

    data = np.full((len(valid_series), max_steps, feature_dim), np.nan, dtype=np.float32)
    theta_data = np.full((len(valid_series), max_steps, theta_dim), np.nan, dtype=np.float32)
    for idx, (trajectory, theta_history) in enumerate(valid_series):
        steps = trajectory.shape[0]
        data[idx, :steps, :] = trajectory
        theta_data[idx, :steps, :] = theta_history
    
    metadata = {
        "config": asdict(config),
        "obj_matrix": obj_matrix,
        "group_matrix": group_matrix,
        "omegas": omegas,
        "theta_data": theta_data,
        "series_lengths": series_lengths,
        "num_requested_series": int(config.num_series),
        "num_retained_series": len(valid_series),
        "num_filtered_zero_length_series": filtered_zero_length,
        "num_filtered_order_parameter_series": filtered_order_parameter,
        "total_generation_attempts": total_generation_attempts,
        "mean_generation_attempts_per_retained_series": total_generation_attempts / float(len(valid_series)),
        "order_parameter_tail_steps": ORDER_PARAMETER_TAIL_STEPS,
        "min_group_order_parameter": MIN_GROUP_ORDER_PARAMETER,
        "max_global_order_parameter": MAX_GLOBAL_ORDER_PARAMETER,
        "accepted_global_order_parameter_means": np.asarray(accepted_global_r_means, dtype=np.float32),
        "accepted_group_order_parameter_means": np.asarray(accepted_group_r_means, dtype=np.float32),
        "min_retained_steps": int(series_lengths.min()),
        "max_retained_steps": max_steps,
    }
    return data, metadata
