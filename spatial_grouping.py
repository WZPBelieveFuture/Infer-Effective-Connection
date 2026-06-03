import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STAGE_DIR = Path(__file__).resolve().parent / "loc_data_kuramoto" / "des" / "16" / "stage1"
DEFAULT_STAGE_WINDOWS = {
    "stage1": (0, 1000),
    "stage5": (4000, 5000),
    "stage11": (10000, 11000),
}


def _read_csv_matrix(path):
    matrix = pd.read_csv(path, header=None).values
    if matrix.ndim != 2:
        raise ValueError(f"{path} must be a 2D CSV matrix")
    return matrix.astype(np.float32)


def _read_csv_shape(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    num_rows = 0
    num_cols = None
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if num_cols is None:
                num_cols = len(row)
            elif len(row) != num_cols:
                raise ValueError(f"{path} contains rows with inconsistent column counts")
            num_rows += 1

    if num_cols is None:
        return (0, 0)
    return (num_rows, num_cols)


def get_stage_sorted_data_shapes(
    stage_dir=DEFAULT_STAGE_DIR,
    data_sort_name="data_sort.csv",
    data_sort_stage_name="data_sort_stage.csv",
):
    stage_dir = Path(stage_dir)
    data_sort_path = stage_dir / data_sort_name
    data_sort_stage_path = stage_dir / data_sort_stage_name

    return {
        "data_sort_shape": _read_csv_shape(data_sort_path),
        "data_sort_stage_shape": _read_csv_shape(data_sort_stage_path),
        "data_sort_path": str(data_sort_path),
        "data_sort_stage_path": str(data_sort_stage_path),
    }


def _detect_node_axis(data, num_nodes):
    row_match = data.shape[0] == int(num_nodes)
    col_match = data.shape[1] == int(num_nodes)
    if row_match and not col_match:
        return 0
    if col_match and not row_match:
        return 1
    if row_match and col_match:
        return 0
    raise ValueError(
        f"Neither axis of data matches the number of nodes={num_nodes}. Got data shape={data.shape}."
    )


def _kmeans_plus_plus_init(points, k, rng):
    num_points = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=np.float32)
    first_index = int(rng.integers(num_points))
    centers[0] = points[first_index]

    closest_dist_sq = ((points - centers[0]) ** 2).sum(axis=1)
    for center_id in range(1, k):
        total_dist = float(closest_dist_sq.sum())
        if total_dist <= 0:
            centers[center_id:] = points[rng.choice(num_points, size=k - center_id, replace=False)]
            break
        probabilities = closest_dist_sq / total_dist
        next_index = int(rng.choice(num_points, p=probabilities))
        centers[center_id] = points[next_index]
        new_dist_sq = ((points - centers[center_id]) ** 2).sum(axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
    return centers


def _run_kmeans(points, k, random_state=42, n_init=20, max_iter=300, tol=1e-4):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if int(k) < 1 or int(k) > points.shape[0]:
        raise ValueError(f"k must be in [1, {points.shape[0]}], but got {k}")

    base_rng = np.random.default_rng(int(random_state))
    best_labels = None
    best_centers = None
    best_inertia = None

    for _ in range(int(n_init)):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        centers = _kmeans_plus_plus_init(points, int(k), rng)

        for _ in range(int(max_iter)):
            distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for group_id in range(int(k)):
                mask = labels == group_id
                if np.any(mask):
                    new_centers[group_id] = points[mask].mean(axis=0)
                else:
                    new_centers[group_id] = points[int(rng.integers(points.shape[0]))]

            center_shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
            centers = new_centers
            if float(center_shift) <= float(tol):
                break

        final_distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = final_distances.argmin(axis=1)
        inertia = float(final_distances[np.arange(points.shape[0]), labels].sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels.astype(np.int32), best_centers.astype(np.float32)


def _relabel_groups_by_center(labels, centers):
    centers = np.asarray(centers, dtype=np.float32)
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    remapped = np.empty_like(labels)
    remapped_centers = np.empty_like(centers)
    for new_label, old_label in enumerate(order):
        remapped[labels == old_label] = new_label
        remapped_centers[new_label] = centers[old_label]
    return remapped.astype(np.int32), remapped_centers.astype(np.float32)


def _slice_time_window(data, node_axis, start_col, end_col):
    if int(start_col) < 0 or int(end_col) <= int(start_col):
        raise ValueError(f"Invalid slice range [{start_col}, {end_col})")

    if node_axis == 0:
        if int(end_col) > data.shape[1]:
            raise ValueError(f"Slice range [{start_col}, {end_col}) exceeds data columns={data.shape[1]}")
        return data[:, int(start_col):int(end_col)]

    if int(end_col) > data.shape[0]:
        raise ValueError(f"Slice range [{start_col}, {end_col}) exceeds data rows={data.shape[0]}")
    return data[int(start_col):int(end_col), :]


def _format_stage_slice(stage_slice, node_axis):
    # Save stage slices in time-major format: [time_steps, num_nodes].
    if node_axis == 0:
        return stage_slice.T
    return stage_slice


def _save_stage_slices(data_sorted, node_axis, stage_root_dir, data_sort_stage_name):
    stage_root_dir = Path(stage_root_dir)
    stage_outputs = {}

    for stage_name, (start_col, end_col) in DEFAULT_STAGE_WINDOWS.items():
        target_dir = stage_root_dir / stage_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / data_sort_stage_name
        stage_slice = _slice_time_window(data_sorted, node_axis, start_col, end_col)
        stage_slice = _format_stage_slice(stage_slice, node_axis)
        pd.DataFrame(stage_slice).to_csv(target_path, header=False, index=False)
        stage_outputs[stage_name] = str(target_path)

    return stage_outputs


def group_nodes_by_kmeans(
    mean_loc_path,
    data_path,
    k,
    data_sort_path=None,
    data_sort_stage_name="data_sort_stage.csv",
    groud_id_path=None,
    group_path=None,
    random_state=42,
    n_init=20,
    max_iter=300,
):
    mean_loc_path = Path(mean_loc_path)
    data_path = Path(data_path)

    coords = _read_csv_matrix(mean_loc_path)
    data = _read_csv_matrix(data_path)
    if coords.shape[1] < 2:
        raise ValueError(f"{mean_loc_path} must contain at least two coordinate columns")

    num_nodes = coords.shape[0]
    node_axis = _detect_node_axis(data, num_nodes)
    labels, centers = _run_kmeans(
        coords[:, :2],
        int(k),
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels, centers = _relabel_groups_by_center(labels, centers)

    sort_index = np.argsort(labels, kind="stable")
    group_sizes = np.bincount(labels, minlength=int(k)).astype(np.int32)

    if node_axis == 0:
        data_sorted = data[sort_index, :]
    else:
        data_sorted = data[:, sort_index]

    stage_dir = data_path.parent
    stage_root_dir = stage_dir.parent
    data_sort_path = Path(data_sort_path) if data_sort_path is not None else stage_dir / "data_sort.csv"
    groud_id_path = Path(groud_id_path) if groud_id_path is not None else stage_dir / "groud_id.csv"
    group_path = Path(group_path) if group_path is not None else stage_dir / "group.csv"

    data_sort_path.parent.mkdir(parents=True, exist_ok=True)
    groud_id_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(data_sorted).to_csv(data_sort_path, header=False, index=False)
    stage_output_paths = _save_stage_slices(
        data_sorted=data_sorted,
        node_axis=node_axis,
        stage_root_dir=stage_root_dir,
        data_sort_stage_name=data_sort_stage_name,
    )

    sorted_position = np.empty(num_nodes, dtype=np.int32)
    sorted_position[sort_index] = np.arange(num_nodes, dtype=np.int32)
    mapping_df = pd.DataFrame(
        {
            "node_id": np.arange(num_nodes, dtype=np.int32),
            "group_id": labels,
            "sorted_position": sorted_position,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    ).sort_values(["group_id", "sorted_position", "node_id"])
    mapping_df.to_csv(groud_id_path, index=False)

    pd.DataFrame(group_sizes).to_csv(group_path, header=False, index=False)

    return {
        "labels": labels,
        "centers": centers,
        "sort_index": sort_index,
        "group_sizes": group_sizes,
        "node_axis": node_axis,
        "data_sort_path": str(data_sort_path),
        "data_sort_stage_paths": stage_output_paths,
        "groud_id_path": str(groud_id_path),
        "group_path": str(group_path),
    }


def group_stage_files(
    stage_dir,
    k,
    data_name="data.csv",
    mean_loc_name="mean_loc.csv",
    data_sort_name="data_sort.csv",
    data_sort_stage_name="data_sort_stage.csv",
    groud_id_name="groud_id.csv",
    group_name="group.csv",
    random_state=42,
):
    stage_dir = Path(stage_dir)
    return group_nodes_by_kmeans(
        mean_loc_path=stage_dir / mean_loc_name,
        data_path=stage_dir / data_name,
        k=k,
        data_sort_path=stage_dir / data_sort_name,
        data_sort_stage_name=data_sort_stage_name,
        groud_id_path=stage_dir / groud_id_name,
        group_path=stage_dir / group_name,
        random_state=random_state,
    )


def main():
    parser = argparse.ArgumentParser(description="Group node coordinates with k-means and sort node time series.")
    parser.add_argument("--stage_dir", type=str, default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--data_name", type=str, default="data.csv")
    parser.add_argument("--mean_loc_name", type=str, default="mean_loc.csv")
    parser.add_argument("--data_sort_name", type=str, default="data_sort.csv")
    parser.add_argument("--data_sort_stage_name", type=str, default="data_sort_stage.csv")
    parser.add_argument("--groud_id_name", type=str, default="groud_id.csv")
    parser.add_argument("--group_name", type=str, default="group.csv")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    result = group_stage_files(
        stage_dir=args.stage_dir,
        k=args.k,
        data_name=args.data_name,
        mean_loc_name=args.mean_loc_name,
        data_sort_name=args.data_sort_name,
        data_sort_stage_name=args.data_sort_stage_name,
        groud_id_name=args.groud_id_name,
        group_name=args.group_name,
        random_state=args.random_state,
    )

    print(f"Saved sorted data to: {result['data_sort_path']}")
    for stage_name, stage_path in result["data_sort_stage_paths"].items():
        print(f"Saved {stage_name} stage data to: {stage_path}")
    print(f"Saved node-group mapping to: {result['groud_id_path']}")
    print(f"Saved group sizes to: {result['group_path']}")


if __name__ == "__main__":
    main()
