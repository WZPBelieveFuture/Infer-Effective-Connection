import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_causal_comparison_ppt import (
    compute_threshold_metrics,
    load_ground_truth_matrix,
    load_square_matrix_npz,
    load_var_ground_truth_matrix,
    resolve_real_fmri_network_path,
)
from compute_var_transport_ei import aggregate_group_transport_ei


class VarGroundTruthLoadingTest(unittest.TestCase):
    def test_load_square_matrix_npz_reads_var_causal_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "generated_data.npz"
            expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.float32)
            np.savez(npz_path, data=np.zeros((5, 3), dtype=np.float32), causal_matrix=expected)

            loaded = load_square_matrix_npz(npz_path, "causal_matrix", "var_ground_truth")

            np.testing.assert_array_equal(loaded, expected)


class TransportMapEITest(unittest.TestCase):
    def test_aggregate_group_transport_ei_outputs_nonnegative_ei(self):
        transport_jacobians = np.full((3, 2, 3), 0.01, dtype=np.float32)
        group_sizes = np.array([2, 1], dtype=np.int64)
        sigma_diag = np.array([0.1, 0.1], dtype=np.float32)

        transport_ei, _, _, _ = aggregate_group_transport_ei(
            transport_jacobians,
            group_sizes=group_sizes,
            sigma_diag=sigma_diag,
            L=1.0,
            eps=1e-12,
        )

        self.assertGreaterEqual(float(transport_ei.min()), 0.0)

    def test_transport_ei_keeps_jacobian_separable_edges_separable(self):
        transport_jacobians = np.array(
            [
                [
                    [0.20, 0.10],
                    [0.19, 0.20],
                ],
                [
                    [0.20, 0.10],
                    [0.19, 0.20],
                ],
            ],
            dtype=np.float32,
        )
        group_sizes = np.array([1, 1], dtype=np.int64)
        sigma_diag = np.array([100.0, 0.001], dtype=np.float32)
        ground_truth = np.eye(2, dtype=np.float32)

        transport_ei, mean_group_gain, _, _ = aggregate_group_transport_ei(
            transport_jacobians,
            group_sizes=group_sizes,
            sigma_diag=sigma_diag,
            L=1.0,
            eps=1e-12,
        )

        jacobian_metrics = compute_threshold_metrics(mean_group_gain, ground_truth)
        ei_metrics = compute_threshold_metrics(transport_ei, ground_truth)
        self.assertGreaterEqual(ei_metrics["f1"], jacobian_metrics["f1"])

    def test_load_square_matrix_npz_requires_square_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "generated_data.npz"
            np.savez(npz_path, causal_matrix=np.zeros((2, 3), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "square matrix"):
                load_square_matrix_npz(npz_path, "causal_matrix", "var_ground_truth")

    def test_load_ground_truth_matrix_dispatches_npz_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "generated_data.npz"
            expected = np.eye(2, dtype=np.float32)
            np.savez(npz_path, causal_matrix=expected)

            loaded = load_ground_truth_matrix(npz_path, "var_ground_truth", "causal_matrix")

            np.testing.assert_array_equal(loaded, expected)

    def test_load_var_ground_truth_matrix_expands_group_graph_to_observed_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "generated_data.npz"
            group_graph = np.array([[1, 0], [1, 1]], dtype=np.float32)
            group_sizes = np.array([2, 1], dtype=np.int32)
            np.savez(npz_path, causal_matrix=group_graph, group=group_sizes)

            loaded = load_var_ground_truth_matrix(npz_path, (3, 3), "causal_matrix")

            expected = np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(loaded, expected)


class RealFmriNetworkPathTest(unittest.TestCase):
    def test_resolve_real_fmri_network_path_auto_prefers_ei(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir)
            ei_path = result_root / "ei_causal_graph_scale1.csv"
            jacobian_path = result_root / "jacobian_mean_abs_scale1.csv"
            ei_path.write_text("0.1\n", encoding="utf-8")
            jacobian_path.write_text("0.2\n", encoding="utf-8")

            resolved_path, resolved_kind = resolve_real_fmri_network_path(result_root, 1, "auto")

            self.assertEqual(resolved_path, ei_path)
            self.assertEqual(resolved_kind, "ei")

    def test_resolve_real_fmri_network_path_can_select_jacobian(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir)
            jacobian_path = result_root / "jacobian_mean_abs_scale1.csv"
            jacobian_path.write_text("0.2\n", encoding="utf-8")

            resolved_path, resolved_kind = resolve_real_fmri_network_path(result_root, 1, "jacobian")

            self.assertEqual(resolved_path, jacobian_path)
            self.assertEqual(resolved_kind, "jacobian")


if __name__ == "__main__":
    unittest.main()
