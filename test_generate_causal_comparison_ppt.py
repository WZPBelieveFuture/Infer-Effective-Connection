import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_causal_comparison_ppt import (
    load_ground_truth_matrix,
    load_square_matrix_npz,
    load_var_ground_truth_matrix,
)


class VarGroundTruthLoadingTest(unittest.TestCase):
    def test_load_square_matrix_npz_reads_var_causal_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path = Path(temp_dir) / "generated_data.npz"
            expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.float32)
            np.savez(npz_path, data=np.zeros((5, 3), dtype=np.float32), causal_matrix=expected)

            loaded = load_square_matrix_npz(npz_path, "causal_matrix", "var_ground_truth")

            np.testing.assert_array_equal(loaded, expected)

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


if __name__ == "__main__":
    unittest.main()
