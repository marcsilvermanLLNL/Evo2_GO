from __future__ import annotations

import unittest

import numpy as np
import torch

from evo2_go.model.training_utils import cafa_gene_averaged_fmax
from evo2_go.training.evaluate import (
    GO_ROOT_IDS,
    StreamingMetricAccumulator,
    evaluation_threshold_grid,
    metrics_at_threshold,
    per_term_counts,
    streaming_term_indices,
    streaming_threshold_rows,
)


class StreamingEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32)
        self.probabilities = torch.tensor(
            [
                [0.20, 0.10, 0.90, 0.50, 0.30],
                [0.81, 0.50, 0.49, 0.10, 0.80],
                [0.05, 0.25, 0.55, 0.85, 0.15],
                [0.95, 0.75, 0.50, 0.20, 0.01],
                [0.33, 0.67, 0.79, 0.80, 0.50],
                [0.11, 0.22, 0.44, 0.66, 0.88],
                [0.90, 0.70, 0.40, 0.30, 0.20],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            [
                [1, 0, 1, 0, 0],
                [1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

    def build_accumulator(self) -> StreamingMetricAccumulator:
        accumulator = StreamingMetricAccumulator(
            thresholds=self.thresholds,
            n_terms=self.probabilities.shape[1],
            calibration_bins=5,
        )
        accumulator.update(self.probabilities[:3], self.targets[:3])
        accumulator.update(self.probabilities[3:], self.targets[3:])
        return accumulator

    def test_fmax_matches_dense_implementation_across_batches(self) -> None:
        streaming = self.build_accumulator().fmax_result()
        dense = cafa_gene_averaged_fmax(
            self.probabilities,
            self.targets,
            thresholds=self.thresholds,
        )
        self.assertAlmostEqual(streaming.fmax, dense.fmax, places=6)
        self.assertAlmostEqual(streaming.threshold, dense.threshold, places=7)
        np.testing.assert_allclose(
            streaming.precision_by_threshold,
            dense.precision_by_threshold,
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            streaming.recall_by_threshold,
            dense.recall_by_threshold,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_per_term_and_micro_counts_match_dense_computation(self) -> None:
        accumulator = self.build_accumulator()
        counts = accumulator.threshold_results()
        threshold_rows, _ = streaming_threshold_rows(accumulator)
        labels = self.targets.to(dtype=torch.bool)

        for threshold_index, threshold in enumerate(self.thresholds):
            tp, fp, fn = per_term_counts(
                self.probabilities,
                self.targets,
                float(threshold),
            )
            np.testing.assert_array_equal(counts.tp[threshold_index], tp.numpy())
            np.testing.assert_array_equal(counts.fp[threshold_index], fp.numpy())
            np.testing.assert_array_equal(counts.fn[threshold_index], fn.numpy())

            predicted = self.probabilities >= threshold
            tn = (~predicted & ~labels).sum(dim=0)
            np.testing.assert_array_equal(counts.tn[threshold_index], tn.numpy())

            total_tp = float(tp.sum())
            total_fp = float(fp.sum())
            expected_micro_precision = total_tp / (total_tp + total_fp)
            self.assertAlmostEqual(
                threshold_rows[threshold_index]["micro_precision"],
                expected_micro_precision,
                places=7,
            )

    def test_calibration_sums_and_brier_are_exact(self) -> None:
        accumulator = self.build_accumulator()
        self.assertEqual(
            int(accumulator.calibration_count.sum()), self.probabilities.numel()
        )
        expected_brier_sum = float(
            torch.sum((self.probabilities - self.targets) ** 2).item()
        )
        self.assertAlmostEqual(
            accumulator.brier_squared_error_sum,
            expected_brier_sum,
            places=6,
        )

    def test_go_roots_are_removed_by_model_index(self) -> None:
        rows = [
            {"go_index": "0", "GO_ID": "GO:1234567"},
            {"go_index": "1", "GO_ID": "GO:0008150"},
            {"go_index": "2", "GO_ID": "GO:7654321"},
            {"go_index": "3", "GO_ID": "GO:0003674"},
            {"go_index": "4", "GO_ID": "GO:0005575"},
        ]
        indices, retained, excluded = streaming_term_indices(
            rows, exclude_go_roots=True
        )
        np.testing.assert_array_equal(indices, np.array([0, 2]))
        self.assertEqual(
            [row["GO_ID"] for row in retained], ["GO:1234567", "GO:7654321"]
        )
        self.assertEqual(set(excluded), set(GO_ROOT_IDS))

    def test_prespecified_operating_threshold_is_added_to_grid(self) -> None:
        thresholds = evaluation_threshold_grid(100, 0.61)
        self.assertTrue(torch.any(torch.isclose(thresholds, torch.tensor(0.61))))
        self.assertTrue(torch.all(thresholds[1:] > thresholds[:-1]))

        result = cafa_gene_averaged_fmax(
            self.probabilities,
            self.targets,
            thresholds=thresholds,
        )
        metrics = metrics_at_threshold(result, 0.61)
        self.assertAlmostEqual(metrics["threshold"], 0.61, places=6)


if __name__ == "__main__":
    unittest.main()
