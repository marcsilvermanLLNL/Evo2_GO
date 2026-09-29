#!/usr/bin/env python3
"""Evaluate an Evo2-GO checkpoint on one split without training."""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evo2_go.model.architecture import (
    DualEncoderConfig,
    DualEncoderModel,
    GoTermGCNConfig,
    SequenceEncoderConfig,
    load_sparse_adjacency_npz,
)
from evo2_go.model.dataset import (
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_GO_VOCABULARY,
    Evo2GoEmbeddingDataset,
    clean_field,
    evo2_go_collate_fn,
    load_evo2_go_dataset,
    tax_ids_from_file,
)
from evo2_go.model.training_utils import (
    FmaxResult,
    cafa_gene_averaged_fmax,
    load_training_checkpoint,
)
from evo2_go.training.config import defaults_from_config, resolve_project_root


DEFAULT_ADJACENCY = Path(
    "outputs/model_vocab/go_adjacency_non_iea_min_count_10/go_adjacency_sparse.npz"
)
DEFAULT_OUTPUT_DIR = Path("outputs/model_training/non_iea_min_count_10/evaluation")
GO_ROOT_IDS = frozenset({"GO:0003674", "GO:0005575", "GO:0008150"})


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    tmp_path.replace(path)


def summarize_tax_ids(tax_ids: list[str] | None) -> str:
    if not tax_ids:
        return ""
    preview = ",".join(tax_ids[:10])
    suffix = ",..." if len(tax_ids) > 10 else ""
    return f"{len(tax_ids)} values: {preview}{suffix}"


def write_threshold_curve(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("threshold", "precision", "recall", "f1"))
        for threshold, precision, recall, f1 in zip(
            result.thresholds,
            result.precision_by_threshold,
            result.recall_by_threshold,
            result.f1_by_threshold,
            strict=True,
        ):
            writer.writerow(
                (
                    f"{threshold:.8f}",
                    f"{precision:.8f}",
                    f"{recall:.8f}",
                    f"{f1:.8f}",
                )
            )
    tmp_path.replace(path)


def safe_metric(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def evaluation_threshold_grid(
    threshold_steps: int,
    operating_threshold: float | None = None,
) -> torch.Tensor:
    """Build the Fmax grid and include an exact prespecified operating point."""
    values = np.linspace(0.01, 0.99, threshold_steps, dtype=np.float32)
    if operating_threshold is not None:
        if not 0.0 <= operating_threshold <= 1.0:
            raise ValueError("--operating-threshold must be between 0 and 1")
        candidate = np.float32(operating_threshold)
        if not np.any(np.isclose(values, candidate, rtol=0.0, atol=1e-7)):
            values = np.sort(np.append(values, candidate)).astype(np.float32)
    return torch.from_numpy(values)


def metrics_at_threshold(result: FmaxResult, threshold: float) -> dict[str, float]:
    """Return gene-averaged metrics at an exact threshold in an Fmax result."""
    thresholds = np.asarray(result.thresholds, dtype=np.float64)
    index = int(np.argmin(np.abs(thresholds - threshold)))
    if not np.isclose(thresholds[index], threshold, rtol=0.0, atol=1e-6):
        raise ValueError(f"Operating threshold {threshold} is absent from the metric grid")
    return {
        "threshold": float(thresholds[index]),
        "precision": float(result.precision_by_threshold[index]),
        "recall": float(result.recall_by_threshold[index]),
        "f1": float(result.f1_by_threshold[index]),
    }


def load_vocabulary_rows(path: Path, n_terms: int) -> list[dict[str, str]]:
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    if len(rows) != n_terms:
        raise ValueError(
            f"Vocabulary row count mismatch: {len(rows)} rows in {path}, "
            f"but model has {n_terms} terms"
        )
    if "go_index" in (reader.fieldnames or []):
        rows.sort(key=lambda row: int(row["go_index"]))
    else:
        for index, row in enumerate(rows):
            row["go_index"] = str(index)
    return rows


def per_term_counts(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_bool = targets.to(dtype=torch.bool)
    predicted = probabilities >= threshold
    tp = (predicted & target_bool).sum(dim=0)
    fp = (predicted & ~target_bool).sum(dim=0)
    fn = (~predicted & target_bool).sum(dim=0)
    return tp, fp, fn


def counts_to_precision_recall_f1(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp_float = tp.to(dtype=torch.float64)
    fp_float = fp.to(dtype=torch.float64)
    fn_float = fn.to(dtype=torch.float64)
    precision_den = tp_float + fp_float
    recall_den = tp_float + fn_float
    f1_den = (2.0 * tp_float) + fp_float + fn_float
    precision = torch.where(
        precision_den > 0,
        tp_float / precision_den.clamp_min(1.0),
        torch.full_like(tp_float, float("nan")),
    )
    recall = torch.where(
        recall_den > 0,
        tp_float / recall_den.clamp_min(1.0),
        torch.full_like(tp_float, float("nan")),
    )
    f1 = torch.where(
        f1_den > 0,
        (2.0 * tp_float) / f1_den.clamp_min(1.0),
        torch.full_like(tp_float, float("nan")),
    )
    return precision, recall, f1


def finite_float(value: float) -> str:
    """Format finite float metrics while preserving NaN for undefined cases."""

    return f"{value:.8f}" if np.isfinite(value) else "nan"


def nan_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def auc_trapezoid(x_values: np.ndarray, y_values: np.ndarray) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2:
        return float("nan")
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    order = np.argsort(x_valid)
    return float(np.trapezoid(y_valid[order], x_valid[order]))


def binary_auc_average_precision_by_rows(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute row-wise AUROC and average precision from dense binary labels.

    Rows can represent genes scored across GO terms, or GO terms scored across
    genes. Undefined AUROC/AP values are returned as NaN when a row lacks
    positives, negatives, or both. Exact tie handling is not material for the
    current neural-network scores, where tied probabilities are rare.
    """

    if scores.shape != labels.shape:
        raise ValueError(
            f"Score/label shape mismatch: scores={scores.shape}, labels={labels.shape}"
        )
    if scores.ndim != 2:
        raise ValueError("Score-based metrics require a 2D matrix")

    n_rows, n_columns = scores.shape
    ranks = np.arange(1, n_columns + 1, dtype=np.float32)
    auroc = np.full(n_rows, np.nan, dtype=np.float32)
    average_precision = np.full(n_rows, np.nan, dtype=np.float32)
    labels_bool = labels.astype(bool, copy=False)

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        score_chunk = scores[start:stop]
        label_chunk = labels_bool[start:stop]
        positive_counts = label_chunk.sum(axis=1).astype(np.float32)
        negative_counts = (n_columns - positive_counts).astype(np.float32)

        order = np.argsort(-score_chunk, axis=1)
        sorted_labels = np.take_along_axis(label_chunk, order, axis=1)
        cumulative_positives = np.cumsum(sorted_labels, axis=1, dtype=np.float32)

        valid_ap = positive_counts > 0
        precision_at_rank = cumulative_positives / ranks
        ap_values = (precision_at_rank * sorted_labels).sum(axis=1)
        ap_chunk = average_precision[start:stop]
        ap_chunk[valid_ap] = ap_values[valid_ap] / positive_counts[valid_ap]

        valid_auc = (positive_counts > 0) & (negative_counts > 0)
        positives_above_negatives = ((~sorted_labels) * cumulative_positives).sum(axis=1)
        denominator = positive_counts * negative_counts
        auc_chunk = auroc[start:stop]
        auc_chunk[valid_auc] = positives_above_negatives[valid_auc] / denominator[valid_auc]

    return auroc, average_precision


def torch_to_numpy_scores(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        probabilities.detach().cpu().to(dtype=torch.float32).numpy(),
        (targets.detach().cpu().to(dtype=torch.float32).numpy() > 0.5),
    )


def write_score_matrix(
    path: Path,
    *,
    records,
    vocabulary_path: Path,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    global_threshold: float,
) -> None:
    """Write reusable raw scores and labels for downstream publication analysis."""

    score_matrix, target_matrix = torch_to_numpy_scores(probabilities, targets)
    vocab_rows = load_vocabulary_rows(vocabulary_path, score_matrix.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            probabilities=score_matrix.astype(np.float32, copy=False),
            targets=target_matrix.astype(np.uint8, copy=False),
            tax_id=np.array([record.tax_id for record in records], dtype=str),
            GeneID=np.array([record.gene_id for record in records], dtype=str),
            cds_id=np.array([record.cds_id for record in records], dtype=str),
            sequence_length=np.array([record.sequence_length for record in records], dtype=np.int64),
            token_count=np.array([record.token_count for record in records], dtype=np.int64),
            go_index=np.array([int(row.get("go_index", index)) for index, row in enumerate(vocab_rows)]),
            go_id=np.array([row.get("GO_ID", "") for row in vocab_rows], dtype=str),
            go_name=np.array([row.get("name", "") for row in vocab_rows], dtype=str),
            go_namespace=np.array([row.get("namespace", "") for row in vocab_rows], dtype=str),
            global_threshold=np.array([global_threshold], dtype=np.float32),
        )
    tmp_path.replace(path)


def write_publication_threshold_metrics(
    path: Path,
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, float]:
    """Write gene-averaged and micro threshold metrics, including ROC/PR curves."""

    score_matrix, target_matrix = torch_to_numpy_scores(probabilities, targets)
    true_counts = target_matrix.sum(axis=1).astype(np.float32)
    negative_counts = (target_matrix.shape[1] - true_counts).astype(np.float32)
    rows: list[dict[str, float]] = []

    for threshold_value in thresholds.detach().cpu().to(dtype=torch.float32).numpy():
        predicted = score_matrix >= float(threshold_value)
        tp = (predicted & target_matrix).sum(axis=1).astype(np.float32)
        fp = (predicted & ~target_matrix).sum(axis=1).astype(np.float32)
        fn = (~predicted & target_matrix).sum(axis=1).astype(np.float32)
        tn = (~predicted & ~target_matrix).sum(axis=1).astype(np.float32)

        predicted_counts = tp + fp
        precision_per_gene = np.divide(
            tp,
            predicted_counts,
            out=np.zeros_like(tp),
            where=predicted_counts > 0,
        )
        recall_per_gene = np.divide(
            tp,
            tp + fn,
            out=np.zeros_like(tp),
            where=true_counts > 0,
        )
        fpr_per_gene = np.divide(
            fp,
            fp + tn,
            out=np.zeros_like(fp),
            where=negative_counts > 0,
        )

        gene_precision = float(precision_per_gene.mean())
        gene_recall = float(recall_per_gene.mean())
        gene_fpr = float(fpr_per_gene.mean())
        gene_f1 = (
            0.0
            if gene_precision + gene_recall == 0.0
            else 2.0 * gene_precision * gene_recall / (gene_precision + gene_recall)
        )

        total_tp = float(tp.sum())
        total_fp = float(fp.sum())
        total_fn = float(fn.sum())
        total_tn = float(tn.sum())
        micro_precision = (
            total_tp / (total_tp + total_fp) if total_tp + total_fp > 0.0 else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn) if total_tp + total_fn > 0.0 else 0.0
        )
        micro_fpr = (
            total_fp / (total_fp + total_tn) if total_fp + total_tn > 0.0 else 0.0
        )
        micro_f1 = (
            0.0
            if micro_precision + micro_recall == 0.0
            else 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        )

        rows.append(
            {
                "threshold": float(threshold_value),
                "gene_averaged_precision": gene_precision,
                "gene_averaged_recall": gene_recall,
                "gene_averaged_f1": gene_f1,
                "gene_averaged_fpr": gene_fpr,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "micro_fpr": micro_fpr,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_float(value) for key, value in row.items()})
    tmp_path.replace(path)

    gene_fpr = np.array([0.0, *[row["gene_averaged_fpr"] for row in rows], 1.0])
    gene_tpr = np.array([0.0, *[row["gene_averaged_recall"] for row in rows], 1.0])
    micro_fpr = np.array([0.0, *[row["micro_fpr"] for row in rows], 1.0])
    micro_tpr = np.array([0.0, *[row["micro_recall"] for row in rows], 1.0])
    gene_recall = np.array([row["gene_averaged_recall"] for row in rows])
    gene_precision = np.array([row["gene_averaged_precision"] for row in rows])
    micro_recall = np.array([row["micro_recall"] for row in rows])
    micro_precision = np.array([row["micro_precision"] for row in rows])
    return {
        "gene_averaged_roc_auc_threshold_grid": auc_trapezoid(gene_fpr, gene_tpr),
        "micro_roc_auc_threshold_grid": auc_trapezoid(micro_fpr, micro_tpr),
        "gene_averaged_pr_auc_threshold_grid": auc_trapezoid(gene_recall, gene_precision),
        "micro_pr_auc_threshold_grid": auc_trapezoid(micro_recall, micro_precision),
    }


def write_publication_gene_metrics(
    path: Path,
    *,
    records,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    global_threshold: float,
) -> dict[str, float]:
    """Write per-gene threshold metrics plus score-ranked AUROC/AP."""

    score_matrix, target_matrix = torch_to_numpy_scores(probabilities, targets)
    gene_auroc, gene_average_precision = binary_auc_average_precision_by_rows(
        score_matrix,
        target_matrix,
        chunk_size=512,
    )
    predicted = score_matrix >= global_threshold
    tp = (predicted & target_matrix).sum(axis=1)
    fp = (predicted & ~target_matrix).sum(axis=1)
    fn = (~predicted & target_matrix).sum(axis=1)
    label_count = target_matrix.sum(axis=1)
    predicted_count = predicted.sum(axis=1)

    precision = np.divide(
        tp,
        tp + fp,
        out=np.full(tp.shape, np.nan, dtype=np.float64),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.full(tp.shape, np.nan, dtype=np.float64),
        where=(tp + fn) > 0,
    )
    f1 = np.divide(
        2 * tp,
        (2 * tp) + fp + fn,
        out=np.full(tp.shape, np.nan, dtype=np.float64),
        where=((2 * tp) + fp + fn) > 0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "tax_id",
                "GeneID",
                "cds_id",
                "sequence_length",
                "token_count",
                "label_count",
                "predicted_positive_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "auroc",
                "average_precision",
                "global_threshold",
            )
        )
        for index, record in enumerate(records):
            writer.writerow(
                (
                    record.tax_id,
                    record.gene_id,
                    record.cds_id,
                    record.sequence_length,
                    record.token_count,
                    int(label_count[index]),
                    int(predicted_count[index]),
                    int(tp[index]),
                    int(fp[index]),
                    int(fn[index]),
                    finite_float(float(precision[index])),
                    finite_float(float(recall[index])),
                    finite_float(float(f1[index])),
                    finite_float(float(gene_auroc[index])),
                    finite_float(float(gene_average_precision[index])),
                    f"{global_threshold:.8f}",
                )
            )
    tmp_path.replace(path)

    auroc_summary = nan_summary(gene_auroc)
    ap_summary = nan_summary(gene_average_precision)
    return {
        "per_gene_auroc_mean": auroc_summary["mean"],
        "per_gene_auroc_median": auroc_summary["median"],
        "per_gene_auroc_q25": auroc_summary["q25"],
        "per_gene_auroc_q75": auroc_summary["q75"],
        "per_gene_average_precision_mean": ap_summary["mean"],
        "per_gene_average_precision_median": ap_summary["median"],
        "per_gene_average_precision_q25": ap_summary["q25"],
        "per_gene_average_precision_q75": ap_summary["q75"],
    }


def write_publication_go_term_metrics(
    path: Path,
    *,
    vocabulary_path: Path,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    global_threshold: float,
    thresholds: torch.Tensor,
) -> dict[str, float]:
    """Write per-GO-term threshold metrics plus score-ranked AUROC/AP."""

    score_matrix, target_matrix = torch_to_numpy_scores(probabilities, targets)
    term_auroc, term_average_precision = binary_auc_average_precision_by_rows(
        score_matrix.T,
        target_matrix.T,
        chunk_size=64,
    )
    vocab_rows = load_vocabulary_rows(vocabulary_path, score_matrix.shape[1])
    target_bool = targets.to(dtype=torch.bool)
    test_positive_count = target_bool.sum(dim=0)
    test_negative_count = target_bool.shape[0] - test_positive_count
    tp, fp, fn = per_term_counts(probabilities, targets, global_threshold)
    precision, recall, f1 = counts_to_precision_recall_f1(tp, fp, fn)

    best_f1 = torch.full((probabilities.shape[1],), float("nan"), dtype=torch.float64)
    best_precision = torch.full_like(best_f1, float("nan"))
    best_recall = torch.full_like(best_f1, float("nan"))
    best_threshold = torch.full_like(best_f1, float("nan"))
    for threshold in thresholds:
        term_tp, term_fp, term_fn = per_term_counts(
            probabilities,
            targets,
            float(threshold.item()),
        )
        term_precision, term_recall, term_f1 = counts_to_precision_recall_f1(
            term_tp,
            term_fp,
            term_fn,
        )
        better = torch.nan_to_num(term_f1, nan=-1.0) > torch.nan_to_num(best_f1, nan=-1.0)
        best_f1 = torch.where(better, term_f1, best_f1)
        best_precision = torch.where(better, term_precision, best_precision)
        best_recall = torch.where(better, term_recall, best_recall)
        best_threshold = torch.where(
            better,
            torch.full_like(best_threshold, float(threshold.item())),
            best_threshold,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        fieldnames = [
            "go_index",
            "GO_ID",
            "name",
            "namespace",
            "database_frequency",
            "test_positive_count",
            "test_negative_count",
            "test_tp_at_global_threshold",
            "test_fp_at_global_threshold",
            "test_fn_at_global_threshold",
            "test_precision_at_global_threshold",
            "test_recall_at_global_threshold",
            "test_f1_at_global_threshold",
            "test_global_threshold",
            "test_term_fmax",
            "test_best_threshold",
            "test_precision_at_best_threshold",
            "test_recall_at_best_threshold",
            "auroc",
            "average_precision",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(vocab_rows):
            writer.writerow(
                {
                    "go_index": row.get("go_index", index),
                    "GO_ID": row.get("GO_ID", ""),
                    "name": row.get("name", ""),
                    "namespace": row.get("namespace", ""),
                    "database_frequency": row.get("trainable_gene_count", ""),
                    "test_positive_count": int(test_positive_count[index].item()),
                    "test_negative_count": int(test_negative_count[index].item()),
                    "test_tp_at_global_threshold": int(tp[index].item()),
                    "test_fp_at_global_threshold": int(fp[index].item()),
                    "test_fn_at_global_threshold": int(fn[index].item()),
                    "test_precision_at_global_threshold": finite_float(
                        float(precision[index].item())
                    ),
                    "test_recall_at_global_threshold": finite_float(
                        float(recall[index].item())
                    ),
                    "test_f1_at_global_threshold": finite_float(float(f1[index].item())),
                    "test_global_threshold": f"{global_threshold:.8f}",
                    "test_term_fmax": finite_float(float(best_f1[index].item())),
                    "test_best_threshold": finite_float(float(best_threshold[index].item())),
                    "test_precision_at_best_threshold": finite_float(
                        float(best_precision[index].item())
                    ),
                    "test_recall_at_best_threshold": finite_float(
                        float(best_recall[index].item())
                    ),
                    "auroc": finite_float(float(term_auroc[index])),
                    "average_precision": finite_float(float(term_average_precision[index])),
                }
            )
    tmp_path.replace(path)

    auroc_summary = nan_summary(term_auroc)
    ap_summary = nan_summary(term_average_precision)
    return {
        "per_go_term_auroc_mean": auroc_summary["mean"],
        "per_go_term_auroc_median": auroc_summary["median"],
        "per_go_term_auroc_q25": auroc_summary["q25"],
        "per_go_term_auroc_q75": auroc_summary["q75"],
        "per_go_term_average_precision_mean": ap_summary["mean"],
        "per_go_term_average_precision_median": ap_summary["median"],
        "per_go_term_average_precision_q25": ap_summary["q25"],
        "per_go_term_average_precision_q75": ap_summary["q75"],
    }


def write_publication_calibration_metrics(
    path: Path,
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 20,
) -> dict[str, float]:
    """Write flattened gene-GO calibration bins and return summary metrics."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    score_matrix, target_matrix = torch_to_numpy_scores(probabilities, targets)
    scores = score_matrix.reshape(-1)
    labels = target_matrix.reshape(-1).astype(np.float32, copy=False)
    brier_score = float(np.mean((scores - labels) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    rows: list[dict[str, float]] = []
    total = float(scores.size)
    ece = 0.0
    mce = 0.0
    for bin_index in range(n_bins):
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        if bin_index == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        count = int(mask.sum())
        if count:
            mean_score = float(scores[mask].mean())
            observed_fraction = float(labels[mask].mean())
            calibration_gap = abs(mean_score - observed_fraction)
        else:
            mean_score = float("nan")
            observed_fraction = float("nan")
            calibration_gap = float("nan")
        if count and np.isfinite(calibration_gap):
            ece += (count / total) * calibration_gap
            mce = max(mce, calibration_gap)
        rows.append(
            {
                "bin_index": float(bin_index),
                "bin_lower": lower,
                "bin_upper": upper,
                "count": float(count),
                "mean_score": mean_score,
                "observed_positive_fraction": observed_fraction,
                "absolute_calibration_gap": calibration_gap,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: int(value) if key in {"bin_index", "count"} else finite_float(value)
                    for key, value in row.items()
                }
            )
    tmp_path.replace(path)

    return {
        "brier_score": brier_score,
        "expected_calibration_error_20bin": ece,
        "maximum_calibration_error_20bin": mce,
    }


def write_publication_summary(
    path: Path,
    rows: list[tuple[object, object]],
) -> None:
    write_key_value_tsv(path, rows)


def write_per_term_metrics(
    path: Path,
    *,
    vocabulary_path: Path,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    global_threshold: float,
    thresholds: torch.Tensor,
) -> None:
    """Write pooled one-vs-rest metrics for each GO term on the evaluated split."""

    vocab_rows = load_vocabulary_rows(vocabulary_path, probabilities.shape[1])
    target_bool = targets.to(dtype=torch.bool)
    test_positive_count = target_bool.sum(dim=0)
    test_negative_count = target_bool.shape[0] - test_positive_count

    tp, fp, fn = per_term_counts(probabilities, targets, global_threshold)
    precision, recall, f1 = counts_to_precision_recall_f1(tp, fp, fn)

    best_f1 = torch.full((probabilities.shape[1],), float("nan"), dtype=torch.float64)
    best_precision = torch.full_like(best_f1, float("nan"))
    best_recall = torch.full_like(best_f1, float("nan"))
    best_threshold = torch.full_like(best_f1, float("nan"))
    for threshold in thresholds:
        term_tp, term_fp, term_fn = per_term_counts(
            probabilities,
            targets,
            float(threshold.item()),
        )
        term_precision, term_recall, term_f1 = counts_to_precision_recall_f1(
            term_tp,
            term_fp,
            term_fn,
        )
        better = torch.nan_to_num(term_f1, nan=-1.0) > torch.nan_to_num(best_f1, nan=-1.0)
        best_f1 = torch.where(better, term_f1, best_f1)
        best_precision = torch.where(better, term_precision, best_precision)
        best_recall = torch.where(better, term_recall, best_recall)
        best_threshold = torch.where(
            better,
            torch.full_like(best_threshold, float(threshold.item())),
            best_threshold,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        fieldnames = [
            "go_index",
            "GO_ID",
            "name",
            "namespace",
            "database_frequency",
            "test_positive_count",
            "test_negative_count",
            "test_tp_at_global_threshold",
            "test_fp_at_global_threshold",
            "test_fn_at_global_threshold",
            "test_precision_at_global_threshold",
            "test_recall_at_global_threshold",
            "test_f1_at_global_threshold",
            "test_global_threshold",
            "test_term_fmax",
            "test_best_threshold",
            "test_precision_at_best_threshold",
            "test_recall_at_best_threshold",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(vocab_rows):
            writer.writerow(
                {
                    "go_index": row.get("go_index", index),
                    "GO_ID": row.get("GO_ID", ""),
                    "name": row.get("name", ""),
                    "namespace": row.get("namespace", ""),
                    "database_frequency": row.get("trainable_gene_count", ""),
                    "test_positive_count": int(test_positive_count[index].item()),
                    "test_negative_count": int(test_negative_count[index].item()),
                    "test_tp_at_global_threshold": int(tp[index].item()),
                    "test_fp_at_global_threshold": int(fp[index].item()),
                    "test_fn_at_global_threshold": int(fn[index].item()),
                    "test_precision_at_global_threshold": f"{precision[index].item():.8f}",
                    "test_recall_at_global_threshold": f"{recall[index].item():.8f}",
                    "test_f1_at_global_threshold": f"{f1[index].item():.8f}",
                    "test_global_threshold": f"{global_threshold:.8f}",
                    "test_term_fmax": f"{best_f1[index].item():.8f}",
                    "test_best_threshold": f"{best_threshold[index].item():.8f}",
                    "test_precision_at_best_threshold": f"{best_precision[index].item():.8f}",
                    "test_recall_at_best_threshold": f"{best_recall[index].item():.8f}",
                }
            )
    tmp_path.replace(path)


def write_per_gene_metrics(
    path: Path,
    *,
    records,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    global_threshold: float,
) -> None:
    """Write thresholded precision/recall/F1 per evaluated gene."""

    if len(records) != probabilities.shape[0]:
        raise ValueError(
            f"Record/probability row mismatch: {len(records)} records, "
            f"{probabilities.shape[0]} probability rows"
        )

    target_bool = targets.to(dtype=torch.bool)
    predicted = probabilities >= global_threshold
    tp = (predicted & target_bool).sum(dim=1)
    fp = (predicted & ~target_bool).sum(dim=1)
    fn = (~predicted & target_bool).sum(dim=1)
    label_count = target_bool.sum(dim=1)
    predicted_positive_count = predicted.sum(dim=1)
    precision, recall, f1 = counts_to_precision_recall_f1(tp, fp, fn)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "tax_id",
                "GeneID",
                "cds_id",
                "sequence_length",
                "token_count",
                "label_count",
                "predicted_positive_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "global_threshold",
            )
        )
        for index, record in enumerate(records):
            writer.writerow(
                (
                    record.tax_id,
                    record.gene_id,
                    record.cds_id,
                    record.sequence_length,
                    record.token_count,
                    int(label_count[index].item()),
                    int(predicted_positive_count[index].item()),
                    int(tp[index].item()),
                    int(fp[index].item()),
                    int(fn[index].item()),
                    f"{precision[index].item():.8f}",
                    f"{recall[index].item():.8f}",
                    f"{f1[index].item():.8f}",
                    f"{global_threshold:.8f}",
                )
            )
    tmp_path.replace(path)


@dataclass(frozen=True)
class StreamingThresholdResults:
    """Final exact threshold-grid counts accumulated from bounded-size batches."""

    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    gene_precision: np.ndarray
    gene_recall: np.ndarray
    gene_fpr: np.ndarray


class StreamingMetricAccumulator:
    """Accumulate exact grid metrics without retaining gene-by-term matrices."""

    def __init__(
        self,
        *,
        thresholds: torch.Tensor | np.ndarray,
        n_terms: int,
        calibration_bins: int = 20,
    ) -> None:
        threshold_array = np.asarray(
            torch.as_tensor(thresholds, dtype=torch.float32).cpu().numpy(),
            dtype=np.float32,
        )
        if threshold_array.ndim != 1 or threshold_array.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array")
        if np.any(np.diff(threshold_array) <= 0.0):
            raise ValueError("thresholds must be strictly increasing")
        if n_terms <= 0:
            raise ValueError("n_terms must be positive")
        if calibration_bins <= 0:
            raise ValueError("calibration_bins must be positive")

        self.thresholds = threshold_array
        self.n_terms = n_terms
        self.calibration_bins = calibration_bins
        histogram_bins = threshold_array.size + 1
        self.term_positive_histogram = np.zeros(
            (n_terms, histogram_bins), dtype=np.int64
        )
        self.term_all_histogram = np.zeros((n_terms, histogram_bins), dtype=np.int64)
        self.gene_precision_sum = np.zeros(threshold_array.size, dtype=np.float64)
        self.gene_recall_sum = np.zeros(threshold_array.size, dtype=np.float64)
        self.gene_fpr_sum = np.zeros(threshold_array.size, dtype=np.float64)
        self.calibration_count = np.zeros(calibration_bins, dtype=np.int64)
        self.calibration_score_sum = np.zeros(calibration_bins, dtype=np.float64)
        self.calibration_positive_sum = np.zeros(calibration_bins, dtype=np.int64)
        self.brier_squared_error_sum = 0.0
        self.records = 0

    def update(self, probabilities: torch.Tensor, targets: torch.Tensor) -> None:
        if probabilities.shape != targets.shape:
            raise ValueError(
                "Streaming score/target shape mismatch: "
                f"{tuple(probabilities.shape)} != {tuple(targets.shape)}"
            )
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_terms:
            raise ValueError(
                "Streaming metrics expect (batch, n_terms) with "
                f"n_terms={self.n_terms}; got {tuple(probabilities.shape)}"
            )
        if probabilities.shape[0] == 0:
            return

        scores = probabilities.detach().cpu().to(dtype=torch.float32)
        labels = targets.detach().cpu().to(dtype=torch.float32) > 0.5
        thresholds = torch.from_numpy(self.thresholds)
        # bucket k means exactly k grid thresholds are <= the score. Because
        # evaluation predicts score >= threshold, right=True preserves equality.
        score_bins = torch.bucketize(scores, thresholds, right=True)
        label_int = labels.to(dtype=torch.int64)

        n_thresholds = int(self.thresholds.size)
        gene_positive_histogram = torch.zeros(
            (scores.shape[0], n_thresholds + 1), dtype=torch.int64
        )
        gene_all_histogram = torch.zeros_like(gene_positive_histogram)
        gene_positive_histogram.scatter_add_(1, score_bins, label_int)
        gene_all_histogram.scatter_add_(
            1, score_bins, torch.ones_like(score_bins, dtype=torch.int64)
        )

        gene_tp = torch.flip(
            torch.cumsum(torch.flip(gene_positive_histogram, dims=(1,)), dim=1),
            dims=(1,),
        )[:, 1:]
        gene_predicted = torch.flip(
            torch.cumsum(torch.flip(gene_all_histogram, dims=(1,)), dim=1),
            dims=(1,),
        )[:, 1:]
        gene_fp = gene_predicted - gene_tp
        true_count = label_int.sum(dim=1, keepdim=True)
        negative_count = self.n_terms - true_count

        gene_tp_float = gene_tp.to(dtype=torch.float64)
        predicted_float = gene_predicted.to(dtype=torch.float64)
        gene_fp_float = gene_fp.to(dtype=torch.float64)
        precision = torch.where(
            gene_predicted > 0,
            gene_tp_float / predicted_float.clamp_min(1.0),
            torch.zeros_like(gene_tp_float),
        )
        recall = torch.where(
            true_count > 0,
            gene_tp_float / true_count.to(dtype=torch.float64).clamp_min(1.0),
            torch.zeros_like(gene_tp_float),
        )
        false_positive_rate = torch.where(
            negative_count > 0,
            gene_fp_float / negative_count.to(dtype=torch.float64).clamp_min(1.0),
            torch.zeros_like(gene_fp_float),
        )
        self.gene_precision_sum += precision.sum(dim=0).numpy()
        self.gene_recall_sum += recall.sum(dim=0).numpy()
        self.gene_fpr_sum += false_positive_rate.sum(dim=0).numpy()

        term_positive_histogram = torch.zeros(
            (self.n_terms, n_thresholds + 1), dtype=torch.int64
        )
        term_all_histogram = torch.zeros_like(term_positive_histogram)
        term_positive_histogram.scatter_add_(1, score_bins.T, label_int.T)
        term_all_histogram.scatter_add_(
            1,
            score_bins.T,
            torch.ones_like(score_bins.T, dtype=torch.int64),
        )
        self.term_positive_histogram += term_positive_histogram.numpy()
        self.term_all_histogram += term_all_histogram.numpy()

        calibration_index = torch.clamp(
            torch.floor(scores.reshape(-1) * self.calibration_bins).to(torch.int64),
            min=0,
            max=self.calibration_bins - 1,
        )
        flat_scores = scores.reshape(-1).to(dtype=torch.float64)
        flat_labels = labels.reshape(-1).to(dtype=torch.int64)
        self.calibration_count += torch.bincount(
            calibration_index, minlength=self.calibration_bins
        ).numpy()
        score_sums = torch.zeros(self.calibration_bins, dtype=torch.float64)
        score_sums.index_add_(0, calibration_index, flat_scores)
        self.calibration_score_sum += score_sums.numpy()
        positive_sums = torch.zeros(self.calibration_bins, dtype=torch.int64)
        positive_sums.index_add_(0, calibration_index, flat_labels)
        self.calibration_positive_sum += positive_sums.numpy()
        self.brier_squared_error_sum += float(
            torch.sum((flat_scores - flat_labels.to(dtype=torch.float64)) ** 2).item()
        )
        self.records += int(scores.shape[0])

    def threshold_results(self) -> StreamingThresholdResults:
        if self.records == 0:
            raise ValueError("Streaming evaluation received no records")

        positive_cumulative = np.flip(
            np.cumsum(np.flip(self.term_positive_histogram, axis=1), axis=1),
            axis=1,
        )[:, 1:].T
        all_cumulative = np.flip(
            np.cumsum(np.flip(self.term_all_histogram, axis=1), axis=1),
            axis=1,
        )[:, 1:].T
        positive_count = self.term_positive_histogram.sum(axis=1)[None, :]
        negative_count = self.records - positive_count
        tp = positive_cumulative
        fp = all_cumulative - positive_cumulative
        fn = positive_count - tp
        tn = negative_count - fp
        return StreamingThresholdResults(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            gene_precision=self.gene_precision_sum / self.records,
            gene_recall=self.gene_recall_sum / self.records,
            gene_fpr=self.gene_fpr_sum / self.records,
        )

    def fmax_result(self) -> FmaxResult:
        counts = self.threshold_results()
        denominator = counts.gene_precision + counts.gene_recall
        f1 = np.divide(
            2.0 * counts.gene_precision * counts.gene_recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        best_index = int(np.argmax(f1))
        return FmaxResult(
            fmax=float(f1[best_index]),
            threshold=float(self.thresholds[best_index]),
            precision=float(counts.gene_precision[best_index]),
            recall=float(counts.gene_recall[best_index]),
            thresholds=tuple(float(value) for value in self.thresholds),
            precision_by_threshold=tuple(float(value) for value in counts.gene_precision),
            recall_by_threshold=tuple(float(value) for value in counts.gene_recall),
            f1_by_threshold=tuple(float(value) for value in f1),
        )


def streaming_term_indices(
    vocabulary_rows: list[dict[str, str]],
    *,
    exclude_go_roots: bool,
) -> tuple[np.ndarray, list[dict[str, str]], list[str]]:
    excluded = [
        row.get("GO_ID", "")
        for row in vocabulary_rows
        if exclude_go_roots and row.get("GO_ID", "") in GO_ROOT_IDS
    ]
    indices = np.array(
        [
            index
            for index, row in enumerate(vocabulary_rows)
            if not exclude_go_roots or row.get("GO_ID", "") not in GO_ROOT_IDS
        ],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError("No GO terms remain after root-term exclusion")
    return indices, [vocabulary_rows[index] for index in indices], excluded


@torch.no_grad()
def collect_streaming_statistics(
    model: DualEncoderModel,
    loader: DataLoader,
    *,
    device: torch.device,
    term_indices: np.ndarray,
    thresholds: torch.Tensor,
    calibration_bins: int,
    max_batches: int | None,
    progress_every_batches: int,
) -> tuple[StreamingMetricAccumulator, int]:
    model.eval()
    # GO embeddings and the scale are invariant during evaluation. Caching
    # them avoids rerunning the GCN for every one of the ~12k test batches.
    go_representations_t = model.go_encoder().transpose(0, 1)
    logit_scale = model.logit_scale.clamp(
        min=model.config.min_logit_scale,
        max=model.config.max_logit_scale,
    )
    accumulator = StreamingMetricAccumulator(
        thresholds=thresholds,
        n_terms=int(term_indices.size),
        calibration_bins=calibration_bins,
    )
    index_tensor = torch.from_numpy(term_indices)
    batches_seen = 0
    for batch in loader:
        embeddings = batch["embeddings"].to(
            device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
        )
        sequence_representations = model.sequence_encoder(embeddings)
        probabilities = torch.sigmoid(
            logit_scale
            * torch.matmul(sequence_representations, go_representations_t)
        ).cpu()[:, index_tensor]
        targets = batch["targets"][:, index_tensor]
        accumulator.update(probabilities, targets)
        batches_seen += 1
        if progress_every_batches > 0 and batches_seen % progress_every_batches == 0:
            print(
                f"Streaming pass 1: {accumulator.records:,} records "
                f"({batches_seen:,} batches)",
                flush=True,
            )
        if max_batches is not None and batches_seen >= max_batches:
            break
    if batches_seen == 0:
        raise ValueError("Evaluation loader produced no batches")
    return accumulator, batches_seen


def streaming_threshold_rows(
    accumulator: StreamingMetricAccumulator,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    counts = accumulator.threshold_results()
    rows: list[dict[str, float]] = []
    for index, threshold in enumerate(accumulator.thresholds):
        total_tp = float(counts.tp[index].sum())
        total_fp = float(counts.fp[index].sum())
        total_fn = float(counts.fn[index].sum())
        total_tn = float(counts.tn[index].sum())
        micro_precision = safe_metric(total_tp, total_tp + total_fp)
        micro_recall = safe_metric(total_tp, total_tp + total_fn)
        micro_fpr = safe_metric(total_fp, total_fp + total_tn)
        if not np.isfinite(micro_precision):
            micro_precision = 0.0
        if not np.isfinite(micro_recall):
            micro_recall = 0.0
        if not np.isfinite(micro_fpr):
            micro_fpr = 0.0
        micro_f1 = (
            0.0
            if micro_precision + micro_recall == 0.0
            else 2.0
            * micro_precision
            * micro_recall
            / (micro_precision + micro_recall)
        )
        gene_precision = float(counts.gene_precision[index])
        gene_recall = float(counts.gene_recall[index])
        gene_f1 = (
            0.0
            if gene_precision + gene_recall == 0.0
            else 2.0 * gene_precision * gene_recall / (gene_precision + gene_recall)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "gene_averaged_precision": gene_precision,
                "gene_averaged_recall": gene_recall,
                "gene_averaged_f1": gene_f1,
                "gene_averaged_fpr": float(counts.gene_fpr[index]),
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "micro_fpr": micro_fpr,
            }
        )

    gene_fpr = np.array([0.0, *[row["gene_averaged_fpr"] for row in rows], 1.0])
    gene_tpr = np.array([0.0, *[row["gene_averaged_recall"] for row in rows], 1.0])
    micro_fpr = np.array([0.0, *[row["micro_fpr"] for row in rows], 1.0])
    micro_tpr = np.array([0.0, *[row["micro_recall"] for row in rows], 1.0])
    summary = {
        "gene_averaged_roc_auc_threshold_grid": auc_trapezoid(gene_fpr, gene_tpr),
        "micro_roc_auc_threshold_grid": auc_trapezoid(micro_fpr, micro_tpr),
        "gene_averaged_pr_auc_threshold_grid": auc_trapezoid(
            np.array([row["gene_averaged_recall"] for row in rows]),
            np.array([row["gene_averaged_precision"] for row in rows]),
        ),
        "micro_pr_auc_threshold_grid": auc_trapezoid(
            np.array([row["micro_recall"] for row in rows]),
            np.array([row["micro_precision"] for row in rows]),
        ),
    }
    return rows, summary


def write_streaming_threshold_metrics(
    path: Path,
    rows: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {key: finite_float(value) for key, value in row.items()} for row in rows
        )
    tmp_path.replace(path)


def write_streaming_term_metrics(
    path: Path,
    *,
    vocabulary_rows: list[dict[str, str]],
    accumulator: StreamingMetricAccumulator,
    global_threshold: float,
    include_rank_columns: bool,
) -> dict[str, float]:
    counts = accumulator.threshold_results()
    threshold_index = int(
        np.argmin(np.abs(accumulator.thresholds.astype(np.float64) - global_threshold))
    )
    tp = counts.tp.astype(np.float64)
    fp = counts.fp.astype(np.float64)
    fn = counts.fn.astype(np.float64)
    precision = np.divide(
        tp,
        tp + fp,
        out=np.full_like(tp, np.nan),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.full_like(tp, np.nan),
        where=(tp + fn) > 0,
    )
    f1 = np.divide(
        2.0 * tp,
        (2.0 * tp) + fp + fn,
        out=np.full_like(tp, np.nan),
        where=((2.0 * tp) + fp + fn) > 0,
    )
    ranked_f1 = np.where(np.isfinite(f1), f1, -np.inf)
    best_index = np.argmax(ranked_f1, axis=0)
    term_indices = np.arange(accumulator.n_terms)
    has_metric = np.isfinite(f1).any(axis=0)
    best_f1 = np.where(has_metric, f1[best_index, term_indices], np.nan)
    best_precision = np.where(
        has_metric, precision[best_index, term_indices], np.nan
    )
    best_recall = np.where(has_metric, recall[best_index, term_indices], np.nan)
    best_threshold = np.where(
        has_metric, accumulator.thresholds[best_index], np.nan
    )
    positive_count = accumulator.term_positive_histogram.sum(axis=1)

    fieldnames = [
        "go_index",
        "GO_ID",
        "name",
        "namespace",
        "database_frequency",
        "test_positive_count",
        "test_negative_count",
        "test_tp_at_global_threshold",
        "test_fp_at_global_threshold",
        "test_fn_at_global_threshold",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_f1_at_global_threshold",
        "test_global_threshold",
        "test_term_fmax",
        "test_best_threshold",
        "test_precision_at_best_threshold",
        "test_recall_at_best_threshold",
    ]
    if include_rank_columns:
        fieldnames.extend(("auroc", "average_precision"))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(vocabulary_rows):
            output = {
                "go_index": row.get("go_index", index),
                "GO_ID": row.get("GO_ID", ""),
                "name": row.get("name", ""),
                "namespace": row.get("namespace", ""),
                "database_frequency": row.get("trainable_gene_count", ""),
                "test_positive_count": int(positive_count[index]),
                "test_negative_count": int(accumulator.records - positive_count[index]),
                "test_tp_at_global_threshold": int(counts.tp[threshold_index, index]),
                "test_fp_at_global_threshold": int(counts.fp[threshold_index, index]),
                "test_fn_at_global_threshold": int(counts.fn[threshold_index, index]),
                "test_precision_at_global_threshold": finite_float(
                    float(precision[threshold_index, index])
                ),
                "test_recall_at_global_threshold": finite_float(
                    float(recall[threshold_index, index])
                ),
                "test_f1_at_global_threshold": finite_float(
                    float(f1[threshold_index, index])
                ),
                "test_global_threshold": f"{global_threshold:.8f}",
                "test_term_fmax": finite_float(float(best_f1[index])),
                "test_best_threshold": finite_float(float(best_threshold[index])),
                "test_precision_at_best_threshold": finite_float(
                    float(best_precision[index])
                ),
                "test_recall_at_best_threshold": finite_float(float(best_recall[index])),
            }
            if include_rank_columns:
                # Exact term-wise ranking requires retaining or externally sorting
                # millions of scores per term. Leave these explicitly undefined.
                output.update({"auroc": "nan", "average_precision": "nan"})
            writer.writerow(output)
    tmp_path.replace(path)
    return {
        "per_go_term_auroc_mean": float("nan"),
        "per_go_term_auroc_median": float("nan"),
        "per_go_term_auroc_q25": float("nan"),
        "per_go_term_auroc_q75": float("nan"),
        "per_go_term_average_precision_mean": float("nan"),
        "per_go_term_average_precision_median": float("nan"),
        "per_go_term_average_precision_q25": float("nan"),
        "per_go_term_average_precision_q75": float("nan"),
    }


def write_streaming_calibration_metrics(
    path: Path,
    accumulator: StreamingMetricAccumulator,
) -> dict[str, float]:
    total = int(accumulator.calibration_count.sum())
    if total == 0:
        raise ValueError("Cannot write calibration metrics without predictions")
    rows: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    for index in range(accumulator.calibration_bins):
        count = int(accumulator.calibration_count[index])
        mean_score = safe_metric(accumulator.calibration_score_sum[index], count)
        observed = safe_metric(accumulator.calibration_positive_sum[index], count)
        gap = abs(mean_score - observed) if count else float("nan")
        if count:
            ece += (count / total) * gap
            mce = max(mce, gap)
        rows.append(
            {
                "bin_index": float(index),
                "bin_lower": index / accumulator.calibration_bins,
                "bin_upper": (index + 1) / accumulator.calibration_bins,
                "count": float(count),
                "mean_score": mean_score,
                "observed_positive_fraction": observed,
                "absolute_calibration_gap": gap,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: int(value)
                    if key in {"bin_index", "count"}
                    else finite_float(value)
                    for key, value in row.items()
                }
            )
    tmp_path.replace(path)
    return {
        "brier_score": accumulator.brier_squared_error_sum / total,
        f"expected_calibration_error_{accumulator.calibration_bins}bin": ece,
        f"maximum_calibration_error_{accumulator.calibration_bins}bin": mce,
    }


@torch.no_grad()
def write_streaming_gene_metrics(
    standard_path: Path,
    publication_path: Path | None,
    *,
    model: DualEncoderModel,
    loader: DataLoader,
    device: torch.device,
    term_indices: np.ndarray,
    global_threshold: float,
    expected_records: int,
    max_batches: int | None,
    progress_every_batches: int,
) -> dict[str, float]:
    model.eval()
    go_representations_t = model.go_encoder().transpose(0, 1)
    logit_scale = model.logit_scale.clamp(
        min=model.config.min_logit_scale,
        max=model.config.max_logit_scale,
    )
    standard_path.parent.mkdir(parents=True, exist_ok=True)
    if publication_path is not None:
        publication_path.parent.mkdir(parents=True, exist_ok=True)
    standard_tmp = standard_path.with_suffix(standard_path.suffix + ".tmp")
    publication_tmp = (
        publication_path.with_suffix(publication_path.suffix + ".tmp")
        if publication_path is not None
        else None
    )
    index_tensor = torch.from_numpy(term_indices)
    gene_auroc = np.full(expected_records, np.nan, dtype=np.float32)
    gene_average_precision = np.full(expected_records, np.nan, dtype=np.float32)
    records_seen = 0
    batches_seen = 0

    standard_header = (
        "tax_id",
        "GeneID",
        "cds_id",
        "sequence_length",
        "token_count",
        "label_count",
        "predicted_positive_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "global_threshold",
    )
    publication_header = (*standard_header[:-1], "auroc", "average_precision", "global_threshold")

    with ExitStack() as stack:
        standard_handle = stack.enter_context(standard_tmp.open("wt", newline=""))
        standard_writer = csv.writer(
            standard_handle, delimiter="\t", lineterminator="\n"
        )
        standard_writer.writerow(standard_header)
        publication_writer = None
        if publication_tmp is not None:
            publication_handle = stack.enter_context(
                publication_tmp.open("wt", newline="")
            )
            publication_writer = csv.writer(
                publication_handle, delimiter="\t", lineterminator="\n"
            )
            publication_writer.writerow(publication_header)

        for batch in loader:
            embeddings = batch["embeddings"].to(
                device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
            )
            sequence_representations = model.sequence_encoder(embeddings)
            scores = torch.sigmoid(
                logit_scale
                * torch.matmul(sequence_representations, go_representations_t)
            ).cpu()[:, index_tensor]
            labels = batch["targets"][:, index_tensor].to(dtype=torch.float32) > 0.5
            score_matrix = scores.numpy()
            target_matrix = labels.numpy()
            batch_auroc, batch_ap = binary_auc_average_precision_by_rows(
                score_matrix, target_matrix, chunk_size=max(1, score_matrix.shape[0])
            )
            predicted = score_matrix >= global_threshold
            tp = (predicted & target_matrix).sum(axis=1)
            fp = (predicted & ~target_matrix).sum(axis=1)
            fn = (~predicted & target_matrix).sum(axis=1)
            label_count = target_matrix.sum(axis=1)
            predicted_count = predicted.sum(axis=1)
            precision = np.divide(
                tp,
                tp + fp,
                out=np.full(tp.shape, np.nan, dtype=np.float64),
                where=(tp + fp) > 0,
            )
            recall = np.divide(
                tp,
                tp + fn,
                out=np.full(tp.shape, np.nan, dtype=np.float64),
                where=(tp + fn) > 0,
            )
            f1 = np.divide(
                2 * tp,
                (2 * tp) + fp + fn,
                out=np.full(tp.shape, np.nan, dtype=np.float64),
                where=((2 * tp) + fp + fn) > 0,
            )
            identities = batch["identities"]
            for index, identity in enumerate(identities):
                base = (
                    identity["tax_id"],
                    identity["GeneID"],
                    identity["cds_id"],
                    identity["sequence_length"],
                    identity["token_count"],
                    int(label_count[index]),
                    int(predicted_count[index]),
                    int(tp[index]),
                    int(fp[index]),
                    int(fn[index]),
                    finite_float(float(precision[index])),
                    finite_float(float(recall[index])),
                    finite_float(float(f1[index])),
                )
                standard_writer.writerow((*base, f"{global_threshold:.8f}"))
                if publication_writer is not None:
                    publication_writer.writerow(
                        (
                            *base,
                            finite_float(float(batch_auroc[index])),
                            finite_float(float(batch_ap[index])),
                            f"{global_threshold:.8f}",
                        )
                    )
            next_record = records_seen + score_matrix.shape[0]
            if next_record > expected_records:
                raise ValueError("Second streaming pass produced too many records")
            gene_auroc[records_seen:next_record] = batch_auroc
            gene_average_precision[records_seen:next_record] = batch_ap
            records_seen = next_record
            batches_seen += 1
            if progress_every_batches > 0 and batches_seen % progress_every_batches == 0:
                print(
                    f"Streaming pass 2: {records_seen:,} records "
                    f"({batches_seen:,} batches)",
                    flush=True,
                )
            if max_batches is not None and batches_seen >= max_batches:
                break

    if records_seen != expected_records:
        raise ValueError(
            f"Streaming passes disagree: pass 1={expected_records:,}, "
            f"pass 2={records_seen:,} records"
        )
    standard_tmp.replace(standard_path)
    if publication_path is not None and publication_tmp is not None:
        publication_tmp.replace(publication_path)

    auroc_summary = nan_summary(gene_auroc)
    ap_summary = nan_summary(gene_average_precision)
    return {
        "per_gene_auroc_mean": auroc_summary["mean"],
        "per_gene_auroc_median": auroc_summary["median"],
        "per_gene_auroc_q25": auroc_summary["q25"],
        "per_gene_auroc_q75": auroc_summary["q75"],
        "per_gene_average_precision_mean": ap_summary["mean"],
        "per_gene_average_precision_median": ap_summary["median"],
        "per_gene_average_precision_q25": ap_summary["q25"],
        "per_gene_average_precision_q75": ap_summary["q75"],
    }


@torch.no_grad()
def collect_streaming_statistics_reusing_gene_rank_metrics(
    standard_path: Path,
    publication_path: Path,
    reuse_publication_path: Path,
    *,
    model: DualEncoderModel,
    loader: DataLoader,
    device: torch.device,
    term_indices: np.ndarray,
    thresholds: torch.Tensor,
    operating_threshold: float,
    calibration_bins: int,
    max_batches: int | None,
    progress_every_batches: int,
    allow_trailing_reuse_rows: bool = False,
) -> tuple[StreamingMetricAccumulator, int, dict[str, float]]:
    """Collect streaming metrics and rewrite gene rows in one inference pass.

    AUROC and average precision do not depend on the operating threshold.  This
    path copies those columns from an earlier evaluation of the same ordered
    test records while recomputing every threshold-dependent gene metric from
    the current model scores.  Identity checks prevent accidental reuse from a
    different evaluation population or ordering.  Truncated smoke evaluations
    may explicitly allow unused trailing rows in the source table; full
    evaluations require the source and current populations to end together.
    """

    model.eval()
    go_representations_t = model.go_encoder().transpose(0, 1)
    logit_scale = model.logit_scale.clamp(
        min=model.config.min_logit_scale,
        max=model.config.max_logit_scale,
    )
    accumulator = StreamingMetricAccumulator(
        thresholds=thresholds,
        n_terms=int(term_indices.size),
        calibration_bins=calibration_bins,
    )
    index_tensor = torch.from_numpy(term_indices)
    standard_path.parent.mkdir(parents=True, exist_ok=True)
    publication_path.parent.mkdir(parents=True, exist_ok=True)
    standard_tmp = standard_path.with_suffix(standard_path.suffix + ".tmp")
    publication_tmp = publication_path.with_suffix(publication_path.suffix + ".tmp")
    standard_header = (
        "tax_id",
        "GeneID",
        "cds_id",
        "sequence_length",
        "token_count",
        "label_count",
        "predicted_positive_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "global_threshold",
    )
    publication_header = (
        *standard_header[:-1],
        "auroc",
        "average_precision",
        "global_threshold",
    )
    expected_records = len(loader.dataset)
    gene_auroc = np.full(expected_records, np.nan, dtype=np.float32)
    gene_average_precision = np.full(expected_records, np.nan, dtype=np.float32)
    records_seen = 0
    batches_seen = 0

    with ExitStack() as stack:
        standard_handle = stack.enter_context(standard_tmp.open("wt", newline=""))
        publication_handle = stack.enter_context(
            publication_tmp.open("wt", newline="")
        )
        reuse_handle = stack.enter_context(reuse_publication_path.open("rt", newline=""))
        standard_writer = csv.writer(
            standard_handle, delimiter="\t", lineterminator="\n"
        )
        publication_writer = csv.writer(
            publication_handle, delimiter="\t", lineterminator="\n"
        )
        standard_writer.writerow(standard_header)
        publication_writer.writerow(publication_header)
        reuse_reader = csv.DictReader(reuse_handle, delimiter="\t")
        required_reuse_columns = {
            "tax_id",
            "GeneID",
            "cds_id",
            "auroc",
            "average_precision",
        }
        missing = required_reuse_columns.difference(reuse_reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{reuse_publication_path} is missing required columns: {sorted(missing)}"
            )

        for batch in loader:
            embeddings = batch["embeddings"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            sequence_representations = model.sequence_encoder(embeddings)
            scores = torch.sigmoid(
                logit_scale
                * torch.matmul(sequence_representations, go_representations_t)
            ).cpu()[:, index_tensor]
            labels = batch["targets"][:, index_tensor].to(dtype=torch.float32) > 0.5
            accumulator.update(scores, labels)

            predicted = scores >= operating_threshold
            tp = (predicted & labels).sum(dim=1)
            fp = (predicted & ~labels).sum(dim=1)
            fn = (~predicted & labels).sum(dim=1)
            label_count = labels.sum(dim=1)
            predicted_count = predicted.sum(dim=1)
            precision, recall, f1 = counts_to_precision_recall_f1(tp, fp, fn)

            for index, identity in enumerate(batch["identities"]):
                try:
                    reused = next(reuse_reader)
                except StopIteration as error:
                    raise ValueError(
                        f"{reuse_publication_path} ended before the current evaluation"
                    ) from error
                observed_identity = (
                    clean_field(reused["tax_id"]),
                    clean_field(reused["GeneID"]),
                    clean_field(reused["cds_id"]),
                )
                expected_identity = (
                    clean_field(identity["tax_id"]),
                    clean_field(identity["GeneID"]),
                    clean_field(identity["cds_id"]),
                )
                if observed_identity != expected_identity:
                    raise ValueError(
                        "Reused publication gene metrics do not match the current "
                        f"record order at row {records_seen + index + 2:,}: "
                        f"expected={expected_identity!r}, observed={observed_identity!r}"
                    )
                auroc = float(reused["auroc"])
                average_precision = float(reused["average_precision"])
                output_index = records_seen + index
                gene_auroc[output_index] = auroc
                gene_average_precision[output_index] = average_precision
                base = (
                    identity["tax_id"],
                    identity["GeneID"],
                    identity["cds_id"],
                    identity["sequence_length"],
                    identity["token_count"],
                    int(label_count[index]),
                    int(predicted_count[index]),
                    int(tp[index]),
                    int(fp[index]),
                    int(fn[index]),
                    finite_float(float(precision[index])),
                    finite_float(float(recall[index])),
                    finite_float(float(f1[index])),
                )
                standard_writer.writerow((*base, f"{operating_threshold:.8f}"))
                publication_writer.writerow(
                    (
                        *base,
                        finite_float(auroc),
                        finite_float(average_precision),
                        f"{operating_threshold:.8f}",
                    )
                )

            records_seen += scores.shape[0]
            batches_seen += 1
            if progress_every_batches > 0 and batches_seen % progress_every_batches == 0:
                print(
                    f"Streaming combined pass: {records_seen:,} records "
                    f"({batches_seen:,} batches)",
                    flush=True,
                )
            if max_batches is not None and batches_seen >= max_batches:
                break

        if next(reuse_reader, None) is not None and not allow_trailing_reuse_rows:
            raise ValueError(
                f"{reuse_publication_path} contains more rows than the current evaluation"
            )

    if records_seen != accumulator.records:
        raise ValueError(
            "Streaming accumulator and gene writer disagree: "
            f"{accumulator.records:,} != {records_seen:,} records"
        )
    standard_tmp.replace(standard_path)
    publication_tmp.replace(publication_path)
    auroc_summary = nan_summary(gene_auroc[:records_seen])
    ap_summary = nan_summary(gene_average_precision[:records_seen])
    gene_summary = {
        "per_gene_auroc_mean": auroc_summary["mean"],
        "per_gene_auroc_median": auroc_summary["median"],
        "per_gene_auroc_q25": auroc_summary["q25"],
        "per_gene_auroc_q75": auroc_summary["q75"],
        "per_gene_average_precision_mean": ap_summary["mean"],
        "per_gene_average_precision_median": ap_summary["median"],
        "per_gene_average_precision_q25": ap_summary["q25"],
        "per_gene_average_precision_q75": ap_summary["q75"],
    }
    return accumulator, batches_seen, gene_summary


def build_loader(
    dataset: Evo2GoEmbeddingDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=evo2_go_collate_fn,
    )


@torch.no_grad()
def collect_probabilities_and_targets(
    model: DualEncoderModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    batches_seen = 0
    for batch in loader:
        embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
        batch_targets = batch["targets"].to(device=device, dtype=torch.float32)
        probabilities.append(model.predict_proba(embeddings).cpu())
        targets.append(batch_targets.cpu())
        batches_seen += 1
        if max_batches is not None and batches_seen >= max_batches:
            break

    if not probabilities:
        raise ValueError("Evaluation loader produced no batches")
    return torch.cat(probabilities, dim=0), torch.cat(targets, dim=0), batches_seen


def run_streaming_evaluation(
    *,
    args: argparse.Namespace,
    bundle,
    loader: DataLoader,
    model: DualEncoderModel,
    checkpoint: dict,
    device: torch.device,
    started: float,
) -> int:
    if args.save_score_matrix:
        raise ValueError(
            "--save-score-matrix is incompatible with --streaming because the "
            "full dense matrix is the source of the OOM risk. Use "
            "--no-save-score-matrix."
        )
    if args.reuse_publication_gene_metrics is not None and not args.write_gene_metrics:
        raise ValueError(
            "--reuse-publication-gene-metrics cannot be combined with "
            "--no-write-gene-metrics"
        )

    evaluation_mode = (
        "streaming_single_pass_reused_rank_metrics"
        if args.reuse_publication_gene_metrics is not None
        else (
            "streaming_two_pass"
            if args.write_gene_metrics
            else "streaming_single_pass_no_gene_metrics"
        )
    )

    model_n_terms = bundle.vocabulary.n_terms
    vocabulary_rows = load_vocabulary_rows(args.vocabulary, model_n_terms)
    term_indices, evaluated_vocabulary_rows, excluded_roots = streaming_term_indices(
        vocabulary_rows,
        exclude_go_roots=args.exclude_go_roots,
    )
    summary_path = args.output_dir / f"{args.split_name}_evaluation_summary.tsv"
    curve_path = args.output_dir / f"{args.split_name}_threshold_curve.tsv"
    per_term_path = args.output_dir / f"{args.split_name}_go_term_fmax_vs_frequency.tsv"
    per_gene_path = args.output_dir / f"{args.split_name}_gene_metrics.tsv"
    publication_summary_path = args.publication_summary_output or (
        args.output_dir / f"{args.split_name}_publication_summary.tsv"
    )
    publication_threshold_path = args.publication_threshold_output or (
        args.output_dir / f"{args.split_name}_publication_threshold_metrics.tsv"
    )
    publication_gene_path = args.publication_gene_output or (
        args.output_dir / f"{args.split_name}_publication_gene_metrics.tsv"
    )
    publication_go_term_path = args.publication_go_term_output or (
        args.output_dir / f"{args.split_name}_publication_go_term_metrics.tsv"
    )
    publication_calibration_path = args.publication_calibration_output or (
        args.output_dir / f"{args.split_name}_publication_calibration_metrics.tsv"
    )

    thresholds = evaluation_threshold_grid(
        args.threshold_steps,
        args.operating_threshold,
    )
    gene_summary: dict[str, float] | None = None
    if args.reuse_publication_gene_metrics is not None:
        if args.operating_threshold is None:
            raise ValueError(
                "--reuse-publication-gene-metrics requires --operating-threshold"
            )
        if not args.publication_metrics:
            raise ValueError(
                "--reuse-publication-gene-metrics requires --publication-metrics"
            )
        accumulator, batches_seen, gene_summary = (
            collect_streaming_statistics_reusing_gene_rank_metrics(
                per_gene_path,
                publication_gene_path,
                args.reuse_publication_gene_metrics,
                model=model,
                loader=loader,
                device=device,
                term_indices=term_indices,
                thresholds=thresholds,
                operating_threshold=float(args.operating_threshold),
                calibration_bins=args.publication_calibration_bins,
                max_batches=args.max_batches,
                progress_every_batches=args.progress_every_batches,
                allow_trailing_reuse_rows=(
                    args.max_records is not None or args.max_batches is not None
                ),
            )
        )
    else:
        accumulator, batches_seen = collect_streaming_statistics(
            model,
            loader,
            device=device,
            term_indices=term_indices,
            thresholds=thresholds,
            calibration_bins=args.publication_calibration_bins,
            max_batches=args.max_batches,
            progress_every_batches=args.progress_every_batches,
        )
    result = accumulator.fmax_result()
    operating_threshold = (
        float(result.threshold)
        if args.operating_threshold is None
        else float(args.operating_threshold)
    )
    operating_metrics = metrics_at_threshold(result, operating_threshold)

    write_threshold_curve(curve_path, result)
    write_streaming_term_metrics(
        per_term_path,
        vocabulary_rows=evaluated_vocabulary_rows,
        accumulator=accumulator,
        global_threshold=operating_threshold,
        include_rank_columns=False,
    )
    threshold_rows, threshold_summary = streaming_threshold_rows(accumulator)
    if args.publication_metrics:
        write_streaming_threshold_metrics(publication_threshold_path, threshold_rows)
        go_term_summary = write_streaming_term_metrics(
            publication_go_term_path,
            vocabulary_rows=evaluated_vocabulary_rows,
            accumulator=accumulator,
            global_threshold=operating_threshold,
            include_rank_columns=True,
        )
        calibration_summary = write_streaming_calibration_metrics(
            publication_calibration_path,
            accumulator,
        )
    else:
        go_term_summary = {}
        calibration_summary = {}

    print(
        f"Streaming pass 1 complete: {accumulator.records:,} records; "
        f"Fmax={result.fmax:.8f} at threshold={result.threshold:.8f}; "
        f"operating threshold={operating_threshold:.8f}",
        flush=True,
    )
    if gene_summary is None and args.write_gene_metrics:
        gene_summary = write_streaming_gene_metrics(
            per_gene_path,
            publication_gene_path if args.publication_metrics else None,
            model=model,
            loader=loader,
            device=device,
            term_indices=term_indices,
            global_threshold=operating_threshold,
            expected_records=accumulator.records,
            max_batches=args.max_batches,
            progress_every_batches=args.progress_every_batches,
        )
    if gene_summary is None:
        gene_summary = {}

    elapsed = time.time() - started
    write_key_value_tsv(
        summary_path,
        [
            ("status", "complete"),
            (
                "evaluation_mode",
                evaluation_mode,
            ),
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", checkpoint.get("step", "")),
            ("checkpoint_best_fmax", checkpoint.get("best_fmax", "")),
            ("embedding_root", args.embedding_root),
            ("vocabulary", args.vocabulary),
            ("adjacency", args.adjacency),
            ("label_cache", args.label_cache),
            ("split_manifest", args.split_manifest),
            ("split_name", args.split_name),
            ("tax_ids_file", args.tax_ids_file or ""),
            ("tax_ids_count", len(args.tax_ids or [])),
            ("records_before_label_filter", bundle.records_before_label_filter),
            ("records_evaluated", accumulator.records),
            ("batches_evaluated_per_pass", batches_seen),
            ("model_n_terms", model_n_terms),
            ("n_terms", accumulator.n_terms),
            ("exclude_go_roots", int(bool(args.exclude_go_roots))),
            ("excluded_go_root_ids", ",".join(excluded_roots)),
            ("batch_size", args.batch_size),
            ("threshold_steps", len(result.thresholds)),
            ("fmax", f"{result.fmax:.8f}"),
            ("threshold", f"{result.threshold:.8f}"),
            ("precision", f"{result.precision:.8f}"),
            ("recall", f"{result.recall:.8f}"),
            ("operating_threshold", f"{operating_threshold:.8f}"),
            ("precision_at_operating_threshold", f"{operating_metrics['precision']:.8f}"),
            ("recall_at_operating_threshold", f"{operating_metrics['recall']:.8f}"),
            ("f1_at_operating_threshold", f"{operating_metrics['f1']:.8f}"),
            ("save_score_matrix", 0),
            ("score_matrix", ""),
            ("publication_metrics", int(bool(args.publication_metrics))),
            (
                "publication_summary",
                publication_summary_path if args.publication_metrics else "",
            ),
            ("device", device),
            ("elapsed_seconds", f"{elapsed:.3f}"),
        ],
    )

    if args.publication_metrics:
        publication_summary_rows: list[tuple[object, object]] = [
            ("status", "complete"),
            (
                "evaluation_mode",
                evaluation_mode,
            ),
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", checkpoint.get("step", "")),
            ("embedding_root", args.embedding_root),
            ("vocabulary", args.vocabulary),
            ("adjacency", args.adjacency),
            ("label_cache", args.label_cache),
            ("split_manifest", args.split_manifest),
            ("split_name", args.split_name),
            ("records_evaluated", accumulator.records),
            ("model_n_terms", model_n_terms),
            ("n_terms", accumulator.n_terms),
            ("exclude_go_roots", int(bool(args.exclude_go_roots))),
            ("excluded_go_root_ids", ",".join(excluded_roots)),
            ("threshold_steps", len(result.thresholds)),
            ("global_threshold", f"{operating_threshold:.8f}"),
            (
                "global_threshold_source",
                "prespecified" if args.operating_threshold is not None else "test_fmax",
            ),
            ("gene_averaged_fmax", f"{result.fmax:.8f}"),
            ("fmax_threshold", f"{result.threshold:.8f}"),
            ("gene_averaged_precision_at_fmax", f"{result.precision:.8f}"),
            ("gene_averaged_recall_at_fmax", f"{result.recall:.8f}"),
            (
                "gene_averaged_precision_at_global_threshold",
                f"{operating_metrics['precision']:.8f}",
            ),
            (
                "gene_averaged_recall_at_global_threshold",
                f"{operating_metrics['recall']:.8f}",
            ),
            (
                "gene_averaged_f1_at_global_threshold",
                f"{operating_metrics['f1']:.8f}",
            ),
            (
                "per_gene_rank_metrics",
                "exact_within_gene" if args.write_gene_metrics else "not_computed",
            ),
            (
                "per_gene_rank_metrics_source",
                (
                    args.reuse_publication_gene_metrics or "current_evaluation"
                    if args.write_gene_metrics
                    else ""
                ),
            ),
            (
                "per_go_term_rank_metrics",
                "not_computed_streaming_requires_external_sort",
            ),
            ("score_matrix", "not_written_streaming"),
            ("publication_threshold_metrics", publication_threshold_path),
            (
                "publication_gene_metrics",
                publication_gene_path if args.write_gene_metrics else "",
            ),
            ("publication_go_term_metrics", publication_go_term_path),
            ("publication_calibration_metrics", publication_calibration_path),
        ]
        for metric_map in (
            threshold_summary,
            gene_summary,
            go_term_summary,
            calibration_summary,
        ):
            publication_summary_rows.extend(
                (key, finite_float(float(value))) for key, value in metric_map.items()
            )
        write_publication_summary(publication_summary_path, publication_summary_rows)
        print(f"Wrote {publication_summary_path}")
        print(f"Wrote {publication_threshold_path}")
        if args.write_gene_metrics:
            print(f"Wrote {publication_gene_path}")
        print(f"Wrote {publication_go_term_path}")
        print(f"Wrote {publication_calibration_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {curve_path}")
    print(f"Wrote {per_term_path}")
    if args.write_gene_metrics:
        print(f"Wrote {per_gene_path}")
    return 0


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    pre_args, _ = pre_parser.parse_known_args()
    project_root = resolve_project_root(pre_args.project_root)
    config_defaults = defaults_from_config(
        pre_args.config,
        project_root=project_root,
        section="evaluation",
    )

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_GO_VOCABULARY)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-cached-shards", type=int, default=2)
    parser.add_argument("--threshold-steps", type=int, default=100)
    parser.add_argument(
        "--operating-threshold",
        type=float,
        help=(
            "Optional prespecified threshold for threshold-dependent metrics; "
            "Fmax is still calculated independently over the threshold grid."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use bounded-memory two-pass evaluation. This is required for the "
            "full test split and cannot write a monolithic score matrix."
        ),
    )
    parser.add_argument(
        "--exclude-go-roots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the three uninformative GO ontology roots from streaming metrics.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=1000,
        help="Print streaming progress every N batches; 0 disables progress messages.",
    )
    parser.add_argument(
        "--save-score-matrix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write raw sigmoid probabilities, binary targets, gene identities, "
            "and GO vocabulary metadata to <split>_score_matrix.npz."
        ),
    )
    parser.add_argument(
        "--score-matrix-output",
        type=Path,
        help="Optional explicit path for --save-score-matrix output.",
    )
    parser.add_argument(
        "--publication-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write publication-oriented gene-level, GO-term-level, and "
            "threshold-grid statistics including AUROC and average precision."
        ),
    )
    parser.add_argument(
        "--write-gene-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write per-gene metric tables. Disable for a one-pass streaming "
            "evaluation when only aggregate and per-GO-term metrics are needed."
        ),
    )
    parser.add_argument(
        "--drop-examples-without-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop records with no retained labels. Disable for source-stratified "
            "evaluation so every source uses the same test population."
        ),
    )
    parser.add_argument("--publication-summary-output", type=Path)
    parser.add_argument("--publication-threshold-output", type=Path)
    parser.add_argument("--publication-gene-output", type=Path)
    parser.add_argument("--publication-go-term-output", type=Path)
    parser.add_argument("--publication-calibration-output", type=Path)
    parser.add_argument(
        "--reuse-publication-gene-metrics",
        type=Path,
        help=(
            "For a streaming fixed-threshold rerun, reuse threshold-independent "
            "per-gene AUROC and average precision from this ordered publication "
            "gene-metrics table and write all gene metrics during the first pass."
        ),
    )
    parser.add_argument("--publication-calibration-bins", type=int, default=20)
    parser.add_argument(
        "--tax-id",
        dest="tax_ids",
        action="append",
        help="Restrict loaded embedding records to these tax IDs. Can be repeated or comma-delimited.",
    )
    parser.add_argument(
        "--tax-ids-file",
        "--tax-id-file",
        dest="tax_ids_file",
        type=Path,
        help="Restrict loaded embedding records to tax IDs listed in this file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved evaluation inputs without loading data or a checkpoint.",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless supplied by --config")
    if args.label_cache is None:
        parser.error("--label-cache is required unless supplied by --config")
    if args.split_manifest is None:
        parser.error("--split-manifest is required unless supplied by --config")

    file_tax_ids = tax_ids_from_file(args.tax_ids_file)
    if file_tax_ids:
        args.tax_ids = [*(args.tax_ids or []), *file_tax_ids]

    if args.threshold_steps <= 1:
        raise ValueError("--threshold-steps must be > 1")
    if args.operating_threshold is not None and not 0.0 <= args.operating_threshold <= 1.0:
        raise ValueError("--operating-threshold must be between 0 and 1")
    if args.dry_run:
        for key in (
            "config",
            "project_root",
            "checkpoint",
            "embedding_root",
            "vocabulary",
            "adjacency",
            "label_cache",
            "split_manifest",
            "split_name",
            "output_dir",
            "batch_size",
            "num_workers",
            "max_batches",
            "max_records",
            "threshold_steps",
            "operating_threshold",
            "device",
            "streaming",
            "exclude_go_roots",
            "progress_every_batches",
            "save_score_matrix",
            "score_matrix_output",
            "publication_metrics",
            "write_gene_metrics",
            "drop_examples_without_labels",
            "publication_summary_output",
            "publication_threshold_output",
            "publication_gene_output",
            "publication_go_term_output",
            "publication_calibration_output",
            "reuse_publication_gene_metrics",
            "publication_calibration_bins",
            "tax_ids_file",
            "tax_ids",
            "tax_ids_count",
        ):
            if key == "tax_ids_count":
                value = len(args.tax_ids or [])
            elif key == "tax_ids":
                value = summarize_tax_ids(args.tax_ids)
            else:
                value = getattr(args, key)
            print(f"{key}\t{value}")
        return 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_evo2_go_dataset(
        embedding_root=args.embedding_root,
        vocabulary_path=args.vocabulary,
        label_index_cache_path=args.label_cache,
        split_manifest_path=args.split_manifest,
        split_name=args.split_name,
        tax_ids=args.tax_ids,
        max_records=args.max_records,
        max_cached_shards=args.max_cached_shards,
        drop_examples_without_labels=args.drop_examples_without_labels,
        preselect_split_records=args.streaming,
    )
    loader = build_loader(
        bundle.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    adjacency = load_sparse_adjacency_npz(args.adjacency)
    n_terms = bundle.vocabulary.n_terms
    model = DualEncoderModel(
        adjacency,
        DualEncoderConfig(
            go_encoder=GoTermGCNConfig(n_terms=n_terms),
            sequence_encoder=SequenceEncoderConfig(),
        ),
    ).to(device)
    checkpoint = load_training_checkpoint(
        args.checkpoint,
        model=model,
        map_location=device,
    )

    if args.streaming:
        return run_streaming_evaluation(
            args=args,
            bundle=bundle,
            loader=loader,
            model=model,
            checkpoint=checkpoint,
            device=device,
            started=started,
        )

    probabilities, targets, batches_seen = collect_probabilities_and_targets(
        model,
        loader,
        device=device,
        max_batches=args.max_batches,
    )
    thresholds = evaluation_threshold_grid(
        args.threshold_steps,
        args.operating_threshold,
    )
    result = cafa_gene_averaged_fmax(probabilities, targets, thresholds=thresholds)
    operating_threshold = (
        float(result.threshold)
        if args.operating_threshold is None
        else float(args.operating_threshold)
    )
    operating_metrics = metrics_at_threshold(result, operating_threshold)

    summary_path = args.output_dir / f"{args.split_name}_evaluation_summary.tsv"
    curve_path = args.output_dir / f"{args.split_name}_threshold_curve.tsv"
    per_term_path = args.output_dir / f"{args.split_name}_go_term_fmax_vs_frequency.tsv"
    per_gene_path = args.output_dir / f"{args.split_name}_gene_metrics.tsv"
    score_matrix_path = args.score_matrix_output or (
        args.output_dir / f"{args.split_name}_score_matrix.npz"
    )
    publication_summary_path = args.publication_summary_output or (
        args.output_dir / f"{args.split_name}_publication_summary.tsv"
    )
    publication_threshold_path = args.publication_threshold_output or (
        args.output_dir / f"{args.split_name}_publication_threshold_metrics.tsv"
    )
    publication_gene_path = args.publication_gene_output or (
        args.output_dir / f"{args.split_name}_publication_gene_metrics.tsv"
    )
    publication_go_term_path = args.publication_go_term_output or (
        args.output_dir / f"{args.split_name}_publication_go_term_metrics.tsv"
    )
    publication_calibration_path = args.publication_calibration_output or (
        args.output_dir / f"{args.split_name}_publication_calibration_metrics.tsv"
    )
    write_key_value_tsv(
        summary_path,
        [
            ("status", "complete"),
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", checkpoint.get("step", "")),
            ("checkpoint_best_fmax", checkpoint.get("best_fmax", "")),
            ("embedding_root", args.embedding_root),
            ("vocabulary", args.vocabulary),
            ("adjacency", args.adjacency),
            ("label_cache", args.label_cache),
            ("split_manifest", args.split_manifest),
            ("split_name", args.split_name),
            ("tax_ids_file", args.tax_ids_file or ""),
            ("tax_ids_count", len(args.tax_ids or [])),
            ("records_before_label_filter", bundle.records_before_label_filter),
            ("records_evaluated", probabilities.shape[0]),
            ("batches_evaluated", batches_seen),
            ("n_terms", n_terms),
            ("batch_size", args.batch_size),
            ("threshold_steps", len(result.thresholds)),
            ("fmax", f"{result.fmax:.8f}"),
            ("threshold", f"{result.threshold:.8f}"),
            ("precision", f"{result.precision:.8f}"),
            ("recall", f"{result.recall:.8f}"),
            ("operating_threshold", f"{operating_threshold:.8f}"),
            ("precision_at_operating_threshold", f"{operating_metrics['precision']:.8f}"),
            ("recall_at_operating_threshold", f"{operating_metrics['recall']:.8f}"),
            ("f1_at_operating_threshold", f"{operating_metrics['f1']:.8f}"),
            ("save_score_matrix", int(bool(args.save_score_matrix))),
            ("score_matrix", score_matrix_path if args.save_score_matrix else ""),
            ("publication_metrics", int(bool(args.publication_metrics))),
            (
                "publication_summary",
                publication_summary_path if args.publication_metrics else "",
            ),
            ("device", device),
            ("elapsed_seconds", f"{time.time() - started:.3f}"),
        ],
    )
    write_threshold_curve(curve_path, result)
    write_per_term_metrics(
        per_term_path,
        vocabulary_path=args.vocabulary,
        probabilities=probabilities,
        targets=targets,
        global_threshold=operating_threshold,
        thresholds=thresholds,
    )
    write_per_gene_metrics(
        per_gene_path,
        records=bundle.dataset.records,
        probabilities=probabilities,
        targets=targets,
        global_threshold=operating_threshold,
    )
    if args.save_score_matrix:
        write_score_matrix(
            score_matrix_path,
            records=bundle.dataset.records,
            vocabulary_path=args.vocabulary,
            probabilities=probabilities,
            targets=targets,
            global_threshold=operating_threshold,
        )
        print(f"Wrote {score_matrix_path}")
    if args.publication_metrics:
        threshold_summary = write_publication_threshold_metrics(
            publication_threshold_path,
            probabilities=probabilities,
            targets=targets,
            thresholds=thresholds,
        )
        gene_summary = write_publication_gene_metrics(
            publication_gene_path,
            records=bundle.dataset.records,
            probabilities=probabilities,
            targets=targets,
            global_threshold=operating_threshold,
        )
        go_term_summary = write_publication_go_term_metrics(
            publication_go_term_path,
            vocabulary_path=args.vocabulary,
            probabilities=probabilities,
            targets=targets,
            global_threshold=operating_threshold,
            thresholds=thresholds,
        )
        calibration_summary = write_publication_calibration_metrics(
            publication_calibration_path,
            probabilities=probabilities,
            targets=targets,
            n_bins=args.publication_calibration_bins,
        )
        publication_summary_rows: list[tuple[object, object]] = [
            ("status", "complete"),
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", checkpoint.get("step", "")),
            ("embedding_root", args.embedding_root),
            ("vocabulary", args.vocabulary),
            ("adjacency", args.adjacency),
            ("label_cache", args.label_cache),
            ("split_manifest", args.split_manifest),
            ("split_name", args.split_name),
            ("records_evaluated", probabilities.shape[0]),
            ("n_terms", n_terms),
            ("threshold_steps", len(result.thresholds)),
            ("global_threshold", f"{operating_threshold:.8f}"),
            (
                "global_threshold_source",
                "prespecified" if args.operating_threshold is not None else "test_fmax",
            ),
            ("gene_averaged_fmax", f"{result.fmax:.8f}"),
            ("fmax_threshold", f"{result.threshold:.8f}"),
            ("gene_averaged_precision_at_fmax", f"{result.precision:.8f}"),
            ("gene_averaged_recall_at_fmax", f"{result.recall:.8f}"),
            (
                "gene_averaged_precision_at_global_threshold",
                f"{operating_metrics['precision']:.8f}",
            ),
            (
                "gene_averaged_recall_at_global_threshold",
                f"{operating_metrics['recall']:.8f}",
            ),
            (
                "gene_averaged_f1_at_global_threshold",
                f"{operating_metrics['f1']:.8f}",
            ),
            ("publication_threshold_metrics", publication_threshold_path),
            ("publication_gene_metrics", publication_gene_path),
            ("publication_go_term_metrics", publication_go_term_path),
            ("publication_calibration_metrics", publication_calibration_path),
        ]
        for metric_map in (
            threshold_summary,
            gene_summary,
            go_term_summary,
            calibration_summary,
        ):
            publication_summary_rows.extend(
                (key, finite_float(float(value))) for key, value in metric_map.items()
            )
        write_publication_summary(publication_summary_path, publication_summary_rows)
        print(f"Wrote {publication_summary_path}")
        print(f"Wrote {publication_threshold_path}")
        print(f"Wrote {publication_gene_path}")
        print(f"Wrote {publication_go_term_path}")
        print(f"Wrote {publication_calibration_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {curve_path}")
    print(f"Wrote {per_term_path}")
    print(f"Wrote {per_gene_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
