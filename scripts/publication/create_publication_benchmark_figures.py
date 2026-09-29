#!/usr/bin/env python3
"""Generate publication-ready benchmark figures from Evo2-GO test metrics.

This script complements figure_01_threshold_curve.py and
figure_02_go_term_frequency.py. It uses only saved evaluation artifacts, so it
does not rerun model inference.
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evo2_go_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "human_test_taxon_val_1919taxa_min_count_15000_bs256_mean_loss_3h"
DEFAULT_BUILD_ID = "all_labels_true_path_1919taxa_human_test_min_count_15000_v1"
DEFAULT_GO_BASIC_OBO = PROJECT_ROOT / "data" / "go" / "go-basic.obo"

INK = "#222222"
MUTED = "#666666"
GRID = "#d6d6d6"
BLUE = "#2f6fae"
ORANGE = "#c45a14"
GREEN = "#4f8f46"
PURPLE = "#7b3f98"

GO_ROOT_LABELS = {
    "GO:0003674": "molecular function (MF)",
    "GO:0005575": "cellular component (CC)",
    "GO:0008150": "biological process (BP)",
}

NAMESPACE_ORDER = [
    "biological_process",
    "cellular_component",
    "molecular_function",
]
NAMESPACE_LABELS = {
    "biological_process": "Biological process",
    "cellular_component": "Cellular component",
    "molecular_function": "Molecular function",
}
NAMESPACE_SHORT = {
    "biological_process": "BP",
    "cellular_component": "CC",
    "molecular_function": "MF",
}
NAMESPACE_COLORS = {
    "biological_process": BLUE,
    "cellular_component": ORANGE,
    "molecular_function": GREEN,
}
HEATMAP_NAMESPACE_ORDER = [
    "molecular_function",
    "biological_process",
    "cellular_component",
]
NAMESPACE_SLUGS = {
    "biological_process": "biological_process",
    "cellular_component": "cellular_component",
    "molecular_function": "molecular_function",
}
OCCURRENCE_BIN_EDGES = [
    0,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    float("inf"),
]
OCCURRENCE_BIN_LABELS = [
    "15-25k",
    "25-50k",
    "50-100k",
    "100-250k",
    "250-500k",
    "500k-1M",
    "1-2.5M",
    "2.5-5M",
    ">=5M",
]


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_key_value_tsv(path: Path) -> dict[str, str]:
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["metric"]: row["value"] for row in reader}


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def write_caption(path: Path, caption: str) -> None:
    path.write_text(
        textwrap.fill(
            caption,
            width=88,
            break_long_words=False,
            break_on_hyphens=False,
        )
        + "\n",
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def as_float(summary: dict[str, str], key: str) -> float:
    return float(summary[key])


def as_int(summary: dict[str, str], key: str) -> int:
    return int(float(summary[key]))


def is_mmseqs_test(summary: dict[str, str]) -> bool:
    return "mmseqs_linclust" in summary.get("split_manifest", "")


def test_set_label(summary: dict[str, str], *, held_out: bool = True) -> str:
    if is_mmseqs_test(summary):
        return "held-out MMseqs-cluster test set" if held_out else "MMseqs-cluster test set"
    return "held-out human test set" if held_out else "human test set"


def test_record_label(summary: dict[str, str]) -> str:
    return "sequences" if is_mmseqs_test(summary) else "genes"


def root_exclusion_note(summary: dict[str, str]) -> str:
    if summary.get("exclude_go_roots", "0").lower() not in {"1", "true", "yes"}:
        return ""
    root_ids = [
        go_id
        for go_id in summary.get("excluded_go_root_ids", "").split(",")
        if go_id
    ]
    labels = [f"{GO_ROOT_LABELS.get(go_id, 'ontology root')} ({go_id})" for go_id in root_ids]
    if not labels:
        return " The three GO ontology root terms were excluded from all reported metrics."
    if len(labels) == 1:
        joined = labels[0]
    else:
        joined = ", ".join(labels[:-1]) + ", and " + labels[-1]
    return f" The ontology root terms {joined} were excluded from all reported metrics."


def figure_03_stem_name(summary: dict[str, str]) -> str:
    if is_mmseqs_test(summary):
        return "figure_03_mmseqs_test_roc_pr_curves"
    return "figure_03_human_test_roc_pr_curves"


def make_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def count_formatter(value: float, _position: int) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def binned_summary(metrics: pd.DataFrame, metric_column: str, bins: int = 8) -> pd.DataFrame:
    frame = metrics[["database_frequency", metric_column]].dropna().copy()
    frame = frame.loc[frame["database_frequency"] > 0]
    frame["bin"] = pd.qcut(frame["database_frequency"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    return grouped.agg(
        frequency_median=("database_frequency", "median"),
        metric_median=(metric_column, "median"),
        metric_q25=(metric_column, lambda values: values.quantile(0.25)),
        metric_q75=(metric_column, lambda values: values.quantile(0.75)),
        terms=(metric_column, "size"),
    ).reset_index(drop=True)


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return values, values
    y = np.arange(1, values.size + 1, dtype=float) / float(values.size)
    return values, y


def load_go_depths_from_obo(go_basic_obo: Path, go_ids: pd.Series) -> pd.DataFrame:
    target_ids = set(go_ids.dropna().astype(str))
    parents_by_id: dict[str, set[str]] = {}
    current_id: str | None = None
    current_parents: set[str] = set()
    in_term = False

    def finish_term() -> None:
        if current_id is not None:
            parents_by_id[current_id] = set(current_parents)

    with go_basic_obo.open("rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finish_term()
                current_id = None
                current_parents = set()
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                finish_term()
                current_id = None
                current_parents = set()
                in_term = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                current_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("is_a: "):
                current_parents.add(line.split("is_a: ", 1)[1].split()[0])
            elif line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of":
                    current_parents.add(parts[2])
        finish_term()

    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(go_id: str) -> int:
        if go_id in memo:
            return memo[go_id]
        if go_id in visiting:
            return 0
        visiting.add(go_id)
        parents = [parent for parent in parents_by_id.get(go_id, set()) if parent in parents_by_id]
        value = 1 + max(depth(parent) for parent in parents) if parents else 0
        visiting.remove(go_id)
        memo[go_id] = value
        return value

    rows = [
        {
            "GO_ID": go_id,
            "go_graph_depth": depth(go_id),
        }
        for go_id in sorted(target_ids)
        if go_id in parents_by_id
    ]
    return pd.DataFrame(rows)


def plot_roc_pr(
    threshold_metrics: pd.DataFrame,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    metrics = make_numeric(
        threshold_metrics.copy(),
        [
            "threshold",
            "gene_averaged_precision",
            "gene_averaged_recall",
            "gene_averaged_fpr",
            "micro_precision",
            "micro_recall",
            "micro_fpr",
        ],
    )
    gene_roc_auc = as_float(publication_summary, "gene_averaged_roc_auc_threshold_grid")
    micro_roc_auc = as_float(publication_summary, "micro_roc_auc_threshold_grid")
    gene_pr_auc = as_float(publication_summary, "gene_averaged_pr_auc_threshold_grid")
    micro_pr_auc = as_float(publication_summary, "micro_pr_auc_threshold_grid")
    records = as_int(publication_summary, "records_evaluated")
    terms = as_int(publication_summary, "n_terms")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    roc = metrics.sort_values("gene_averaged_fpr")
    axes[0].plot(
        roc["gene_averaged_fpr"],
        roc["gene_averaged_recall"],
        color=BLUE,
        linewidth=2.0,
        label=f"Gene-averaged AUROC {gene_roc_auc:.3f}",
    )
    micro_roc = metrics.sort_values("micro_fpr")
    axes[0].plot(
        micro_roc["micro_fpr"],
        micro_roc["micro_recall"],
        color=PURPLE,
        linewidth=1.8,
        label=f"Micro AUROC {micro_roc_auc:.3f}",
    )
    axes[0].plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].set_title("A  ROC threshold curve", loc="left", fontweight="semibold")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("Recall")

    pr = metrics.sort_values("gene_averaged_recall")
    axes[1].plot(
        pr["gene_averaged_recall"],
        pr["gene_averaged_precision"],
        color=BLUE,
        linewidth=2.0,
        label=f"Gene-averaged AUPRC {gene_pr_auc:.3f}",
    )
    micro_pr = metrics.sort_values("micro_recall")
    axes[1].plot(
        micro_pr["micro_recall"],
        micro_pr["micro_precision"],
        color=PURPLE,
        linewidth=1.8,
        label=f"Micro AUPRC {micro_pr_auc:.3f}",
    )
    axes[1].set_title("B  Precision-recall threshold curve", loc="left", fontweight="semibold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="both", color=GRID, linewidth=0.35, alpha=0.85)
        ax.legend(frameon=False, loc="lower right")

    fig.text(
        0.5,
        -0.01,
        f"{test_set_label(publication_summary, held_out=False)}: "
        f"{records:,} {test_record_label(publication_summary)} and "
        f"{terms:,} retained non-root true-path GO terms",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.92, wspace=0.26)

    stem = output_dir / figure_03_stem_name(publication_summary)
    save_figure(fig, stem, dpi=dpi)
    write_key_value_tsv(
        stem.with_name(stem.name + "_summary.tsv"),
        [
            ("gene_averaged_roc_auc_threshold_grid", f"{gene_roc_auc:.8f}"),
            ("micro_roc_auc_threshold_grid", f"{micro_roc_auc:.8f}"),
            ("gene_averaged_pr_auc_threshold_grid", f"{gene_pr_auc:.8f}"),
            ("micro_pr_auc_threshold_grid", f"{micro_pr_auc:.8f}"),
            ("threshold_points", len(metrics)),
        ],
    )
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            f"Figure 3. ROC and precision-recall benchmark curves for the "
            f"{test_set_label(publication_summary)}. The best checkpoint was evaluated "
            f"on {records:,} {test_record_label(publication_summary)} and {terms:,} "
            "retained non-root true-path GO terms. The ROC panel shows recall "
            "as a function of false positive rate for the same 0.01 to 0.99 "
            "threshold grid used for descriptive (oracle) test-set Fmax; the "
            "precision-recall panel shows the "
            "corresponding precision-recall tradeoff. Gene-averaged curves first "
            "compute each quantity per gene and then average across genes, whereas "
            "micro curves pool gene-GO decisions before computing each quantity. "
            f"Threshold-grid AUROC was {gene_roc_auc:.3f} gene-averaged and "
            f"{micro_roc_auc:.3f} micro; threshold-grid AUPRC was {gene_pr_auc:.3f} "
            f"gene-averaged and {micro_pr_auc:.3f} micro. Fmax maximizes "
            "gene-averaged F1 over this test-set grid and should not be interpreted "
            "as performance at a validation-selected operating threshold."
            + root_exclusion_note(publication_summary)
        ),
    )


def plot_calibration(
    calibration_metrics: pd.DataFrame,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    metrics = make_numeric(
        calibration_metrics.copy(),
        [
            "bin_lower",
            "bin_upper",
            "count",
            "mean_score",
            "observed_positive_fraction",
            "absolute_calibration_gap",
        ],
    )
    metrics = metrics.loc[metrics["count"].fillna(0) > 0].copy()
    metrics["bin_width"] = metrics["bin_upper"] - metrics["bin_lower"]
    brier = as_float(publication_summary, "brier_score")
    ece = as_float(publication_summary, "expected_calibration_error_20bin")
    mce = as_float(publication_summary, "maximum_calibration_error_20bin")
    records = as_int(publication_summary, "records_evaluated")
    terms = as_int(publication_summary, "n_terms")

    fig = plt.figure(figsize=(7.1, 3.15))
    grid = GridSpec(1, 2, width_ratios=(1.25, 1.0), wspace=0.28, figure=fig)
    ax = fig.add_subplot(grid[0, 0])
    hist_ax = fig.add_subplot(grid[0, 1])

    sizes = 30.0 + 130.0 * np.sqrt(metrics["count"] / metrics["count"].max())
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.plot(
        metrics["mean_score"],
        metrics["observed_positive_fraction"],
        color=BLUE,
        linewidth=1.7,
        alpha=0.9,
    )
    ax.scatter(
        metrics["mean_score"],
        metrics["observed_positive_fraction"],
        s=sizes,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.set_title("A  Reliability curve", loc="left", fontweight="semibold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="both", color=GRID, linewidth=0.35, alpha=0.85)
    ax.text(
        0.04,
        0.96,
        f"Brier {brier:.3f}\nECE {ece:.3f}\nMCE {mce:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color=INK,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#dddddd"},
    )

    hist_ax.bar(
        metrics["bin_lower"],
        metrics["count"],
        width=metrics["bin_width"],
        align="edge",
        color=ORANGE,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )
    hist_ax.set_title("B  Score-bin occupancy", loc="left", fontweight="semibold")
    hist_ax.set_xlabel("Predicted probability bin")
    hist_ax.set_ylabel("Gene-GO decisions")
    hist_ax.set_xlim(0, 1)
    hist_ax.set_yscale("log")
    hist_ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    hist_ax.yaxis.set_major_formatter(FuncFormatter(count_formatter))
    hist_ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)

    stem = output_dir / "figure_04_probability_calibration"
    save_figure(fig, stem, dpi=dpi)
    write_key_value_tsv(
        stem.with_name(stem.name + "_summary.tsv"),
        [
            ("brier_score", f"{brier:.8f}"),
            ("expected_calibration_error_20bin", f"{ece:.8f}"),
            ("maximum_calibration_error_20bin", f"{mce:.8f}"),
            ("calibration_bins_with_observations", len(metrics)),
            ("gene_go_decisions", int(metrics["count"].sum())),
        ],
    )
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            f"Figure 4. Probability calibration on the {test_set_label(publication_summary)}. "
            f"Predicted probabilities for all pairs of test {test_record_label(publication_summary)} "
            f"and retained non-root GO terms ({records:,} "
            f"{test_record_label(publication_summary)} and {terms:,} terms) were grouped into "
            "20 equal-width score bins. The reliability curve compares each bin's "
            "mean predicted probability with its observed positive-label fraction; "
            "circle area is proportional to the number of gene-GO decisions in the "
            "bin. The histogram shows the same score-bin occupancy on a logarithmic "
            f"scale. The Brier score was {brier:.3f}, expected calibration error "
            f"was {ece:.3f}, and maximum calibration error was {mce:.3f}, indicating "
            "that the discriminative threshold scores should not be interpreted as "
            "well-calibrated posterior probabilities without calibration."
            + root_exclusion_note(publication_summary)
        ),
    )


def label_count_deciles(gene_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = gene_metrics[["label_count", "f1", "average_precision"]].dropna().copy()
    frame["label_count"] = pd.to_numeric(frame["label_count"], errors="coerce")
    frame = frame.dropna(subset=["label_count"])
    frame["label_decile"] = pd.qcut(frame["label_count"], q=10, duplicates="drop")
    grouped = frame.groupby("label_decile", observed=True)
    return grouped.agg(
        genes=("label_count", "size"),
        label_count_median=("label_count", "median"),
        label_count_min=("label_count", "min"),
        label_count_max=("label_count", "max"),
        median_f1=("f1", "median"),
        f1_q25=("f1", lambda values: values.quantile(0.25)),
        f1_q75=("f1", lambda values: values.quantile(0.75)),
        median_average_precision=("average_precision", "median"),
    ).reset_index(drop=True)


def plot_gene_level_discrimination(
    gene_metrics: pd.DataFrame,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    metrics = make_numeric(
        gene_metrics.copy(),
        ["label_count", "precision", "recall", "f1", "auroc", "average_precision"],
    )
    auroc = finite_values(metrics["auroc"])
    ap = finite_values(metrics["average_precision"])
    f1 = finite_values(metrics["f1"])
    x_auroc, y_auroc = ecdf(auroc)
    x_ap, y_ap = ecdf(ap)
    deciles = label_count_deciles(metrics)

    records = as_int(publication_summary, "records_evaluated")
    per_gene_auroc_median = as_float(publication_summary, "per_gene_auroc_median")
    per_gene_ap_median = as_float(publication_summary, "per_gene_average_precision_median")

    fig = plt.figure(figsize=(7.25, 3.25))
    grid = GridSpec(1, 2, width_ratios=(1.05, 1.15), wspace=0.28, figure=fig)
    ax = fig.add_subplot(grid[0, 0])
    decile_ax = fig.add_subplot(grid[0, 1])

    ax.plot(
        x_auroc,
        y_auroc,
        color=BLUE,
        linewidth=2.0,
        label=f"AUROC median {np.median(auroc):.3f}",
    )
    ax.plot(
        x_ap,
        y_ap,
        color=ORANGE,
        linewidth=2.0,
        label=f"Average precision median {np.median(ap):.3f}",
    )
    ax.set_title("A  Per-gene score-ranked metrics", loc="left", fontweight="semibold")
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Cumulative fraction of genes")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="both", color=GRID, linewidth=0.35, alpha=0.85)
    ax.legend(frameon=False, loc="lower right")

    decile_ax.fill_between(
        deciles["label_count_median"],
        deciles["f1_q25"],
        deciles["f1_q75"],
        color=BLUE,
        alpha=0.14,
        linewidth=0,
    )
    decile_ax.plot(
        deciles["label_count_median"],
        deciles["median_f1"],
        color=BLUE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Median F1",
    )
    decile_ax.plot(
        deciles["label_count_median"],
        deciles["median_average_precision"],
        color=ORANGE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Median AP",
    )
    decile_ax.set_title("B  Annotation-count dependence", loc="left", fontweight="semibold")
    decile_ax.set_xlabel("Median true labels per gene in decile")
    decile_ax.set_ylabel("Per-gene metric")
    decile_ax.set_ylim(-0.02, 1.02)
    decile_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    decile_ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)
    decile_ax.legend(frameon=False, loc="lower right")

    stem = output_dir / "figure_05_gene_level_discrimination"
    save_figure(fig, stem, dpi=dpi)
    deciles.to_csv(stem.with_name(stem.name + "_deciles.tsv"), sep="\t", index=False)
    write_key_value_tsv(
        stem.with_name(stem.name + "_summary.tsv"),
        [
            ("genes", records),
            ("median_f1_at_global_threshold", f"{np.median(f1):.8f}"),
            ("median_per_gene_auroc", f"{per_gene_auroc_median:.8f}"),
            ("median_per_gene_average_precision", f"{per_gene_ap_median:.8f}"),
            ("mean_true_labels_per_gene", f"{metrics['label_count'].mean():.8f}"),
            ("max_true_labels_per_gene", int(metrics["label_count"].max())),
        ],
    )
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            f"Figure 5. Sequence-level discrimination on the "
            f"{test_set_label(publication_summary)}. "
            f"The empirical cumulative distribution functions summarize per-gene "
            f"AUROC and average precision across {records:,} "
            f"{test_record_label(publication_summary)} using "
            "score-ranked GO-term probabilities. The second panel groups genes by "
            "deciles of true retained-label count and reports median F1 at the "
            "test-selected global Fmax threshold together with median average "
            "precision; the "
            "shaded band shows the interquartile range for F1. Median per-gene "
            f"AUROC was {per_gene_auroc_median:.3f}, median per-gene average "
            f"precision was {per_gene_ap_median:.3f}, and median F1 at the global "
            f"threshold was {np.median(f1):.3f}. The global threshold was selected "
            "to maximize gene-averaged F1 using test labels, so threshold-dependent "
            "F1 values are descriptive (oracle) test-set results rather than "
            "performance at a validation-selected operating threshold. Per-gene "
            "AUROC and average precision are score-ranked metrics and do not use "
            "that threshold."
            + root_exclusion_note(publication_summary)
        ),
    )


def namespace_summary(per_term: pd.DataFrame) -> pd.DataFrame:
    frame = per_term.loc[per_term["test_positive_count"] > 0].copy()
    frame["namespace"] = frame["namespace"].fillna("other")
    grouped = (
        frame.groupby("namespace", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            human_test_positive_labels=("test_positive_count", "sum"),
            median_f1_at_global_threshold=("test_f1_at_global_threshold", "median"),
            f1_q25=("test_f1_at_global_threshold", lambda values: values.quantile(0.25)),
            f1_q75=("test_f1_at_global_threshold", lambda values: values.quantile(0.75)),
            median_per_term_fmax=("test_term_fmax", "median"),
            fmax_q25=("test_term_fmax", lambda values: values.quantile(0.25)),
            fmax_q75=("test_term_fmax", lambda values: values.quantile(0.75)),
            median_auroc=("auroc", "median"),
            median_average_precision=("average_precision", "median"),
        )
        .reset_index()
    )
    order = {namespace: index for index, namespace in enumerate(NAMESPACE_ORDER)}
    grouped["sort_key"] = grouped["namespace"].map(order).fillna(99)
    return grouped.sort_values(["sort_key", "namespace"]).drop(columns=["sort_key"])


def depth_summary(per_term: pd.DataFrame, go_basic_obo: Path) -> pd.DataFrame:
    depths = load_go_depths_from_obo(go_basic_obo, per_term["GO_ID"])
    frame = per_term.merge(depths, on="GO_ID", how="inner")
    frame = frame.loc[frame["test_positive_count"] > 0].copy()
    return (
        frame.groupby("go_graph_depth", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            human_test_positive_labels=("test_positive_count", "sum"),
            median_f1_at_global_threshold=("test_f1_at_global_threshold", "median"),
            f1_q25=("test_f1_at_global_threshold", lambda values: values.quantile(0.25)),
            f1_q75=("test_f1_at_global_threshold", lambda values: values.quantile(0.75)),
            median_per_term_fmax=("test_term_fmax", "median"),
            fmax_q25=("test_term_fmax", lambda values: values.quantile(0.25)),
            fmax_q75=("test_term_fmax", lambda values: values.quantile(0.75)),
            median_auroc=("auroc", "median"),
            median_average_precision=("average_precision", "median"),
        )
        .reset_index()
        .sort_values("go_graph_depth")
    )


def plot_namespace_box(ax: plt.Axes, per_term: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    frame = per_term.loc[per_term["test_positive_count"] > 0].copy()
    categories = [namespace for namespace in NAMESPACE_ORDER if namespace in set(frame["namespace"])]
    values = [frame.loc[frame["namespace"].eq(namespace), metric].dropna().to_numpy() for namespace in categories]
    box = ax.boxplot(
        values,
        positions=np.arange(len(categories)),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.35},
        whiskerprops={"color": "#777777", "linewidth": 0.9},
        capprops={"color": "#777777", "linewidth": 0.9},
    )
    for patch, namespace in zip(box["boxes"], categories):
        color = NAMESPACE_COLORS.get(namespace, "#777777")
        patch.set_facecolor(color)
        patch.set_alpha(0.26)
        patch.set_edgecolor(color)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([NAMESPACE_SHORT.get(namespace, namespace) for namespace in categories])
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("GO namespace")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)


def plot_depth_trend(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    median_col: str,
    q25_col: str,
    q75_col: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    x = summary["go_graph_depth"].to_numpy(dtype=float)
    median = summary[median_col].to_numpy(dtype=float)
    q25 = summary[q25_col].to_numpy(dtype=float)
    q75 = summary[q75_col].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, median, color=color, linewidth=2.0, marker="o", markersize=3.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("GO graph depth")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)


def plot_go_namespace_depth(
    per_term_metrics: pd.DataFrame,
    publication_summary: dict[str, str],
    go_basic_obo: Path,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    per_term = make_numeric(
        per_term_metrics.copy(),
        [
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
            "auroc",
            "average_precision",
        ],
    )
    per_term = per_term.loc[per_term["test_positive_count"] > 0].copy()
    ns_summary = namespace_summary(per_term)
    depth_term_frame = depth_namespace_term_frame(per_term_metrics, go_basic_obo)
    depth_by_namespace = summarize_depth_by_namespace(depth_term_frame)

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.50))
    plot_namespace_box(
        axes[0, 0],
        per_term,
        "test_f1_at_global_threshold",
        "A  Namespace: global threshold",
        "GO-term F1",
    )
    plot_namespace_box(
        axes[0, 1],
        per_term,
        "test_term_fmax",
        "B  Namespace: per-term optimum",
        "GO-term Fmax",
    )
    plot_depth_namespace_trends(
        axes[1, 0],
        depth_by_namespace,
        median_col="median_f1_at_global_threshold",
        q25_col="f1_q25",
        q75_col="f1_q75",
        title="C  Depth by namespace: global threshold",
        ylabel="Median GO-term F1",
    )
    plot_depth_namespace_trends(
        axes[1, 1],
        depth_by_namespace,
        median_col="median_per_term_fmax",
        q25_col="fmax_q25",
        q75_col="fmax_q75",
        title="D  Depth by namespace: per-term optimum",
        ylabel="Median GO-term Fmax",
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles, labels, loc="upper right", frameon=False, ncol=1)
    max_depth = int(depth_term_frame["go_graph_depth"].max())
    for ax in axes[1, :]:
        ax.set_xlim(-0.7, max_depth + 0.7)
        ax.set_xticks(list(range(0, max_depth + 1, 2)))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.94, wspace=0.24, hspace=0.52)

    stem = output_dir / "figure_06_go_namespace_depth_performance"
    save_figure(fig, stem, dpi=dpi)
    ns_summary.to_csv(stem.with_name(stem.name + "_namespace_summary.tsv"), sep="\t", index=False)
    depth_by_namespace.to_csv(
        stem.with_name(stem.name + "_depth_summary.tsv"),
        sep="\t",
        index=False,
    )
    depth_by_namespace.to_csv(
        stem.with_name(stem.name + "_depth_namespace_summary.tsv"),
        sep="\t",
        index=False,
    )
    depth_term_frame[
        [
            "GO_ID",
            "name",
            "namespace",
            "namespace_short",
            "go_graph_depth",
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
            "auroc",
            "average_precision",
        ]
    ].to_csv(stem.with_name(stem.name + "_depth_namespace_term_table.tsv"), sep="\t", index=False)

    records = as_int(publication_summary, "records_evaluated")
    terms = as_int(publication_summary, "n_terms")
    bp_median = ns_summary.loc[
        ns_summary["namespace"].eq("biological_process"),
        "median_per_term_fmax",
    ]
    cc_median = ns_summary.loc[
        ns_summary["namespace"].eq("cellular_component"),
        "median_per_term_fmax",
    ]
    mf_median = ns_summary.loc[
        ns_summary["namespace"].eq("molecular_function"),
        "median_per_term_fmax",
    ]
    bp_text = f"{float(bp_median.iloc[0]):.3f}" if not bp_median.empty else "NA"
    cc_text = f"{float(cc_median.iloc[0]):.3f}" if not cc_median.empty else "NA"
    mf_text = f"{float(mf_median.iloc[0]):.3f}" if not mf_median.empty else "NA"
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 6. GO-term performance stratified by ontology namespace and "
            "graph depth. The top row compares one-vs-rest GO-term F1 at the "
            "single test-selected gene-averaged global threshold with per-term "
            "optimal Fmax "
            "across biological process, cellular component, and molecular function "
            "terms. The bottom row summarizes the same metrics by GO graph depth "
            "within each namespace; lines show namespace-specific depth-wise "
            "medians and shaded bands show interquartile ranges. GO graph depth "
            "was computed from is_a and part_of parent links in go-basic.obo. "
            f"The analysis uses retained non-root true-path GO terms with at least one "
            f"positive label among {records:,} {test_record_label(publication_summary)} "
            f"in the {test_set_label(publication_summary)} "
            f"({len(per_term):,} of {terms:,} vocabulary terms). Median per-term "
            f"optimal Fmax by namespace was BP={bp_text}, CC={cc_text}, and "
            f"MF={mf_text}, underscoring that gene-averaged Fmax summarizes a "
            "different quantity than one-vs-rest GO-term performance. These are "
            "descriptive (oracle) test-set analyses: the global threshold was "
            "selected using all test labels, and each per-term Fmax independently "
            "selects that term's best test-set threshold. Neither represents "
            "performance at a validation-selected operating threshold."
            + root_exclusion_note(publication_summary)
        ),
    )


def depth_namespace_term_frame(per_term_metrics: pd.DataFrame, go_basic_obo: Path) -> pd.DataFrame:
    per_term = make_numeric(
        per_term_metrics.copy(),
        [
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
            "auroc",
            "average_precision",
        ],
    )
    depths = load_go_depths_from_obo(go_basic_obo, per_term["GO_ID"])
    frame = per_term.merge(depths, on="GO_ID", how="inner")
    frame = frame.loc[
        frame["namespace"].isin(NAMESPACE_ORDER)
        & (frame["test_positive_count"] > 0)
        & frame["test_f1_at_global_threshold"].notna()
        & frame["test_term_fmax"].notna()
    ].copy()
    frame["namespace_label"] = frame["namespace"].map(NAMESPACE_LABELS)
    frame["namespace_short"] = frame["namespace"].map(NAMESPACE_SHORT)
    frame["go_graph_depth"] = frame["go_graph_depth"].astype(int)
    return frame


def summarize_depth_by_namespace(term_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        term_frame.groupby(["namespace", "namespace_short", "go_graph_depth"], observed=True)
        .agg(
            terms=("GO_ID", "count"),
            human_test_positive_labels=("test_positive_count", "sum"),
            median_f1_at_global_threshold=("test_f1_at_global_threshold", "median"),
            f1_q25=("test_f1_at_global_threshold", lambda values: values.quantile(0.25)),
            f1_q75=("test_f1_at_global_threshold", lambda values: values.quantile(0.75)),
            median_per_term_fmax=("test_term_fmax", "median"),
            fmax_q25=("test_term_fmax", lambda values: values.quantile(0.25)),
            fmax_q75=("test_term_fmax", lambda values: values.quantile(0.75)),
            median_auroc=("auroc", "median"),
            median_average_precision=("average_precision", "median"),
        )
        .reset_index()
        .sort_values(["namespace", "go_graph_depth"])
    )


def plot_depth_namespace_trends(
    ax: plt.Axes,
    depth_by_namespace: pd.DataFrame,
    *,
    median_col: str,
    q25_col: str,
    q75_col: str,
    title: str,
    ylabel: str,
) -> None:
    for namespace in NAMESPACE_ORDER:
        subset = depth_by_namespace.loc[depth_by_namespace["namespace"].eq(namespace)].copy()
        if subset.empty:
            continue
        color = NAMESPACE_COLORS[namespace]
        x = subset["go_graph_depth"].to_numpy(dtype=float)
        median = subset[median_col].to_numpy(dtype=float)
        q25 = subset[q25_col].to_numpy(dtype=float)
        q75 = subset[q75_col].to_numpy(dtype=float)
        ax.fill_between(x, q25, q75, color=color, alpha=0.12, linewidth=0)
        ax.plot(
            x,
            median,
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=3.4,
            label=NAMESPACE_SHORT[namespace],
        )
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("GO graph depth")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)


def plot_depth_term_scatter(
    ax: plt.Axes,
    term_frame: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    ylabel: str,
) -> None:
    for namespace in NAMESPACE_ORDER:
        subset = term_frame.loc[term_frame["namespace"].eq(namespace)].copy()
        if subset.empty:
            continue
        hashes = pd.util.hash_pandas_object(subset["GO_ID"].astype(str), index=False)
        offsets = ((hashes.astype(float) % 100) / 100.0 - 0.5) * 0.54
        sizes = (subset["test_positive_count"].clip(lower=1) ** 0.45).clip(5, 34)
        ax.scatter(
            subset["go_graph_depth"] + offsets,
            subset[metric_col],
            s=sizes,
            alpha=0.30,
            color=NAMESPACE_COLORS[namespace],
            edgecolors="none",
            rasterized=True,
        )
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("GO graph depth")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)


def plot_go_depth_namespace_terms(
    per_term_metrics: pd.DataFrame,
    publication_summary: dict[str, str],
    go_basic_obo: Path,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    term_frame = depth_namespace_term_frame(per_term_metrics, go_basic_obo)
    depth_by_namespace = summarize_depth_by_namespace(term_frame)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharey=True)
    plot_depth_namespace_trends(
        axes[0],
        depth_by_namespace,
        median_col="median_f1_at_global_threshold",
        q25_col="f1_q25",
        q75_col="f1_q75",
        title="A  Global threshold",
        ylabel="Median GO-term F1",
    )
    plot_depth_namespace_trends(
        axes[1],
        depth_by_namespace,
        median_col="median_per_term_fmax",
        q25_col="fmax_q25",
        q75_col="fmax_q75",
        title="B  Per-term optimal threshold",
        ylabel="Median GO-term Fmax",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.005),
        ncol=3,
        frameon=False,
    )
    max_depth = int(term_frame["go_graph_depth"].max())
    for ax in axes:
        ax.set_xlim(-0.7, max_depth + 0.7)
        ax.set_xticks(list(range(0, max_depth + 1, 2)))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.78, wspace=0.18)

    stem = output_dir / "figure_07_go_depth_namespace_term_performance"
    save_figure(fig, stem, dpi=dpi)
    depth_by_namespace.to_csv(
        stem.with_name(stem.name + "_depth_namespace_summary.tsv"),
        sep="\t",
        index=False,
    )
    term_frame[
        [
            "GO_ID",
            "name",
            "namespace",
            "namespace_short",
            "go_graph_depth",
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
            "auroc",
            "average_precision",
        ]
    ].to_csv(stem.with_name(stem.name + "_term_table.tsv"), sep="\t", index=False)

    records = as_int(publication_summary, "records_evaluated")
    terms = as_int(publication_summary, "n_terms")
    namespace_counts = (
        term_frame.groupby("namespace_short", observed=True)
        .agg(terms=("GO_ID", "count"))
        .reindex(["BP", "CC", "MF"])
        .fillna(0)
    )
    bp_terms = int(namespace_counts.loc["BP", "terms"])
    cc_terms = int(namespace_counts.loc["CC", "terms"])
    mf_terms = int(namespace_counts.loc["MF", "terms"])
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 7. Ontology-depth benchmark stratified by GO namespace and "
            "graph depth. Lines show depth-wise median performance separately "
            "for biological process, cellular component, and molecular function "
            "terms; shaded bands are namespace-specific interquartile ranges. "
            "The left panel uses the single global threshold selected by the "
            "gene-averaged test-set Fmax sweep, whereas the right panel uses "
            "each GO term's best test-set threshold over the same grid. Both are "
            "descriptive (oracle) test-set analyses rather than performance at a "
            "validation-selected operating threshold. The analysis includes "
            f"{len(term_frame):,} supported GO terms from {records:,} "
            f"{test_record_label(publication_summary)} in the "
            f"{test_set_label(publication_summary)} ({bp_terms:,} BP, "
            f"{cc_terms:,} CC, and {mf_terms:,} "
            f"MF terms out of {terms:,} vocabulary terms), with graph depth "
            "computed from is_a and part_of links in go-basic.obo."
            + root_exclusion_note(publication_summary)
        ),
    )


def namespace_gene_metrics_from_score_matrix(score_matrix_path: Path) -> tuple[pd.DataFrame, float]:
    with np.load(score_matrix_path, allow_pickle=False) as score_data:
        probabilities = score_data["probabilities"]
        targets = score_data["targets"].astype(bool)
        namespaces = score_data["go_namespace"].astype(str)
        threshold = float(score_data["global_threshold"][0])
        tax_id = score_data["tax_id"].astype(str)
        gene_id = score_data["GeneID"].astype(str)
        cds_id = score_data["cds_id"].astype(str)

        rows: list[pd.DataFrame] = []
        predicted = probabilities >= threshold
        for namespace in NAMESPACE_ORDER:
            mask = namespaces == namespace
            if not np.any(mask):
                continue
            namespace_targets = targets[:, mask]
            namespace_predicted = predicted[:, mask]
            tp = np.logical_and(namespace_predicted, namespace_targets).sum(axis=1)
            fp = np.logical_and(namespace_predicted, ~namespace_targets).sum(axis=1)
            fn = np.logical_and(~namespace_predicted, namespace_targets).sum(axis=1)
            label_count = namespace_targets.sum(axis=1)
            predicted_positive_count = namespace_predicted.sum(axis=1)
            precision = np.divide(
                tp,
                tp + fp,
                out=np.zeros(tp.shape, dtype=float),
                where=(tp + fp) > 0,
            )
            recall = np.divide(
                tp,
                tp + fn,
                out=np.zeros(tp.shape, dtype=float),
                where=(tp + fn) > 0,
            )
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros(precision.shape, dtype=float),
                where=(precision + recall) > 0,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "namespace": namespace,
                        "namespace_short": NAMESPACE_SHORT[namespace],
                        "tax_id": tax_id,
                        "GeneID": gene_id,
                        "cds_id": cds_id,
                        "label_count": label_count,
                        "predicted_positive_count": predicted_positive_count,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(), threshold
    return pd.concat(rows, ignore_index=True), threshold


def summarize_namespace_gene_annotation_bins(namespace_metrics: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for namespace in NAMESPACE_ORDER:
        metrics = namespace_metrics.loc[
            namespace_metrics["namespace"].eq(namespace)
            & (namespace_metrics["label_count"] > 0)
        ].copy()
        if metrics.empty:
            continue
        metrics["label_count_bin"] = pd.qcut(
            metrics["label_count"].rank(method="first"),
            q=min(10, len(metrics)),
            labels=False,
        )
        summary = (
            metrics.groupby("label_count_bin", dropna=False)
            .agg(
                namespace=("namespace", "first"),
                namespace_short=("namespace_short", "first"),
                genes=("label_count", "count"),
                median_label_count=("label_count", "median"),
                mean_label_count=("label_count", "mean"),
                median_predicted_positive_count=("predicted_positive_count", "median"),
                median_precision=("precision", "median"),
                median_recall=("recall", "median"),
                median_f1=("f1", "median"),
                mean_f1=("f1", "mean"),
            )
            .reset_index()
        )
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_namespace_gene_annotation_count_performance(
    score_matrix_path: Path,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    namespace_metrics, threshold = namespace_gene_metrics_from_score_matrix(score_matrix_path)
    bin_summary = summarize_namespace_gene_annotation_bins(namespace_metrics)
    stem = output_dir / "figure_08_namespace_gene_annotation_count_performance"
    if namespace_metrics.empty or bin_summary.empty:
        raise ValueError("No namespace-specific gene metrics were available.")

    bin_summary.to_csv(stem.with_name(stem.name + "_deciles.tsv"), sep="\t", index=False)
    namespace_metrics.loc[namespace_metrics["label_count"] > 0, :].to_csv(
        stem.with_name(stem.name + "_gene_metrics.tsv.gz"),
        sep="\t",
        index=False,
        compression="gzip",
    )
    summary = (
        namespace_metrics.loc[namespace_metrics["label_count"] > 0]
        .groupby(["namespace", "namespace_short"], observed=True)
        .agg(
            genes_with_namespace_labels=("GeneID", "count"),
            median_label_count=("label_count", "median"),
            median_predicted_positive_count=("predicted_positive_count", "median"),
            median_precision=("precision", "median"),
            median_recall=("recall", "median"),
            median_f1=("f1", "median"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(stem.with_name(stem.name + "_summary.tsv"), sep="\t", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 2.95), sharey=True)
    metric_styles = [
        ("median_precision", "Precision", BLUE),
        ("median_recall", "Recall", ORANGE),
        ("median_f1", "F1", GREEN),
    ]
    for ax, namespace in zip(axes, NAMESPACE_ORDER, strict=True):
        subset = bin_summary.loc[bin_summary["namespace"].eq(namespace)].copy()
        for metric_col, label, color in metric_styles:
            ax.plot(
                subset["median_label_count"],
                subset[metric_col],
                marker="o",
                markersize=3.8,
                linewidth=1.8,
                color=color,
                label=label,
            )
        ax.set_title(NAMESPACE_LABELS[namespace], loc="left", fontweight="semibold")
        ax.set_xlabel("Median true labels per gene")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)
    axes[0].set_ylabel("Median gene-level score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.78, wspace=0.18)
    save_figure(fig, stem, dpi=dpi)

    records = as_int(publication_summary, "records_evaluated")
    terms = as_int(publication_summary, "n_terms")
    summary_by_namespace = {
        row["namespace_short"]: row
        for row in summary.to_dict(orient="records")
    }
    bp_genes = int(summary_by_namespace.get("BP", {}).get("genes_with_namespace_labels", 0))
    cc_genes = int(summary_by_namespace.get("CC", {}).get("genes_with_namespace_labels", 0))
    mf_genes = int(summary_by_namespace.get("MF", {}).get("genes_with_namespace_labels", 0))
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 8. Gene-level performance as a function of namespace-specific "
            "annotation density. The annotation-count benchmark was computed "
            "separately for biological process, cellular component, and molecular "
            "function terms from the saved test score matrix and "
            f"the test-selected global threshold {threshold:.3f}. Within each "
            "namespace, genes "
            "were binned into deciles by the number of retained true-path labels in "
            "that namespace; points show decile medians for precision, recall, and "
            "F1 computed using only GO terms from that namespace. The analysis uses "
            f"{records:,} {test_record_label(publication_summary)} in the "
            f"{test_set_label(publication_summary)} and {terms:,} retained GO terms; "
            f"genes with at least one namespace-specific label were BP={bp_genes:,}, "
            f"CC={cc_genes:,}, and MF={mf_genes:,}. Because the global threshold "
            "was selected using test labels, the threshold-dependent precision, "
            "recall, and F1 values are descriptive (oracle) test-set results rather "
            "than performance at a validation-selected operating threshold."
            + root_exclusion_note(publication_summary)
        ),
    )


def frequency_depth_frame(per_term_metrics: pd.DataFrame, go_basic_obo: Path) -> pd.DataFrame:
    per_term = make_numeric(
        per_term_metrics.copy(),
        [
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
            "auroc",
            "average_precision",
        ],
    )
    depths = load_go_depths_from_obo(go_basic_obo, per_term["GO_ID"])
    frame = per_term.merge(depths, on="GO_ID", how="inner")
    frame = frame.loc[
        (frame["database_frequency"] > 0)
        & (frame["test_positive_count"] > 0)
        & frame["test_f1_at_global_threshold"].notna()
        & frame["test_term_fmax"].notna()
    ].copy()
    frame["go_graph_depth"] = frame["go_graph_depth"].astype(int)
    return frame


def depth_occurrence_heatmap_frame(per_term_metrics: pd.DataFrame, go_basic_obo: Path) -> pd.DataFrame:
    term_frame = frequency_depth_frame(per_term_metrics, go_basic_obo).copy()
    term_frame["namespace"] = term_frame["namespace"].astype(str).str.strip()
    term_frame["occurrence_bin"] = pd.cut(
        term_frame["database_frequency"],
        bins=OCCURRENCE_BIN_EDGES,
        labels=OCCURRENCE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    return term_frame.loc[term_frame["occurrence_bin"].notna()].copy()


def depth_occurrence_heatmap_summary(term_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        term_frame.groupby(["namespace", "go_graph_depth", "occurrence_bin"], observed=True)
        .agg(
            terms=("GO_ID", "count"),
            database_frequency_median=("database_frequency", "median"),
            database_frequency_min=("database_frequency", "min"),
            database_frequency_max=("database_frequency", "max"),
            human_test_positive_labels=("test_positive_count", "sum"),
            median_term_f1_at_global_threshold=("test_f1_at_global_threshold", "median"),
            mean_term_f1_at_global_threshold=("test_f1_at_global_threshold", "mean"),
            term_f1_q25=("test_f1_at_global_threshold", lambda values: values.quantile(0.25)),
            term_f1_q75=("test_f1_at_global_threshold", lambda values: values.quantile(0.75)),
            median_term_fmax=("test_term_fmax", "median"),
            mean_term_fmax=("test_term_fmax", "mean"),
            term_fmax_q25=("test_term_fmax", lambda values: values.quantile(0.25)),
            term_fmax_q75=("test_term_fmax", lambda values: values.quantile(0.75)),
        )
        .reset_index()
        .sort_values(["namespace", "go_graph_depth", "occurrence_bin"])
    )


def plot_depth_occurrence_heatmap(
    ax: plt.Axes,
    namespace_summary: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    depth_values: list[int],
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_xticklabels: bool = True,
) -> matplotlib.image.AxesImage:
    heat = (
        namespace_summary.pivot(
            index="go_graph_depth",
            columns="occurrence_bin",
            values=metric_column,
        )
        .reindex(index=depth_values, columns=OCCURRENCE_BIN_LABELS)
        .astype(float)
    )
    counts = (
        namespace_summary.pivot(
            index="go_graph_depth",
            columns="occurrence_bin",
            values="terms",
        )
        .reindex(index=depth_values, columns=OCCURRENCE_BIN_LABELS)
        .fillna(0)
        .astype(int)
    )

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(
        np.ma.masked_invalid(heat.to_numpy(dtype=float)),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(OCCURRENCE_BIN_LABELS)))
    ax.set_xticklabels(OCCURRENCE_BIN_LABELS, rotation=35, ha="right")
    ax.tick_params(axis="x", labelbottom=show_xticklabels)
    ax.set_yticks(np.arange(len(depth_values)))
    ax.set_yticklabels(depth_values)
    ax.set_xlabel("Binned occurrences (dataset sequences per GO term)" if show_xlabel else "")
    ax.set_ylabel("GO graph depth" if show_ylabel else "")
    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xticks(np.arange(-0.5, len(OCCURRENCE_BIN_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(depth_values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(counts.shape[0]):
        for column_index in range(counts.shape[1]):
            terms = int(counts.iat[row_index, column_index])
            if 0 < terms < 5:
                value = heat.iat[row_index, column_index]
                label_color = "white" if np.isfinite(value) and value >= 0.55 else INK
                ax.text(
                    column_index,
                    row_index,
                    f"n={terms}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=label_color,
                )

    return image


def plot_combined_go_depth_occurrence_f1_heatmap(
    term_frame: pd.DataFrame,
    summary: pd.DataFrame,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    """Combine namespace heatmaps into one figure with shared labels."""
    depth_values = list(range(0, int(term_frame["go_graph_depth"].max()) + 1))
    threshold = as_float(publication_summary, "global_threshold")
    stem = output_dir / "figure_09_go_depth_occurrence_f1_heatmap"

    fig, axes = plt.subplots(
        len(HEATMAP_NAMESPACE_ORDER),
        2,
        figsize=(9.8, 10.2),
        sharex=True,
        sharey=True,
    )
    image = None
    row_summaries: list[tuple[str, int, float, float]] = []
    figure_letters = {
        "molecular_function": "A",
        "biological_process": "B",
        "cellular_component": "C",
    }

    for row_index, namespace in enumerate(HEATMAP_NAMESPACE_ORDER):
        namespace_frame = term_frame.loc[term_frame["namespace"].eq(namespace)].copy()
        namespace_summary = summary.loc[summary["namespace"].eq(namespace)].copy()
        if namespace_frame.empty or namespace_summary.empty:
            continue

        for column_index, (metric_column, column_title) in enumerate(
            (
                ("median_term_f1_at_global_threshold", "Global threshold"),
                ("median_term_fmax", "Per-term optimal threshold"),
            )
        ):
            image = plot_depth_occurrence_heatmap(
                axes[row_index, column_index],
                namespace_summary,
                metric_column=metric_column,
                title="",
                depth_values=depth_values,
                show_ylabel=False,
                show_xlabel=False,
                show_xticklabels=row_index == len(HEATMAP_NAMESPACE_ORDER) - 1,
            )
            axes[row_index, column_index].tick_params(
                axis="y",
                labelleft=column_index == 0,
            )

        letter = figure_letters[namespace]
        axes[row_index, 0].text(
            0.0,
            1.035,
            f"{letter}  {NAMESPACE_LABELS[namespace]}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontweight="semibold",
            fontsize=9,
            color=INK,
        )
        row_summaries.append(
            (
                NAMESPACE_SHORT[namespace],
                len(namespace_frame),
                float(namespace_summary["median_term_f1_at_global_threshold"].median()),
                float(namespace_summary["median_term_fmax"].median()),
            )
        )

    fig.subplots_adjust(
        left=0.09,
        right=0.87,
        bottom=0.12,
        top=0.93,
        wspace=0.07,
        hspace=0.20,
    )
    for column_index, column_title in enumerate(
        ("Global threshold", "Per-term optimal threshold")
    ):
        position = axes[0, column_index].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.977,
            column_title,
            ha="center",
            va="top",
            fontweight="semibold",
            fontsize=9,
            color=INK,
        )
    fig.supxlabel("Binned occurrences (dataset sequences per GO term)", y=0.025)
    fig.supylabel("GO graph depth", x=0.015)
    if image is not None:
        colorbar_axis = fig.add_axes([0.90, 0.12, 0.018, 0.83])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Median GO-term score")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    save_figure(fig, stem, dpi=dpi)

    row_text = "; ".join(
        f"{namespace}={terms:,} terms" for namespace, terms, _f1, _fmax in row_summaries
    )
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 9. GO-term performance by ontology namespace, graph depth, and "
            "binned dataset occurrence. Rows show (A) molecular function, (B) "
            "biological process, and (C) cellular component "
            f"({row_text}). Each cell groups GO terms by depth and occurrence and "
            "reports the median F1 calculated independently for each term across "
            "all test sequences. The left column uses the global threshold selected "
            f"by descriptive test-set Fmax ({threshold:.3f}); the right column shows "
            "per-term descriptive test-set Fmax. Gray cells contain no terms, and "
            "n denotes term counts in sparse cells."
            + root_exclusion_note(publication_summary)
        ),
    )


def plot_go_depth_occurrence_f1_heatmaps(
    per_term_metrics: pd.DataFrame,
    publication_summary: dict[str, str],
    go_basic_obo: Path,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    term_frame = depth_occurrence_heatmap_frame(per_term_metrics, go_basic_obo)
    summary = depth_occurrence_heatmap_summary(term_frame)
    base_stem = output_dir / "figure_09_go_depth_occurrence_f1_heatmap"
    term_frame[
        [
            "GO_ID",
            "name",
            "namespace",
            "go_graph_depth",
            "database_frequency",
            "occurrence_bin",
            "test_positive_count",
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
    ].to_csv(base_stem.with_name(base_stem.name + "_term_table.tsv"), sep="\t", index=False)
    summary.to_csv(base_stem.with_name(base_stem.name + "_summary.tsv"), sep="\t", index=False)

    depth_values = list(range(0, int(term_frame["go_graph_depth"].max()) + 1))
    threshold = as_float(publication_summary, "global_threshold")
    records = as_int(publication_summary, "records_evaluated")
    figure_letters = {
        "molecular_function": "A",
        "biological_process": "B",
        "cellular_component": "C",
    }

    for namespace in HEATMAP_NAMESPACE_ORDER:
        namespace_frame = term_frame.loc[term_frame["namespace"].eq(namespace)].copy()
        namespace_summary = summary.loc[summary["namespace"].eq(namespace)].copy()
        if namespace_frame.empty or namespace_summary.empty:
            continue

        letter = figure_letters[namespace]
        stem = output_dir / (
            f"figure_09{letter.lower()}_"
            f"{NAMESPACE_SLUGS[namespace]}_depth_occurrence_f1_heatmap"
        )
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.55), sharey=True)
        plot_depth_occurrence_heatmap(
            axes[0],
            namespace_summary,
            metric_column="median_term_f1_at_global_threshold",
            title="Global threshold",
            depth_values=depth_values,
            show_ylabel=True,
        )
        image = plot_depth_occurrence_heatmap(
            axes[1],
            namespace_summary,
            metric_column="median_term_fmax",
            title="Per-term optimal threshold",
            depth_values=depth_values,
            show_ylabel=False,
        )
        fig.subplots_adjust(left=0.07, right=0.86, bottom=0.22, top=0.88, wspace=0.10)
        colorbar_axis = fig.add_axes([0.902, 0.22, 0.018, 0.66])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Median GO-term score")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        nonempty_cells = len(namespace_summary)
        median_cell_f1 = float(namespace_summary["median_term_f1_at_global_threshold"].median())
        median_cell_fmax = float(namespace_summary["median_term_fmax"].median())
        fig.text(
            0.08,
            0.965,
            f"{letter}  {NAMESPACE_LABELS[namespace]}",
            ha="left",
            va="top",
            fontweight="semibold",
            fontsize=10,
            color=INK,
        )
        fig.text(
            0.87,
            0.965,
            f"{len(namespace_frame):,} terms; {nonempty_cells:,} occupied cells",
            ha="right",
            va="top",
            color=MUTED,
        )
        save_figure(fig, stem, dpi=dpi)

        write_caption(
            stem.with_name(stem.name + "_caption.txt"),
            (
                f"Figure 9{letter}. {NAMESPACE_LABELS[namespace]} GO-term performance "
                "by GO graph depth and binned dataset occurrence. Each heatmap cell "
                "groups retained true-path GO terms by ontology depth and by the "
                "number of sequences annotated to the term in the dataset used for "
                "vocabulary construction. The left panel "
                "shows the median one-vs-rest GO-term F1 at the single global "
                f"test-selected global threshold ({threshold:.3f}); for each GO "
                "term, F1 is computed "
                "across the held-out sequences as 2TP/(2TP+FP+FN), treating that "
                "term as the positive class rather than averaging gene-level scores. "
                "The right panel shows the median per-term optimal Fmax, where each "
                "GO term is evaluated at its best threshold over the test-set "
                "threshold grid. Both panels are descriptive (oracle) test-set "
                "analyses: the global threshold was selected using all test labels, "
                "and each per-term Fmax independently selects that term's best "
                "test-set threshold. Neither represents performance at a "
                "validation-selected operating threshold. "
                "Empty gray cells contain no retained terms. Sparse cells are labeled "
                "n=1 through n=4, where n is the number of GO terms in that depth-by-"
                "occurrence bin. The heatmap includes "
                f"{len(namespace_frame):,} {NAMESPACE_LABELS[namespace].lower()} "
                f"terms across {records:,} {test_record_label(publication_summary)} in the "
                f"{test_set_label(publication_summary)}; the median occupied "
                f"cell score was {median_cell_f1:.3f} for global-threshold F1 and "
                f"{median_cell_fmax:.3f} for per-term optimal Fmax. Exact cell "
                "medians, occurrence ranges, and term counts are provided in "
                "figure_09_go_depth_occurrence_f1_heatmap_summary.tsv."
                + root_exclusion_note(publication_summary)
            ),
        )

    plot_combined_go_depth_occurrence_f1_heatmap(
        term_frame,
        summary,
        publication_summary,
        output_dir,
        dpi=dpi,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument(
        "--score-matrix",
        type=Path,
        help=(
            "Saved score matrix from evaluate-model. Defaults to "
            "<eval-dir>/test_score_matrix.npz and is used for Figure 8."
        ),
    )
    parser.add_argument(
        "--skip-namespace-gene-annotation-count",
        action="store_true",
        help="Skip Figure 8, which derives namespace-specific per-gene metrics from the score matrix.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir or (
        PROJECT_ROOT / "outputs" / "builds" / args.build_id / "runs" / args.run_name / "test_eval"
    )
    output_dir = args.output_dir or PROJECT_ROOT / "figures" / args.run_name / "publication_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "test_publication_summary.tsv"
    threshold_path = eval_dir / "test_publication_threshold_metrics.tsv"
    gene_path = eval_dir / "test_publication_gene_metrics.tsv"
    go_term_path = eval_dir / "test_publication_go_term_metrics.tsv"
    calibration_path = eval_dir / "test_publication_calibration_metrics.tsv"
    score_matrix_path = args.score_matrix or (eval_dir / "test_score_matrix.npz")

    publication_summary = read_key_value_tsv(summary_path)
    threshold_metrics = pd.read_csv(threshold_path, sep="\t")
    gene_metrics = pd.read_csv(gene_path, sep="\t")
    go_term_metrics = pd.read_csv(go_term_path, sep="\t")
    calibration_metrics = pd.read_csv(calibration_path, sep="\t")

    set_publication_style()
    plot_roc_pr(threshold_metrics, publication_summary, output_dir, dpi=args.dpi)
    plot_calibration(calibration_metrics, publication_summary, output_dir, dpi=args.dpi)
    plot_gene_level_discrimination(gene_metrics, publication_summary, output_dir, dpi=args.dpi)
    plot_go_namespace_depth(
        go_term_metrics,
        publication_summary,
        args.go_basic_obo,
        output_dir,
        dpi=args.dpi,
    )
    if not args.skip_namespace_gene_annotation_count:
        plot_namespace_gene_annotation_count_performance(
            score_matrix_path,
            publication_summary,
            output_dir,
            dpi=args.dpi,
        )
    plot_go_depth_occurrence_f1_heatmaps(
        go_term_metrics,
        publication_summary,
        args.go_basic_obo,
        output_dir,
        dpi=args.dpi,
    )

    manifest_rows = [
        ("figure_03_roc_pr", output_dir / f"{figure_03_stem_name(publication_summary)}.png"),
        ("figure_04_calibration", output_dir / "figure_04_probability_calibration.png"),
        ("figure_05_gene_level", output_dir / "figure_05_gene_level_discrimination.png"),
        ("figure_06_go_namespace_depth", output_dir / "figure_06_go_namespace_depth_performance.png"),
        (
            "figure_09_go_depth_occurrence_f1_heatmap",
            output_dir / "figure_09_go_depth_occurrence_f1_heatmap.png",
        ),
        (
            "figure_09a_molecular_function_depth_occurrence_f1_heatmap",
            output_dir / "figure_09a_molecular_function_depth_occurrence_f1_heatmap.png",
        ),
        (
            "figure_09b_biological_process_depth_occurrence_f1_heatmap",
            output_dir / "figure_09b_biological_process_depth_occurrence_f1_heatmap.png",
        ),
        (
            "figure_09c_cellular_component_depth_occurrence_f1_heatmap",
            output_dir / "figure_09c_cellular_component_depth_occurrence_f1_heatmap.png",
        ),
    ]
    if not args.skip_namespace_gene_annotation_count:
        manifest_rows.insert(
            4,
            (
                "figure_08_namespace_gene_annotation_count",
                output_dir / "figure_08_namespace_gene_annotation_count_performance.png",
            ),
        )
    write_key_value_tsv(output_dir / "additional_publication_benchmark_manifest.tsv", manifest_rows)
    print(f"Wrote additional publication benchmark figures to: {output_dir}")
    for name, path in manifest_rows:
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
