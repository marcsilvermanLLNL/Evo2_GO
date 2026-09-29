#!/usr/bin/env python3
"""Generate a publication-ready GO-term performance versus frequency figure."""

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
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "human_test_taxon_val_1919taxa_min_count_15000_bs256_mean_loss_3h"
DEFAULT_BUILD_ID = "all_labels_true_path_1919taxa_human_test_min_count_15000_v1"

GO_ROOT_LABELS = {
    "GO:0003674": "molecular function (MF)",
    "GO:0005575": "cellular component (CC)",
    "GO:0008150": "biological process (BP)",
}

NAMESPACE_LABELS = {
    "biological_process": "Biological process",
    "cellular_component": "Cellular component",
    "molecular_function": "Molecular function",
}
NAMESPACE_ORDER = [
    "biological_process",
    "cellular_component",
    "molecular_function",
]
NAMESPACE_COLORS = {
    "biological_process": "#3b6ea8",
    "cellular_component": "#c45a14",
    "molecular_function": "#4f8f46",
}


def is_mmseqs_test(summary: dict[str, str]) -> bool:
    return "mmseqs_linclust" in summary.get("split_manifest", "")


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
        return " The three GO ontology root terms were excluded from this analysis."
    if len(labels) == 1:
        joined = labels[0]
    else:
        joined = ", ".join(labels[:-1]) + ", and " + labels[-1]
    return f" The ontology root terms {joined} were excluded from this analysis."


def read_summary(path: Path) -> dict[str, str]:
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["metric"]: row["value"] for row in reader}


def count_formatter(value: float, _position: int) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def load_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path, sep="\t")
    numeric_columns = [
        "database_frequency",
        "test_positive_count",
        "test_f1_at_global_threshold",
        "test_term_fmax",
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.loc[
        (metrics["database_frequency"] > 0)
        & (metrics["test_positive_count"] > 0)
        & metrics["test_f1_at_global_threshold"].notna()
        & metrics["test_term_fmax"].notna()
    ].copy()
    metrics["namespace"] = metrics["namespace"].fillna("other")
    return metrics


def binned_summary(metrics: pd.DataFrame, metric_column: str, bins: int = 8) -> pd.DataFrame:
    frame = metrics[["database_frequency", metric_column]].dropna().copy()
    frame["bin"] = pd.qcut(frame["database_frequency"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    summary = grouped.agg(
        frequency_median=("database_frequency", "median"),
        metric_median=(metric_column, "median"),
        metric_q25=(metric_column, lambda values: values.quantile(0.25)),
        metric_q75=(metric_column, lambda values: values.quantile(0.75)),
        terms=(metric_column, "size"),
    ).reset_index(drop=True)
    return summary


def spearman_rho(metrics: pd.DataFrame, metric_column: str) -> float:
    ranks = metrics[["database_frequency", metric_column]].rank(method="average")
    return float(ranks["database_frequency"].corr(ranks[metric_column]))


def plot_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    ylabel: str,
) -> None:
    for namespace in NAMESPACE_ORDER:
        subset = metrics.loc[metrics["namespace"] == namespace]
        if subset.empty:
            continue
        ax.scatter(
            subset["database_frequency"],
            subset[metric_column],
            s=14,
            alpha=0.45,
            linewidths=0,
            color=NAMESPACE_COLORS[namespace],
            label=NAMESPACE_LABELS[namespace],
        )

    trend = binned_summary(metrics, metric_column)
    ax.fill_between(
        trend["frequency_median"].to_numpy(dtype=float),
        trend["metric_q25"].to_numpy(dtype=float),
        trend["metric_q75"].to_numpy(dtype=float),
        color="#222222",
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(
        trend["frequency_median"],
        trend["metric_median"],
        color="#222222",
        linewidth=2.0,
        label="Binned median",
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("Dataset sequences annotated to GO term")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(count_formatter))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#b8b8b8", alpha=0.8)


def write_caption(
    path: Path,
    *,
    metrics: pd.DataFrame,
    summary: dict[str, str],
    training_universe_label: str,
) -> None:
    records = int(summary["records_evaluated"])
    vocab_terms = int(summary["n_terms"])
    shown_terms = len(metrics)
    threshold = float(summary["threshold"])
    f1_median = float(metrics["test_f1_at_global_threshold"].median())
    fmax_median = float(metrics["test_term_fmax"].median())
    f1_rho = spearman_rho(metrics, "test_f1_at_global_threshold")
    fmax_rho = spearman_rho(metrics, "test_term_fmax")
    test_record_label = "test sequences" if is_mmseqs_test(summary) else "held-out human genes"
    caption = (
        "Figure 2. GO-term-level performance increases with GO-term dataset "
        "frequency. Each point represents one retained true-path GO term with "
        f"at least one positive {test_record_label[:-1]}; terms are colored by GO "
        "namespace. The x-axis shows the number of dataset sequences annotated to "
        f"the term in {training_universe_label}. The left "
        f"panel reports one-vs-rest GO-term F1 at the single global threshold "
        f"selected by the gene-averaged test-set threshold sweep "
        f"({threshold:.3f}); the right panel reports the best one-vs-rest F1 "
        "for each GO term over the same threshold grid. Black lines show "
        "frequency-binned medians and shaded bands show interquartile ranges. "
        f"The analysis includes {shown_terms:,} of {vocab_terms:,} vocabulary "
        f"terms with test-set support across {records:,} {test_record_label}. "
        f"Median F1 at the global threshold was {f1_median:.3f}, whereas median "
        f"per-term optimal Fmax was {fmax_median:.3f}; Spearman correlations "
        f"with dataset occurrence were rho={f1_rho:.2f} and rho={fmax_rho:.2f}, "
        "respectively. Both panels are descriptive (oracle) test-set analyses: "
        "the global threshold was selected using all test labels, and each "
        "per-term Fmax independently selects that term's best test-set threshold. "
        "Neither represents performance at a validation-selected operating threshold."
        + root_exclusion_note(summary)
    )
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


def write_summary(path: Path, metrics: pd.DataFrame) -> None:
    rows = [
        ("terms_with_test_support", len(metrics)),
        ("median_f1_at_global_threshold", f"{metrics['test_f1_at_global_threshold'].median():.8f}"),
        ("median_per_term_fmax", f"{metrics['test_term_fmax'].median():.8f}"),
        ("spearman_frequency_f1_at_global_threshold", f"{spearman_rho(metrics, 'test_f1_at_global_threshold'):.8f}"),
        ("spearman_frequency_per_term_fmax", f"{spearman_rho(metrics, 'test_term_fmax'):.8f}"),
    ]
    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--per-term-metrics", type=Path)
    parser.add_argument("--evaluation-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--training-universe-label",
        default="the frozen 1,919-taxon min-count-15,000 build",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir or (
        PROJECT_ROOT / "outputs" / "builds" / args.build_id / "runs" / args.run_name / "test_eval"
    )
    per_term_metrics = args.per_term_metrics or eval_dir / "test_publication_go_term_metrics.tsv"
    if not per_term_metrics.exists():
        per_term_metrics = eval_dir / "test_go_term_fmax_vs_frequency.tsv"
    evaluation_summary = args.evaluation_summary or eval_dir / "test_evaluation_summary.tsv"
    output_dir = args.output_dir or PROJECT_ROOT / "figures" / args.run_name / "publication_ready"

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(per_term_metrics)
    summary = read_summary(evaluation_summary)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    plot_panel(
        axes[0],
        metrics,
        metric_column="test_f1_at_global_threshold",
        title="A  Global threshold",
        ylabel="GO-term F1",
    )
    plot_panel(
        axes[1],
        metrics,
        metric_column="test_term_fmax",
        title="B  Per-term optimal threshold",
        ylabel="GO-term Fmax",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    wanted = ["Biological process", "Cellular component", "Molecular function", "Binned median"]
    by_label = {label: handle for handle, label in zip(handles, labels, strict=False)}
    fig.legend(
        [by_label[label] for label in wanted if label in by_label],
        [label for label in wanted if label in by_label],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.03),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.80, wspace=0.14)

    stem = output_dir / "figure_02_go_term_frequency_performance"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        metrics=metrics,
        summary=summary,
        training_universe_label=args.training_universe_label,
    )
    write_summary(stem.with_name(stem.name + "_summary.tsv"), metrics)


if __name__ == "__main__":
    main()
