#!/usr/bin/env python3
"""Create reusable Evo2-GO result figures from training/evaluation TSV files."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evo2_go_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "outputs/model_training/non_iea_genes_all_labels_min_count_10/"
    "runs/human_taxon_holdout_bs512_80k_pbatch4h"
)
DEFAULT_SPLIT_SUMMARY = Path(
    "outputs/model_training/non_iea_genes_all_labels_min_count_10/"
    "splits/human_taxon_holdout_summary.tsv"
)
DEFAULT_SPLIT_MANIFEST = Path(
    "outputs/model_training/non_iea_genes_all_labels_min_count_10/"
    "splits/human_taxon_holdout_manifest.tsv"
)
DEFAULT_LABEL_CACHE = Path(
    "outputs/model_training/non_iea_genes_all_labels_min_count_10/label_index.tsv.gz"
)
DEFAULT_GO_BASIC_OBO = Path("data/go/go-basic.obo")

GO_CATEGORY_ORDER = ["BP", "CC", "MF"]
GO_CATEGORY_COLORS = {"BP": "#4c78a8", "CC": "#f58518", "MF": "#54a24b"}

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "figure.facecolor": "white",
        "font.size": 10,
        "legend.frameon": False,
    }
)


def count_formatter(value: float, _position: int) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


COUNT_FORMATTER = FuncFormatter(count_formatter)


def rank_formatter(value: float, _position: int) -> str:
    value = float(value)
    if abs(value) >= 1_000:
        compact = f"{value / 1_000:.1f}k"
        return compact.replace(".0k", "k")
    return f"{value:.0f}"


RANK_FORMATTER = FuncFormatter(rank_formatter)


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.45)


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "evo2_go_results"


def read_key_value_tsv(path: Path) -> pd.Series:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if not {"metric", "value"}.issubset(frame.columns):
        raise ValueError(f"Expected metric/value columns in {path}")
    return frame.set_index("metric")["value"]


def resolve_recorded_path(path: Path | str | None, *, project_root: Path) -> Path | None:
    """Resolve paths recorded on another mounted spelling of the same repo."""

    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        rooted = project_root / candidate
        if rooted.exists():
            return rooted

    text = str(candidate)
    marker = "Evo2-GO-Remodel/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        rooted = project_root / suffix
        if rooted.exists():
            return rooted

    ws2_prefix = "/usr/WS2/"
    workspace_prefix = "/usr/workspace/"
    if text.startswith(ws2_prefix):
        swapped = Path(workspace_prefix + text[len(ws2_prefix) :])
        if swapped.exists():
            return swapped

    return candidate


def infer_summary_path(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None
    name = manifest_path.name
    if name.endswith("_manifest.tsv"):
        candidate = manifest_path.with_name(name.replace("_manifest.tsv", "_summary.tsv"))
        if candidate.exists():
            return candidate
    return None


def infer_true_path_term_counts(vocabulary_path: Path | None) -> Path | None:
    if vocabulary_path is None:
        return None
    if vocabulary_path.name != "go_vocabulary.tsv":
        return None
    build_root = vocabulary_path.parent.parent
    candidate = build_root / "data" / "labels" / "true_path_go_term_counts.tsv"
    if candidate.exists():
        return candidate
    return None


def namespace_short(namespace: object) -> str:
    return {
        "biological_process": "BP",
        "cellular_component": "CC",
        "molecular_function": "MF",
    }.get(str(namespace), str(namespace))


def go_category_order(categories: pd.Series) -> list[str]:
    present = [category for category in GO_CATEGORY_ORDER if category in set(categories)]
    present.extend(sorted(set(categories.dropna().astype(str)).difference(present)))
    return present


def numeric_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column != "event":
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> Path:
    path = output_dir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def parse_comparison(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--comparison-metrics must use NAME=PATH syntax"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Comparison name cannot be empty")
    return name, Path(path)


def plot_split_sizes(split_summary: pd.Series, output_dir: Path, dpi: int) -> Path | None:
    rows: list[tuple[str, int]] = []
    for split in ("train", "val", "test"):
        labeled_key = f"{split}_records_with_retained_labels"
        fallback_key = f"{split}_records"
        if labeled_key in split_summary.index:
            rows.append((split, int(split_summary.loc[labeled_key])))
        elif fallback_key in split_summary.index:
            rows.append((split, int(split_summary.loc[fallback_key])))
    if not rows:
        return None

    frame = pd.DataFrame(rows, columns=("split", "records"))
    frame.to_csv(output_dir / "split_sizes.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    ax.bar(frame["split"], frame["records"], color=colors[: len(frame)])
    ax.set_title("Training, validation, and human test set sizes")
    ax.set_xlabel("Split")
    ax.set_ylabel("Genes with retained GO labels")
    ax.yaxis.set_major_formatter(COUNT_FORMATTER)
    clean_axis(ax)
    for index, row in frame.iterrows():
        ax.text(index, row["records"], f"{row['records']:,}", ha="center", va="bottom")
    return save_figure(fig, output_dir, "split_sizes", dpi)


def plot_training_diagnostics(metrics: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    train_metrics = metrics[metrics["event"] == "train"].copy()
    validation_metrics = metrics[metrics["event"] == "validation"].copy()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not validation_metrics.empty:
        ax.plot(
            validation_metrics["step"],
            validation_metrics["fmax"],
            marker="o",
            linewidth=2.0,
            label="rat validation",
        )
        best = validation_metrics.loc[validation_metrics["fmax"].idxmax()]
        ax.scatter(
            [best["step"]],
            [best["fmax"]],
            s=90,
            color="#e45756",
            zorder=3,
            label=f"best val step {int(best['step']):,}",
        )
        ax.annotate(
            f"Fmax {best['fmax']:.3f}",
            xy=(best["step"], best["fmax"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )
    elif not train_metrics.empty:
        ax.plot(
            train_metrics["step"],
            train_metrics["loss"].rolling(10, min_periods=1).mean(),
            linewidth=2.0,
            color="#4c78a8",
            label="training loss",
        )
    ax.set_title("Validation performance during training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gene-averaged Fmax")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(COUNT_FORMATTER)
    clean_axis(ax)
    ax.legend()

    return save_figure(fig, output_dir, "training_diagnostics", dpi)


def plot_loss_diagnostics(metrics: pd.DataFrame, output_dir: Path, dpi: int) -> Path | None:
    train_metrics = metrics[metrics["event"] == "train"].copy()
    required = {
        "positive_loss",
        "negative_loss",
        "positive_labels_per_example",
        "mean_positive_probability",
        "mean_negative_probability",
        "logit_scale",
    }
    if train_metrics.empty or not required.issubset(train_metrics.columns):
        return None

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(train_metrics["step"], train_metrics["positive_loss"], label="positive")
    ax.plot(train_metrics["step"], train_metrics["negative_loss"], label="negative")
    ax.set_title("Asymmetric loss components")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss contribution")
    ax.xaxis.set_major_formatter(COUNT_FORMATTER)
    clean_axis(ax)
    ax.legend()

    return save_figure(fig, output_dir, "loss_diagnostics", dpi)


def plot_validation_metrics(
    validation_metrics: pd.DataFrame,
    test_summary: pd.Series | None,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    if validation_metrics.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(validation_metrics["step"], validation_metrics["precision"], marker="o", label="validation precision")
    ax.plot(validation_metrics["step"], validation_metrics["recall"], marker="o", label="validation recall")
    if test_summary is not None and {"precision", "recall"}.issubset(test_summary.index):
        ax.axhline(
            float(test_summary.loc["precision"]),
            color="#4c78a8",
            linestyle="--",
            alpha=0.6,
            label="human test precision",
        )
        ax.axhline(
            float(test_summary.loc["recall"]),
            color="#f58518",
            linestyle="--",
            alpha=0.6,
            label="human test recall",
        )
    ax.set_title("Precision and recall across validation checkpoints")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gene-averaged score")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(COUNT_FORMATTER)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    ax.legend()

    return save_figure(fig, output_dir, "validation_metrics", dpi)


def plot_test_threshold_curve(curve: pd.DataFrame, output_dir: Path, dpi: int) -> Path | None:
    if curve.empty:
        return None
    for column in curve.columns:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")

    best = curve.loc[curve["f1"].idxmax()]
    best.to_frame("best_test_threshold").to_csv(
        output_dir / "best_test_threshold.tsv", sep="\t"
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(curve["threshold"], curve["f1"], label="F1", linewidth=2.0)
    ax.plot(curve["threshold"], curve["precision"], label="precision")
    ax.plot(curve["threshold"], curve["recall"], label="recall")
    ax.axvline(
        best["threshold"],
        color="#e45756",
        linestyle="--",
        label=f"best F1 threshold {best['threshold']:.3f}",
    )
    ax.set_title("Human test threshold sweep")
    ax.set_xlabel("Prediction probability threshold")
    ax.set_ylabel("Gene-averaged score")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    ax.legend()

    return save_figure(fig, output_dir, "test_threshold_curve", dpi)


def plot_per_term_frequency_metric(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    *,
    y_column: str,
    stem: str,
    title: str,
    ylabel: str,
    colorbar_label: str,
) -> Path | None:
    if per_term.empty:
        return None
    required = {"database_frequency", y_column, "test_positive_count"}
    if not required.issubset(per_term.columns):
        missing = ", ".join(sorted(required.difference(per_term.columns)))
        print(f"Skipping per-term plot; missing columns: {missing}")
        return None

    plot_terms = per_term.copy()
    plot_terms["database_frequency"] = pd.to_numeric(plot_terms["database_frequency"], errors="coerce")
    plot_terms[y_column] = pd.to_numeric(plot_terms[y_column], errors="coerce").fillna(0.0)
    plot_terms["test_positive_count"] = pd.to_numeric(plot_terms["test_positive_count"], errors="coerce")
    plot_terms = plot_terms.dropna(subset=["database_frequency", y_column])
    if plot_terms.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    has_human_positives = plot_terms["test_positive_count"] > 0

    ax.scatter(
        plot_terms.loc[~has_human_positives, "database_frequency"],
        plot_terms.loc[~has_human_positives, y_column],
        s=10,
        alpha=0.15,
        color="#c7c7c7",
        label="no positive test labels",
    )
    scatter = ax.scatter(
        plot_terms.loc[has_human_positives, "database_frequency"],
        plot_terms.loc[has_human_positives, y_column],
        s=(plot_terms.loc[has_human_positives, "test_positive_count"].clip(lower=1) ** 0.5) * 9,
        alpha=0.70,
        c=plot_terms.loc[has_human_positives, y_column],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        label="has positive test labels",
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.set_xlabel("GO term frequency in selected label database (log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label)

    plot_terms[["database_frequency", "test_positive_count", y_column]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).to_csv(output_dir / f"{stem}_summary.tsv", sep="\t")
    return save_figure(fig, output_dir, stem, dpi)


def plot_per_term_metrics(per_term: pd.DataFrame, output_dir: Path, dpi: int) -> list[tuple[str, Path]]:
    written: list[tuple[str, Path]] = []
    path = plot_per_term_frequency_metric(
        per_term,
        output_dir,
        dpi,
        y_column="test_term_fmax",
        stem="per_go_term_fmax_vs_frequency",
        title="Per-GO-term Fmax vs GO term frequency",
        ylabel="Per-GO-term optimal F1 on human test",
        colorbar_label="Per-GO-term optimal F1",
    )
    if path:
        written.append(("per_go_term_fmax_vs_frequency", path))

    path = plot_per_term_frequency_metric(
        per_term,
        output_dir,
        dpi,
        y_column="test_f1_at_global_threshold",
        stem="per_go_term_f1_at_gene_averaged_global_threshold_vs_frequency",
        title="GO-term F1 at the gene-averaged global threshold",
        ylabel="GO-term F1 using the global threshold",
        colorbar_label="F1 at global threshold",
    )
    if path:
        written.append(("per_go_term_f1_at_gene_averaged_global_threshold_vs_frequency", path))
    return written


def per_term_global_f1_frame(per_term: pd.DataFrame) -> pd.DataFrame:
    return per_term_metric_frame(per_term, "test_f1_at_global_threshold")


def per_term_fmax_frame(per_term: pd.DataFrame) -> pd.DataFrame:
    return per_term_metric_frame(per_term, "test_term_fmax")


def per_term_metric_frame(per_term: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    required = {
        "GO_ID",
        "name",
        "namespace",
        "test_positive_count",
        metric_column,
    }
    if per_term.empty or not required.issubset(per_term.columns):
        return pd.DataFrame()

    metrics = per_term.copy()
    metrics["category"] = metrics["namespace"].map(namespace_short)
    metrics["test_positive_count"] = pd.to_numeric(
        metrics["test_positive_count"], errors="coerce"
    ).fillna(0.0)
    metrics[metric_column] = pd.to_numeric(metrics[metric_column], errors="coerce").fillna(0.0)
    if "database_frequency" in metrics.columns:
        metrics["database_frequency"] = pd.to_numeric(
            metrics["database_frequency"], errors="coerce"
        )

    metrics = metrics[metrics["test_positive_count"] > 0].copy()
    return metrics


def plot_go_namespace_f1(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    metrics = per_term_global_f1_frame(per_term)
    if metrics.empty:
        return None

    categories = go_category_order(metrics["category"])
    summary = (
        metrics.groupby("category", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_f1=("test_f1_at_global_threshold", "median"),
            mean_f1=("test_f1_at_global_threshold", "mean"),
            f1_p25=("test_f1_at_global_threshold", lambda s: s.quantile(0.25)),
            f1_p75=("test_f1_at_global_threshold", lambda s: s.quantile(0.75)),
            total_human_test_positive_labels=("test_positive_count", "sum"),
        )
        .reindex(categories)
        .reset_index()
    )
    summary.to_csv(output_dir / "go_namespace_f1_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    grouped_values = [
        metrics.loc[metrics["category"].eq(category), "test_f1_at_global_threshold"].to_numpy()
        for category in categories
    ]
    box = ax.boxplot(
        grouped_values,
        positions=range(len(categories)),
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": "#777777"},
        capprops={"color": "#777777"},
    )
    for patch, category in zip(box["boxes"], categories, strict=True):
        patch.set_facecolor(GO_CATEGORY_COLORS.get(category, "#777777"))
        patch.set_alpha(0.28)
        patch.set_edgecolor(GO_CATEGORY_COLORS.get(category, "#777777"))

    for index, category in enumerate(categories):
        subset = metrics.loc[metrics["category"].eq(category)]
        if subset.empty:
            continue
        hashes = pd.util.hash_pandas_object(subset["GO_ID"].astype(str), index=False)
        offsets = ((hashes.astype(float) % 100) / 100.0 - 0.5) * 0.44
        ax.scatter(
            index + offsets,
            subset["test_f1_at_global_threshold"],
            s=14,
            alpha=0.42,
            color=GO_CATEGORY_COLORS.get(category, "#777777"),
            edgecolors="none",
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GO-term F1 by GO namespace")
    ax.set_xlabel("GO namespace")
    ax.set_ylabel("GO-term F1 at the gene-averaged global threshold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    return save_figure(fig, output_dir, "go_namespace_f1_at_gene_averaged_global_threshold", dpi)


def plot_go_namespace_fmax(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    metrics = per_term_fmax_frame(per_term)
    if metrics.empty:
        return None

    metric_column = "test_term_fmax"
    categories = go_category_order(metrics["category"])
    summary = (
        metrics.groupby("category", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_fmax=(metric_column, "median"),
            mean_fmax=(metric_column, "mean"),
            fmax_p25=(metric_column, lambda s: s.quantile(0.25)),
            fmax_p75=(metric_column, lambda s: s.quantile(0.75)),
            total_human_test_positive_labels=("test_positive_count", "sum"),
        )
        .reindex(categories)
        .reset_index()
    )
    summary.to_csv(output_dir / "go_namespace_fmax_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    grouped_values = [
        metrics.loc[metrics["category"].eq(category), metric_column].to_numpy()
        for category in categories
    ]
    box = ax.boxplot(
        grouped_values,
        positions=range(len(categories)),
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": "#777777"},
        capprops={"color": "#777777"},
    )
    for patch, category in zip(box["boxes"], categories, strict=True):
        patch.set_facecolor(GO_CATEGORY_COLORS.get(category, "#777777"))
        patch.set_alpha(0.28)
        patch.set_edgecolor(GO_CATEGORY_COLORS.get(category, "#777777"))

    for index, category in enumerate(categories):
        subset = metrics.loc[metrics["category"].eq(category)]
        if subset.empty:
            continue
        hashes = pd.util.hash_pandas_object(subset["GO_ID"].astype(str), index=False)
        offsets = ((hashes.astype(float) % 100) / 100.0 - 0.5) * 0.44
        ax.scatter(
            index + offsets,
            subset[metric_column],
            s=14,
            alpha=0.42,
            color=GO_CATEGORY_COLORS.get(category, "#777777"),
            edgecolors="none",
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GO-term Fmax by GO namespace")
    ax.set_xlabel("GO namespace")
    ax.set_ylabel("GO-term Fmax using per-term optimal thresholds")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    return save_figure(fig, output_dir, "go_namespace_fmax_at_per_go_term_optimal_threshold", dpi)


def load_go_graph_depths(go_basic_obo: Path | None, go_ids: pd.Series) -> pd.DataFrame:
    if go_basic_obo is None or not go_basic_obo.exists():
        return pd.DataFrame()
    try:
        from goatools.obo_parser import GODag
    except ImportError:
        return load_go_graph_depths_from_obo(go_basic_obo, go_ids)

    go_dag = GODag(str(go_basic_obo), optional_attrs={"relationship"})
    rows: list[dict[str, object]] = []
    for go_id in sorted(set(go_ids.dropna().astype(str))):
        term = go_dag.get(go_id)
        if term is None:
            continue
        depth = getattr(term, "depth", None)
        level = getattr(term, "level", None)
        if depth is None:
            depth = level
        rows.append(
            {
                "GO_ID": go_id,
                "go_graph_depth": depth,
                "go_graph_level": level,
            }
        )
    return pd.DataFrame(rows)


def load_go_graph_depths_from_obo(go_basic_obo: Path, go_ids: pd.Series) -> pd.DataFrame:
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
        if parents:
            value = 1 + max(depth(parent) for parent in parents)
        else:
            value = 0
        visiting.remove(go_id)
        memo[go_id] = value
        return value

    rows = [
        {
            "GO_ID": go_id,
            "go_graph_depth": depth(go_id),
            "go_graph_level": depth(go_id),
        }
        for go_id in sorted(target_ids)
        if go_id in parents_by_id
    ]
    return pd.DataFrame(rows)


def plot_go_depth_f1(
    per_term: pd.DataFrame,
    go_basic_obo: Path | None,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    metrics = per_term_global_f1_frame(per_term)
    if metrics.empty:
        return None

    depths = load_go_graph_depths(go_basic_obo, metrics["GO_ID"])
    if depths.empty:
        print("Skipping GO-depth plot; no GO depth metadata was available.")
        return None

    plot_terms = metrics.merge(depths, on="GO_ID", how="left")
    plot_terms["go_graph_depth"] = pd.to_numeric(
        plot_terms["go_graph_depth"], errors="coerce"
    )
    plot_terms = plot_terms.dropna(subset=["go_graph_depth"])
    if plot_terms.empty:
        return None
    plot_terms["go_graph_depth"] = plot_terms["go_graph_depth"].astype(int)

    plot_terms[
        [
            "GO_ID",
            "name",
            "category",
            "go_graph_depth",
            "go_graph_level",
            "test_positive_count",
            "test_f1_at_global_threshold",
        ]
    ].to_csv(output_dir / "go_depth_f1_table.tsv", sep="\t", index=False)

    depth_summary = (
        plot_terms.groupby(["go_graph_depth", "category"], dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_f1=("test_f1_at_global_threshold", "median"),
            mean_f1=("test_f1_at_global_threshold", "mean"),
            total_human_test_positive_labels=("test_positive_count", "sum"),
        )
        .reset_index()
    )
    depth_summary.to_csv(output_dir / "go_depth_f1_summary.tsv", sep="\t", index=False)
    median_by_depth = (
        plot_terms.groupby("go_graph_depth", dropna=False)
        .agg(median_f1=("test_f1_at_global_threshold", "median"))
        .reset_index()
        .sort_values("go_graph_depth")
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    categories = go_category_order(plot_terms["category"])
    for category in categories:
        subset = plot_terms.loc[plot_terms["category"].eq(category)].copy()
        if subset.empty:
            continue
        hashes = pd.util.hash_pandas_object(subset["GO_ID"].astype(str), index=False)
        offsets = ((hashes.astype(float) % 100) / 100.0 - 0.5) * 0.42
        ax.scatter(
            subset["go_graph_depth"] + offsets,
            subset["test_f1_at_global_threshold"],
            s=(subset["test_positive_count"].clip(lower=1) ** 0.5).clip(8, 70),
            alpha=0.45,
            color=GO_CATEGORY_COLORS.get(category, "#777777"),
            label=category,
            edgecolors="none",
        )
    ax.plot(
        median_by_depth["go_graph_depth"],
        median_by_depth["median_f1"],
        color="#222222",
        marker="o",
        linewidth=2.0,
        label="median by depth",
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GO-term F1 by GO graph depth")
    ax.set_xlabel("GO graph depth in go-basic.obo")
    ax.set_ylabel("GO-term F1 at the gene-averaged global threshold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    ax.legend(title="GO namespace", frameon=False)
    return save_figure(fig, output_dir, "go_depth_f1_at_gene_averaged_global_threshold", dpi)


def plot_go_depth_fmax(
    per_term: pd.DataFrame,
    go_basic_obo: Path | None,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    metrics = per_term_fmax_frame(per_term)
    if metrics.empty:
        return None

    depths = load_go_graph_depths(go_basic_obo, metrics["GO_ID"])
    if depths.empty:
        print("Skipping GO-depth Fmax plot; no GO depth metadata was available.")
        return None

    metric_column = "test_term_fmax"
    plot_terms = metrics.merge(depths, on="GO_ID", how="left")
    plot_terms["go_graph_depth"] = pd.to_numeric(
        plot_terms["go_graph_depth"], errors="coerce"
    )
    plot_terms = plot_terms.dropna(subset=["go_graph_depth"])
    if plot_terms.empty:
        return None
    plot_terms["go_graph_depth"] = plot_terms["go_graph_depth"].astype(int)

    plot_terms[
        [
            "GO_ID",
            "name",
            "category",
            "go_graph_depth",
            "go_graph_level",
            "test_positive_count",
            metric_column,
        ]
    ].to_csv(output_dir / "go_depth_fmax_table.tsv", sep="\t", index=False)

    depth_summary = (
        plot_terms.groupby(["go_graph_depth", "category"], dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_fmax=(metric_column, "median"),
            mean_fmax=(metric_column, "mean"),
            total_human_test_positive_labels=("test_positive_count", "sum"),
        )
        .reset_index()
    )
    depth_summary.to_csv(output_dir / "go_depth_fmax_summary.tsv", sep="\t", index=False)
    median_by_depth = (
        plot_terms.groupby("go_graph_depth", dropna=False)
        .agg(median_fmax=(metric_column, "median"))
        .reset_index()
        .sort_values("go_graph_depth")
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    categories = go_category_order(plot_terms["category"])
    for category in categories:
        subset = plot_terms.loc[plot_terms["category"].eq(category)].copy()
        if subset.empty:
            continue
        hashes = pd.util.hash_pandas_object(subset["GO_ID"].astype(str), index=False)
        offsets = ((hashes.astype(float) % 100) / 100.0 - 0.5) * 0.42
        ax.scatter(
            subset["go_graph_depth"] + offsets,
            subset[metric_column],
            s=(subset["test_positive_count"].clip(lower=1) ** 0.5).clip(8, 70),
            alpha=0.45,
            color=GO_CATEGORY_COLORS.get(category, "#777777"),
            label=category,
            edgecolors="none",
        )
    ax.plot(
        median_by_depth["go_graph_depth"],
        median_by_depth["median_fmax"],
        color="#222222",
        marker="o",
        linewidth=2.0,
        label="median by depth",
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GO-term Fmax by GO graph depth")
    ax.set_xlabel("GO graph depth in go-basic.obo")
    ax.set_ylabel("GO-term Fmax using per-term optimal thresholds")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    ax.legend(title="GO namespace", frameon=False)
    return save_figure(fig, output_dir, "go_depth_fmax_at_per_go_term_optimal_threshold", dpi)


def plot_gene_annotation_count_performance(
    gene_metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    required = {"label_count", "precision", "recall", "f1", "predicted_positive_count"}
    if gene_metrics.empty or not required.issubset(gene_metrics.columns):
        return None

    metrics = gene_metrics.copy()
    for column in ("label_count", "precision", "recall", "f1", "predicted_positive_count"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.dropna(subset=["label_count", "precision", "recall", "f1"])
    metrics = metrics[metrics["label_count"] > 0].copy()
    if metrics.empty:
        return None

    metrics["label_count_bin"] = pd.qcut(
        metrics["label_count"].rank(method="first"),
        q=min(10, len(metrics)),
        labels=False,
    )
    bin_summary = (
        metrics.groupby("label_count_bin", dropna=False)
        .agg(
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
    bin_summary.to_csv(
        output_dir / "gene_annotation_count_performance_deciles.tsv",
        sep="\t",
        index=False,
    )
    metrics[
        [
            "tax_id",
            "GeneID",
            "cds_id",
            "label_count",
            "predicted_positive_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]
    ].to_csv(output_dir / "gene_metrics_numeric.tsv", sep="\t", index=False)

    spearman_rows = []
    for column in ("precision", "recall", "f1", "predicted_positive_count"):
        spearman_rows.append(
            {
                "x": "label_count",
                "y": column,
                "spearman_r": metrics["label_count"].corr(metrics[column], method="spearman"),
            }
        )
    pd.DataFrame(spearman_rows).to_csv(
        output_dir / "gene_annotation_count_spearman.tsv",
        sep="\t",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        bin_summary["median_label_count"],
        bin_summary["median_precision"],
        marker="o",
        label="precision",
    )
    ax.plot(
        bin_summary["median_label_count"],
        bin_summary["median_recall"],
        marker="o",
        label="recall",
    )
    ax.plot(
        bin_summary["median_label_count"],
        bin_summary["median_f1"],
        marker="o",
        label="F1",
    )
    ax.set_title("Performance changes with annotation density")
    ax.set_xlabel("Median true GO labels per human gene")
    ax.set_ylabel("Median gene-level score")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    ax.legend(frameon=False)

    return save_figure(fig, output_dir, "gene_annotation_count_performance", dpi)


def add_train_test_term_counts(
    per_term: pd.DataFrame,
    split_manifest_path: Path,
    label_cache_path: Path,
) -> pd.DataFrame:
    split_manifest = pd.read_csv(
        split_manifest_path,
        sep="\t",
        dtype={"tax_id": str, "GeneID": str, "split": str},
        usecols=["split", "tax_id", "GeneID"],
    ).drop_duplicates()

    train_keys = set(
        map(
            tuple,
            split_manifest.loc[split_manifest["split"].eq("train"), ["tax_id", "GeneID"]].to_numpy(),
        )
    )
    test_keys = set(
        map(
            tuple,
            split_manifest.loc[split_manifest["split"].eq("test"), ["tax_id", "GeneID"]].to_numpy(),
        )
    )

    train_counts: Counter[int] = Counter()
    test_counts: Counter[int] = Counter()
    label_cache = pd.read_csv(
        label_cache_path,
        sep="\t",
        dtype={"tax_id": str, "GeneID": str, "go_indices": str},
        usecols=["tax_id", "GeneID", "go_indices"],
    )
    for row in label_cache.itertuples(index=False):
        key = (row.tax_id, row.GeneID)
        if not isinstance(row.go_indices, str) or not row.go_indices:
            continue
        indices = [int(value) for value in row.go_indices.split(",") if value]
        if key in train_keys:
            train_counts.update(indices)
        if key in test_keys:
            test_counts.update(indices)

    output = per_term.copy()
    output["go_index"] = pd.to_numeric(output["go_index"], errors="coerce").astype("Int64")
    output["train_examples"] = output["go_index"].map(train_counts).fillna(0).astype(int)
    output["test_examples"] = output["go_index"].map(test_counts).fillna(0).astype(int)
    return output


def write_per_term_tables(per_term: pd.DataFrame, output_dir: Path) -> None:
    if per_term.empty or "test_term_fmax" not in per_term.columns:
        return
    table = per_term.copy()
    table["test_term_fmax"] = pd.to_numeric(table["test_term_fmax"], errors="coerce")
    for column in (
        "test_f1_at_global_threshold",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_positive_count",
        "database_frequency",
        "train_examples",
        "test_examples",
    ):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    if "train_examples" not in table.columns:
        table["train_examples"] = table.get("database_frequency", pd.NA)
    if "test_examples" not in table.columns:
        table["test_examples"] = table.get("test_positive_count", pd.NA)

    common_columns = [
        "GO_ID",
        "name",
        "namespace",
        "test_term_fmax",
        "train_examples",
        "test_examples",
        "test_best_threshold",
        "test_precision_at_best_threshold",
        "test_recall_at_best_threshold",
    ]
    present_columns = [column for column in common_columns if column in table.columns]

    top100_columns = [
        "GO_ID",
        "name",
        "namespace",
        "train_examples",
        "test_examples",
        "database_frequency",
        "test_positive_count",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_f1_at_global_threshold",
        "test_term_fmax",
        "test_best_threshold",
    ]
    top100_present = [column for column in top100_columns if column in table.columns]
    f1_sort_column = (
        "test_f1_at_global_threshold"
        if "test_f1_at_global_threshold" in table.columns
        else "test_term_fmax"
    )
    table.sort_values(
        [f1_sort_column, "test_examples", "train_examples"],
        ascending=[False, False, False],
    ).head(100)[top100_present].to_csv(
        output_dir / "top100_go_terms_by_test_f1.tsv",
        sep="\t",
        index=False,
    )

    high_f1 = table.loc[table["test_term_fmax"] >= 0.8, present_columns].sort_values(
        ["test_term_fmax", "test_examples", "train_examples"],
        ascending=[False, False, False],
    )
    high_f1.to_csv(output_dir / "go_terms_f1_ge_0.8.tsv", sep="\t", index=False)

    if "train_examples" in table.columns:
        top_train = table.loc[:, present_columns].sort_values(
            "train_examples", ascending=False
        ).head(10)
        top_train.to_csv(output_dir / "top10_go_terms_by_train_examples.tsv", sep="\t", index=False)


def plot_comparisons(
    comparisons: list[tuple[str, Path]],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    rows: list[dict[str, object]] = []
    for name, path in comparisons:
        if not path.exists():
            print(f"Skipping missing comparison metrics: {path}")
            continue
        frame = numeric_metrics(pd.read_csv(path, sep="\t"))
        validation = frame[frame["event"] == "validation"].copy()
        if validation.empty:
            continue
        best = validation.loc[validation["fmax"].idxmax()]
        rows.append(
            {
                "run": name,
                "best_step": int(best["step"]),
                "best_val_fmax": best["fmax"],
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "metrics_path": str(path),
            }
        )
    if not rows:
        return None

    comparison = pd.DataFrame(rows).sort_values("best_val_fmax", ascending=False)
    comparison.to_csv(output_dir / "comparison_best_validation_fmax.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(comparison["run"], comparison["best_val_fmax"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_title("Best validation Fmax across runs")
    ax.set_xlabel("best validation Fmax")
    for index, row in comparison.reset_index(drop=True).iterrows():
        ax.text(
            row["best_val_fmax"],
            index,
            f" {row['best_val_fmax']:.4f} @ {row['best_step']:,}",
            va="center",
        )
    return save_figure(fig, output_dir, "comparison_best_validation_fmax", dpi)


def load_go_benchmark_table(
    vocabulary_path: Path,
    term_counts_path: Path | None,
    output_dir: Path,
) -> pd.DataFrame:
    vocab = pd.read_csv(vocabulary_path, sep="\t")
    vocab["go_index"] = pd.to_numeric(vocab["go_index"], errors="coerce").astype("Int64")
    vocab["trainable_gene_count"] = pd.to_numeric(
        vocab["trainable_gene_count"], errors="coerce"
    ).fillna(0)
    vocab["category"] = vocab["namespace"].map(namespace_short)

    if term_counts_path is not None and term_counts_path.exists():
        counts = pd.read_csv(term_counts_path, sep="\t")
        keep_columns = [
            "GO_ID",
            "explicit_gene_count",
            "inferred_ancestor_gene_count",
            "explicit_only_gene_count",
            "inferred_only_gene_count",
            "explicit_and_inferred_gene_count",
        ]
        present = [column for column in keep_columns if column in counts.columns]
        if "GO_ID" in present:
            vocab = vocab.merge(counts[present], on="GO_ID", how="left")

    for column in (
        "explicit_gene_count",
        "inferred_ancestor_gene_count",
        "explicit_only_gene_count",
        "inferred_only_gene_count",
        "explicit_and_inferred_gene_count",
    ):
        if column in vocab.columns:
            vocab[column] = pd.to_numeric(vocab[column], errors="coerce").fillna(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(output_dir / "go_vocabulary_benchmark_table.tsv", sep="\t", index=False)
    return vocab


def plot_go_vocabulary_frequency(
    go_terms: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    if go_terms.empty:
        return None

    namespace_summary = (
        go_terms.groupby("category", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_trainable_gene_count=("trainable_gene_count", "median"),
            mean_trainable_gene_count=("trainable_gene_count", "mean"),
            total_term_gene_labels=("trainable_gene_count", "sum"),
        )
        .reset_index()
        .sort_values("category")
    )
    namespace_summary.to_csv(
        output_dir / "01_namespace_vocabulary_summary.tsv",
        sep="\t",
        index=False,
    )

    top_terms = go_terms.sort_values("trainable_gene_count", ascending=False).head(25)
    top_terms[
        ["go_index", "GO_ID", "name", "category", "trainable_gene_count"]
    ].to_csv(output_dir / "01_top25_go_terms_by_frequency.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    cumulative = go_terms.sort_values("trainable_gene_count", ascending=False).copy()
    total_labels = float(cumulative["trainable_gene_count"].sum())
    cumulative["term_rank"] = range(1, len(cumulative) + 1)
    cumulative["cumulative_label_fraction"] = (
        cumulative["trainable_gene_count"].cumsum() / max(total_labels, 1.0)
    )
    ax.plot(
        cumulative["term_rank"],
        cumulative["cumulative_label_fraction"],
        color="#54a24b",
        linewidth=2.0,
    )
    for rank in (25, 100, 250, 1000):
        if rank <= len(cumulative):
            ax.axvline(rank, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_title("Training labels are concentrated in common GO terms")
    ax.set_xlabel("GO terms ordered from most to least frequent")
    ax.set_ylabel("Share of training gene-GO labels covered")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    return save_figure(fig, output_dir, "01_go_vocabulary_frequency", dpi)


def gaussian_kde_1d(values: pd.Series, grid: np.ndarray) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.zeros_like(grid)
    if clean.size == 1 or float(np.std(clean)) == 0.0:
        center = float(clean[0])
        bandwidth = 0.05
    else:
        bandwidth = 1.06 * float(np.std(clean, ddof=1)) * (clean.size ** (-1.0 / 5.0))
        bandwidth = max(bandwidth, 0.03)
        center = 0.0
    if clean.size == 1:
        clean = np.array([center])
    scaled = (grid[:, None] - clean[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).sum(axis=1)
    density /= clean.size * bandwidth * np.sqrt(2.0 * np.pi)
    return density


def plot_go_term_appearance_kde(
    *,
    term_counts_path: Path | None,
    retained_go_terms: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    if term_counts_path is None or not term_counts_path.exists():
        return None
    counts = pd.read_csv(term_counts_path, sep="\t", usecols=["GO_ID", "trainable_gene_count"])
    counts["trainable_gene_count"] = pd.to_numeric(
        counts["trainable_gene_count"], errors="coerce"
    )
    counts = counts[counts["trainable_gene_count"] > 0].copy()
    if counts.empty:
        return None

    retained_ids = set(retained_go_terms["GO_ID"].astype(str))
    counts["retained_in_vocabulary"] = counts["GO_ID"].astype(str).isin(retained_ids)
    counts["log10_trainable_gene_count"] = np.log10(counts["trainable_gene_count"])
    counts.to_csv(output_dir / "00_go_term_appearance_counts.tsv", sep="\t", index=False)

    retained = counts[counts["retained_in_vocabulary"]].copy()
    dropped = counts[~counts["retained_in_vocabulary"]].copy()
    summary = pd.DataFrame(
        [
            {
                "group": "all_observed_true_path_terms",
                "terms": len(counts),
                "min_count": counts["trainable_gene_count"].min(),
                "median_count": counts["trainable_gene_count"].median(),
                "mean_count": counts["trainable_gene_count"].mean(),
                "max_count": counts["trainable_gene_count"].max(),
            },
            {
                "group": "retained_min_count_vocabulary",
                "terms": len(retained),
                "min_count": retained["trainable_gene_count"].min() if not retained.empty else np.nan,
                "median_count": retained["trainable_gene_count"].median() if not retained.empty else np.nan,
                "mean_count": retained["trainable_gene_count"].mean() if not retained.empty else np.nan,
                "max_count": retained["trainable_gene_count"].max() if not retained.empty else np.nan,
            },
            {
                "group": "dropped_below_vocabulary_threshold",
                "terms": len(dropped),
                "min_count": dropped["trainable_gene_count"].min() if not dropped.empty else np.nan,
                "median_count": dropped["trainable_gene_count"].median() if not dropped.empty else np.nan,
                "mean_count": dropped["trainable_gene_count"].mean() if not dropped.empty else np.nan,
                "max_count": dropped["trainable_gene_count"].max() if not dropped.empty else np.nan,
            },
        ]
    )
    summary.to_csv(output_dir / "00_go_term_appearance_kde_summary.tsv", sep="\t", index=False)

    x_min = max(0.0, float(counts["log10_trainable_gene_count"].min()) - 0.1)
    x_max = float(counts["log10_trainable_gene_count"].max()) + 0.1
    grid = np.linspace(x_min, x_max, 400)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    all_density = gaussian_kde_1d(counts["log10_trainable_gene_count"], grid)
    ax.plot(grid, all_density, color="#4c78a8", label=f"all observed terms, n={len(counts):,}")
    if not retained.empty:
        retained_density = gaussian_kde_1d(retained["log10_trainable_gene_count"], grid)
        ax.plot(
            grid,
            retained_density,
            color="#54a24b",
            label=f"retained vocabulary, n={len(retained):,}",
        )
    if not dropped.empty:
        dropped_density = gaussian_kde_1d(dropped["log10_trainable_gene_count"], grid)
        ax.plot(
            grid,
            dropped_density,
            color="#e45756",
            alpha=0.75,
            label=f"dropped terms, n={len(dropped):,}",
        )
    if not retained.empty:
        threshold = float(retained["trainable_gene_count"].min())
        if threshold > 0:
            ax.axvline(
                np.log10(threshold),
                color="#222222",
                linestyle="--",
                linewidth=1.2,
                label=f"vocabulary cutoff: {threshold:,.0f} genes",
            )
    ax.set_title("GO term frequency before and after vocabulary filtering")
    ax.set_xlabel("log10(number of training genes annotated with GO term)")
    ax.set_ylabel("kernel density")
    ax.legend(frameon=False)
    clean_axis(ax)

    return save_figure(fig, output_dir, "00_go_term_appearance_kde", dpi)


def plot_true_path_composition(
    go_terms: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    required = {
        "explicit_only_gene_count",
        "inferred_only_gene_count",
        "explicit_and_inferred_gene_count",
    }
    if go_terms.empty or not required.issubset(go_terms.columns):
        return None

    composition = (
        go_terms.groupby("category", dropna=False)
        .agg(
            explicit_only=("explicit_only_gene_count", "sum"),
            inferred_only=("inferred_only_gene_count", "sum"),
            explicit_and_inferred=("explicit_and_inferred_gene_count", "sum"),
            terms=("GO_ID", "count"),
        )
        .reset_index()
        .sort_values("category")
    )
    composition.to_csv(
        output_dir / "02_true_path_composition_by_namespace.tsv",
        sep="\t",
        index=False,
    )

    plot_terms = go_terms.copy()
    denom = plot_terms["trainable_gene_count"].replace(0, pd.NA)
    plot_terms["inferred_only_fraction"] = plot_terms["inferred_only_gene_count"] / denom
    plot_terms["explicit_only_fraction"] = plot_terms["explicit_only_gene_count"] / denom
    plot_terms[
        [
            "go_index",
            "GO_ID",
            "name",
            "category",
            "trainable_gene_count",
            "explicit_only_gene_count",
            "inferred_only_gene_count",
            "explicit_and_inferred_gene_count",
            "inferred_only_fraction",
            "explicit_only_fraction",
        ]
    ].to_csv(output_dir / "02_true_path_term_composition.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = [0.0] * len(composition)
    for column, label, color in (
        ("explicit_only", "explicit only", "#4c78a8"),
        ("inferred_only", "inferred ancestor only", "#f58518"),
        ("explicit_and_inferred", "explicit and inferred", "#54a24b"),
    ):
        values = composition[column].astype(float).to_numpy()
        ax.bar(composition["category"], values, bottom=bottom, label=label, color=color)
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    ax.set_title("Where true-path labels come from")
    ax.set_xlabel("GO namespace")
    ax.set_ylabel("Training gene-GO label assignments")
    ax.yaxis.set_major_formatter(COUNT_FORMATTER)
    ax.legend(frameon=False)
    clean_axis(ax)

    return save_figure(fig, output_dir, "02_true_path_label_composition", dpi)


def holdout_keys_from_split_manifest(
    split_manifest_path: Path,
    *,
    chunksize: int = 500_000,
) -> dict[tuple[str, str], str]:
    holdout: dict[tuple[str, str], str] = {}
    for chunk in pd.read_csv(
        split_manifest_path,
        sep="\t",
        dtype={"split": str, "tax_id": str, "GeneID": str},
        usecols=["split", "tax_id", "GeneID"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["split"].isin(["val", "test"])]
        for row in chunk.drop_duplicates(["tax_id", "GeneID"]).itertuples(index=False):
            holdout[(str(row.tax_id), str(row.GeneID))] = str(row.split)
    return holdout


def build_holdout_go_label_counts(
    go_terms: pd.DataFrame,
    split_manifest_path: Path,
    label_cache_path: Path,
    output_dir: Path,
    *,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    cache_path = output_dir / "03_holdout_go_label_counts.tsv"
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")

    holdout = holdout_keys_from_split_manifest(split_manifest_path)
    val_counts: Counter[int] = Counter()
    test_counts: Counter[int] = Counter()

    for chunk in pd.read_csv(
        label_cache_path,
        sep="\t",
        dtype={"tax_id": str, "GeneID": str, "go_indices": str},
        usecols=["tax_id", "GeneID", "go_indices"],
        chunksize=chunksize,
    ):
        for row in chunk.itertuples(index=False):
            split = holdout.get((str(row.tax_id), str(row.GeneID)))
            if split is None or not isinstance(row.go_indices, str) or not row.go_indices:
                continue
            indices = [int(value) for value in row.go_indices.split(",") if value]
            if split == "val":
                val_counts.update(indices)
            elif split == "test":
                test_counts.update(indices)

    output = go_terms.copy()
    output["go_index_int"] = pd.to_numeric(output["go_index"], errors="coerce").astype("Int64")
    output["val_positive_genes"] = output["go_index_int"].map(val_counts).fillna(0).astype(int)
    output["test_positive_genes"] = output["go_index_int"].map(test_counts).fillna(0).astype(int)
    output["train_positive_genes_estimated"] = (
        output["trainable_gene_count"]
        - output["val_positive_genes"]
        - output["test_positive_genes"]
    ).clip(lower=0).astype(int)
    output = output.drop(columns=["go_index_int"])
    output.to_csv(cache_path, sep="\t", index=False)
    return output


def plot_holdout_go_label_coverage(
    go_counts: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    if go_counts.empty:
        return None

    coverage = (
        go_counts.groupby("category", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            terms_with_val_labels=("val_positive_genes", lambda s: int((s > 0).sum())),
            terms_with_test_labels=("test_positive_genes", lambda s: int((s > 0).sum())),
            val_positive_gene_term_pairs=("val_positive_genes", "sum"),
            test_positive_gene_term_pairs=("test_positive_genes", "sum"),
        )
        .reset_index()
        .sort_values("category")
    )
    coverage.to_csv(output_dir / "03_holdout_go_coverage_by_namespace.tsv", sep="\t", index=False)

    absent_test = go_counts[go_counts["test_positive_genes"].eq(0)].copy()
    absent_test[
        ["go_index", "GO_ID", "name", "category", "trainable_gene_count", "val_positive_genes"]
    ].to_csv(output_dir / "03_go_terms_absent_from_test.tsv", sep="\t", index=False)
    go_counts.sort_values("test_positive_genes", ascending=False).head(25)[
        ["go_index", "GO_ID", "name", "category", "trainable_gene_count", "test_positive_genes"]
    ].to_csv(output_dir / "03_top25_go_terms_by_test_labels.tsv", sep="\t", index=False)

    coverage["val_term_fraction"] = (
        coverage["terms_with_val_labels"] / coverage["terms"].replace(0, pd.NA)
    )
    coverage["test_term_fraction"] = (
        coverage["terms_with_test_labels"] / coverage["terms"].replace(0, pd.NA)
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = list(range(len(coverage)))
    width = 0.35
    ax.bar(
        [pos - width / 2 for pos in positions],
        coverage["val_term_fraction"],
        width=width,
        label="rat validation",
        color="#f58518",
    )
    ax.bar(
        [pos + width / 2 for pos in positions],
        coverage["test_term_fraction"],
        width=width,
        label="human test",
        color="#54a24b",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(coverage["category"])
    ax.set_ylim(0, 1.02)
    ax.set_title("How much of the GO vocabulary appears in each holdout set")
    ax.set_xlabel("GO namespace")
    ax.set_ylabel("Vocabulary terms with at least one positive gene")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False)
    clean_axis(ax)

    return save_figure(fig, output_dir, "03_holdout_go_label_coverage", dpi)


def plot_per_go_term_prediction_benchmarks(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    required = {
        "GO_ID",
        "namespace",
        "database_frequency",
        "test_positive_count",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_f1_at_global_threshold",
        "test_term_fmax",
    }
    if per_term.empty or not required.issubset(per_term.columns):
        return None

    metrics = per_term.copy()
    metrics["category"] = metrics["namespace"].map(namespace_short)
    numeric_columns = [
        "database_frequency",
        "test_positive_count",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_f1_at_global_threshold",
        "test_term_fmax",
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics[metrics["test_positive_count"].fillna(0) > 0].copy()
    if metrics.empty:
        return None

    namespace_summary = (
        metrics.groupby("category", dropna=False)
        .agg(
            terms_with_test_labels=("GO_ID", "count"),
            median_global_f1=("test_f1_at_global_threshold", "median"),
            median_global_precision=("test_precision_at_global_threshold", "median"),
            median_global_recall=("test_recall_at_global_threshold", "median"),
            median_term_fmax=("test_term_fmax", "median"),
            total_test_positive_labels=("test_positive_count", "sum"),
        )
        .reset_index()
        .sort_values("category")
    )
    namespace_summary.to_csv(
        output_dir / "04_per_go_term_predictive_summary_by_namespace.tsv",
        sep="\t",
        index=False,
    )

    metrics["frequency_decile"] = pd.qcut(
        metrics["database_frequency"].rank(method="first"),
        q=min(10, len(metrics)),
        labels=False,
    )
    frequency_deciles = (
        metrics.groupby("frequency_decile", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_database_frequency=("database_frequency", "median"),
            median_test_positive_count=("test_positive_count", "median"),
            median_global_f1=("test_f1_at_global_threshold", "median"),
            median_global_precision=("test_precision_at_global_threshold", "median"),
            median_global_recall=("test_recall_at_global_threshold", "median"),
            median_term_fmax=("test_term_fmax", "median"),
        )
        .reset_index()
    )
    frequency_deciles.to_csv(
        output_dir / "04_per_go_term_predictive_frequency_deciles.tsv",
        sep="\t",
        index=False,
    )

    table_columns = [
        "GO_ID",
        "name",
        "category",
        "database_frequency",
        "test_positive_count",
        "test_f1_at_global_threshold",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_term_fmax",
        "test_best_threshold",
    ]
    present_columns = [column for column in table_columns if column in metrics.columns]
    top_terms = metrics.sort_values(
        ["test_f1_at_global_threshold", "test_positive_count"],
        ascending=[False, False],
    ).head(25)
    bottom_terms = metrics.sort_values(
        ["test_f1_at_global_threshold", "test_positive_count"],
        ascending=[True, False],
    ).head(25)
    pd.concat(
        [
            top_terms.assign(rank_group="top_global_f1"),
            bottom_terms.assign(rank_group="bottom_global_f1"),
        ],
        ignore_index=True,
    )[["rank_group", *present_columns]].to_csv(
        output_dir / "04_top_bottom_go_terms_by_global_f1.tsv",
        sep="\t",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"BP": "#4c78a8", "CC": "#f58518", "MF": "#54a24b"}
    for category, subset in metrics.groupby("category", dropna=False):
        ax.scatter(
            subset["test_positive_count"].clip(lower=1),
            subset["test_f1_at_global_threshold"],
            s=18,
            alpha=0.55,
            color=colors.get(str(category), "#777777"),
            label=str(category),
            edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-GO-term performance on human test genes")
    ax.set_xlabel("Human test genes annotated with this GO term")
    ax.set_ylabel("F1 for this GO term at the global test threshold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axis(ax)
    ax.legend(title="Namespace", frameon=False)

    return save_figure(fig, output_dir, "04_per_go_term_predictive_performance", dpi)


def plot_go_prediction_concentration(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    required = {
        "GO_ID",
        "name",
        "namespace",
        "test_positive_count",
        "test_tp_at_global_threshold",
        "test_fp_at_global_threshold",
        "test_fn_at_global_threshold",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_f1_at_global_threshold",
    }
    if per_term.empty or not required.issubset(per_term.columns):
        return None

    metrics = per_term.copy()
    metrics["category"] = metrics["namespace"].map(namespace_short)
    numeric_columns = [
        "test_positive_count",
        "test_tp_at_global_threshold",
        "test_fp_at_global_threshold",
        "test_fn_at_global_threshold",
        "test_precision_at_global_threshold",
        "test_recall_at_global_threshold",
        "test_f1_at_global_threshold",
    ]
    for column in numeric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce").fillna(0.0)
    metrics = metrics[metrics["test_positive_count"] > 0].copy()
    if metrics.empty:
        return None

    ranked = metrics.sort_values("test_positive_count", ascending=False).reset_index(drop=True)
    ranked["rank_by_test_positive_count"] = ranked.index + 1

    total_positive = float(ranked["test_positive_count"].sum())
    total_tp = float(ranked["test_tp_at_global_threshold"].sum())
    total_fp = float(ranked["test_fp_at_global_threshold"].sum())
    ranked["cumulative_positive_share"] = ranked["test_positive_count"].cumsum() / max(
        total_positive, 1.0
    )
    ranked["cumulative_true_positive_share"] = ranked[
        "test_tp_at_global_threshold"
    ].cumsum() / max(total_tp, 1.0)
    ranked["cumulative_false_positive_share"] = ranked[
        "test_fp_at_global_threshold"
    ].cumsum() / max(total_fp, 1.0)
    ranked["cumulative_tp"] = ranked["test_tp_at_global_threshold"].cumsum()
    ranked["cumulative_fp"] = ranked["test_fp_at_global_threshold"].cumsum()
    ranked["cumulative_positive_count"] = ranked["test_positive_count"].cumsum()
    ranked["cumulative_micro_precision"] = ranked["cumulative_tp"] / (
        ranked["cumulative_tp"] + ranked["cumulative_fp"]
    ).replace(0, pd.NA)
    ranked["cumulative_recall_within_included_terms"] = ranked["cumulative_tp"] / ranked[
        "cumulative_positive_count"
    ].replace(0, pd.NA)
    ranked[
        [
            "rank_by_test_positive_count",
            "GO_ID",
            "name",
            "category",
            "test_positive_count",
            "test_tp_at_global_threshold",
            "test_fp_at_global_threshold",
            "test_fn_at_global_threshold",
            "test_precision_at_global_threshold",
            "test_recall_at_global_threshold",
            "test_f1_at_global_threshold",
            "cumulative_positive_share",
            "cumulative_true_positive_share",
            "cumulative_false_positive_share",
            "cumulative_micro_precision",
            "cumulative_recall_within_included_terms",
        ]
    ].to_csv(output_dir / "05_go_term_prediction_concentration_ranked.tsv", sep="\t", index=False)

    top_n_values = [25, 100, 250]
    top_rows: list[dict[str, float | int]] = []
    for top_n in top_n_values:
        subset = ranked.head(min(top_n, len(ranked)))
        tp = float(subset["test_tp_at_global_threshold"].sum())
        fp = float(subset["test_fp_at_global_threshold"].sum())
        positives = float(subset["test_positive_count"].sum())
        top_rows.append(
            {
                "top_n": top_n,
                "terms_included": len(subset),
                "positive_label_share": positives / max(total_positive, 1.0),
                "true_positive_share": tp / max(total_tp, 1.0),
                "false_positive_share": fp / max(total_fp, 1.0),
                "micro_precision_within_top_n": tp / max(tp + fp, 1.0),
                "recall_within_top_n_terms": tp / max(positives, 1.0),
                "overall_recall_contributed": tp / max(total_positive, 1.0),
            }
        )
    top_summary = pd.DataFrame(top_rows)
    top_summary.to_csv(output_dir / "05_top_n_go_term_concentration.tsv", sep="\t", index=False)

    ranked.head(50)[
        [
            "rank_by_test_positive_count",
            "GO_ID",
            "name",
            "category",
            "test_positive_count",
            "test_tp_at_global_threshold",
            "test_fp_at_global_threshold",
            "test_recall_at_global_threshold",
            "test_f1_at_global_threshold",
        ]
    ].to_csv(output_dir / "05_top50_go_terms_by_test_label_count.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        ranked["rank_by_test_positive_count"],
        ranked["cumulative_positive_count"],
        label="positive labels in human test",
        color="#4c78a8",
        linewidth=2.0,
    )
    ax.plot(
        ranked["rank_by_test_positive_count"],
        ranked["cumulative_tp"],
        label="correct positive predictions",
        color="#54a24b",
        linewidth=2.0,
    )
    for top_n in top_n_values:
        ax.axvline(top_n, color="#999999", linestyle=":", linewidth=1.0)
        if top_n <= len(ranked):
            row = ranked.iloc[top_n - 1]
            ax.annotate(
                f"top {top_n}",
                xy=(top_n, row["cumulative_tp"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=9,
                color="#444444",
            )
    ax.set_xscale("log")
    ax.set_title("Correct predictions concentrate in frequent GO terms")
    ax.set_xlabel("Top N GO terms, ordered by human-test positive labels")
    ax.set_ylabel("Cumulative human gene-GO pairs")
    ax.yaxis.set_major_formatter(COUNT_FORMATTER)
    clean_axis(ax)
    ax.legend(frameon=False)

    return save_figure(fig, output_dir, "05_go_term_prediction_concentration", dpi)


def benchmark_per_term_metrics(per_term: pd.DataFrame) -> pd.DataFrame:
    required = {
        "GO_ID",
        "name",
        "namespace",
        "database_frequency",
        "test_positive_count",
        "test_f1_at_global_threshold",
        "test_term_fmax",
    }
    if per_term.empty or not required.issubset(per_term.columns):
        return pd.DataFrame()

    metrics = per_term.copy()
    metrics["category"] = metrics["namespace"].map(namespace_short)
    for column in (
        "database_frequency",
        "test_positive_count",
        "test_f1_at_global_threshold",
        "test_term_fmax",
    ):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    metrics = metrics[
        (metrics["database_frequency"] > 0)
        & (metrics["test_positive_count"].fillna(0) > 0)
    ].copy()
    if metrics.empty:
        return metrics

    # Per-term F1 is defined at the selected global test threshold. Terms with
    # positives but no positive predictions are plotted as F1=0.
    metrics["test_f1_at_global_threshold"] = metrics[
        "test_f1_at_global_threshold"
    ].fillna(0.0)
    metrics["test_term_fmax"] = metrics["test_term_fmax"].fillna(0.0)
    return metrics


def plot_benchmark_f1_vs_go_frequency(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    metrics = benchmark_per_term_metrics(per_term)
    if metrics.empty:
        return None

    metrics["frequency_decile"] = pd.qcut(
        metrics["database_frequency"].rank(method="first"),
        q=min(10, len(metrics)),
        labels=False,
    )
    deciles = (
        metrics.groupby("frequency_decile", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_dataset_appearances=("database_frequency", "median"),
            median_human_test_positives=("test_positive_count", "median"),
            median_f1=("test_f1_at_global_threshold", "median"),
            f1_p25=("test_f1_at_global_threshold", lambda s: s.quantile(0.25)),
            f1_p75=("test_f1_at_global_threshold", lambda s: s.quantile(0.75)),
            median_fmax=("test_term_fmax", "median"),
        )
        .reset_index()
    )
    metrics[
        [
            "GO_ID",
            "name",
            "category",
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
        ]
    ].to_csv(output_dir / "01_go_term_f1_frequency_table.tsv", sep="\t", index=False)
    deciles.to_csv(output_dir / "01_f1_vs_go_term_frequency_deciles.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {"BP": "#4c78a8", "CC": "#f58518", "MF": "#54a24b"}
    for category, subset in metrics.groupby("category", dropna=False):
        ax.scatter(
            subset["database_frequency"],
            subset["test_f1_at_global_threshold"],
            s=18,
            alpha=0.42,
            color=colors.get(str(category), "#777777"),
            label=str(category),
            edgecolors="none",
        )

    ax.plot(
        deciles["median_dataset_appearances"],
        deciles["median_f1"],
        color="#222222",
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        label="median by frequency decile",
    )
    ax.vlines(
        deciles["median_dataset_appearances"],
        deciles["f1_p25"],
        deciles["f1_p75"],
        color="#222222",
        linewidth=1.1,
        alpha=0.55,
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GO-term F1 versus how often the term appears")
    ax.set_xlabel("Genes in the dataset annotated with this GO term")
    ax.set_ylabel("Human test F1 for this GO term")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(COUNT_FORMATTER)
    clean_axis(ax)
    ax.legend(title="GO namespace", frameon=False)
    return save_figure(fig, output_dir, "01_f1_vs_go_term_frequency", dpi)


def plot_benchmark_explicit_vs_inferred_counts(
    go_terms: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    required = {"GO_ID", "name", "category", "explicit_gene_count", "inferred_ancestor_gene_count"}
    if go_terms.empty or not required.issubset(go_terms.columns):
        return None

    terms = go_terms.copy()
    for column in ("explicit_gene_count", "inferred_ancestor_gene_count", "trainable_gene_count"):
        if column in terms.columns:
            terms[column] = pd.to_numeric(terms[column], errors="coerce").fillna(0.0)
    terms = terms[
        (terms["explicit_gene_count"] > 0) | (terms["inferred_ancestor_gene_count"] > 0)
    ].copy()
    if terms.empty:
        return None

    terms[
        [
            "GO_ID",
            "name",
            "category",
            "trainable_gene_count",
            "explicit_gene_count",
            "inferred_ancestor_gene_count",
            "explicit_only_gene_count",
            "inferred_only_gene_count",
            "explicit_and_inferred_gene_count",
        ]
    ].to_csv(output_dir / "02_explicit_vs_inferred_counts_table.tsv", sep="\t", index=False)

    summary = (
        terms.groupby("category", dropna=False)
        .agg(
            terms=("GO_ID", "count"),
            median_explicit_genes=("explicit_gene_count", "median"),
            median_inferred_ancestor_genes=("inferred_ancestor_gene_count", "median"),
            total_explicit_genes=("explicit_gene_count", "sum"),
            total_inferred_ancestor_genes=("inferred_ancestor_gene_count", "sum"),
        )
        .reset_index()
        .sort_values("category")
    )
    summary.to_csv(output_dir / "02_explicit_vs_inferred_counts_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = {"BP": "#4c78a8", "CC": "#f58518", "MF": "#54a24b"}
    for category, subset in terms.groupby("category", dropna=False):
        ax.scatter(
            subset["explicit_gene_count"],
            subset["inferred_ancestor_gene_count"],
            s=18,
            alpha=0.42,
            color=colors.get(str(category), "#777777"),
            label=str(category),
            edgecolors="none",
        )

    max_count = max(
        float(terms["explicit_gene_count"].max()),
        float(terms["inferred_ancestor_gene_count"].max()),
        1.0,
    )
    ax.plot([0, max_count], [0, max_count], color="#222222", linestyle=":", linewidth=1.2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Explicit versus ancestor-inferred GO-term appearances")
    ax.set_xlabel("Genes explicitly annotated to this GO term")
    ax.set_ylabel("Genes where this GO term is inferred from descendants")
    ax.xaxis.set_major_formatter(COUNT_FORMATTER)
    ax.yaxis.set_major_formatter(COUNT_FORMATTER)
    clean_axis(ax)
    ax.legend(title="GO namespace", frameon=False)
    return save_figure(fig, output_dir, "02_explicit_vs_inferred_counts", dpi)


def plot_benchmark_ranked_go_term_f1(
    per_term: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    metrics = benchmark_per_term_metrics(per_term)
    if metrics.empty:
        return None

    ranked = metrics.sort_values(
        ["test_f1_at_global_threshold", "test_positive_count", "database_frequency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked["rank_by_human_test_f1"] = ranked.index + 1
    ranked[
        [
            "rank_by_human_test_f1",
            "GO_ID",
            "name",
            "category",
            "database_frequency",
            "test_positive_count",
            "test_f1_at_global_threshold",
            "test_term_fmax",
        ]
    ].to_csv(output_dir / "03_go_term_f1_ranked.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = {"BP": "#4c78a8", "CC": "#f58518", "MF": "#54a24b"}
    for category, subset in ranked.groupby("category", dropna=False):
        ax.scatter(
            subset["rank_by_human_test_f1"],
            subset["test_f1_at_global_threshold"],
            s=14,
            alpha=0.50,
            color=colors.get(str(category), "#777777"),
            label=str(category),
            edgecolors="none",
        )
    median_f1 = ranked["test_f1_at_global_threshold"].median()
    ax.axhline(
        median_f1,
        color="#222222",
        linewidth=1.2,
        linestyle="--",
        label=f"median F1 {median_f1:.2f}",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("F1 for each GO term with human-test labels")
    ax.set_xlabel("GO terms sorted from highest to lowest F1")
    ax.set_ylabel("Human test F1 for this GO term")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(RANK_FORMATTER)
    clean_axis(ax)
    ax.legend(title="GO namespace", frameon=False)
    return save_figure(fig, output_dir, "03_go_term_f1_ranked", dpi)


def write_go_benchmark_summary(
    output_dir: Path,
    go_terms: pd.DataFrame,
    holdout_counts: pd.DataFrame | None,
    per_term: pd.DataFrame | None,
    test_summary: pd.Series | None,
    written: list[tuple[object, object]],
) -> None:
    rows: list[tuple[object, object]] = [
        ("vocabulary_terms", len(go_terms)),
        ("min_trainable_gene_count", int(go_terms["trainable_gene_count"].min())),
        ("median_trainable_gene_count", f"{go_terms['trainable_gene_count'].median():.3f}"),
        ("max_trainable_gene_count", int(go_terms["trainable_gene_count"].max())),
    ]
    if holdout_counts is not None and not holdout_counts.empty:
        rows.extend(
            [
                ("terms_with_human_test_labels", int((holdout_counts["test_positive_genes"] > 0).sum())),
                ("terms_without_human_test_labels", int((holdout_counts["test_positive_genes"] == 0).sum())),
                ("human_test_gene_term_pairs", int(holdout_counts["test_positive_genes"].sum())),
                ("rat_validation_gene_term_pairs", int(holdout_counts["val_positive_genes"].sum())),
            ]
        )
    if per_term is not None and not per_term.empty and "test_positive_count" in per_term.columns:
        test_positive_count = pd.to_numeric(
            per_term["test_positive_count"], errors="coerce"
        ).fillna(0)
        rows.extend(
            [
                ("per_term_metric_terms", len(per_term)),
                ("per_term_metric_terms_with_test_labels", int((test_positive_count > 0).sum())),
            ]
        )
    if test_summary is not None:
        for key in ("fmax", "threshold", "precision", "recall", "records_evaluated"):
            if key in test_summary.index:
                rows.append((f"test_{key}", test_summary.loc[key]))
    rows.extend((f"figure_{name}", path) for name, path in written)
    write_key_value_tsv(output_dir / "benchmark_manifest.tsv", rows)


def build_go_benchmarks(
    *,
    vocabulary_path: Path | None,
    term_counts_path: Path | None,
    split_manifest_path: Path | None,
    label_cache_path: Path | None,
    per_term_metrics: pd.DataFrame | None,
    test_summary: pd.Series | None,
    output_dir: Path,
    dpi: int,
    skip_holdout_counts: bool,
) -> list[tuple[object, object]]:
    written: list[tuple[object, object]] = []
    if vocabulary_path is None or not vocabulary_path.exists():
        print("Skipping GO benchmarks; vocabulary path is unavailable.")
        return written

    output_dir.mkdir(parents=True, exist_ok=True)
    go_terms = load_go_benchmark_table(vocabulary_path, term_counts_path, output_dir)

    holdout_counts = None
    if not skip_holdout_counts and split_manifest_path is not None and label_cache_path is not None and split_manifest_path.exists() and label_cache_path.exists():
        holdout_counts_path = output_dir / "03_holdout_go_label_counts.tsv"
        if holdout_counts_path.exists():
            holdout_counts = pd.read_csv(holdout_counts_path, sep="\t")

    if per_term_metrics is not None and not per_term_metrics.empty:
        path = plot_benchmark_f1_vs_go_frequency(
            per_term_metrics,
            output_dir,
            dpi,
        )
        if path:
            written.append(("f1_vs_go_term_frequency", path))

        path = plot_benchmark_ranked_go_term_f1(
            per_term_metrics,
            output_dir,
            dpi,
        )
        if path:
            written.append(("go_term_f1_ranked", path))

    path = plot_benchmark_explicit_vs_inferred_counts(go_terms, output_dir, dpi)
    if path:
        written.append(("explicit_vs_inferred_counts", path))

    write_go_benchmark_summary(
        output_dir,
        go_terms,
        holdout_counts,
        per_term_metrics,
        test_summary,
        written,
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT_SUMMARY)
    parser.add_argument("--test-summary", type=Path)
    parser.add_argument("--test-threshold-curve", type=Path)
    parser.add_argument("--per-term-metrics", type=Path)
    parser.add_argument("--gene-metrics", type=Path)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--label-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--go-term-counts", type=Path)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument(
        "--comparison-metrics",
        action="append",
        type=parse_comparison,
        default=[],
        metavar="NAME=PATH",
        help="Optional comparison run metrics in NAME=PATH format. If omitted, no comparison figure is written.",
    )
    parser.add_argument(
        "--include-default-comparisons",
        action="store_true",
        help="Include the historical default random-split IEA and non-IEA comparisons.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("figures"))
    parser.add_argument(
        "--figure-folder",
        help="Subdirectory under --output-root. Defaults to --run-dir basename.",
    )
    parser.add_argument(
        "--skip-go-benchmarks",
        action="store_true",
        help="Skip the nested benchmark/ GO-term visualization folder.",
    )
    parser.add_argument(
        "--skip-holdout-label-counts",
        action="store_true",
        help="Skip val/test GO-label coverage counts from split manifest and label cache.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    run_dir = args.run_dir
    project_root = Path.cwd()
    run_metadata_path = run_dir / "training_run_metadata.tsv"
    run_metadata = read_key_value_tsv(run_metadata_path) if run_metadata_path.exists() else None

    training_metrics_path = args.training_metrics or run_dir / "training_metrics.tsv"
    test_summary_path = args.test_summary or run_dir / "test_eval/test_evaluation_summary.tsv"
    test_curve_path = args.test_threshold_curve or run_dir / "test_eval/test_threshold_curve.tsv"
    per_term_path = args.per_term_metrics or run_dir / "test_eval/test_go_term_fmax_vs_frequency.tsv"
    gene_metrics_path = args.gene_metrics or run_dir / "test_eval/test_gene_metrics.tsv"

    metadata_vocabulary = None
    metadata_label_cache = None
    metadata_split_manifest = None
    if run_metadata is not None:
        metadata_vocabulary = resolve_recorded_path(
            run_metadata.get("vocabulary"),
            project_root=project_root,
        )
        metadata_label_cache = resolve_recorded_path(
            run_metadata.get("label_cache"),
            project_root=project_root,
        )
        metadata_split_manifest = resolve_recorded_path(
            run_metadata.get("split_manifest"),
            project_root=project_root,
        )

    vocabulary_path = resolve_recorded_path(args.vocabulary, project_root=project_root)
    if vocabulary_path is None:
        vocabulary_path = metadata_vocabulary

    label_cache_path = resolve_recorded_path(args.label_cache, project_root=project_root)
    if not label_cache_path.exists() and metadata_label_cache is not None:
        label_cache_path = metadata_label_cache

    split_manifest_path = resolve_recorded_path(args.split_manifest, project_root=project_root)
    if not split_manifest_path.exists() and metadata_split_manifest is not None:
        split_manifest_path = metadata_split_manifest

    split_summary_path = resolve_recorded_path(args.split_summary, project_root=project_root)
    if not split_summary_path.exists():
        inferred = infer_summary_path(split_manifest_path)
        if inferred is not None:
            split_summary_path = inferred

    go_term_counts_path = resolve_recorded_path(args.go_term_counts, project_root=project_root)
    if go_term_counts_path is None:
        go_term_counts_path = infer_true_path_term_counts(vocabulary_path)
    go_basic_obo_path = resolve_recorded_path(args.go_basic_obo, project_root=project_root)

    folder = sanitize_folder_name(args.figure_folder or run_dir.name)
    output_dir = args.output_root / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[object, object]] = []

    metrics = numeric_metrics(pd.read_csv(training_metrics_path, sep="\t"))
    metrics.to_csv(output_dir / "training_metrics_numeric.tsv", sep="\t", index=False)
    validation_metrics = metrics[metrics["event"] == "validation"].copy()

    split_summary = None
    if split_summary_path.exists():
        split_summary = read_key_value_tsv(split_summary_path)
        path = plot_split_sizes(split_summary, output_dir, args.dpi)
        if path:
            written.append(("split_sizes", path))

    test_summary = None
    if test_summary_path.exists():
        test_summary = read_key_value_tsv(test_summary_path)

    written.append(
        ("training_diagnostics", plot_training_diagnostics(metrics, output_dir, args.dpi))
    )
    path = plot_loss_diagnostics(metrics, output_dir, args.dpi)
    if path:
        written.append(("loss_diagnostics", path))
    path = plot_validation_metrics(validation_metrics, test_summary, output_dir, args.dpi)
    if path:
        written.append(("validation_metrics", path))

    if test_curve_path.exists():
        curve = pd.read_csv(test_curve_path, sep="\t")
        path = plot_test_threshold_curve(curve, output_dir, args.dpi)
        if path:
            written.append(("test_threshold_curve", path))
    else:
        print(f"Skipping test threshold curve; file not found: {test_curve_path}")

    per_term = pd.DataFrame()
    if per_term_path.exists():
        per_term = pd.read_csv(per_term_path, sep="\t")
        if split_manifest_path.exists() and label_cache_path.exists():
            per_term = add_train_test_term_counts(
                per_term,
                split_manifest_path,
                label_cache_path,
            )
            per_term.to_csv(output_dir / "go_term_metrics_with_split_counts.tsv", sep="\t", index=False)
            write_per_term_tables(per_term, output_dir)
        written.extend(plot_per_term_metrics(per_term, output_dir, args.dpi))
        path = plot_go_namespace_f1(per_term, output_dir, args.dpi)
        if path:
            written.append(("go_namespace_f1_at_gene_averaged_global_threshold", path))
        path = plot_go_depth_f1(per_term, go_basic_obo_path, output_dir, args.dpi)
        if path:
            written.append(("go_depth_f1_at_gene_averaged_global_threshold", path))
        path = plot_go_namespace_fmax(per_term, output_dir, args.dpi)
        if path:
            written.append(("go_namespace_fmax_at_per_go_term_optimal_threshold", path))
        path = plot_go_depth_fmax(per_term, go_basic_obo_path, output_dir, args.dpi)
        if path:
            written.append(("go_depth_fmax_at_per_go_term_optimal_threshold", path))
    else:
        print(f"Skipping per-term plot; file not found: {per_term_path}")

    if gene_metrics_path.exists():
        gene_metrics = pd.read_csv(gene_metrics_path, sep="\t")
        path = plot_gene_annotation_count_performance(
            gene_metrics,
            output_dir,
            args.dpi,
        )
        if path:
            written.append(("gene_annotation_count_performance", path))
    else:
        print(f"Skipping gene annotation-count plot; file not found: {gene_metrics_path}")

    comparisons = list(args.comparison_metrics)
    if args.include_default_comparisons:
        default_random_iea = Path(
            "outputs/model_training/non_iea_genes_all_labels_min_count_10/"
            "runs/random_split_bs512_80k_pbatch2h/training_metrics.tsv"
        )
        default_random_non_iea = Path(
            "outputs/model_training/non_iea_min_count_10/"
            "runs/random_split_bs512_80k_pbatch2h/training_metrics.tsv"
        )
        comparisons = [
            ("current_run", training_metrics_path),
            ("random_split_iea_same_genes", default_random_iea),
            ("random_split_non_iea", default_random_non_iea),
        ]
    if comparisons:
        path = plot_comparisons(comparisons, output_dir, args.dpi)
        if path:
            written.append(("comparison_best_validation_fmax", path))

    if not args.skip_go_benchmarks:
        benchmark_dir = output_dir / "benchmark"
        benchmark_written = build_go_benchmarks(
            vocabulary_path=vocabulary_path,
            term_counts_path=go_term_counts_path,
            split_manifest_path=split_manifest_path,
            label_cache_path=label_cache_path,
            per_term_metrics=per_term,
            test_summary=test_summary,
            output_dir=benchmark_dir,
            dpi=args.dpi,
            skip_holdout_counts=args.skip_holdout_label_counts,
        )
        written.extend((f"benchmark_{name}", path) for name, path in benchmark_written)

    summary_rows: list[tuple[object, object]] = [
        ("run_dir", run_dir),
        ("training_metrics", training_metrics_path),
        ("split_summary", split_summary_path if split_summary_path.exists() else ""),
        ("test_summary", test_summary_path if test_summary_path.exists() else ""),
        ("test_threshold_curve", test_curve_path if test_curve_path.exists() else ""),
        ("per_term_metrics", per_term_path if per_term_path.exists() else ""),
        ("gene_metrics", gene_metrics_path if gene_metrics_path.exists() else ""),
        ("vocabulary", vocabulary_path if vocabulary_path is not None and vocabulary_path.exists() else ""),
        ("go_term_counts", go_term_counts_path if go_term_counts_path is not None and go_term_counts_path.exists() else ""),
        ("go_basic_obo", go_basic_obo_path if go_basic_obo_path is not None and go_basic_obo_path.exists() else ""),
        ("split_manifest", split_manifest_path if split_manifest_path.exists() else ""),
        ("label_cache", label_cache_path if label_cache_path.exists() else ""),
        ("output_dir", output_dir),
        ("benchmark_dir", output_dir / "benchmark" if not args.skip_go_benchmarks else ""),
        ("validation_rows", len(validation_metrics)),
    ]
    if not validation_metrics.empty:
        best = validation_metrics.loc[validation_metrics["fmax"].idxmax()]
        summary_rows.extend(
            [
                ("best_validation_step", int(best["step"])),
                ("best_validation_fmax", f"{best['fmax']:.8f}"),
                ("best_validation_threshold", f"{best['threshold']:.8f}"),
                ("best_validation_precision", f"{best['precision']:.8f}"),
                ("best_validation_recall", f"{best['recall']:.8f}"),
            ]
        )
    if test_summary is not None:
        for key in ("fmax", "threshold", "precision", "recall", "records_evaluated"):
            if key in test_summary.index:
                summary_rows.append((f"test_{key}", test_summary.loc[key]))
    summary_rows.extend((f"figure_{name}", path) for name, path in written)
    write_key_value_tsv(output_dir / "figure_manifest.tsv", summary_rows)

    print(f"Wrote figures and tables to: {output_dir}")
    for name, path in written:
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
