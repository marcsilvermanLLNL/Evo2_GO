#!/usr/bin/env python3
"""Plot whole-dataset GO label composition and labels per sequence."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import textwrap
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evo2_go_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "human_test_taxon_val_1919taxa_min_count_15000_bs256_mean_loss_3h"
DEFAULT_BUILD_ID = "all_labels_true_path_1919taxa_human_test_min_count_15000_v1"

GO_ROOT_IDS = {
    "GO:0003674",  # molecular function
    "GO:0005575",  # cellular component
    "GO:0008150",  # biological process
}

INK = "#222222"
MUTED = "#666666"
GRID = "#d6d6d6"
BLUE = "#2f6fae"
ORANGE = "#c45a14"
GREEN = "#4f8f46"
GRAY = "#8a8a8a"

NAMESPACE_ORDER = [
    "biological_process",
    "cellular_component",
    "molecular_function",
    "other",
]
NAMESPACE_LABELS = {
    "biological_process": "Biological process",
    "cellular_component": "Cellular component",
    "molecular_function": "Molecular function",
    "other": "Other / missing",
}
NAMESPACE_SHORT = {
    "biological_process": "BP",
    "cellular_component": "CC",
    "molecular_function": "MF",
    "other": "Other",
}
NAMESPACE_COLORS = {
    "biological_process": BLUE,
    "cellular_component": ORANGE,
    "molecular_function": GREEN,
    "other": GRAY,
}


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


def count_formatter(value: float, _position: int) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


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


def load_namespace_summary(
    vocabulary_path: Path,
    *,
    exclude_go_roots: bool,
) -> pd.DataFrame:
    vocabulary = pd.read_csv(vocabulary_path, sep="\t")
    if exclude_go_roots:
        vocabulary = vocabulary.loc[~vocabulary["GO_ID"].isin(GO_ROOT_IDS)].copy()
    vocabulary["trainable_gene_count"] = pd.to_numeric(
        vocabulary["trainable_gene_count"],
        errors="coerce",
    ).fillna(0)
    namespace = vocabulary["namespace"].where(vocabulary["namespace"].notna(), "other")
    vocabulary["namespace"] = namespace.astype(str).str.strip()
    vocabulary.loc[~vocabulary["namespace"].isin(NAMESPACE_ORDER), "namespace"] = "other"

    summary = (
        vocabulary.groupby("namespace", dropna=False)
        .agg(
            retained_labels=("trainable_gene_count", "sum"),
            vocabulary_terms=("GO_ID", "count"),
        )
        .reindex(NAMESPACE_ORDER, fill_value=0)
        .reset_index()
    )
    total = float(summary["retained_labels"].sum())
    summary["fraction"] = np.where(total > 0, summary["retained_labels"] / total, 0.0)
    summary["namespace_label"] = summary["namespace"].map(NAMESPACE_LABELS)
    summary["namespace_short"] = summary["namespace"].map(NAMESPACE_SHORT)
    return summary


def go_root_indices(vocabulary_path: Path) -> set[int]:
    vocabulary = pd.read_csv(vocabulary_path, sep="\t", usecols=["go_index", "GO_ID"])
    roots = vocabulary.loc[vocabulary["GO_ID"].isin(GO_ROOT_IDS), "go_index"]
    indices = {int(value) for value in roots}
    if len(indices) != len(GO_ROOT_IDS):
        found_ids = set(vocabulary.loc[vocabulary["GO_ID"].isin(GO_ROOT_IDS), "GO_ID"])
        missing = ", ".join(sorted(GO_ROOT_IDS - found_ids))
        raise ValueError(f"GO vocabulary is missing expected ontology root terms: {missing}")
    return indices


def count_labels_per_sequence(
    label_index_path: Path,
    *,
    excluded_go_indices: set[int] | None = None,
    progress_interval: int = 5_000_000,
) -> dict[str, object]:
    counts: Counter[int] = Counter()
    rows = 0
    total_labels = 0
    excluded_labels = 0
    excluded_tokens = tuple("," + str(index) + "," for index in sorted(excluded_go_indices or set()))

    with gzip.open(label_index_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "go_indices" not in (reader.fieldnames or []):
            raise ValueError(f"{label_index_path} does not contain a go_indices column.")
        for row in reader:
            value = row.get("go_indices", "")
            label_count = 0 if value == "" else value.count(",") + 1
            if value and excluded_tokens:
                padded_value = "," + value + ","
                removed = sum(token in padded_value for token in excluded_tokens)
                label_count -= removed
                excluded_labels += removed
            counts[label_count] += 1
            rows += 1
            total_labels += label_count
            if progress_interval > 0 and rows % progress_interval == 0:
                print(f"Read {rows:,} label-index rows...", flush=True)

    if rows == 0:
        raise ValueError(f"{label_index_path} did not contain any label rows.")

    return {
        "counts": counts,
        "sequences": rows,
        "labels": total_labels,
        "excluded_labels": excluded_labels,
    }


def percentile_from_counts(counts: Counter[int], quantile: float) -> int:
    if not counts:
        return 0
    total = sum(counts.values())
    threshold = quantile * total
    cumulative = 0
    for label_count in sorted(counts):
        cumulative += counts[label_count]
        if cumulative >= threshold:
            return int(label_count)
    return int(max(counts))


def distribution_table(counts: Counter[int]) -> pd.DataFrame:
    total = float(sum(counts.values()))
    rows = []
    cumulative = 0
    for label_count in sorted(counts):
        sequences = counts[label_count]
        cumulative += sequences
        rows.append(
            {
                "labels_per_sequence": label_count,
                "sequences": sequences,
                "fraction": sequences / total,
                "cumulative_fraction": cumulative / total,
            }
        )
    return pd.DataFrame(rows)


def label_count_stats(counts: Counter[int]) -> dict[str, float | int]:
    total_sequences = sum(counts.values())
    total_labels = sum(label_count * sequences for label_count, sequences in counts.items())
    mean = total_labels / total_sequences
    variance = (
        sum(sequences * (label_count - mean) ** 2 for label_count, sequences in counts.items())
        / total_sequences
    )
    return {
        "sequences": total_sequences,
        "retained_labels": total_labels,
        "mean": mean,
        "sd": float(np.sqrt(variance)),
        "min": min(counts),
        "q25": percentile_from_counts(counts, 0.25),
        "median": percentile_from_counts(counts, 0.50),
        "q75": percentile_from_counts(counts, 0.75),
        "p90": percentile_from_counts(counts, 0.90),
        "p95": percentile_from_counts(counts, 0.95),
        "p99": percentile_from_counts(counts, 0.99),
        "p995": percentile_from_counts(counts, 0.995),
        "max": max(counts),
    }


def plot_namespace_allocation(
    ax: plt.Axes,
    namespace_summary: pd.DataFrame,
    *,
    exclude_go_roots: bool,
) -> None:
    left = 0.0
    total = float(namespace_summary["retained_labels"].sum())
    for row in namespace_summary.to_dict(orient="records"):
        width = float(row["retained_labels"])
        if width <= 0:
            continue
        namespace = row["namespace"]
        ax.barh(
            [0],
            [width],
            left=[left],
            height=0.42,
            color=NAMESPACE_COLORS[namespace],
            edgecolor="white",
            linewidth=0.8,
        )
        fraction = float(row["fraction"])
        if fraction >= 0.045:
            ax.text(
                left + width / 2,
                0,
                f"{row['namespace_short']}\n{fraction:.1%}",
                ha="center",
                va="center",
                color="white",
                fontweight="semibold",
                linespacing=1.0,
            )
        left += width

    ax.set_yticks([])
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlim(0, total)
    ax.set_xlabel(
        "Retained non-root true-path labels"
        if exclude_go_roots
        else "Retained true-path labels"
    )
    ax.xaxis.set_major_formatter(FuncFormatter(count_formatter))
    ax.set_title("A  Aggregate label allocation", loc="left", fontweight="semibold")
    ax.grid(axis="x", color=GRID, linewidth=0.35, alpha=0.85)
    ax.text(
        0.02,
        0.82,
        f"{total / 1_000_000:.1f}M retained gene-GO labels",
        ha="left",
        va="bottom",
        color=MUTED,
        transform=ax.transAxes,
    )


def plot_label_count_distribution(
    ax: plt.Axes,
    counts: Counter[int],
    stats: dict[str, float | int],
    *,
    histogram_cap: int | None,
    exclude_go_roots: bool,
) -> int:
    if histogram_cap is None:
        histogram_cap = int(min(max(int(stats["p995"]), 25), 100, int(stats["max"])))
    histogram_cap = max(histogram_cap, int(stats["min"]))

    total = float(stats["sequences"])
    xs = np.arange(int(stats["min"]), histogram_cap + 1)
    ys = np.array([counts.get(int(x), 0) / total for x in xs], dtype=float)
    ax.bar(xs, ys, width=0.82, color=BLUE, edgecolor="white", linewidth=0.25)

    tail_sequences = sum(sequences for label_count, sequences in counts.items() if label_count > histogram_cap)
    if tail_sequences:
        ax.bar(
            [histogram_cap + 1],
            [tail_sequences / total],
            width=0.82,
            color=GRAY,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.set_xlim(int(stats["min"]) - 0.8, histogram_cap + 2.2)
        ax.set_xticks([int(stats["min"]), int(stats["median"]), int(stats["p95"]), histogram_cap + 1])
        ax.set_xticklabels(
            [
                str(int(stats["min"])),
                str(int(stats["median"])),
                str(int(stats["p95"])),
                f">{histogram_cap}",
            ]
        )
    else:
        ax.set_xlim(int(stats["min"]) - 0.8, histogram_cap + 0.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))

    ax.set_xlabel(
        "Retained non-root GO labels per sequence"
        if exclude_go_roots
        else "Retained GO labels per sequence"
    )
    ax.set_ylabel("Sequences")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("B  Label density per sequence", loc="left", fontweight="semibold")
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)
    ax.text(
        0.98,
        0.95,
        (
            f"n={int(stats['sequences']):,}\n"
            f"median={int(stats['median'])}\n"
            f"IQR={int(stats['q25'])}-{int(stats['q75'])}\n"
            f"p95={int(stats['p95'])}"
        ),
        ha="right",
        va="top",
        color=INK,
        transform=ax.transAxes,
        linespacing=1.25,
    )
    return histogram_cap


def plot_figure(
    namespace_summary: pd.DataFrame,
    counts: Counter[int],
    stats: dict[str, float | int],
    output_dir: Path,
    *,
    dpi: int,
    histogram_cap: int | None,
    dataset_label: str,
    exclude_go_roots: bool,
    excluded_root_labels: int,
) -> Path:
    stem = output_dir / "figure_10_dataset_label_composition"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.45, 3.05),
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.30},
    )
    plot_namespace_allocation(
        axes[0],
        namespace_summary,
        exclude_go_roots=exclude_go_roots,
    )
    used_cap = plot_label_count_distribution(
        axes[1],
        counts,
        stats,
        histogram_cap=histogram_cap,
        exclude_go_roots=exclude_go_roots,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.90)
    save_figure(fig, stem, dpi=dpi)

    namespace_records = {
        row["namespace_short"]: row
        for row in namespace_summary.to_dict(orient="records")
    }
    bp_fraction = float(namespace_records["BP"]["fraction"])
    cc_fraction = float(namespace_records["CC"]["fraction"])
    mf_fraction = float(namespace_records["MF"]["fraction"])
    tail_text = ""
    if int(stats["max"]) > used_cap:
        tail_fraction = sum(
            sequences for label_count, sequences in counts.items() if label_count > used_cap
        ) / float(stats["sequences"])
        tail_text = f" Counts above {used_cap} labels are shown as a single overflow bin ({tail_fraction:.2%} of sequences)."
    root_note = ""
    if exclude_go_roots:
        root_note = (
            " The ontology root terms molecular function (GO:0003674), cellular "
            "component (GO:0005575), and biological process (GO:0008150) were "
            f"excluded from both panels; this removed {excluded_root_labels:,} "
            "root-label occurrences before the per-sequence distribution was computed."
        )
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 10. Whole-dataset retained GO label composition. Panel A shows "
            "the aggregate allocation of retained true-path gene-GO labels across "
            f"GO namespaces after vocabulary filtering, summed over {dataset_label}. "
            "Panel B shows the distribution of retained "
            "true-path GO labels per sequence in label_index.tsv.gz, including "
            "explicit terms and inferred is_a/part_of ancestors. Namespace label "
            f"allocation was BP={bp_fraction:.1%}, CC={cc_fraction:.1%}, and "
            f"MF={mf_fraction:.1%}. The label index contains "
            f"{int(stats['sequences']):,} sequences and "
            f"{int(stats['retained_labels']):,} retained labels; the median was "
            f"{int(stats['median'])} labels per sequence, the interquartile range "
            f"was {int(stats['q25'])}-{int(stats['q75'])}, and the 95th percentile "
            f"was {int(stats['p95'])}.{tail_text}{root_note}"
        ),
    )
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--go-vocabulary", type=Path)
    parser.add_argument("--label-index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset-label",
        default="the selected dataset",
        help="Human-readable dataset description used in the caption.",
    )
    parser.add_argument(
        "--exclude-go-roots",
        action="store_true",
        help=(
            "Exclude GO:0003674, GO:0005575, and GO:0008150 from namespace "
            "totals and per-sequence label counts."
        ),
    )
    parser.add_argument(
        "--histogram-cap",
        type=int,
        help=(
            "Largest exact label count to show in panel B. Higher counts are "
            "pooled into an overflow bin. Defaults to min(p99.5, 100)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dir = PROJECT_ROOT / "outputs" / "builds" / args.build_id
    vocabulary_path = args.go_vocabulary or build_dir / "adjacency" / "go_vocabulary.tsv"
    label_index_path = args.label_index or build_dir / "data" / "labels" / "label_index.tsv.gz"
    output_dir = args.output_dir or PROJECT_ROOT / "figures" / args.run_name / "publication_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_publication_style()
    namespace_summary = load_namespace_summary(
        vocabulary_path,
        exclude_go_roots=args.exclude_go_roots,
    )
    excluded_indices = go_root_indices(vocabulary_path) if args.exclude_go_roots else set()
    label_counts = count_labels_per_sequence(
        label_index_path,
        excluded_go_indices=excluded_indices,
    )
    counts = label_counts["counts"]
    stats = label_count_stats(counts)

    stem = output_dir / "figure_10_dataset_label_composition"
    namespace_summary.to_csv(
        stem.with_name(stem.name + "_namespace_summary.tsv"),
        sep="\t",
        index=False,
    )
    distribution_table(counts).to_csv(
        stem.with_name(stem.name + "_label_count_distribution.tsv"),
        sep="\t",
        index=False,
    )
    write_key_value_tsv(
        stem.with_name(stem.name + "_summary.tsv"),
        [
            ("sequences", int(stats["sequences"])),
            ("retained_labels_from_label_index", int(stats["retained_labels"])),
            ("exclude_go_roots", int(args.exclude_go_roots)),
            ("excluded_go_root_ids", ",".join(sorted(GO_ROOT_IDS)) if args.exclude_go_roots else ""),
            ("excluded_go_root_indices", ",".join(str(value) for value in sorted(excluded_indices))),
            ("excluded_go_root_label_occurrences", int(label_counts["excluded_labels"])),
            (
                "retained_labels_from_go_vocabulary",
                int(namespace_summary["retained_labels"].sum()),
            ),
            ("labels_per_sequence_mean", f"{float(stats['mean']):.6f}"),
            ("labels_per_sequence_sd", f"{float(stats['sd']):.6f}"),
            ("labels_per_sequence_min", int(stats["min"])),
            ("labels_per_sequence_q25", int(stats["q25"])),
            ("labels_per_sequence_median", int(stats["median"])),
            ("labels_per_sequence_q75", int(stats["q75"])),
            ("labels_per_sequence_p90", int(stats["p90"])),
            ("labels_per_sequence_p95", int(stats["p95"])),
            ("labels_per_sequence_p99", int(stats["p99"])),
            ("labels_per_sequence_p995", int(stats["p995"])),
            ("labels_per_sequence_max", int(stats["max"])),
        ],
    )
    plot_stem = plot_figure(
        namespace_summary,
        counts,
        stats,
        output_dir,
        dpi=args.dpi,
        histogram_cap=args.histogram_cap,
        dataset_label=args.dataset_label,
        exclude_go_roots=args.exclude_go_roots,
        excluded_root_labels=int(label_counts["excluded_labels"]),
    )

    print(f"Wrote dataset label composition figure to: {plot_stem.with_suffix('.png')}")
    print(f"Wrote dataset label composition PDF to: {plot_stem.with_suffix('.pdf')}")
    print(f"Wrote exact label-count distribution to: {plot_stem.with_name(plot_stem.name + '_label_count_distribution.tsv')}")


if __name__ == "__main__":
    main()
