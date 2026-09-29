#!/usr/bin/env python3
"""Generate a publication-ready evaluation threshold sweep figure."""

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
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "human_test_taxon_val_1919taxa_min_count_15000_bs256_mean_loss_3h"
DEFAULT_BUILD_ID = "all_labels_true_path_1919taxa_human_test_min_count_15000_v1"

GO_ROOT_LABELS = {
    "GO:0003674": "molecular function (MF)",
    "GO:0005575": "cellular component (CC)",
    "GO:0008150": "biological process (BP)",
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
        return " The three GO ontology root terms were excluded from all reported metrics."
    if len(labels) == 1:
        joined = labels[0]
    else:
        joined = ", ".join(labels[:-1]) + ", and " + labels[-1]
    return f" The ontology root terms {joined} were excluded from all reported metrics."


def read_threshold_curve(path: Path) -> list[dict[str, float]]:
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [
            {
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_summary(path: Path) -> dict[str, str]:
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["metric"]: row["value"] for row in reader}


def write_caption(
    path: Path,
    *,
    summary: dict[str, str],
    threshold_count: int,
    figure_number: int,
    highlight_operating_threshold: bool,
) -> None:
    records = int(summary["records_evaluated"])
    terms = int(summary["n_terms"])
    fmax = float(summary["fmax"])
    threshold = float(summary["threshold"])
    precision = float(summary["precision"])
    recall = float(summary["recall"])
    checkpoint_step = int(summary["checkpoint_step"])
    split_name = summary.get("split_name", "test").lower()
    record_label = "sequences" if is_mmseqs_test(summary) else "genes"
    if highlight_operating_threshold:
        operating_threshold = float(summary["operating_threshold"])
        operating_f1 = float(summary["f1_at_operating_threshold"])
        operating_precision = float(summary["precision_at_operating_threshold"])
        operating_recall = float(summary["recall_at_operating_threshold"])
        caption = (
            f"Figure {figure_number}. Evo2-GO performance on the held-out "
            "MMseqs-cluster test set across probability thresholds. The "
            f"validation-selected threshold of {operating_threshold:.2f} produced "
            f"gene-averaged F1 of {operating_f1:.3f}, precision of "
            f"{operating_precision:.3f}, and recall of {operating_recall:.3f} across "
            f"{records:,} sequences and {terms:,} retained non-root GO terms. The "
            "validation-selected threshold remained fixed for the primary test "
            "evaluation and is not a calibrated posterior probability; the three GO "
            "ontology root terms were excluded from all reported metrics."
        )
    elif split_name in {"val", "validation"}:
        caption = (
            f"Figure {figure_number}. Fixed-panel validation threshold sweep for "
            "Evo2-GO function prediction. Predicted GO-term probabilities from the "
            f"selected checkpoint (training step {checkpoint_step:,}) were evaluated "
            f"over {threshold_count:,} evenly spaced cutoffs from 0.01 to 0.99 on the "
            f"deterministic validation panel ({records:,} {record_label}; {terms:,} "
            "retained true-path GO terms). Precision and recall were computed per gene "
            "and averaged across genes, and F1 was calculated from those averages at "
            f"each threshold; validation Fmax was {fmax:.3f} at threshold "
            f"{threshold:.3f}, with precision {precision:.3f} and recall {recall:.3f}. "
            "The selected threshold was fixed before primary evaluation on the held-out "
            "MMseqs-cluster test set and is not a calibrated posterior probability."
            + root_exclusion_note(summary)
        )
    else:
        test_label = "MMseqs-cluster test set" if is_mmseqs_test(summary) else "human-test set"
        caption = (
            f"Figure {figure_number}. {test_label} threshold sweep for Evo2-GO function "
            "prediction. Predicted GO-term probabilities from the best checkpoint "
            f"(training step {checkpoint_step:,}) were thresholded over "
            f"{threshold_count:,} evenly spaced cutoffs from 0.01 to 0.99 and "
            f"evaluated on the held-out test partition ({records:,} {record_label}; "
            f"{terms:,} retained non-root true-path GO terms). "
            "Precision and recall were computed per gene and averaged across genes; "
            "F1 was then computed from the averaged precision and recall at each "
            f"threshold. The maximum gene-averaged F1 was {fmax:.3f} at threshold "
            f"{threshold:.3f}, with precision {precision:.3f} and recall "
            f"{recall:.3f}. Because threshold {threshold:.3f} was selected to maximize "
            "F1 using the test labels, this value is a descriptive (oracle) test-set "
            "Fmax rather than performance at a validation-selected operating threshold. "
            "The selected cutoff is also not a calibrated posterior-probability threshold."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--figure-number", type=int, default=1)
    parser.add_argument("--output-stem")
    parser.add_argument(
        "--highlight-operating-threshold",
        action="store_true",
        help=(
            "Highlight the prespecified operating threshold as the primary point and "
            "show test-set Fmax as a secondary statistic."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir or (
        PROJECT_ROOT / "outputs" / "builds" / args.build_id / "runs" / args.run_name / "test_eval"
    )
    output_dir = args.output_dir or PROJECT_ROOT / "figures" / args.run_name / "publication_ready"
    threshold_curve = eval_dir / f"{args.split_name}_threshold_curve.tsv"
    evaluation_summary = eval_dir / f"{args.split_name}_evaluation_summary.tsv"

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_threshold_curve(threshold_curve)
    summary = read_summary(evaluation_summary)

    best = max(rows, key=lambda row: row["f1"])
    fmax = float(summary["fmax"])
    threshold = float(summary["threshold"])
    precision = float(summary["precision"])
    recall = float(summary["recall"])
    records = int(summary["records_evaluated"])
    terms = int(summary["n_terms"])
    roots_excluded = summary.get("exclude_go_roots", "0").lower() in {"1", "true", "yes"}
    term_count_label = "non-root GO terms" if roots_excluded else "GO terms"
    if args.highlight_operating_threshold:
        required = (
            "operating_threshold",
            "f1_at_operating_threshold",
            "precision_at_operating_threshold",
            "recall_at_operating_threshold",
        )
        missing = [field for field in required if not summary.get(field)]
        if missing:
            raise ValueError(
                "Operating-threshold highlighting requires summary fields: "
                + ", ".join(missing)
            )
        display_threshold = float(summary["operating_threshold"])
        display_f1 = float(summary["f1_at_operating_threshold"])
        display_precision = float(summary["precision_at_operating_threshold"])
        display_recall = float(summary["recall_at_operating_threshold"])
    else:
        display_threshold = threshold
        display_f1 = fmax
        display_precision = precision
        display_recall = recall

    colors = {
        "F1": "#2f6fae",
        "Precision": "#7b3f98",
        "Recall": "#c65f16",
        "Ink": "#222222",
    }

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

    fig = plt.figure(figsize=(5.8, 2.85))
    grid = GridSpec(
        1,
        2,
        width_ratios=(4.25, 1.35),
        wspace=0.16,
        figure=fig,
    )
    ax = fig.add_subplot(grid[0, 0])
    side_ax = fig.add_subplot(grid[0, 1])
    x = [row["threshold"] for row in rows]
    ax.plot(x, [row["f1"] for row in rows], color=colors["F1"], linewidth=2.1, label="F1")
    ax.plot(
        x,
        [row["precision"] for row in rows],
        color=colors["Precision"],
        linewidth=1.7,
        label="Precision",
    )
    ax.plot(
        x,
        [row["recall"] for row in rows],
        color=colors["Recall"],
        linewidth=1.7,
        label="Recall",
    )
    ax.axvline(
        display_threshold,
        color=colors["Ink"],
        linewidth=0.9,
        linestyle=(0, (4, 3)),
    )
    ax.scatter(
        [display_threshold],
        [display_f1],
        s=30,
        color=colors["F1"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Gene-averaged score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", linestyle="-", linewidth=0.35, color="#d8d8d8", alpha=0.9)
    ax.tick_params(axis="both", width=0.8, length=3)

    side_ax.axis("off")
    side_ax.set_xlim(0.0, 1.0)
    side_ax.set_ylim(0.0, 1.0)
    side_ax.text(
        0.0,
        0.98,
        (
            "Validation panel"
            if summary.get("split_name", "test").lower() in {"val", "validation"}
            else ("MMseqs test" if is_mmseqs_test(summary) else "Human test")
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="semibold",
        color=colors["Ink"],
    )
    side_ax.text(
        0.0,
        0.89,
        f"{records:,} {'sequences' if is_mmseqs_test(summary) else 'genes'}\n"
        f"{terms:,} {term_count_label}",
        ha="left",
        va="top",
        fontsize=7.5,
        linespacing=1.25,
        color="#555555",
    )

    legend_y = 0.70
    for offset, label in enumerate(("F1", "Precision", "Recall")):
        y = legend_y - 0.085 * offset
        side_ax.plot(
            [0.0, 0.24],
            [y, y],
            color=colors[label],
            linewidth=2.0,
            solid_capstyle="round",
        )
        side_ax.text(0.30, y, label, ha="left", va="center", fontsize=7.8, color=colors["Ink"])

    side_ax.plot(
        [0.0, 0.24],
        [0.39, 0.39],
        color=colors["Ink"],
        linewidth=0.9,
        linestyle=(0, (4, 3)),
    )
    side_ax.text(
        0.30,
        0.39,
        "Validation-selected" if args.highlight_operating_threshold else "Best threshold",
        ha="left",
        va="center",
        fontsize=7.8,
        color=colors["Ink"],
    )
    metric_rows = (
        ("F1", f"{display_f1:.3f}"),
        ("Threshold", f"{display_threshold:.3f}"),
        ("Precision", f"{display_precision:.3f}"),
        ("Recall", f"{display_recall:.3f}"),
    ) if args.highlight_operating_threshold else (
        ("Fmax", f"{fmax:.3f}"),
        ("Threshold", f"{threshold:.3f}"),
        ("Precision", f"{precision:.3f}"),
        ("Recall", f"{recall:.3f}"),
    )
    for offset, (label, value) in enumerate(metric_rows):
        y = 0.28 - 0.075 * offset
        side_ax.text(0.0, y, label, ha="left", va="top", fontsize=7.5, color=colors["Ink"])
        side_ax.text(0.95, y, value, ha="right", va="top", fontsize=7.5, color=colors["Ink"])

    if args.output_stem:
        stem_name = args.output_stem
    elif summary.get("split_name", "test").lower() in {"val", "validation"}:
        stem_name = "figure_01_validation_threshold_sweep"
    else:
        stem_name = (
            "figure_01_mmseqs_test_threshold_sweep"
            if is_mmseqs_test(summary)
            else "figure_01_human_test_threshold_sweep"
        )
    stem = output_dir / stem_name
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.96)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    threshold_count = int(summary.get("threshold_steps") or len(rows))
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        summary=summary,
        threshold_count=threshold_count,
        figure_number=args.figure_number,
        highlight_operating_threshold=args.highlight_operating_threshold,
    )


if __name__ == "__main__":
    main()
