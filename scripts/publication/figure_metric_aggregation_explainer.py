#!/usr/bin/env python3
"""Explain why gene-level CAFA Fmax can exceed median GO-term F1/Fmax."""

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
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "mmseqs_linclust_full_85_5_10_min_count_15000_bs256_mean_loss_fixed_val50k_v2"
BUILD_ID = "all_labels_true_path_full_cleaned_min_count_15000_v1"
FIGURE_FOLDER = PROJECT_ROOT / "figures" / RUN_NAME / "publication_ready"
DEFAULT_TEST_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "builds"
    / BUILD_ID
    / "runs"
    / RUN_NAME
    / "test_eval_cpu"
    / "test_evaluation_summary.tsv"
)
DEFAULT_TERM_SUMMARY = FIGURE_FOLDER / "figure_02_go_term_frequency_performance_summary.tsv"

INK = "#222222"
MUTED = "#666666"
LIGHT = "#F5F6F7"
GRID = "#D7DCE1"
BLUE = "#2F6FAE"
BLUE_LIGHT = "#EAF2F8"
GREEN = "#4F8F46"
GREEN_LIGHT = "#EDF5EA"
ORANGE = "#C45A14"
ORANGE_LIGHT = "#FCEFE6"
PURPLE = "#7B3F98"
PURPLE_LIGHT = "#F4ECF7"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_key_value_tsv(path: Path) -> dict[str, str]:
    with path.open("rt", newline="") as handle:
        return {
            row["metric"]: row["value"]
            for row in csv.DictReader(handle, delimiter="\t")
        }


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.4,
    connectionstyle: str = "arc3",
    mutation_scale: float = 11,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def rounded_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edge: str,
    fill: str,
    label: str,
    title: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.11",
            linewidth=1.25,
            edgecolor=edge,
            facecolor=fill,
            zorder=1,
        )
    )
    ax.text(
        x + 0.28,
        y + height - 0.34,
        label,
        color=edge,
        fontsize=13,
        weight="bold",
        ha="left",
        va="center",
        zorder=8,
    )
    ax.text(
        x + 0.72,
        y + height - 0.34,
        title,
        color=INK,
        fontsize=11.4,
        weight="bold",
        ha="left",
        va="center",
        zorder=8,
    )


def illustrative_states() -> np.ndarray:
    """Return schematic TN/TP/FN/FP states for genes by GO terms.

    Codes are 0=unlabeled/negative, 1=true positive, 2=false negative,
    and 3=false positive. The pattern is explanatory, not sampled data.
    """

    states = np.zeros((8, 12), dtype=np.int8)
    true_terms = [
        (0, 1, 2, 4),
        (0, 1, 3, 5),
        (0, 2, 3, 6),
        (0, 1, 2, 3, 7),
        (0, 1, 8),
        (0, 2, 3, 9),
        (0, 1, 2, 10),
        (0, 1, 3, 11),
    ]
    predicted_terms = [
        (0, 1, 2, 3),
        (0, 1, 3),
        (0, 2),
        (0, 1, 2, 3),
        (0, 1),
        (0, 2, 3, 8),
        (0, 1, 2),
        (0, 1, 3),
    ]
    for gene_index, (true_set, predicted_set) in enumerate(
        zip(true_terms, predicted_terms, strict=True)
    ):
        true = set(true_set)
        predicted = set(predicted_set)
        for term_index in true & predicted:
            states[gene_index, term_index] = 1
        for term_index in true - predicted:
            states[gene_index, term_index] = 2
        for term_index in predicted - true:
            states[gene_index, term_index] = 3
    return states


def draw_matrix_panel(ax: plt.Axes) -> None:
    x, y, width, height = 0.48, 1.62, 8.25, 6.14
    rounded_panel(
        ax,
        x,
        y,
        width,
        height,
        edge="#818A93",
        fill="white",
        label="A",
        title="One prediction matrix, two evaluation directions",
    )
    ax.text(
        x + width - 0.28,
        y + height - 0.34,
        "schematic",
        color=MUTED,
        fontsize=7.2,
        style="italic",
        ha="right",
        va="center",
    )

    states = illustrative_states()
    rows, columns = states.shape
    x0, y0 = 1.58, 2.42
    cell_w, cell_h = 0.49, 0.49
    color_for_state = {
        0: "white",
        1: BLUE,
        2: ORANGE,
        3: "#CDB5D8",
    }

    # Column-group backgrounds distinguish frequent broad terms from the long tail.
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 0.12, y0 - 0.13),
            4 * cell_w + 0.20,
            rows * cell_h + 0.26,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0,
            facecolor=BLUE_LIGHT,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 4 * cell_w - 0.02, y0 - 0.13),
            8 * cell_w + 0.14,
            rows * cell_h + 0.26,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0,
            facecolor=LIGHT,
            zorder=2,
        )
    )

    for row in range(rows):
        for column in range(columns):
            state = int(states[row, column])
            cell_x = x0 + column * cell_w
            cell_y = y0 + (rows - 1 - row) * cell_h
            ax.add_patch(
                Rectangle(
                    (cell_x, cell_y),
                    cell_w * 0.82,
                    cell_h * 0.82,
                    facecolor=color_for_state[state],
                    edgecolor=GRID if state == 0 else "white",
                    linewidth=0.65,
                    zorder=4,
                )
            )

    # Highlight one gene (row) and one term (column) to expose the aggregation axes.
    highlighted_row = 3
    row_y = y0 + (rows - 1 - highlighted_row) * cell_h
    ax.add_patch(
        Rectangle(
            (x0 - 0.07, row_y - 0.07),
            columns * cell_w - 0.03,
            cell_h * 0.82 + 0.14,
            facecolor="none",
            edgecolor=BLUE,
            linewidth=2.0,
            zorder=6,
        )
    )
    highlighted_column = 8
    column_x = x0 + highlighted_column * cell_w
    ax.add_patch(
        Rectangle(
            (column_x - 0.07, y0 - 0.07),
            cell_w * 0.82 + 0.14,
            rows * cell_h - 0.03,
            facecolor="none",
            edgecolor=GREEN,
            linewidth=2.0,
            zorder=7,
        )
    )

    ax.text(
        x0 + 2 * cell_w - 0.05,
        y0 + rows * cell_h + 0.43,
        "Common / broad terms",
        color=BLUE,
        fontsize=8.1,
        weight="bold",
        ha="center",
    )
    ax.text(
        x0 + 8 * cell_w,
        y0 + rows * cell_h + 0.43,
        "Specific / rare terms",
        color=MUTED,
        fontsize=8.1,
        weight="bold",
        ha="center",
    )
    ax.text(
        x0 + columns * cell_w / 2,
        y0 + rows * cell_h + 0.79,
        "GO terms",
        color=INK,
        fontsize=9.3,
        weight="bold",
        ha="center",
    )
    ax.text(
        x0 - 0.78,
        y0 + rows * cell_h / 2,
        "Genes",
        color=INK,
        fontsize=9.3,
        weight="bold",
        rotation=90,
        ha="center",
        va="center",
    )

    arrow(
        ax,
        (x0 - 0.55, row_y + cell_h * 0.41),
        (x0 - 0.10, row_y + cell_h * 0.41),
        color=BLUE,
    )
    ax.text(
        x0 - 0.60,
        row_y + cell_h * 0.41,
        "one label set",
        color=BLUE,
        fontsize=7.2,
        weight="bold",
        ha="right",
        va="center",
    )
    arrow(
        ax,
        (column_x + cell_w * 0.41, y0 - 0.54),
        (column_x + cell_w * 0.41, y0 - 0.10),
        color=GREEN,
    )
    ax.text(
        column_x + cell_w * 0.41,
        y0 - 0.75,
        "one binary task",
        color=GREEN,
        fontsize=7.2,
        weight="bold",
        ha="center",
        va="center",
    )

    # Compact legend.
    legend_y = 1.95
    legend_items = (
        (BLUE, "correct label"),
        (ORANGE, "missed label"),
        ("#CDB5D8", "extra prediction"),
        ("white", "negative"),
    )
    legend_x = 1.38
    for color, label in legend_items:
        ax.add_patch(
            Rectangle(
                (legend_x, legend_y),
                0.19,
                0.19,
                facecolor=color,
                edgecolor=GRID if color == "white" else "white",
                linewidth=0.7,
                zorder=7,
            )
        )
        ax.text(
            legend_x + 0.27,
            legend_y + 0.095,
            label,
            color=MUTED,
            fontsize=6.9,
            ha="left",
            va="center",
        )
        legend_x += 1.55


def draw_gene_metric_panel(
    ax: plt.Axes,
    *,
    fmax: float,
    threshold: float,
    precision: float,
    recall: float,
) -> None:
    x, y, width, height = 9.06, 4.74, 6.48, 3.02
    rounded_panel(
        ax,
        x,
        y,
        width,
        height,
        edge=BLUE,
        fill=BLUE_LIGHT,
        label="B",
        title="Gene-level CAFA Fmax",
    )
    ax.text(
        x + 0.43,
        y + 1.92,
        "For each gene",
        color=BLUE,
        fontsize=8.2,
        weight="bold",
        ha="left",
    )
    ax.text(
        x + 0.43,
        y + 1.51,
        r"compute $P_i(t)$ and $R_i(t)$ over its GO-label set",
        color=INK,
        fontsize=8.3,
        ha="left",
    )
    ax.text(
        x + 0.43,
        y + 1.08,
        r"average across genes  $\rightarrow$  $\bar{P}(t)$ and $\bar{R}(t)$",
        color=INK,
        fontsize=8.3,
        ha="left",
    )
    ax.text(
        x + 0.43,
        y + 0.60,
        r"$F(t)=2\bar{P}(t)\bar{R}(t)/(\bar{P}(t)+\bar{R}(t))$",
        color=MUTED,
        fontsize=8.0,
        ha="left",
    )

    ax.add_patch(
        FancyBboxPatch(
            (x + 4.43, y + 0.48),
            1.67,
            1.70,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0,
            facecolor=BLUE,
            zorder=4,
        )
    )
    ax.text(
        x + 5.265,
        y + 1.78,
        "max over one\nglobal threshold",
        color="#DCECF8",
        fontsize=7.1,
        weight="bold",
        ha="center",
        va="center",
        linespacing=1.2,
        zorder=6,
    )
    ax.text(
        x + 5.265,
        y + 1.17,
        f"{fmax:.3f}",
        color="white",
        fontsize=20,
        weight="bold",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        x + 5.265,
        y + 0.71,
        f"P {precision:.3f}  •  R {recall:.3f}\nt = {threshold:.3f}",
        color="white",
        fontsize=6.8,
        ha="center",
        va="center",
        linespacing=1.22,
        zorder=6,
    )


def draw_term_metric_panel(
    ax: plt.Axes,
    *,
    terms: int,
    median_global_f1: float,
    median_term_fmax: float,
) -> None:
    x, y, width, height = 9.06, 1.62, 6.48, 2.78
    rounded_panel(
        ax,
        x,
        y,
        width,
        height,
        edge=GREEN,
        fill=GREEN_LIGHT,
        label="C",
        title="GO-term median",
    )
    ax.text(
        x + width - 0.28,
        y + height - 0.34,
        f"{terms:,} TERMS",
        color=GREEN,
        fontsize=7.2,
        weight="bold",
        ha="right",
        va="center",
    )
    ax.text(
        x + 0.43,
        y + 1.70,
        "For each GO term",
        color=GREEN,
        fontsize=8.2,
        weight="bold",
        ha="left",
    )
    ax.text(
        x + 0.43,
        y + 1.30,
        r"count $TP_j$, $FP_j$, and $FN_j$ across genes",
        color=INK,
        fontsize=8.3,
        ha="left",
    )
    ax.text(
        x + 0.43,
        y + 0.88,
        r"compute $F1_j$  $\rightarrow$  median across terms",
        color=INK,
        fontsize=8.3,
        ha="left",
    )
    ax.text(
        x + 0.43,
        y + 0.48,
        "Each term has equal weight—\nrare and common alike",
        color=MUTED,
        fontsize=7.2,
        ha="left",
        va="center",
        linespacing=1.25,
    )

    card_width = 1.35
    for card_x, value, top, bottom, fill in (
        (
            x + 3.55,
            median_global_f1,
            "one global t",
            "median F1",
            GREEN,
        ),
        (
            x + 5.02,
            median_term_fmax,
            "best t / term",
            "median Fmax",
            PURPLE,
        ),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (card_x, y + 0.43),
                card_width,
                1.49,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                linewidth=0,
                facecolor=fill,
                zorder=4,
            )
        )
        ax.text(
            card_x + card_width / 2,
            y + 1.66,
            top,
            color="#E8F2E5" if fill == GREEN else "#EEE5F2",
            fontsize=6.7,
            weight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            card_x + card_width / 2,
            y + 1.14,
            f"{value:.3f}",
            color="white",
            fontsize=18,
            weight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            card_x + card_width / 2,
            y + 0.68,
            bottom,
            color="white",
            fontsize=6.8,
            ha="center",
            va="center",
            zorder=6,
        )


def draw_takeaway(ax: plt.Axes) -> None:
    x, y, width, height = 0.48, 0.42, 15.06, 0.86
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=1.1,
            edgecolor=ORANGE,
            facecolor=ORANGE_LIGHT,
            zorder=1,
        )
    )
    ax.text(
        x + 0.35,
        y + height / 2,
        "WHY THE GAP?",
        color=ORANGE,
        fontsize=8.0,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        x + 2.02,
        y + height / 2,
        "Common true-path ancestors improve many gene label sets",
        color=INK,
        fontsize=8.0,
        ha="left",
        va="center",
    )
    ax.text(
        x + 7.03,
        y + height / 2,
        "+",
        color=ORANGE,
        fontsize=13,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        x + 7.35,
        y + height / 2,
        "Numerous rare terms remain difficult",
        color=INK,
        fontsize=8.0,
        ha="left",
        va="center",
    )
    ax.text(
        x + 11.03,
        y + height / 2,
        "→",
        color=ORANGE,
        fontsize=13,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        x + 11.38,
        y + height / 2,
        "stronger gene-level than median term-level score",
        color=INK,
        fontsize=8.0,
        weight="bold",
        ha="left",
        va="center",
    )


def make_figure(
    *,
    fmax: float,
    threshold: float,
    precision: float,
    recall: float,
    terms: int,
    median_global_f1: float,
    median_term_fmax: float,
) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.48,
        8.62,
        "Why gene-level CAFA Fmax exceeds median GO-term Fmax",
        color=INK,
        fontsize=19.5,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.50,
        8.25,
        "The metrics summarize the same predictions along different axes and weight common versus rare labels differently",
        color=MUTED,
        fontsize=10.0,
        ha="left",
        va="center",
    )

    draw_matrix_panel(ax)
    draw_gene_metric_panel(
        ax,
        fmax=fmax,
        threshold=threshold,
        precision=precision,
        recall=recall,
    )
    draw_term_metric_panel(
        ax,
        terms=terms,
        median_global_f1=median_global_f1,
        median_term_fmax=median_term_fmax,
    )
    draw_takeaway(ax)
    return fig


def write_caption(
    path: Path,
    *,
    fmax: float,
    threshold: float,
    precision: float,
    recall: float,
    terms: int,
    median_global_f1: float,
    median_term_fmax: float,
) -> None:
    caption = (
        "Figure. Gene-level and GO-term-level metrics summarize the same multilabel "
        "prediction matrix along different axes. Panel A is a schematic illustration, "
        "not a sample of test observations. Gene-level CAFA evaluation treats each row as "
        "one predicted label set: precision and recall are calculated for each gene, "
        "averaged across genes, combined into F1, and maximized over one global threshold. "
        f"On the MMseqs-cluster test set, this produced Fmax={fmax:.3f} at threshold "
        f"{threshold:.3f} (precision={precision:.3f}, recall={recall:.3f}). GO-term-level "
        "evaluation treats each column as a binary classification task across all genes and "
        f"then takes the median across {terms:,} terms. The median term F1 was "
        f"{median_global_f1:.3f} at the gene-optimized global threshold and the median "
        f"per-term optimal Fmax was {median_term_fmax:.3f}. True-path propagation supplies "
        "many genes with broad, frequent ancestor terms that the model predicts relatively "
        "well, improving row-wise label-set overlap. In contrast, term-wise aggregation "
        "weights every GO term equally; numerous rare or specific terms with few correct "
        "predictions therefore lower the median. The gene-level and term-level results are "
        "complementary rather than contradictory. The reported test threshold sweep is "
        "descriptive; an operational threshold should be selected using validation data."
    )
    path.write_text(
        textwrap.fill(caption, width=92, break_long_words=False, break_on_hyphens=False)
        + "\n",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    *,
    fmax: float,
    threshold: float,
    precision: float,
    recall: float,
    terms: int,
    median_global_f1: float,
    median_term_fmax: float,
) -> None:
    rows = (
        ("gene_averaged_fmax", fmax),
        ("gene_averaged_fmax_threshold", threshold),
        ("gene_averaged_precision_at_fmax", precision),
        ("gene_averaged_recall_at_fmax", recall),
        ("terms_with_test_support", terms),
        ("median_term_f1_at_global_threshold", median_global_f1),
        ("median_per_term_optimal_fmax", median_term_fmax),
        ("matrix_is_schematic", 1),
    )
    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-summary", type=Path, default=DEFAULT_TEST_SUMMARY)
    parser.add_argument("--term-summary", type=Path, default=DEFAULT_TERM_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_FOLDER)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_summary = read_key_value_tsv(args.test_summary.resolve())
    term_summary = read_key_value_tsv(args.term_summary.resolve())

    values = {
        "fmax": float(test_summary["fmax"]),
        "threshold": float(test_summary["threshold"]),
        "precision": float(test_summary["precision"]),
        "recall": float(test_summary["recall"]),
        "terms": int(term_summary["terms_with_test_support"]),
        "median_global_f1": float(term_summary["median_f1_at_global_threshold"]),
        "median_term_fmax": float(term_summary["median_per_term_fmax"]),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_metric_aggregation_explainer"

    fig = make_figure(**values)
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_caption(stem.with_name(stem.name + "_caption").with_suffix(".txt"), **values)
    write_summary(stem.with_name(stem.name + "_summary").with_suffix(".tsv"), **values)

    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
