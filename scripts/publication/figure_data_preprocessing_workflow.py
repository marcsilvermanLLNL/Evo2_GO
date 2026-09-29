#!/usr/bin/env python3
"""Create a publication-style overview of the Evo2-GO preprocessing workflow."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evo2_go_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "data_preprocessing_workflow"

INK = "#17212B"
MUTED = "#52606D"
LIGHT_TEXT = "#F8FAFC"
SOURCE_FILL = "#F3F5F7"
SOURCE_EDGE = "#7B8794"
BLUE = "#2166AC"
BLUE_FILL = "#EAF2F8"
GREEN = "#2D7D59"
GREEN_FILL = "#EAF5EF"
ORANGE = "#C65D18"
ORANGE_FILL = "#FCEFE6"
PURPLE = "#6250A5"
PURPLE_FILL = "#F0EDFA"
LINE = "#75808A"


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


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    number: str,
    edge: str,
    fill: str,
    title_size: float = 10.4,
    body_size: float = 8.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.11",
        linewidth=1.45,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(patch)

    badge = FancyBboxPatch(
        (x + 0.16, y + height - 0.43),
        0.46,
        0.29,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=0,
        facecolor=edge,
        zorder=4,
    )
    ax.add_patch(badge)
    ax.text(
        x + 0.39,
        y + height - 0.285,
        number,
        color=LIGHT_TEXT,
        fontsize=8.1,
        weight="bold",
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        x + 0.76,
        y + height - 0.285,
        title,
        color=INK,
        fontsize=title_size,
        weight="bold",
        ha="left",
        va="center",
        zorder=5,
    )
    ax.text(
        x + 0.20,
        y + height - 0.59,
        body,
        color=MUTED,
        fontsize=body_size,
        ha="left",
        va="top",
        linespacing=1.35,
        zorder=5,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    connectionstyle: str = "arc3",
    linewidth: float = 1.5,
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def source_card(
    ax: plt.Axes,
    x: float,
    y: float,
    title: str,
    detail: str,
    accent: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        2.35,
        0.73,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=0.85,
        edgecolor="#D4D9DE",
        facecolor="white",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.10,
            0.73,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            linewidth=0,
            facecolor=accent,
            zorder=4,
        )
    )
    ax.text(x + 0.20, y + 0.47, title, color=INK, fontsize=9.1, weight="bold", va="center")
    ax.text(x + 0.20, y + 0.22, detail, color=MUTED, fontsize=7.25, va="center")


def save_caption(path: Path) -> None:
    caption = (
        "Figure. Evo2-GO data preprocessing workflow. Downloaded NCBI Gene2GO, "
        "RefSeq CDS FASTA, and Gene Ontology files form the preserved "
        "source-data snapshot. Positive Gene2GO annotations were retained, including "
        "IEA evidence, while negated annotations were removed. One representative CDS "
        "was selected per (tax_id, GeneID), sequences longer than 8,192 characters were "
        "excluded, and the remaining 30,921,763 sequences were converted to 4,096-"
        "dimensional FP16 Evo2 embeddings. Explicit GO labels were propagated through "
        "is_a and part_of ancestors, and terms supported by at least 15,000 selected "
        "genes defined a 3,384-term vocabulary and normalized GO graph. In parallel, "
        "MMseqs2 linclust grouped nucleotide CDSs at 70% sequence identity and 50% "
        "coverage; whole clusters were assigned to 85% training, 5% validation, and "
        "10% test partitions. Embeddings, propagated targets, GO adjacency, and split "
        "assignments were then joined into model-ready records. The remote acquisition "
        "stage is not yet fully automated in Evo2-GO-Remodel, so reproducing the "
        "workflow currently begins from the preserved source files."
    )
    path.write_text(
        textwrap.fill(caption, width=92, break_long_words=False, break_on_hyphens=False)
        + "\n",
        encoding="utf-8",
    )


def draw_table_icon(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    width, height = 0.78 * scale, 0.62 * scale
    ax.add_patch(Rectangle((cx - width / 2, cy - height / 2), width, height,
                           facecolor="white", edgecolor=color, linewidth=1.5, zorder=5))
    for fraction in (0.33, 0.66):
        ax.plot([cx - width / 2, cx + width / 2],
                [cy - height / 2 + height * fraction] * 2,
                color=color, linewidth=0.85, zorder=6)
    ax.plot([cx - 0.12 * scale] * 2, [cy - height / 2, cy + height / 2],
            color=color, linewidth=0.85, zorder=6)
    for offset in (-0.20, 0.0, 0.20):
        ax.add_patch(Circle((cx - 0.25 * scale, cy + offset * scale),
                            0.035 * scale, facecolor=color, edgecolor="none", zorder=7))


def draw_dna_icon(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    import math

    ys = [cy - 0.42 * scale + index * 0.035 * scale for index in range(25)]
    phase = [index / 24 * 3.0 * math.pi for index in range(25)]
    left = [cx + 0.18 * scale * math.sin(value) for value in phase]
    right = [cx - 0.18 * scale * math.sin(value) for value in phase]
    ax.plot(left, ys, color=color, linewidth=1.7, zorder=6)
    ax.plot(right, ys, color=color, linewidth=1.7, zorder=6)
    for index in range(2, 24, 4):
        ax.plot([left[index], right[index]], [ys[index], ys[index]],
                color=color, linewidth=1.0, alpha=0.8, zorder=5)


def draw_tree_icon(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    nodes = [
        (cx, cy + 0.35 * scale),
        (cx - 0.25 * scale, cy),
        (cx + 0.25 * scale, cy),
        (cx - 0.38 * scale, cy - 0.34 * scale),
        (cx - 0.10 * scale, cy - 0.34 * scale),
        (cx + 0.25 * scale, cy - 0.34 * scale),
    ]
    for first, second in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)):
        ax.plot([nodes[first][0], nodes[second][0]],
                [nodes[first][1], nodes[second][1]],
                color=color, linewidth=1.35, zorder=5)
    for index, (x_value, y_value) in enumerate(nodes):
        radius = 0.105 * scale if index == 0 else 0.085 * scale
        ax.add_patch(Circle((x_value, y_value), radius, facecolor="white",
                            edgecolor=color, linewidth=1.4, zorder=6))


def draw_representative_icon(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    lengths = (0.48, 0.78, 0.60)
    for index, length in enumerate(lengths):
        y_value = cy + (0.27 - index * 0.27) * scale
        selected = index == 1
        ax.plot([cx - 0.38 * scale, cx - 0.38 * scale + length * scale],
                [y_value, y_value], color=color if selected else "#A9B4BE",
                linewidth=5.0 if selected else 3.0, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle((cx + 0.48 * scale, cy), 0.16 * scale,
                        facecolor=color, edgecolor="none", zorder=6))
    ax.plot([cx + 0.40 * scale, cx + 0.47 * scale, cx + 0.58 * scale],
            [cy, cy - 0.08 * scale, cy + 0.09 * scale], color="white",
            linewidth=1.6, solid_capstyle="round", zorder=7)


def draw_filter_icon(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    cutoff = cx + 0.17 * scale
    ax.plot([cutoff, cutoff], [cy - 0.42 * scale, cy + 0.42 * scale],
            color=MUTED, linewidth=1.1, linestyle=(0, (3, 2)), zorder=5)
    for index, length in enumerate((0.42, 0.60, 0.88)):
        y_value = cy + (0.28 - index * 0.28) * scale
        end = cx - 0.42 * scale + length * scale
        accepted = end <= cutoff
        ax.plot([cx - 0.42 * scale, end], [y_value, y_value],
                color=color if accepted else "#C2C8CE", linewidth=4,
                solid_capstyle="round", zorder=6)
    ax.plot([cx + 0.31 * scale, cx + 0.49 * scale],
            [cy - 0.36 * scale, cy - 0.18 * scale], color=ORANGE, linewidth=1.8, zorder=7)
    ax.plot([cx + 0.31 * scale, cx + 0.49 * scale],
            [cy - 0.18 * scale, cy - 0.36 * scale], color=ORANGE, linewidth=1.8, zorder=7)


def draw_embedding_icon(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    draw_dna_icon(ax, cx - 0.28 * scale, cy, 0.68 * scale, color)
    arrow(ax, (cx - 0.02 * scale, cy), (cx + 0.18 * scale, cy), color=color,
          linewidth=1.0, zorder=6)
    shades = ("#D9EAF5", "#9CC7E4", "#579BCB", color)
    for row in range(4):
        for column in range(3):
            ax.add_patch(Rectangle(
                (cx + (0.20 + column * 0.15) * scale, cy + (0.24 - row * 0.16) * scale),
                0.11 * scale, 0.11 * scale,
                facecolor=shades[(row + column) % len(shades)], edgecolor="white",
                linewidth=0.35, zorder=6,
            ))


def draw_cluster_icon(ax: plt.Axes, cx: float, cy: float, scale: float) -> None:
    groups = (
        (cx - 0.33 * scale, cy + 0.12 * scale, BLUE),
        (cx + 0.25 * scale, cy + 0.20 * scale, GREEN),
        (cx + 0.10 * scale, cy - 0.30 * scale, ORANGE),
    )
    offsets = ((-0.10, 0.02), (0.08, 0.10), (0.0, -0.10), (0.15, -0.08))
    for group_x, group_y, group_color in groups:
        for x_offset, y_offset in offsets:
            ax.add_patch(Circle(
                (group_x + x_offset * scale, group_y + y_offset * scale),
                0.075 * scale, facecolor=group_color, edgecolor="white",
                linewidth=0.6, zorder=6,
            ))


def workflow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    number: str,
    title: str,
    subtitle: str,
    edge: str,
    fill: str,
    icon,
) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.45, edgecolor=edge, facecolor=fill, zorder=3,
    ))
    ax.add_patch(Circle((x + 0.30, y + height - 0.28), 0.18,
                        facecolor=edge, edgecolor="none", zorder=7))
    ax.text(x + 0.30, y + height - 0.28, number, color="white", fontsize=8.2,
            weight="bold", ha="center", va="center", zorder=8)
    icon(ax, x + 0.70, y + height / 2 - 0.06, 0.82, edge)
    ax.text(x + 1.32, y + height - 0.36, title, color=INK, fontsize=10.6,
            weight="bold", ha="left", va="center", zorder=7)
    ax.text(x + 1.32, y + 0.38, subtitle, color=MUTED, fontsize=8.4,
            ha="left", va="center", linespacing=1.32, zorder=7)


def input_card(
    ax: plt.Axes,
    x: float,
    y: float,
    title: str,
    color: str,
    icon,
) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), 2.38, 0.87,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=0.9, edgecolor="#D4D9DE", facecolor="white", zorder=3,
    ))
    icon(ax, x + 0.48, y + 0.43, 0.68, color)
    ax.text(x + 0.92, y + 0.43, title, color=INK, fontsize=9.0,
            weight="bold", ha="left", va="center", zorder=6)


def make_figure() -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.52, 8.62, "Evo2–GO data preprocessing workflow", fontsize=21,
            weight="bold", color=INK, ha="left", va="center")
    ax.text(0.54, 8.25,
            "Downloaded biological data are transformed into sequence features, functional labels, and leakage-aware data partitions",
            fontsize=10.3, color=MUTED, ha="left", va="center")

    # Compact biological inputs.
    source_panel = FancyBboxPatch(
        (0.48, 3.45), 2.72, 4.36,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.2, edgecolor=SOURCE_EDGE, facecolor=SOURCE_FILL, zorder=1,
    )
    ax.add_patch(source_panel)
    ax.text(0.72, 7.52, "INPUT DATA", fontsize=8.0, weight="bold",
            color=SOURCE_EDGE, ha="left", va="center")
    input_card(ax, 0.66, 6.25, "Gene2GO\nannotations", GREEN, draw_table_icon)
    input_card(ax, 0.66, 5.05, "RefSeq CDS\nsequences", BLUE, draw_dna_icon)
    input_card(ax, 0.66, 3.85, "GO ontology", GREEN, draw_tree_icon)
    ax.text(1.84, 3.62, "Preserved local files", fontsize=7.2, color=MUTED,
            ha="center", va="center")

    workflow_box(
        ax, 3.70, 6.15, 3.25, 1.40,
        number="1", title="Curate GO labels",
        subtitle="Positive annotations\nRetain evidence\nRemove NOT annotations",
        edge=GREEN, fill=GREEN_FILL, icon=draw_table_icon,
    )
    workflow_box(
        ax, 7.45, 6.15, 3.60, 1.40,
        number="2", title="Representative CDS",
        subtitle="One longest CDS per gene\nDeterministic tie-breaks",
        edge=BLUE, fill=BLUE_FILL, icon=draw_representative_icon,
    )
    workflow_box(
        ax, 11.55, 6.15, 3.92, 1.40,
        number="3", title="Length filter",
        subtitle="≤8,192 nt\n30.9 million sequences retained",
        edge=BLUE, fill=BLUE_FILL, icon=draw_filter_icon,
    )

    ax.text(5.33, 5.56, "FUNCTION", color=GREEN, fontsize=7.8,
            weight="bold", ha="center")
    ax.text(9.25, 5.56, "PARTITION", color=ORANGE, fontsize=7.8,
            weight="bold", ha="center")
    ax.text(13.51, 5.56, "SEQUENCE", color=BLUE, fontsize=7.8,
            weight="bold", ha="center")

    workflow_box(
        ax, 3.70, 3.56, 3.25, 1.66,
        number="5", title="GO labels + graph",
        subtitle="True-path propagation\n3,384 GO terms retained",
        edge=GREEN, fill=GREEN_FILL, icon=draw_tree_icon,
    )
    workflow_box(
        ax, 7.45, 3.56, 3.60, 1.66,
        number="6", title="Cluster split",
        subtitle="MMseqs2 clusters\n85% train • 5% val • 10% test",
        edge=ORANGE, fill=ORANGE_FILL,
        icon=lambda axis, cx, cy, scale, _color: draw_cluster_icon(axis, cx, cy, scale),
    )
    workflow_box(
        ax, 11.55, 3.56, 3.92, 1.66,
        number="4", title="Evo2 embeddings",
        subtitle="Mean-pooled Evo2-7B features\n4,096 dimensions per gene",
        edge=BLUE, fill=BLUE_FILL, icon=draw_embedding_icon,
    )

    # Final model-input bundle.
    ax.add_patch(FancyBboxPatch(
        (4.85, 1.16), 9.05, 1.43,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor=PURPLE, facecolor=PURPLE_FILL, zorder=3,
    ))
    ax.add_patch(Circle((5.18, 2.27), 0.18, facecolor=PURPLE, edgecolor="none", zorder=7))
    ax.text(5.18, 2.27, "7", color="white", fontsize=8.2, weight="bold",
            ha="center", va="center", zorder=8)
    draw_embedding_icon(ax, 5.72, 1.83, 0.68, PURPLE)
    draw_tree_icon(ax, 6.73, 1.83, 0.60, PURPLE)
    draw_cluster_icon(ax, 7.72, 1.83, 0.55)
    ax.text(8.35, 2.10, "Model-ready dataset", color=INK, fontsize=12.0,
            weight="bold", ha="left", va="center")
    ax.text(8.35, 1.66, "Embedding  +  GO targets/graph  +  split assignment",
            color=MUTED, fontsize=9.1, ha="left", va="center")

    model_box = FancyBboxPatch(
        (14.47, 1.40), 1.08, 0.92,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=0, facecolor=INK, zorder=3,
    )
    ax.add_patch(model_box)
    ax.text(15.01, 1.96, "READY", color="#B8E1CC", fontsize=7.0,
            weight="bold", ha="center")
    ax.text(15.01, 1.70, "Evo2–GO\nmodel", color="white", fontsize=8.5,
            weight="bold", ha="center", va="center")

    # Workflow connections.
    arrow(ax, (3.18, 6.84), (3.68, 6.84), color=SOURCE_EDGE)
    arrow(ax, (6.97, 6.84), (7.43, 6.84), color=LINE)
    arrow(ax, (11.07, 6.84), (11.53, 6.84), color=LINE)
    arrow(ax, (5.33, 6.13), (5.33, 5.24), color=GREEN)
    arrow(ax, (3.18, 4.20), (3.68, 4.20), color=GREEN)
    arrow(ax, (13.51, 6.13), (13.51, 5.24), color=BLUE)
    arrow(ax, (12.34, 6.13), (10.72, 5.24), color=ORANGE,
          connectionstyle="arc3,rad=0.17")
    arrow(ax, (5.33, 3.54), (6.78, 2.61), color=GREEN,
          connectionstyle="arc3,rad=-0.12")
    arrow(ax, (9.25, 3.54), (9.47, 2.61), color=ORANGE,
          connectionstyle="arc3,rad=0.02")
    arrow(ax, (13.51, 3.54), (12.30, 2.61), color=BLUE,
          connectionstyle="arc3,rad=0.10")
    arrow(ax, (13.92, 1.86), (14.44, 1.86), color=PURPLE)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_data_preprocessing_workflow"

    fig = make_figure()
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    save_caption(stem.with_name(stem.name + "_caption").with_suffix(".txt"))

    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
