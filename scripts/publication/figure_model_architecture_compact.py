#!/usr/bin/env python3
"""Create a compact, more graphical Evo2-GO architecture figure."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "model_architecture"

INK = "#17212B"
MUTED = "#52606D"
LIGHT_TEXT = "#F8FAFC"
PANEL_FILL = "#F3F5F7"
PANEL_EDGE = "#7B8794"
BLUE = "#2166AC"
BLUE_MID = "#6AA6CF"
BLUE_LIGHT = "#D8EAF5"
BLUE_FILL = "#EAF2F8"
GREEN = "#2D7D59"
GREEN_MID = "#72AF90"
GREEN_LIGHT = "#D8EDE2"
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


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    linewidth: float = 1.5,
    connectionstyle: str = "arc3",
    mutation_scale: float = 11,
    zorder: int = 4,
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


def badge(ax: plt.Axes, x: float, y: float, number: str, color: str) -> None:
    ax.add_patch(Circle((x, y), 0.18, facecolor=color, edgecolor="none", zorder=8))
    ax.text(
        x,
        y,
        number,
        color="white",
        fontsize=8.2,
        weight="bold",
        ha="center",
        va="center",
        zorder=9,
    )


def panel(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
    fill: str,
    number: str,
    title: str,
    side_label: str = "",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.45,
            edgecolor=color,
            facecolor=fill,
            zorder=2,
        )
    )
    badge(ax, x + 0.30, y + height - 0.30, number, color)
    ax.text(
        x + 0.62,
        y + height - 0.30,
        title,
        color=INK,
        fontsize=11.0,
        weight="bold",
        ha="left",
        va="center",
        zorder=8,
    )
    if side_label:
        ax.text(
            x + width - 0.22,
            y + height - 0.30,
            side_label,
            color=color,
            fontsize=7.4,
            weight="bold",
            ha="right",
            va="center",
            zorder=8,
        )


def draw_feature_array(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    rows: int,
    columns: int,
    cell: float,
    color: str,
    pale: str,
) -> None:
    width = columns * cell
    height = rows * cell
    for row in range(rows):
        for column in range(columns):
            shade = color if (row + 2 * column) % 4 in (0, 1) else pale
            ax.add_patch(
                Rectangle(
                    (
                        cx - width / 2 + column * cell,
                        cy - height / 2 + row * cell,
                    ),
                    cell * 0.78,
                    cell * 0.78,
                    facecolor=shade,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=6,
                )
            )


def draw_graph_state(
    ax: plt.Axes,
    cx: float,
    cy: float,
    scale: float,
    *,
    color: str,
    fill: str,
    halo: bool,
) -> None:
    nodes = [
        (cx, cy + 0.42 * scale),
        (cx - 0.31 * scale, cy + 0.05 * scale),
        (cx + 0.31 * scale, cy + 0.05 * scale),
        (cx - 0.43 * scale, cy - 0.38 * scale),
        (cx - 0.08 * scale, cy - 0.38 * scale),
        (cx + 0.35 * scale, cy - 0.38 * scale),
    ]
    if halo:
        for x_value, y_value in nodes:
            ax.add_patch(
                Circle(
                    (x_value, y_value),
                    0.17 * scale,
                    facecolor=fill,
                    edgecolor="none",
                    alpha=0.95,
                    zorder=3,
                )
            )
    for first, second in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)):
        ax.plot(
            [nodes[first][0], nodes[second][0]],
            [nodes[first][1], nodes[second][1]],
            color=color,
            linewidth=1.4,
            zorder=5,
        )
    for index, (x_value, y_value) in enumerate(nodes):
        ax.add_patch(
            Circle(
                (x_value, y_value),
                (0.10 if index == 0 else 0.085) * scale,
                facecolor=color if halo else "white",
                edgecolor=color,
                linewidth=1.25,
                zorder=6,
            )
        )


def draw_lock(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.085, y - 0.075),
            0.17,
            0.15,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor=color,
            edgecolor="none",
            zorder=8,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.057, y + 0.035),
            0.114,
            0.125,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor="none",
            edgecolor=color,
            linewidth=1.4,
            zorder=7,
        )
    )


def draw_binary_target(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    for index, value in enumerate((1, 0, 1, 1, 0, 1)):
        ax.add_patch(
            FancyBboxPatch(
                (cx + (index - 2.5) * 0.13 - 0.045, cy - 0.12),
                0.09,
                0.24,
                boxstyle="round,pad=0.005,rounding_size=0.018",
                facecolor=color if value else "white",
                edgecolor=color,
                linewidth=0.75,
                zorder=6,
            )
        )


def draw_input_panel(ax: plt.Axes) -> None:
    x, y, width, height = 0.50, 1.38, 2.48, 6.36
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=PANEL_EDGE,
            facecolor=PANEL_FILL,
            zorder=1,
        )
    )
    badge(ax, x + 0.30, y + height - 0.30, "1", PANEL_EDGE)
    ax.text(
        x + 0.62,
        y + height - 0.30,
        "INPUTS",
        color=PANEL_EDGE,
        fontsize=8.2,
        weight="bold",
        ha="left",
        va="center",
    )

    # Evo2 feature card.
    ax.add_patch(
        FancyBboxPatch(
            (0.72, 5.22),
            2.05,
            1.70,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=0.9,
            edgecolor="#D4D9DE",
            facecolor="white",
            zorder=3,
        )
    )
    draw_feature_array(
        ax,
        1.22,
        6.05,
        rows=6,
        columns=4,
        cell=0.115,
        color=BLUE,
        pale=BLUE_LIGHT,
    )
    draw_lock(ax, 1.45, 5.51, BLUE)
    ax.text(1.98, 6.37, "Evo2", fontsize=11.2, weight="bold", color=INK, ha="center")
    ax.text(1.98, 6.06, "4,096-D", fontsize=9.0, weight="bold", color=BLUE, ha="center")
    ax.text(1.98, 5.76, "mean pooled", fontsize=7.1, color=MUTED, ha="center")

    # GO graph card.
    ax.add_patch(
        FancyBboxPatch(
            (0.72, 3.03),
            2.05,
            1.70,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=0.9,
            edgecolor="#D4D9DE",
            facecolor="white",
            zorder=3,
        )
    )
    draw_graph_state(ax, 1.22, 3.93, 0.72, color=GREEN, fill=GREEN_LIGHT, halo=False)
    draw_lock(ax, 1.45, 3.32, GREEN)
    ax.text(2.08, 4.18, "GO graph", fontsize=10.0, weight="bold", color=INK, ha="center")
    ax.text(2.08, 3.86, "3,384 terms", fontsize=8.7, weight="bold", color=GREEN, ha="center")
    ax.text(2.08, 3.59, "is_a • part_of", fontsize=6.8, color=MUTED, ha="center")
    ax.text(2.08, 3.38, "self-loops", fontsize=6.8, color=MUTED, ha="center")

    # Target card.
    ax.add_patch(
        FancyBboxPatch(
            (0.72, 1.72),
            2.05,
            0.82,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=0.9,
            edgecolor="#D4D9DE",
            facecolor="white",
            zorder=3,
        )
    )
    draw_binary_target(ax, 1.20, 2.13, ORANGE)
    ax.text(1.75, 2.24, "True-path", fontsize=8.5, weight="bold", color=INK, ha="left")
    ax.text(1.75, 1.98, "targets", fontsize=8.0, color=MUTED, ha="left")


def draw_sequence_lane(ax: plt.Axes) -> None:
    x, y, width, height = 3.40, 5.13, 7.47, 2.61
    panel(
        ax,
        x,
        y,
        width,
        height,
        color=BLUE,
        fill=BLUE_FILL,
        number="2",
        title="Sequence encoder",
        side_label="MLP PROJECTION",
    )

    centers = (4.25, 6.72, 9.35)
    draw_feature_array(
        ax,
        centers[0],
        6.28,
        rows=8,
        columns=5,
        cell=0.105,
        color=BLUE,
        pale=BLUE_LIGHT,
    )
    draw_feature_array(
        ax,
        centers[1],
        6.28,
        rows=6,
        columns=4,
        cell=0.105,
        color=BLUE,
        pale=BLUE_LIGHT,
    )
    draw_feature_array(
        ax,
        centers[2],
        6.28,
        rows=4,
        columns=3,
        cell=0.115,
        color=BLUE,
        pale=BLUE_LIGHT,
    )
    ax.text(centers[0], 5.62, "4,096", color=INK, fontsize=9.2, weight="bold", ha="center")
    ax.text(centers[1], 5.62, "2,048", color=INK, fontsize=9.2, weight="bold", ha="center")
    ax.text(centers[2], 5.62, "1,024", color=INK, fontsize=9.2, weight="bold", ha="center")
    ax.text(centers[0], 5.42, "input", color=MUTED, fontsize=6.8, ha="center")
    ax.text(centers[1], 5.42, "hidden", color=MUTED, fontsize=6.8, ha="center")
    ax.text(centers[2], 5.42, "unit vector", color=MUTED, fontsize=6.8, ha="center")

    arrow(ax, (4.70, 6.28), (6.27, 6.28), color=BLUE, linewidth=2.0)
    arrow(ax, (7.11, 6.28), (9.00, 6.28), color=BLUE, linewidth=2.0)
    ax.text(5.49, 6.62, "LN  •  FC  •  GELU", color=BLUE, fontsize=7.5, weight="bold", ha="center")
    ax.text(5.49, 6.00, "dropout 0.1", color=MUTED, fontsize=6.7, ha="center")
    ax.text(8.06, 6.62, "LN  •  FC", color=BLUE, fontsize=7.5, weight="bold", ha="center")
    ax.text(8.06, 6.00, "dropout 0.1  •  L2", color=MUTED, fontsize=6.7, ha="center")


def draw_go_lane(ax: plt.Axes) -> None:
    x, y, width, height = 3.40, 2.10, 7.47, 2.61
    panel(
        ax,
        x,
        y,
        width,
        height,
        color=GREEN,
        fill=GREEN_FILL,
        number="3",
        title="GO graph encoder",
        side_label="2 MESSAGE-PASSING HOPS",
    )
    centers = (4.25, 6.72, 9.35)
    draw_graph_state(ax, centers[0], 3.25, 0.78, color=GREEN, fill=GREEN_LIGHT, halo=False)
    draw_graph_state(ax, centers[1], 3.25, 0.78, color=GREEN, fill=GREEN_LIGHT, halo=True)
    draw_graph_state(ax, centers[2], 3.25, 0.78, color=GREEN, fill="#BFE0CE", halo=True)
    ax.text(centers[0], 2.56, "256", color=INK, fontsize=9.2, weight="bold", ha="center")
    ax.text(centers[1], 2.56, "512", color=INK, fontsize=9.2, weight="bold", ha="center")
    ax.text(centers[2], 2.56, "1,024", color=INK, fontsize=9.2, weight="bold", ha="center")
    ax.text(centers[0], 2.35, "learned nodes", color=MUTED, fontsize=6.8, ha="center")
    ax.text(centers[1], 2.35, "hidden", color=MUTED, fontsize=6.8, ha="center")
    ax.text(centers[2], 2.35, "unit vectors", color=MUTED, fontsize=6.8, ha="center")

    arrow(ax, (4.77, 3.25), (6.18, 3.25), color=GREEN, linewidth=2.0)
    arrow(ax, (7.25, 3.25), (8.81, 3.25), color=GREEN, linewidth=2.0)
    ax.text(5.49, 3.61, "ÂH  •  FC  •  LN", color=GREEN, fontsize=7.4, weight="bold", ha="center")
    ax.text(5.49, 2.91, "GELU", color=MUTED, fontsize=6.8, ha="center")
    ax.text(8.05, 3.61, "ÂH  •  FC  •  LN", color=GREEN, fontsize=7.4, weight="bold", ha="center")
    ax.text(8.05, 2.91, "L2", color=MUTED, fontsize=6.8, ha="center")


def draw_scorer(ax: plt.Axes) -> None:
    x, y, width, height = 11.25, 2.10, 2.16, 5.64
    panel(
        ax,
        x,
        y,
        width,
        height,
        color=PURPLE,
        fill=PURPLE_FILL,
        number="4",
        title="Cosine match",
    )
    draw_feature_array(
        ax,
        12.32,
        6.35,
        rows=3,
        columns=5,
        cell=0.10,
        color=BLUE,
        pale=BLUE_LIGHT,
    )
    draw_feature_array(
        ax,
        12.32,
        3.45,
        rows=5,
        columns=3,
        cell=0.10,
        color=GREEN,
        pale=GREEN_LIGHT,
    )
    ax.text(12.32, 5.92, "sequence", color=BLUE, fontsize=7.0, weight="bold", ha="center")
    ax.text(12.32, 3.01, "GO terms", color=GREEN, fontsize=7.0, weight="bold", ha="center")

    ax.add_patch(Circle((12.32, 4.78), 0.60, facecolor="white", edgecolor=PURPLE, linewidth=1.7, zorder=5))
    ax.text(12.32, 4.91, "cos", color=INK, fontsize=14.0, weight="bold", ha="center", va="center", zorder=7)
    ax.text(12.32, 4.58, "× s", color=PURPLE, fontsize=10.0, weight="bold", ha="center", va="center", zorder=7)
    arrow(ax, (12.32, 5.90), (12.32, 5.42), color=PURPLE)
    arrow(ax, (12.32, 3.80), (12.32, 4.13), color=PURPLE)


def draw_output(ax: plt.Axes) -> None:
    x, y, width, height = 13.82, 3.22, 1.77, 3.39
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.45,
            edgecolor=INK,
            facecolor="white",
            zorder=2,
        )
    )
    badge(ax, x + 0.30, y + height - 0.30, "5", INK)
    ax.text(x + 0.60, y + height - 0.30, "GO output", color=INK, fontsize=9.5, weight="bold", va="center")
    ax.text(x + width / 2, 5.52, "3,384 logits", color=INK, fontsize=9.0, weight="bold", ha="center")
    arrow(ax, (x + width / 2, 5.24), (x + width / 2, 4.84), color=INK, linewidth=1.0)
    ax.text(x + width / 2, 4.67, "sigmoid", color=MUTED, fontsize=7.2, weight="bold", ha="center")

    shades = ("#F4EAF1", "#E4BED2", "#CF719A", "#A4265A")
    for row in range(6):
        for column in range(4):
            ax.add_patch(
                Rectangle(
                    (14.25 + column * 0.20, 3.68 + row * 0.15),
                    0.15,
                    0.105,
                    facecolor=shades[(row + column) % len(shades)],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=6,
                )
            )
    ax.text(x + width / 2, 3.45, "multilabel probabilities", color=MUTED, fontsize=6.8, ha="center")


def draw_training_footer(ax: plt.Axes) -> None:
    x, y, width, height = 3.40, 0.48, 12.19, 1.12
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.35,
            edgecolor=ORANGE,
            facecolor=ORANGE_FILL,
            zorder=2,
        )
    )
    badge(ax, x + 0.30, y + height - 0.30, "6", ORANGE)
    draw_binary_target(ax, 4.37, 1.02, ORANGE)
    ax.text(4.91, 1.02, "+", color=ORANGE, fontsize=14, weight="bold", ha="center", va="center")
    ax.text(5.45, 1.02, "logits", color=INK, fontsize=8.3, weight="bold", ha="center", va="center")
    arrow(ax, (5.94, 1.02), (6.66, 1.02), color=ORANGE)
    ax.text(7.55, 1.02, "Asymmetric loss", color=INK, fontsize=10.0, weight="bold", ha="center", va="center")
    ax.text(9.62, 1.02, "γ−=4  •  γ+=0  •  clip=.05", color=MUTED, fontsize=7.2, ha="center", va="center")
    arrow(ax, (10.72, 1.02), (11.31, 1.02), color=ORANGE)
    ax.text(13.05, 1.02, "backprop through both encoders + scale", color=MUTED, fontsize=7.55, ha="center", va="center")
    draw_lock(ax, 15.15, 1.02, ORANGE)
    ax.text(15.15, 0.70, "Evo2 + Â fixed", color=ORANGE, fontsize=6.2, weight="bold", ha="center")


def save_caption(path: Path) -> None:
    caption = (
        "Figure. Compact schematic of the Evo2-GO dual encoder. Precomputed mean-pooled "
        "4,096-dimensional Evo2-7B activations from blocks.28.mlp.l3 are stored in FP16, "
        "cast to FP32, and passed through LayerNorm, a 4,096-to-2,048 linear layer, GELU, "
        "0.1 dropout, a second LayerNorm, a 2,048-to-1,024 linear layer, 0.1 dropout, and "
        "L2 normalization. The ontology branch begins with learned 256-dimensional vectors "
        "for 3,384 GO terms. Two GCN message-passing layers use a fixed symmetrically "
        "normalized adjacency with self-loops and bidirectional is_a and part_of edges. "
        "Bias-free projections map 256 to 512 and 512 to 1,024 dimensions; the first is "
        "followed by LayerNorm and GELU, and the second by LayerNorm and L2 normalization. "
        "The two 1,024-dimensional representations are compared by cosine similarity and a "
        "learned scale initialized to 20 and clamped to [1, 100]. GO embeddings act directly "
        "as output heads, producing 3,384 logits and sigmoid probabilities without a separate "
        "classifier matrix or term-bias layer. Mean-reduced asymmetric loss uses true-path "
        "binary targets, gamma-negative 4, gamma-positive 0, and probability clip 0.05. "
        "Training updates both encoders, GO node features, and the scale; precomputed Evo2 "
        "features and the normalized GO adjacency remain fixed."
    )
    path.write_text(
        textwrap.fill(caption, width=92, break_long_words=False, break_on_hyphens=False)
        + "\n",
        encoding="utf-8",
    )


def make_figure() -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        0.52,
        8.62,
        "Evo2–GO dual-encoder architecture",
        fontsize=21,
        weight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    ax.text(
        0.54,
        8.25,
        "Two learned encoders align representative-CDS features and ontology terms in a shared latent space",
        fontsize=10.3,
        color=MUTED,
        ha="left",
        va="center",
    )

    draw_input_panel(ax)
    draw_sequence_lane(ax)
    draw_go_lane(ax)
    draw_scorer(ax)
    draw_output(ax)
    draw_training_footer(ax)

    arrow(ax, (2.99, 6.06), (3.38, 6.28), color=BLUE, connectionstyle="arc3,rad=-0.08")
    arrow(ax, (2.99, 3.88), (3.38, 3.25), color=GREEN, connectionstyle="arc3,rad=0.10")
    arrow(ax, (10.89, 6.28), (11.23, 6.28), color=BLUE)
    arrow(ax, (10.89, 3.25), (11.23, 3.45), color=GREEN, connectionstyle="arc3,rad=-0.08")
    arrow(ax, (13.43, 4.78), (13.80, 4.78), color=PURPLE)
    arrow(ax, (2.78, 2.06), (3.38, 1.05), color=ORANGE, connectionstyle="arc3,rad=0.18")
    arrow(ax, (14.70, 3.20), (14.70, 1.62), color=ORANGE, connectionstyle="arc3,rad=-0.08")

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
    stem = output_dir / "figure_model_architecture_compact"

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
