#!/usr/bin/env python3
"""Create the publication-style Evo2-GO dual-encoder architecture figure."""

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


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    connectionstyle: str = "arc3",
    linewidth: float = 1.45,
    mutation_scale: float = 11,
    zorder: int = 2,
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


def number_badge(ax: plt.Axes, x: float, y: float, number: str, color: str) -> None:
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


def draw_embedding_icon(
    ax: plt.Axes, cx: float, cy: float, scale: float, color: str
) -> None:
    shades = ("#D9EAF5", "#9CC7E4", "#579BCB", color)
    for row in range(4):
        for column in range(4):
            ax.add_patch(
                Rectangle(
                    (
                        cx + (column - 1.5) * 0.16 * scale,
                        cy + (1.5 - row) * 0.16 * scale,
                    ),
                    0.12 * scale,
                    0.12 * scale,
                    facecolor=shades[(row + 2 * column) % len(shades)],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=6,
                )
            )


def draw_graph_icon(
    ax: plt.Axes, cx: float, cy: float, scale: float, color: str
) -> None:
    nodes = [
        (cx, cy + 0.36 * scale),
        (cx - 0.27 * scale, cy + 0.02 * scale),
        (cx + 0.27 * scale, cy + 0.02 * scale),
        (cx - 0.39 * scale, cy - 0.34 * scale),
        (cx - 0.08 * scale, cy - 0.34 * scale),
        (cx + 0.31 * scale, cy - 0.34 * scale),
    ]
    for first, second in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)):
        ax.plot(
            [nodes[first][0], nodes[second][0]],
            [nodes[first][1], nodes[second][1]],
            color=color,
            linewidth=1.35,
            zorder=5,
        )
    for x_value, y_value in nodes:
        ax.add_patch(
            Circle(
                (x_value, y_value),
                0.09 * scale,
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=6,
            )
        )


def draw_target_icon(
    ax: plt.Axes, cx: float, cy: float, scale: float, color: str
) -> None:
    values = (1, 0, 1, 1, 0)
    for index, value in enumerate(values):
        x_value = cx + (index - 2) * 0.17 * scale
        ax.add_patch(
            FancyBboxPatch(
                (x_value - 0.06 * scale, cy - 0.14 * scale),
                0.12 * scale,
                0.28 * scale,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=0.7,
                edgecolor=color,
                facecolor=color if value else "white",
                zorder=6,
            )
        )


def input_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    color: str,
    icon,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=0.9,
            edgecolor="#D4D9DE",
            facecolor="white",
            zorder=3,
        )
    )
    icon(ax, x + 0.38, y + height / 2, 0.62, color)
    ax.text(
        x + 0.76,
        y + height - 0.30,
        title,
        color=INK,
        fontsize=8.7,
        weight="bold",
        ha="left",
        va="center",
        zorder=7,
    )
    ax.text(
        x + 0.76,
        y + height - 0.56,
        body,
        color=MUTED,
        fontsize=6.55,
        ha="left",
        va="top",
        linespacing=1.28,
        zorder=7,
    )


def lane(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    number: str,
    title: str,
    subtitle: str,
    color: str,
    fill: str,
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
            zorder=3,
        )
    )
    number_badge(ax, x + 0.30, y + height - 0.30, number, color)
    ax.text(
        x + 0.62,
        y + height - 0.30,
        title,
        color=INK,
        fontsize=11.0,
        weight="bold",
        ha="left",
        va="center",
        zorder=7,
    )
    ax.text(
        x + width - 0.22,
        y + height - 0.30,
        subtitle,
        color=color,
        fontsize=7.25,
        weight="bold",
        ha="right",
        va="center",
        zorder=7,
    )


def stage_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    emphasis: bool = False,
    font_size: float = 7.2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.055",
            linewidth=1.0 if emphasis else 0.75,
            edgecolor=color if emphasis else "#C6D0D8",
            facecolor="white",
            zorder=5,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        color=INK if emphasis else MUTED,
        fontsize=font_size,
        weight="bold" if emphasis else "normal",
        ha="center",
        va="center",
        linespacing=1.22,
        zorder=7,
    )


def draw_stage_chain(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    widths: list[float],
    height: float,
    gap: float,
    labels: list[str],
    color: str,
    emphasized: set[int],
    font_size: float,
) -> None:
    current_x = x
    for index, (width, label) in enumerate(zip(widths, labels, strict=True)):
        stage_box(
            ax,
            current_x,
            y,
            width,
            height,
            label,
            color,
            emphasis=index in emphasized,
            font_size=font_size,
        )
        if index < len(labels) - 1:
            arrow(
                ax,
                (current_x + width + 0.01, y + height / 2),
                (current_x + width + gap - 0.01, y + height / 2),
                color=color,
                linewidth=1.0,
                mutation_scale=8,
                zorder=6,
            )
        current_x += width + gap


def scorer_panel(ax: plt.Axes) -> None:
    x, y, width, height = 10.95, 2.43, 2.33, 5.32
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=PURPLE,
            facecolor=PURPLE_FILL,
            zorder=3,
        )
    )
    number_badge(ax, x + 0.30, y + height - 0.30, "4", PURPLE)
    ax.text(
        x + 0.62,
        y + height - 0.30,
        "Shared-space scorer",
        color=INK,
        fontsize=10.1,
        weight="bold",
        ha="left",
        va="center",
        zorder=7,
    )

    stage_box(
        ax,
        x + 0.25,
        y + 3.65,
        width - 0.50,
        0.72,
        "Sequence vectors\nB × 1,024 • unit norm",
        BLUE,
        emphasis=True,
        font_size=7.5,
    )
    stage_box(
        ax,
        x + 0.25,
        y + 0.89,
        width - 0.50,
        0.72,
        "GO vectors\n3,384 × 1,024 • unit norm",
        GREEN,
        emphasis=True,
        font_size=7.5,
    )
    ax.text(
        x + width / 2,
        y + 2.78,
        "COSINE SIMILARITY",
        color=PURPLE,
        fontsize=7.3,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        x + width / 2,
        y + 2.35,
        "s · Zseq ZGOᵀ",
        color=INK,
        fontsize=13.0,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        x + width / 2,
        y + 1.95,
        "learned scale s: 20 → [1, 100]",
        color=MUTED,
        fontsize=7.1,
        ha="center",
        va="center",
    )
    ax.text(
        x + width / 2,
        y + 0.43,
        "GO vectors are the output heads\n(no separate classifier matrix)",
        color=MUTED,
        fontsize=7.0,
        ha="center",
        va="center",
        linespacing=1.25,
    )
    arrow(ax, (x + width / 2, y + 3.63), (x + width / 2, y + 3.05), color=PURPLE)
    arrow(ax, (x + width / 2, y + 1.63), (x + width / 2, y + 1.86), color=PURPLE)


def output_panel(ax: plt.Axes) -> None:
    x, y, width, height = 13.55, 3.55, 2.20, 3.00
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.45,
            edgecolor=INK,
            facecolor="white",
            zorder=3,
        )
    )
    number_badge(ax, x + 0.30, y + height - 0.30, "5", INK)
    ax.text(
        x + 0.60,
        y + height - 0.30,
        "GO output",
        color=INK,
        fontsize=9.2,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        x + width / 2,
        y + 1.90,
        "raw logits",
        color=MUTED,
        fontsize=7.4,
        ha="center",
        va="center",
    )
    ax.text(
        x + width / 2,
        y + 1.59,
        "B × 3,384",
        color=INK,
        fontsize=10.1,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.37, y + 0.72),
            width - 0.74,
            0.46,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            linewidth=0,
            facecolor=INK,
            zorder=5,
        )
    )
    ax.text(
        x + width / 2,
        y + 0.95,
        "SIGMOID",
        color=LIGHT_TEXT,
        fontsize=7.6,
        weight="bold",
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 0.35,
        "GO probabilities\n→ predicted label set",
        color=MUTED,
        fontsize=7.2,
        ha="center",
        va="center",
        linespacing=1.25,
    )


def training_panel(ax: plt.Axes) -> None:
    x, y, width, height = 3.45, 0.48, 12.11, 1.42
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.45,
            edgecolor=ORANGE,
            facecolor=ORANGE_FILL,
            zorder=3,
        )
    )
    number_badge(ax, x + 0.30, y + height - 0.30, "6", ORANGE)
    ax.text(
        x + 0.62,
        y + height - 0.30,
        "Training objective",
        color=INK,
        fontsize=10.2,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        x + 2.88,
        y + height - 0.30,
        "raw logits  +  true-path binary targets",
        color=MUTED,
        fontsize=7.8,
        ha="left",
        va="center",
    )
    ax.text(
        x + 6.18,
        y + height - 0.30,
        "→",
        color=ORANGE,
        fontsize=13,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        x + 6.52,
        y + height - 0.30,
        "Asymmetric loss",
        color=INK,
        fontsize=9.0,
        weight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        x + 8.35,
        y + height - 0.30,
        "γ−=4  •  γ+=0  •  clip=0.05  •  mean",
        color=MUTED,
        fontsize=7.4,
        ha="left",
        va="center",
    )
    ax.plot(
        [x + 0.30, x + width - 0.30],
        [y + 0.67, y + 0.67],
        color="#E4C5B1",
        linewidth=0.8,
        zorder=4,
    )
    ax.text(
        x + 0.31,
        y + 0.34,
        "AdamW backprop updates both encoders, GO node features, and logit scale",
        color=MUTED,
        fontsize=7.35,
        ha="left",
        va="center",
    )
    ax.text(
        x + width - 0.31,
        y + 0.34,
        "Fixed: Evo2 embeddings + normalized GO adjacency",
        color=ORANGE,
        fontsize=7.35,
        weight="bold",
        ha="right",
        va="center",
    )


def save_caption(path: Path) -> None:
    caption = (
        "Figure. Evo2-GO dual-encoder architecture. Each representative CDS is "
        "represented by a precomputed, mean-pooled 4,096-dimensional activation "
        "from layer blocks.28.mlp.l3 of Evo2-7B; stored FP16 vectors are cast to "
        "FP32 at the trainable model boundary. The sequence encoder applies "
        "LayerNorm, a 4,096-to-2,048 linear projection, GELU, 0.1 dropout, a "
        "second LayerNorm, a 2,048-to-1,024 linear projection, 0.1 dropout, and "
        "L2 normalization. In parallel, 3,384 learned 256-dimensional GO node "
        "vectors are encoded by a two-layer GCN over a fixed, symmetrically "
        "normalized ontology adjacency containing self-loops and bidirectional "
        "is_a and part_of edges. Each graph layer aggregates with the adjacency; "
        "bias-free projections map 256 to 512 and then 512 to 1,024 dimensions, "
        "with LayerNorm and GELU after the first projection and LayerNorm followed "
        "by L2 normalization after the second. The GCN therefore supplies two-hop "
        "ontology context and is recomputed on each training forward pass. Unit-"
        "normalized sequence and GO representations are compared by a scaled dot "
        "product, equivalent to cosine similarity. The learned scale is initialized "
        "to 20 and clamped to [1, 100], producing one raw logit per retained GO "
        "term; sigmoid converts logits to multilabel probabilities. GO vectors "
        "serve directly as output heads, with no separate classifier matrix or "
        "term-bias layer. Training uses dense true-path binary targets and mean-"
        "reduced asymmetric loss (gamma-negative 4, gamma-positive 0, probability "
        "clip 0.05). Backpropagation updates both encoders, the GO node features, "
        "and the logit scale, while the precomputed Evo2 embeddings and normalized "
        "GO adjacency remain fixed."
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
        "Sequence features and the Gene Ontology graph are mapped into one normalized latent space for multilabel prediction",
        fontsize=10.3,
        color=MUTED,
        ha="left",
        va="center",
    )

    # Inputs and supervision.
    ax.add_patch(
        FancyBboxPatch(
            (0.48, 1.90),
            2.54,
            5.85,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=PANEL_EDGE,
            facecolor=PANEL_FILL,
            zorder=1,
        )
    )
    number_badge(ax, 0.79, 7.43, "1", PANEL_EDGE)
    ax.text(
        1.10,
        7.43,
        "MODEL INPUTS",
        fontsize=8.0,
        weight="bold",
        color=PANEL_EDGE,
        ha="left",
        va="center",
    )
    input_card(
        ax,
        0.68,
        5.55,
        2.13,
        1.43,
        title="Evo2 features",
        body="Frozen Evo2-7B\nMLP layer 28\nMean pooled • 4,096-D\nFP16 → FP32",
        color=BLUE,
        icon=draw_embedding_icon,
    )
    input_card(
        ax,
        0.68,
        3.68,
        2.13,
        1.43,
        title="GO graph",
        body="3,384 retained terms\nis_a + part_of\nself-loops • fixed Â",
        color=GREEN,
        icon=draw_graph_icon,
    )
    input_card(
        ax,
        0.68,
        2.27,
        2.13,
        0.95,
        title="Training targets",
        body="True-path binary labels",
        color=ORANGE,
        icon=draw_target_icon,
    )

    # Trainable sequence lane.
    lane(
        ax,
        3.45,
        5.25,
        7.08,
        2.50,
        number="2",
        title="Sequence encoder",
        subtitle="TRAINABLE PROJECTION",
        color=BLUE,
        fill=BLUE_FILL,
    )
    draw_stage_chain(
        ax,
        x=3.75,
        y=5.69,
        widths=[0.83, 1.03, 1.12, 0.83, 1.05, 1.04],
        height=1.15,
        gap=0.13,
        labels=[
            "LayerNorm\n4,096",
            "Linear\n4,096 → 2,048",
            "GELU\nDropout 0.1",
            "LayerNorm\n2,048",
            "Linear\n2,048 → 1,024",
            "Dropout 0.1\nL2 normalize",
        ],
        color=BLUE,
        emphasized={1, 4},
        font_size=7.15,
    )

    # Trainable GO graph lane.
    lane(
        ax,
        3.45,
        2.43,
        7.08,
        2.47,
        number="3",
        title="GO term graph encoder",
        subtitle="2-LAYER GCN • 2-HOP CONTEXT",
        color=GREEN,
        fill=GREEN_FILL,
    )
    draw_stage_chain(
        ax,
        x=3.72,
        y=2.84,
        widths=[0.89, 0.71, 0.82, 0.78, 0.71, 0.86, 0.91],
        height=1.16,
        gap=0.10,
        labels=[
            "Learned nodes\n3,384 × 256",
            "Â · H\naggregate",
            "Linear\n256 → 512",
            "LayerNorm\n+ GELU",
            "Â · H\naggregate",
            "Linear\n512 → 1,024",
            "LayerNorm\n+ L2 norm",
        ],
        color=GREEN,
        emphasized={0, 2, 5},
        font_size=6.85,
    )
    ax.text(
        3.76,
        2.62,
        "Learned node features • bias-free GCN projections • graph representations recomputed during training",
        color=MUTED,
        fontsize=6.8,
        ha="left",
        va="center",
    )

    scorer_panel(ax)
    output_panel(ax)
    training_panel(ax)

    # Major data-flow connections.
    arrow(ax, (3.03, 6.27), (3.43, 6.27), color=BLUE)
    arrow(ax, (3.03, 4.38), (3.43, 3.68), color=GREEN, connectionstyle="arc3,rad=0.15")
    arrow(ax, (10.55, 6.25), (10.93, 6.25), color=BLUE)
    arrow(ax, (10.55, 3.63), (10.93, 3.63), color=GREEN)
    arrow(ax, (13.30, 5.07), (13.53, 5.07), color=PURPLE)
    arrow(
        ax,
        (14.66, 3.53),
        (14.66, 1.92),
        color=ORANGE,
        connectionstyle="arc3,rad=-0.10",
    )
    arrow(
        ax,
        (2.82, 2.72),
        (3.43, 1.33),
        color=ORANGE,
        connectionstyle="arc3,rad=0.24",
    )

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
    stem = output_dir / "figure_model_architecture"

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
