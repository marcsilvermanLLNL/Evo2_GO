#!/usr/bin/env python3
"""Visualize semantic themes among the top predicted GO terms."""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evo2_go_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "human_test_taxon_val_1919taxa_min_count_15000_bs256_mean_loss_3h"
DEFAULT_BUILD_ID = "all_labels_true_path_1919taxa_human_test_min_count_15000_v1"
DEFAULT_GO_BASIC_OBO = PROJECT_ROOT / "data" / "go" / "go-basic.obo"

INK = "#222222"
MUTED = "#666666"
PANEL_EDGE = "#ffffff"

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


@dataclass(frozen=True)
class GoTerm:
    go_id: str
    name: str
    namespace: str
    parents: frozenset[str]


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def parse_go_basic_obo(go_basic_obo: Path) -> dict[str, GoTerm]:
    terms: dict[str, GoTerm] = {}
    current_id: str | None = None
    current_name: str | None = None
    current_namespace: str | None = None
    current_parents: set[str] = set()
    in_term = False

    def finish_term() -> None:
        if current_id is None:
            return
        terms[current_id] = GoTerm(
            go_id=current_id,
            name=current_name or current_id,
            namespace=current_namespace or "",
            parents=frozenset(current_parents),
        )

    with go_basic_obo.open("rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finish_term()
                current_id = None
                current_name = None
                current_namespace = None
                current_parents = set()
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                finish_term()
                current_id = None
                current_name = None
                current_namespace = None
                current_parents = set()
                in_term = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                current_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                current_name = line.split("name: ", 1)[1].strip()
            elif line.startswith("namespace: "):
                current_namespace = line.split("namespace: ", 1)[1].strip()
            elif line.startswith("is_a: "):
                current_parents.add(line.split("is_a: ", 1)[1].split()[0])
            elif line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of":
                    current_parents.add(parts[2])
        finish_term()

    return terms


def build_go_helpers(terms: dict[str, GoTerm]):
    @lru_cache(maxsize=None)
    def depth(go_id: str) -> int:
        parents = [
            parent
            for parent in terms.get(go_id, GoTerm(go_id, go_id, "", frozenset())).parents
            if parent in terms
        ]
        if not parents:
            return 0
        return 1 + max(depth(parent) for parent in parents)

    @lru_cache(maxsize=None)
    def ancestors(go_id: str) -> frozenset[str]:
        out: set[str] = set()
        for parent in terms.get(go_id, GoTerm(go_id, go_id, "", frozenset())).parents:
            if parent in terms:
                out.add(parent)
                out.update(ancestors(parent))
        return frozenset(out)

    def rollup_ancestor(go_id: str, target_depth: int) -> str:
        term = terms.get(go_id)
        if term is None:
            return go_id
        candidates = [
            candidate
            for candidate in ancestors(go_id) | {go_id}
            if candidate in terms
            and terms[candidate].namespace == term.namespace
            and depth(candidate) <= target_depth
        ]
        if not candidates:
            return go_id
        return max(candidates, key=lambda candidate: (depth(candidate), terms[candidate].name))

    return depth, rollup_ancestor


def top_predicted_terms(
    score_matrix_path: Path,
    terms: dict[str, GoTerm],
    *,
    top_n: int,
    min_depth: int,
    rollup_depth: int,
) -> tuple[pd.DataFrame, float, int]:
    depth, rollup_ancestor = build_go_helpers(terms)
    with np.load(score_matrix_path, allow_pickle=False) as score_data:
        probabilities = score_data["probabilities"]
        threshold = float(score_data["global_threshold"][0])
        predicted_positive_count = (probabilities >= threshold).sum(axis=0)
        mean_probability = probabilities.mean(axis=0)
        go_ids = score_data["go_id"].astype(str)
        go_names = score_data["go_name"].astype(str)
        namespaces = score_data["go_namespace"].astype(str)
        records = int(probabilities.shape[0])

    rows: list[dict[str, object]] = []
    for go_id, name, namespace, predicted_count, mean_prob in zip(
        go_ids,
        go_names,
        namespaces,
        predicted_positive_count,
        mean_probability,
        strict=True,
    ):
        if namespace not in NAMESPACE_ORDER:
            continue
        if int(predicted_count) <= 0:
            continue
        go_depth = depth(go_id)
        if go_depth < min_depth:
            continue
        rollup_id = rollup_ancestor(go_id, rollup_depth)
        rollup_term = terms.get(rollup_id)
        rows.append(
            {
                "GO_ID": go_id,
                "name": name,
                "namespace": namespace,
                "namespace_short": NAMESPACE_SHORT[namespace],
                "go_graph_depth": go_depth,
                "predicted_positive_count": int(predicted_count),
                "mean_probability": float(mean_prob),
                "rollup_GO_ID": rollup_id,
                "rollup_name": rollup_term.name if rollup_term else rollup_id,
                "rollup_depth": depth(rollup_id),
            }
        )

    frame = pd.DataFrame(rows)
    selected = []
    for namespace in NAMESPACE_ORDER:
        subset = frame.loc[frame["namespace"].eq(namespace)].sort_values(
            ["predicted_positive_count", "mean_probability", "GO_ID"],
            ascending=[False, False, True],
        )
        selected.append(subset.head(top_n))
    return pd.concat(selected, ignore_index=True), threshold, records


def summarize_rollups(top_terms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (namespace, namespace_short, rollup_id, rollup_name, rollup_depth), group in top_terms.groupby(
        ["namespace", "namespace_short", "rollup_GO_ID", "rollup_name", "rollup_depth"],
        observed=True,
    ):
        ordered = group.sort_values(
            ["predicted_positive_count", "mean_probability", "name"],
            ascending=[False, False, True],
        )
        example_terms = ordered.loc[
            ~ordered["name"].astype(str).str.casefold().eq(str(rollup_name).casefold()),
            "name",
        ].head(3)
        if example_terms.empty:
            example_terms = ordered["name"].head(3)
        rows.append(
            {
                "namespace": namespace,
                "namespace_short": namespace_short,
                "rollup_GO_ID": rollup_id,
                "rollup_name": rollup_name,
                "rollup_depth": rollup_depth,
                "top_predicted_terms": len(group),
                "predicted_positive_calls": int(group["predicted_positive_count"].sum()),
                "example_terms": "; ".join(example_terms.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["namespace", "top_predicted_terms", "predicted_positive_calls", "rollup_name"],
        ascending=[True, False, False, True],
    )


def treemap_layout(
    items: list[dict[str, object]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[dict[str, object]]:
    if not items:
        return []
    if len(items) == 1:
        item = dict(items[0])
        item.update({"x": x, "y": y, "width": width, "height": height})
        return [item]

    total = sum(float(item["value"]) for item in items)
    half = total / 2.0
    cumulative = 0.0
    split_index = 1
    best_distance = float("inf")
    for index in range(1, len(items)):
        cumulative += float(items[index - 1]["value"])
        distance = abs(cumulative - half)
        if distance < best_distance:
            best_distance = distance
            split_index = index

    first = items[:split_index]
    second = items[split_index:]
    first_total = sum(float(item["value"]) for item in first)
    ratio = first_total / total if total else 0.5

    if width >= height:
        first_width = width * ratio
        return treemap_layout(first, x, y, first_width, height) + treemap_layout(
            second,
            x + first_width,
            y,
            width - first_width,
            height,
        )
    first_height = height * ratio
    return treemap_layout(first, x, y, width, first_height) + treemap_layout(
        second,
        x,
        y + first_height,
        width,
        height - first_height,
    )


def wrap_label(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def build_plot_items(namespace_summary: pd.DataFrame, *, max_themes: int) -> pd.DataFrame:
    ordered = namespace_summary.sort_values(
        ["top_predicted_terms", "predicted_positive_calls", "rollup_name"],
        ascending=[False, False, True],
    ).copy()
    return ordered.head(max_themes)


def plot_namespace_treemap(
    ax: plt.Axes,
    namespace_summary: pd.DataFrame,
    *,
    title: str,
    max_themes: int,
) -> None:
    plot_frame = build_plot_items(namespace_summary, max_themes=max_themes)
    plot_frame = plot_frame.sort_values(
        ["top_predicted_terms", "predicted_positive_calls", "rollup_name"],
        ascending=[False, False, True],
    )
    values = plot_frame["top_predicted_terms"].astype(float).to_numpy()
    items = []
    for index, row in enumerate(plot_frame.to_dict(orient="records")):
        items.append(
            {
                "value": max(float(row["top_predicted_terms"]), 0.01),
                "row": row,
                "index": index,
            }
        )

    rects = treemap_layout(items, 0.0, 0.0, 1.0, 1.0)
    colors = plt.get_cmap("tab20c")(np.linspace(0.05, 0.95, max(len(rects), 2)))
    for rect in rects:
        row = rect["row"]
        x = float(rect["x"])
        y = float(rect["y"])
        width = float(rect["width"])
        height = float(rect["height"])
        area = width * height
        color = colors[int(rect["index"]) % len(colors)]
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=color,
                edgecolor=PANEL_EDGE,
                linewidth=1.0,
            )
        )

        if area < 0.035:
            continue
        label_width = max(9, int(width * 45))
        theme = wrap_label(str(row["rollup_name"]), label_width, 3)
        label = theme
        if area >= 0.13:
            examples = str(row["example_terms"])
            if examples and examples != "nan":
                label += "\n" + wrap_label(f"e.g. {examples}", label_width, 2)
        ax.text(
            x + 0.012,
            y + height - 0.018,
            label,
            ha="left",
            va="top",
            color=INK,
            fontsize=6.8 if area >= 0.085 else 6.3,
            linespacing=1.08,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="semibold", pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_semantic_rollup(
    top_terms: pd.DataFrame,
    rollup_summary: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int,
    min_depth: int,
    rollup_depth: int,
    threshold: float,
    records: int,
    max_themes: int,
    dpi: int,
) -> None:
    stem = output_dir / "figure_11_top_predicted_semantic_rollup"
    top_terms.to_csv(stem.with_name(stem.name + "_top_terms.tsv"), sep="\t", index=False)
    rollup_summary.to_csv(stem.with_name(stem.name + "_theme_summary.tsv"), sep="\t", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.85))
    for ax, namespace in zip(axes, NAMESPACE_ORDER, strict=True):
        subset = rollup_summary.loc[rollup_summary["namespace"].eq(namespace)].copy()
        plot_namespace_treemap(
            ax,
            subset,
            title=f"{NAMESPACE_SHORT[namespace]}  {NAMESPACE_LABELS[namespace]}",
            max_themes=max_themes,
        )
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.04, top=0.88, wspace=0.05)
    save_figure(fig, stem, dpi=dpi)

    namespace_counts = top_terms.groupby("namespace_short", observed=True).size().to_dict()
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 11. Semantic roll-up of the most frequently predicted GO terms. "
            f"For each GO namespace, the {top_n} non-root retained GO terms with "
            f"graph depth at least {min_depth} and the largest number of human-test "
            f"genes predicted positive at the global threshold ({threshold:.3f}) "
            "were mapped to higher-level ancestors at GO graph depth "
            f"{rollup_depth}. The plotted rectangles show the largest ancestor-level "
            "semantic themes; rectangle area is proportional to the number of displayed "
            "top predicted terms assigned to the theme, not to model accuracy. Text "
            "inside each block lists the ancestor theme and, for larger blocks, "
            "representative child terms. The full long-tail roll-up is retained in "
            "the output tables. The analysis "
            f"used predictions for {records:,} held-out human genes and includes "
            f"{namespace_counts.get('BP', 0):,} BP, {namespace_counts.get('CC', 0):,} "
            f"CC, and {namespace_counts.get('MF', 0):,} MF terms. Exact top-term "
            "membership and roll-up assignments are provided in "
            "figure_11_top_predicted_semantic_rollup_top_terms.tsv and "
            "figure_11_top_predicted_semantic_rollup_theme_summary.tsv."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--score-matrix", type=Path)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--rollup-depth", type=int, default=3)
    parser.add_argument("--max-themes", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = (
        PROJECT_ROOT
        / "outputs"
        / "builds"
        / args.build_id
        / "runs"
        / args.run_name
        / "test_eval"
    )
    score_matrix_path = args.score_matrix or (eval_dir / "test_score_matrix.npz")
    output_dir = args.output_dir or PROJECT_ROOT / "figures" / args.run_name / "publication_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_publication_style()
    go_terms = parse_go_basic_obo(args.go_basic_obo)
    top_terms, threshold, records = top_predicted_terms(
        score_matrix_path,
        go_terms,
        top_n=args.top_n,
        min_depth=args.min_depth,
        rollup_depth=args.rollup_depth,
    )
    rollup_summary = summarize_rollups(top_terms)
    plot_semantic_rollup(
        top_terms,
        rollup_summary,
        output_dir,
        top_n=args.top_n,
        min_depth=args.min_depth,
        rollup_depth=args.rollup_depth,
        threshold=threshold,
        records=records,
        max_themes=args.max_themes,
        dpi=args.dpi,
    )

    stem = output_dir / "figure_11_top_predicted_semantic_rollup"
    print(f"Wrote semantic roll-up figure to: {stem.with_suffix('.png')}")
    print(f"Wrote semantic roll-up PDF to: {stem.with_suffix('.pdf')}")
    print(f"Wrote top-term table to: {stem.with_name(stem.name + '_top_terms.tsv')}")


if __name__ == "__main__":
    main()
