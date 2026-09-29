#!/usr/bin/env python3
"""Plot GO-term performance by annotation provenance.

Evidence provenance is reconstructed for each held-out gene--GO-term label.
Every explicit Gene2GO annotation starts in one of five evidence families.  The
legacy true-path mode carries that family through ``is_a`` and ``part_of``
propagation and uses a prespecified priority to make provenance mutually
exclusive.  Explicit mode keeps only the original Gene2GO terms and places a GO
term in every source family directly observed for that term among held-out
genes; these source groups are intentionally non-exclusive.

The output is three namespace-specific figures (BP, MF, and CC), each containing
the global-threshold GO-term F1 and per-term optimal Fmax boxplots used in the
top row of publication Figure 6.  Audit tables retain all per-term source counts.

Pass ``--annotation-label-mode explicit`` when the evaluation targets contain
only original Gene2GO terms and must not be expanded to ontology ancestors.
"""

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
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "mmseqs_linclust_full_85_5_10_min_count_15000_bs256_mean_loss_fixed_val50k_v2"
DEFAULT_BUILD_ID = "all_labels_true_path_full_cleaned_min_count_15000_v1"
DEFAULT_EVAL_DIR = (
    PROJECT_ROOT
    / "outputs/builds"
    / DEFAULT_BUILD_ID
    / "runs"
    / DEFAULT_RUN_NAME
    / "test_eval_cpu"
)
DEFAULT_TEST_GENE_METRICS = DEFAULT_EVAL_DIR / "test_publication_gene_metrics.tsv"
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz"
)
DEFAULT_GO_BASIC_OBO = PROJECT_ROOT / "data/go/go-basic.obo"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / DEFAULT_RUN_NAME / "publication_ready"

INK = "#222222"
MUTED = "#666666"
GRID = "#d6d6d6"
GO_ROOT_IDS = {"GO:0003674", "GO:0005575", "GO:0008150"}

# This is a display hierarchy, not a claim that GO evidence codes form a
# universal quality ranking.  It is applied within a gene--term label only.
EVIDENCE_GROUPS = [
    (
        "experimental",
        "Conventional experimental",
        "Conventional\nexperimental",
        ("EXP", "IDA", "IPI", "IMP", "IGI", "IEP"),
        "#2f6fae",
    ),
    (
        "high_throughput_experimental",
        "High-throughput experimental",
        "High-throughput\nexperimental",
        ("HTP", "HDA", "HMP", "HGI", "HEP"),
        "#c45a14",
    ),
    (
        "phylogenetic_computational",
        "Curated computational/phylogenetic inference",
        "Computational /\nphylogenetic",
        ("IBA", "IBD", "IKR", "IRD", "ISS", "ISO", "ISA", "ISM", "IGC", "RCA"),
        "#4f8f46",
    ),
    (
        "author_curator",
        "Author/curator statements",
        "Author / curator\nstatements",
        ("TAS", "NAS", "IC", "ND"),
        "#7b3f98",
    ),
    (
        "electronic_annotation",
        "Electronic annotation",
        "Electronic\n(IEA only)",
        ("IEA",),
        "#b44747",
    ),
]

GROUP_ORDER = [row[0] for row in EVIDENCE_GROUPS]
GROUP_LABELS = {row[0]: row[1] for row in EVIDENCE_GROUPS}
GROUP_TICK_LABELS = {row[0]: row[2] for row in EVIDENCE_GROUPS}
GROUP_CODES = {row[0]: row[3] for row in EVIDENCE_GROUPS}
GROUP_COLORS = {row[0]: row[4] for row in EVIDENCE_GROUPS}
GROUP_PRIORITY = {group: index + 1 for index, group in enumerate(GROUP_ORDER)}
EVIDENCE_TO_GROUP = {
    evidence: group
    for group, _label, _tick, evidence_codes, _color in EVIDENCE_GROUPS
    for evidence in evidence_codes
}


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def read_key_value_tsv(path: Path) -> dict[str, str]:
    with path.open("rt", newline="") as handle:
        return {
            row["metric"]: row["value"]
            for row in csv.DictReader(handle, delimiter="\t")
        }


def parse_obo_graph(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, str], set[str]]:
    """Return canonical parent links, alt-ID aliases, and obsolete IDs."""

    parents_by_id: dict[str, set[str]] = {}
    alt_to_primary: dict[str, str] = {}
    obsolete: set[str] = set()
    current_id: str | None = None
    current_parents: set[str] = set()
    current_alt_ids: set[str] = set()
    current_obsolete = False
    in_term = False

    def finish_term() -> None:
        if current_id is None:
            return
        parents_by_id[current_id] = set(current_parents)
        for alt_id in current_alt_ids:
            alt_to_primary[alt_id] = current_id
        if current_obsolete:
            obsolete.add(current_id)

    with path.open("rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finish_term()
                current_id = None
                current_parents = set()
                current_alt_ids = set()
                current_obsolete = False
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                finish_term()
                current_id = None
                current_parents = set()
                current_alt_ids = set()
                current_obsolete = False
                in_term = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                current_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("alt_id: "):
                current_alt_ids.add(line.split("alt_id: ", 1)[1].strip())
            elif line.startswith("is_a: "):
                current_parents.add(line.split("is_a: ", 1)[1].split()[0])
            elif line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of":
                    current_parents.add(parts[2])
            elif line == "is_obsolete: true":
                current_obsolete = True
        finish_term()

    for go_id in obsolete:
        parents_by_id[go_id] = set()
    return parents_by_id, alt_to_primary, obsolete


def ancestor_closure(
    go_id: str,
    parents_by_id: dict[str, set[str]],
    memo: dict[str, frozenset[str]],
    visiting: set[str] | None = None,
) -> frozenset[str]:
    """Return the term and all is_a/part_of ancestors."""

    if go_id in memo:
        return memo[go_id]
    if visiting is None:
        visiting = set()
    if go_id in visiting:
        return frozenset({go_id})
    visiting.add(go_id)
    values = {go_id}
    for parent in parents_by_id.get(go_id, set()):
        values.update(ancestor_closure(parent, parents_by_id, memo, visiting))
    visiting.remove(go_id)
    memo[go_id] = frozenset(values)
    return memo[go_id]


def graph_depths(
    go_ids: set[str],
    parents_by_id: dict[str, set[str]],
) -> dict[str, int]:
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

    return {go_id: depth(go_id) for go_id in go_ids if go_id in parents_by_id}


def packed_gene_identity(tax_id: str, gene_id: str) -> int:
    """Pack the numeric NCBI taxon/GeneID pair into one memory-efficient integer."""

    tax_value = int(tax_id)
    gene_value = int(gene_id)
    if tax_value < 0 or tax_value >= 2**32 or gene_value < 0 or gene_value >= 2**32:
        raise ValueError(f"Identity cannot be packed into uint64: {tax_id!r}, {gene_id!r}")
    return (tax_value << 32) | gene_value


def load_test_gene_identities(path: Path, *, progress_every: int) -> tuple[set[int], int]:
    """Load held-out gene identities without materializing the large metrics table."""

    identities: set[int] = set()
    rows_seen = 0
    with path.open("rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        try:
            tax_index = header.index("tax_id")
            gene_index = header.index("GeneID")
        except ValueError as exc:
            raise ValueError(f"{path} must contain tax_id and GeneID columns") from exc
        needed_fields = max(tax_index, gene_index) + 1
        for rows_seen, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t", needed_fields)
            identities.add(packed_gene_identity(fields[tax_index], fields[gene_index]))
            if progress_every > 0 and rows_seen % progress_every == 0:
                print(f"Loaded {rows_seen:,} held-out gene identities...", flush=True)
    if len(identities) != rows_seen:
        raise ValueError(
            f"Held-out gene metrics contain duplicate tax_id/GeneID identities: "
            f"{rows_seen:,} rows versus {len(identities):,} unique identities"
        )
    return identities, rows_seen


def scan_test_gene_term_sources(
    annotations_path: Path,
    test_gene_identities: set[int],
    retained_go_ids: set[str],
    parents_by_id: dict[str, set[str]],
    alt_to_primary: dict[str, str],
    *,
    annotation_label_mode: str,
    progress_every: int,
) -> tuple[dict[str, list[int]], dict[str, int], Counter[str], Counter[str], int, int, int]:
    """Count evidence families supporting each held-out gene--term label.

    In explicit mode, a gene--term pair contributes once to every directly
    observed evidence family supporting it. In true-path mode, the historical
    priority rule keeps each propagated gene--term label mutually exclusive.
    """

    group_count = len(GROUP_ORDER)
    term_group_counts = {go_id: [0] * group_count for go_id in retained_go_ids}
    term_unique_counts = {go_id: 0 for go_id in retained_go_ids}
    all_group_rows: Counter[str] = Counter()
    test_group_rows: Counter[str] = Counter()
    closure_memo: dict[str, frozenset[str]] = {}
    retained_closure_cache: dict[str, tuple[str, ...]] = {}
    current_identity: int | None = None
    current_term_priorities: dict[str, set[int]] = {}
    rows_seen = 0
    test_annotation_rows = 0
    test_genes_with_annotations = 0

    def flush_current_gene() -> None:
        nonlocal test_genes_with_annotations
        if current_identity is None:
            return
        for go_id, priorities in current_term_priorities.items():
            term_unique_counts[go_id] += 1
            selected_priorities = (
                priorities
                if annotation_label_mode == "explicit"
                else {min(priorities)}
            )
            for priority in selected_priorities:
                term_group_counts[go_id][priority - 1] += 1
        test_genes_with_annotations += 1

    opener = gzip.open if ".gz" in annotations_path.suffixes else open
    with opener(annotations_path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = ("tax_id", "GeneID", "GO_ID", "Evidence")
        try:
            tax_index, gene_index, go_index, evidence_index = (
                header.index(column) for column in expected
            )
        except ValueError as exc:
            raise ValueError(
                f"{annotations_path} must contain {', '.join(expected)} columns"
            ) from exc

        for rows_seen, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            evidence = fields[evidence_index].strip().upper()
            group = EVIDENCE_TO_GROUP.get(evidence)
            if group is None:
                raise ValueError(
                    f"Unmapped GO evidence code {evidence!r} at annotation row {rows_seen + 1:,}"
                )
            all_group_rows[group] += 1
            identity = packed_gene_identity(fields[tax_index], fields[gene_index])
            if identity not in test_gene_identities:
                if progress_every > 0 and rows_seen % progress_every == 0:
                    print(
                        f"Scanned {rows_seen:,} annotations; retained "
                        f"{test_annotation_rows:,} held-out rows...",
                        flush=True,
                    )
                continue

            if identity != current_identity:
                flush_current_gene()
                current_identity = identity
                current_term_priorities = {}

            test_annotation_rows += 1
            test_group_rows[group] += 1
            go_id = alt_to_primary.get(fields[go_index].strip(), fields[go_index].strip())
            targets = retained_closure_cache.get(go_id)
            if targets is None:
                if annotation_label_mode == "explicit":
                    targets = (go_id,) if go_id in retained_go_ids else ()
                else:
                    targets = tuple(
                        target
                        for target in ancestor_closure(go_id, parents_by_id, closure_memo)
                        if target in retained_go_ids
                    )
                retained_closure_cache[go_id] = targets
            priority = GROUP_PRIORITY[group]
            for target_go_id in targets:
                current_term_priorities.setdefault(target_go_id, set()).add(priority)

            if progress_every > 0 and rows_seen % progress_every == 0:
                print(
                    f"Scanned {rows_seen:,} annotations; retained "
                    f"{test_annotation_rows:,} held-out rows...",
                    flush=True,
                )
    flush_current_gene()
    return (
        term_group_counts,
        term_unique_counts,
        all_group_rows,
        test_group_rows,
        rows_seen,
        test_annotation_rows,
        test_genes_with_annotations,
    )


def source_assignment_table(
    term_group_counts: dict[str, list[int]],
    term_unique_counts: dict[str, int],
) -> pd.DataFrame:
    """Build a one-row-per-term provenance audit table."""

    rows: list[dict[str, object]] = []
    for go_id in sorted(term_group_counts):
        counts = term_group_counts[go_id]
        assigned = next(
            (group for group, count in zip(GROUP_ORDER, counts, strict=True) if count > 0),
            "unassigned",
        )
        total = term_unique_counts[go_id]
        assigned_count = (
            counts[GROUP_ORDER.index(assigned)] if assigned != "unassigned" else 0
        )
        row: dict[str, object] = {
            "GO_ID": go_id,
            "assigned_source_group": assigned,
            "assigned_source_label": GROUP_LABELS.get(assigned, "Unassigned"),
            "assigned_priority": GROUP_PRIORITY.get(assigned, ""),
            "test_gene_term_labels_with_provenance": total,
            "assigned_source_gene_term_labels": assigned_count,
            "assigned_source_fraction": assigned_count / total if total else np.nan,
            "source_groups_observed": sum(count > 0 for count in counts),
        }
        for group, count in zip(GROUP_ORDER, counts, strict=True):
            row[f"test_gene_term_labels_{group}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def expand_explicit_source_memberships(frame: pd.DataFrame) -> pd.DataFrame:
    """Place each GO term in every directly observed source family."""

    groups: list[pd.DataFrame] = []
    for group in GROUP_ORDER:
        count_column = f"test_gene_term_labels_{group}"
        subset = frame.loc[frame[count_column].gt(0)].copy()
        if subset.empty:
            continue
        subset["assigned_source_group"] = group
        subset["assigned_source_label"] = GROUP_LABELS[group]
        subset["assigned_priority"] = GROUP_PRIORITY[group]
        subset["assigned_source_gene_term_labels"] = subset[count_column]
        subset["assigned_source_fraction"] = (
            subset[count_column] / subset["test_gene_term_labels_with_provenance"]
        )
        groups.append(subset)
    if not groups:
        return frame.iloc[0:0].copy()
    return pd.concat(groups, ignore_index=True)


def summarize_sources(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "namespace",
                "assigned_priority",
                "assigned_source_group",
                "assigned_source_label",
            ]
        )
        .agg(
            terms=("GO_ID", "count"),
            dataset_gene_go_labels=("database_frequency", "sum"),
            test_positive_labels=("test_positive_count", "sum"),
            median_f1_at_global_threshold=("test_f1_at_global_threshold", "median"),
            f1_q25=("test_f1_at_global_threshold", lambda values: values.quantile(0.25)),
            f1_q75=("test_f1_at_global_threshold", lambda values: values.quantile(0.75)),
            median_per_term_fmax=("test_term_fmax", "median"),
            fmax_q25=("test_term_fmax", lambda values: values.quantile(0.25)),
            fmax_q75=("test_term_fmax", lambda values: values.quantile(0.75)),
        )
        .reset_index()
        .sort_values(["namespace", "assigned_priority"])
    )


def summarize_depth(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.dropna(subset=["go_graph_depth"])
        .groupby(
            [
                "assigned_priority",
                "assigned_source_group",
                "assigned_source_label",
                "go_graph_depth",
            ]
        )
        .agg(
            terms=("GO_ID", "count"),
            test_positive_labels=("test_positive_count", "sum"),
            median_f1_at_global_threshold=("test_f1_at_global_threshold", "median"),
            f1_q25=("test_f1_at_global_threshold", lambda values: values.quantile(0.25)),
            f1_q75=("test_f1_at_global_threshold", lambda values: values.quantile(0.75)),
            median_per_term_fmax=("test_term_fmax", "median"),
            fmax_q25=("test_term_fmax", lambda values: values.quantile(0.25)),
            fmax_q75=("test_term_fmax", lambda values: values.quantile(0.75)),
        )
        .reset_index()
        .sort_values(["assigned_priority", "go_graph_depth"])
    )


def plot_source_box(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    xlabel: str,
) -> None:
    present = [group for group in GROUP_ORDER if group in set(frame["assigned_source_group"])]
    positions = [GROUP_ORDER.index(group) for group in present]
    values = [
        frame.loc[frame["assigned_source_group"].eq(group), metric].dropna()
        for group in present
    ]
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.35},
        whiskerprops={"color": "#777777", "linewidth": 0.9},
        capprops={"color": "#777777", "linewidth": 0.9},
    )
    for patch, group in zip(box["boxes"], present, strict=True):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.28)
        patch.set_edgecolor(GROUP_COLORS[group])
    group_sizes = frame["assigned_source_group"].value_counts().to_dict()
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(
        [
            f"{GROUP_TICK_LABELS[group]}\n(n={int(group_sizes.get(group, 0)):,})"
            for group in GROUP_ORDER
        ]
    )
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_xlim(-0.65, len(GROUP_ORDER) - 0.35)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)


def plot_depth_sources(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    median_col: str,
    q25_col: str,
    q75_col: str,
    title: str,
    ylabel: str,
) -> None:
    for group in GROUP_ORDER:
        subset = summary.loc[summary["assigned_source_group"].eq(group)].copy()
        if subset.empty:
            continue
        x = subset["go_graph_depth"].to_numpy(dtype=float)
        median = subset[median_col].to_numpy(dtype=float)
        q25 = subset[q25_col].to_numpy(dtype=float)
        q75 = subset[q75_col].to_numpy(dtype=float)
        color = GROUP_COLORS[group]
        ax.fill_between(x, q25, q75, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x,
            median,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label=GROUP_LABELS[group],
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("GO graph depth")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.35, alpha=0.85)


NAMESPACE_FIGURES = [
    ("biological_process", "11a", "biological_process", "Biological process"),
    ("molecular_function", "11b", "molecular_function", "Molecular function"),
    ("cellular_component", "11c", "cellular_component", "Cellular component"),
]


def plot_figures(
    frame: pd.DataFrame,
    source_summary: pd.DataFrame,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    annotation_label_mode: str,
    annotation_rows: int,
    test_annotation_rows: int,
    test_gene_count: int,
    dpi: int,
) -> list[Path]:
    records = int(float(publication_summary["records_evaluated"]))
    global_threshold = float(publication_summary["global_threshold"])
    threshold_source = publication_summary.get("global_threshold_source", "test_fmax")
    source_axis_label = (
        "Annotation source"
        if annotation_label_mode == "explicit"
        else "Hierarchical annotation source"
    )
    if threshold_source == "prespecified":
        threshold_title = "validation-selected threshold"
        threshold_caption = (
            f"the validation-selected threshold of {global_threshold:.2f}"
        )
    else:
        threshold_title = "test-selected global threshold"
        threshold_caption = (
            f"the test-selected gene-averaged threshold of {global_threshold:.3f}"
        )
    stems: list[Path] = []
    for namespace, figure_number, filename_namespace, namespace_label in NAMESPACE_FIGURES:
        namespace_frame = frame.loc[frame["namespace"].eq(namespace)].copy()
        namespace_summary = source_summary.loc[source_summary["namespace"].eq(namespace)]
        fig, axes = plt.subplots(1, 2, figsize=(11.20, 3.35))
        plot_source_box(
            axes[0],
            namespace_frame,
            "test_f1_at_global_threshold",
            title=f"A  {namespace_label}: {threshold_title}",
            ylabel="GO-term F1",
            xlabel=source_axis_label,
        )
        plot_source_box(
            axes[1],
            namespace_frame,
            "test_term_fmax",
            title=f"B  {namespace_label}: per-term optimum",
            ylabel="GO-term Fmax",
            xlabel=source_axis_label,
        )
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.31, top=0.91, wspace=0.24)

        stem = output_dir / f"figure_{figure_number}_{filename_namespace}_annotation_source_performance"
        save_figure(fig, stem, dpi=dpi)
        stems.append(stem)

        group_counts = {
            row["assigned_source_group"]: int(row["terms"])
            for row in namespace_summary.to_dict(orient="records")
        }
        count_text = ", ".join(
            f"{GROUP_LABELS[group]}={group_counts.get(group, 0):,}"
            for group in GROUP_ORDER
        )
        unique_term_count = int(namespace_frame["GO_ID"].nunique())
        if annotation_label_mode == "explicit":
            provenance_text = (
                "Evaluation targets and evidence provenance were restricted to the "
                "original processed Gene2GO gene--term pairs; no is_a or part_of "
                "ancestor labels were added. Each GO term was included in every "
                "evidence family directly represented among its positive held-out "
                "gene annotations, so source groups are not mutually exclusive"
            )
            analysis_label = "explicit Gene2GO label"
        else:
            provenance_text = (
                "Each explicit annotation carried its evidence family through is_a "
                "and part_of true-path propagation. When multiple annotations "
                "supported the same propagated gene--term label, one mutually "
                "exclusive family was chosen using the prespecified hierarchy"
            )
            analysis_label = "true-path label"
        write_caption(
            stem.with_name(stem.name + "_caption.txt"),
            (
                f"Figure {figure_number}. {namespace_label} GO-term performance "
                f"stratified by {analysis_label} provenance. "
                f"Panel A shows GO-term F1 at {threshold_caption}; panel B shows "
                "each GO term's independently test-optimized Fmax. Evidence "
                f"provenance was reconstructed from {test_annotation_rows:,} rows "
                "matching the held-out test-gene population in the original "
                "processed Gene2GO table "
                f"({annotation_rows:,} total rows). {provenance_text}: conventional "
                "experimental [EXP, IDA, IPI, "
                "IMP, IGI, IEP]; high-throughput experimental [HTP, HDA, HMP, HGI, "
                "HEP]; curated computational/phylogenetic [IBA, IBD, IKR, IRD, ISS, "
                "ISO, ISA, ISM, IGC, RCA]; author/curator statements [TAS, NAS, IC, "
                "ND]; and IEA-only electronic annotation. Boxes show "
                "the distribution across GO terms represented in each source family "
                f"({count_text}; {unique_term_count:,} unique terms). The metrics "
                f"use {records:,} MMseqs-cluster test sequences. Per-term Fmax is a "
                "test-set-optimized analysis. The three ontology root terms were excluded."
            ),
        )
    return stems


def plot_combined_figure(
    frame: pd.DataFrame,
    publication_summary: dict[str, str],
    output_dir: Path,
    *,
    annotation_label_mode: str,
    annotation_rows: int,
    test_annotation_rows: int,
    dpi: int,
) -> Path:
    """Combine the three namespace views into one publication figure."""

    global_threshold = float(publication_summary["global_threshold"])
    records = int(float(publication_summary["records_evaluated"]))
    threshold_source = publication_summary.get("global_threshold_source", "test_fmax")
    if threshold_source == "prespecified":
        threshold_column_title = (
            f"F1 at validation-selected threshold ({global_threshold:.2f})"
        )
        threshold_caption = f"the validation-selected threshold of {global_threshold:.2f}"
    else:
        threshold_column_title = (
            f"F1 at test-selected global threshold ({global_threshold:.3f})"
        )
        threshold_caption = f"the test-selected gene-averaged threshold of {global_threshold:.3f}"
    source_axis_label = "Annotation source"
    combined_order = [
        ("molecular_function", "A", "Molecular function"),
        ("biological_process", "B", "Biological process"),
        ("cellular_component", "C", "Cellular component"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 9.2), sharey=True)
    for row_index, (namespace, panel_letter, namespace_label) in enumerate(combined_order):
        namespace_frame = frame.loc[frame["namespace"].eq(namespace)].copy()
        plot_source_box(
            axes[row_index, 0],
            namespace_frame,
            "test_f1_at_global_threshold",
            title="",
            ylabel="",
            xlabel="",
        )
        plot_source_box(
            axes[row_index, 1],
            namespace_frame,
            "test_term_fmax",
            title="",
            ylabel="",
            xlabel="",
        )
        axes[row_index, 1].tick_params(axis="y", labelleft=False)
        axes[row_index, 0].text(
            0.0,
            1.035,
            f"{panel_letter}  {namespace_label}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontweight="semibold",
            fontsize=9,
            color=INK,
        )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.93,
        hspace=0.68,
        wspace=0.10,
    )
    for column_index, column_title in enumerate(
        (threshold_column_title, "Per-term optimal Fmax")
    ):
        position = axes[0, column_index].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.982,
            column_title,
            ha="center",
            va="top",
            fontweight="semibold",
            fontsize=9,
            color=INK,
        )
    fig.supxlabel(source_axis_label, y=0.025)
    fig.supylabel("GO-term score", x=0.015)

    stem = output_dir / "figure_11_annotation_source_performance"
    save_figure(fig, stem, dpi=dpi)
    write_caption(
        stem.with_name(stem.name + "_caption.txt"),
        (
            "Figure 11. GO-term performance by annotation source for (A) molecular "
            "function, (B) biological process, and (C) cellular component. The left "
            f"column shows GO-term F1 at {threshold_caption}; the right column shows "
            "each GO term's independently test-optimized Fmax. Only original "
            "processed Gene2GO gene--term annotations were used; is_a and part_of "
            "ancestor terms were not propagated. Evidence codes were grouped as "
            "conventional experimental (EXP, IDA, IPI, IMP, IGI, IEP), "
            "high-throughput experimental (HTP, HDA, HMP, HGI, HEP), "
            "computational/phylogenetic (IBA, IBD, IKR, IRD, ISS, ISO, ISA, ISM, "
            "IGC, RCA), author/curator (TAS, NAS, IC, ND), and electronic (IEA). A "
            "GO term appears in every source group directly represented among its "
            "held-out annotations, so groups are non-exclusive. Here, n is the "
            "number of distinct GO terms in each box, not the number of genes or "
            "annotation records. Boxes show medians and interquartile ranges; "
            "whiskers extend to 1.5 times the interquartile range. The three GO "
            "ontology root terms were excluded."
        ),
    )
    return stem


def update_manifest(path: Path, figure_stem: Path) -> None:
    rows: list[tuple[str, str]] = []
    replaced_metrics = {
        "figure_11_go_evidence_source_depth",
        "figure_11a_biological_process_annotation_source",
        "figure_11b_molecular_function_annotation_source",
        "figure_11c_cellular_component_annotation_source",
        "figure_11_annotation_source_performance",
    }
    if path.is_file():
        with path.open("rt", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("metric") not in replaced_metrics:
                    rows.append((row.get("metric", ""), row.get("value", "")))
    rows.append(("figure_11_annotation_source_performance", str(figure_stem.with_suffix(".png"))))
    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-term-metrics",
        type=Path,
        default=DEFAULT_EVAL_DIR / "test_publication_go_term_metrics.tsv",
    )
    parser.add_argument(
        "--publication-summary",
        type=Path,
        default=DEFAULT_EVAL_DIR / "test_publication_summary.tsv",
    )
    parser.add_argument(
        "--test-gene-metrics",
        type=Path,
        default=DEFAULT_TEST_GENE_METRICS,
        help="Held-out per-gene metrics used to identify the exact test gene universe.",
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--annotation-label-mode",
        choices=("true-path", "explicit"),
        default="true-path",
        help="Propagate annotation provenance to ancestors or retain direct terms only.",
    )
    parser.add_argument(
        "--reuse-provenance-counts",
        action="store_true",
        help="Reuse the previously audited per-term gene-label source counts.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.per_term_metrics, sep="\t")
    for column in (
        "database_frequency",
        "test_positive_count",
        "test_f1_at_global_threshold",
        "test_term_fmax",
    ):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics.loc[
        ~metrics["GO_ID"].isin(GO_ROOT_IDS)
        & (metrics["test_positive_count"] > 0)
        & metrics["test_f1_at_global_threshold"].notna()
        & metrics["test_term_fmax"].notna()
    ].copy()
    retained_go_ids = set(metrics["GO_ID"].astype(str))
    publication_summary = read_key_value_tsv(args.publication_summary)
    stem = args.output_dir / "figure_11_go_evidence_source_performance"
    counts_path = stem.with_name(stem.name + "_gene_term_source_counts.tsv")
    definitions_path = stem.with_name(stem.name + "_evidence_group_definition.tsv")

    if args.reuse_provenance_counts:
        cached = pd.read_csv(counts_path, sep="\t")
        cached = cached.set_index("GO_ID")
        if set(cached.index) != retained_go_ids:
            raise ValueError("Cached provenance GO-term universe does not match evaluation metrics")
        term_group_counts = {
            go_id: [
                int(cached.at[go_id, f"test_gene_term_labels_{group}"])
                for group in GROUP_ORDER
            ]
            for go_id in sorted(retained_go_ids)
        }
        term_unique_counts = {
            go_id: int(cached.at[go_id, "test_gene_term_labels_with_provenance"])
            for go_id in sorted(retained_go_ids)
        }
        definitions = pd.read_csv(definitions_path, sep="\t").set_index("source_group")
        annotation_group_rows = Counter(
            {
                group: int(definitions.at[group, "source_annotation_rows"])
                for group in GROUP_ORDER
            }
        )
        test_annotation_group_rows = Counter(
            {
                group: int(definitions.at[group, "held_out_source_annotation_rows"])
                for group in GROUP_ORDER
            }
        )
        annotation_rows = sum(annotation_group_rows.values())
        test_annotation_rows = sum(test_annotation_group_rows.values())
        test_gene_count = int(float(publication_summary["records_evaluated"]))
        test_genes_with_annotations = test_gene_count
    else:
        parents_by_id, alt_to_primary, _obsolete = parse_obo_graph(args.go_basic_obo)
        test_gene_identities, test_gene_count = load_test_gene_identities(
            args.test_gene_metrics,
            progress_every=max(args.progress_every // 10, 1),
        )
        (
            term_group_counts,
            term_unique_counts,
            annotation_group_rows,
            test_annotation_group_rows,
            annotation_rows,
            test_annotation_rows,
            test_genes_with_annotations,
        ) = scan_test_gene_term_sources(
            args.annotations,
            test_gene_identities,
            retained_go_ids,
            parents_by_id,
            alt_to_primary,
            annotation_label_mode=args.annotation_label_mode,
            progress_every=args.progress_every,
        )
    if test_genes_with_annotations != test_gene_count:
        raise ValueError(
            f"Only {test_genes_with_annotations:,} of {test_gene_count:,} held-out genes "
            "had source annotations"
        )
    source_assignments = source_assignment_table(term_group_counts, term_unique_counts)
    unassigned = source_assignments.loc[
        source_assignments["assigned_source_group"].eq("unassigned"), "GO_ID"
    ].tolist()
    if unassigned:
        preview = ", ".join(unassigned[:10])
        raise ValueError(f"No source-evidence provenance found for {len(unassigned)} terms: {preview}")

    base_frame = metrics.merge(
        source_assignments,
        on="GO_ID",
        how="left",
        validate="one_to_one",
    )
    base_frame["provenance_count_difference"] = (
        base_frame["test_gene_term_labels_with_provenance"]
        - base_frame["test_positive_count"]
    )
    mismatched = base_frame.loc[base_frame["provenance_count_difference"].ne(0)]
    if not mismatched.empty:
        preview = ", ".join(
            f"{row.GO_ID}:{int(row.provenance_count_difference):+d}"
            for row in mismatched.head(10).itertuples()
        )
        raise ValueError(
            f"Provenance reconstruction disagrees with evaluation targets for "
            f"{len(mismatched):,} GO terms ({preview})"
        )
    frame = (
        expand_explicit_source_memberships(base_frame)
        if args.annotation_label_mode == "explicit"
        else base_frame
    )
    source_summary = summarize_sources(frame)

    source_assignments.to_csv(
        stem.with_name(stem.name + "_gene_term_source_counts.tsv"),
        sep="\t",
        index=False,
    )
    frame.to_csv(
        stem.with_name(stem.name + "_term_table.tsv"),
        sep="\t",
        index=False,
    )
    source_summary.to_csv(
        stem.with_name(stem.name + "_source_summary.tsv"),
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "priority": GROUP_PRIORITY[group],
                "source_group": group,
                "source_label": GROUP_LABELS[group],
                "evidence_codes": ",".join(GROUP_CODES[group]),
                "source_annotation_rows": int(annotation_group_rows[group]),
                "held_out_source_annotation_rows": int(test_annotation_group_rows[group]),
            }
            for group in GROUP_ORDER
        ]
    ).to_csv(
        stem.with_name(stem.name + "_evidence_group_definition.tsv"),
        sep="\t",
        index=False,
    )

    set_publication_style()
    plot_stems = plot_figures(
        frame,
        source_summary,
        publication_summary,
        args.output_dir,
        annotation_label_mode=args.annotation_label_mode,
        annotation_rows=annotation_rows,
        test_annotation_rows=test_annotation_rows,
        test_gene_count=test_gene_count,
        dpi=args.dpi,
    )
    combined_stem = plot_combined_figure(
        frame,
        publication_summary,
        args.output_dir,
        annotation_label_mode=args.annotation_label_mode,
        annotation_rows=annotation_rows,
        test_annotation_rows=test_annotation_rows,
        dpi=args.dpi,
    )
    update_manifest(
        args.output_dir / "additional_publication_benchmark_manifest.tsv",
        combined_stem,
    )
    for plot_stem in plot_stems:
        print(f"Wrote {plot_stem.with_suffix('.png')}")
        print(f"Wrote {plot_stem.with_suffix('.pdf')}")
        print(f"Wrote {plot_stem.with_name(plot_stem.name + '_caption.txt')}")
    print(f"Wrote {combined_stem.with_suffix('.png')}")
    print(f"Wrote {combined_stem.with_suffix('.pdf')}")
    print(f"Wrote {combined_stem.with_name(combined_stem.name + '_caption.txt')}")


if __name__ == "__main__":
    main()
