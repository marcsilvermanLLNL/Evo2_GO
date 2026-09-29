#!/usr/bin/env python3
"""Build true-path-consistent GO labels for selected Evo2-GO genes.

This script leaves the raw gene2go-derived annotation artifacts untouched. It
streams the evidence-bearing positive annotation table, keeps annotations for a
selected set of (tax_id, GeneID) keys, and expands every retained positive GO
term to its is_a and part_of ancestors from go-basic.obo.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_SELECTED_GENES = Path(
    "outputs/model_training/non_iea_min_count_10/non_iea_embeddable_gene_keys.tsv"
)
DEFAULT_ANNOTATIONS = Path(
    "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz"
)
DEFAULT_GO_BASIC_OBO = Path("data/go/go-basic.obo")
DEFAULT_OUTPUT_DIR = Path("outputs/model_training/non_iea_genes_all_labels_true_path")


def log_progress(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def clean_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def open_text(path: Path) -> Iterator[object]:
    if ".gz" in path.suffixes:
        return gzip.open(path, "rt", newline="")
    return path.open("rt", newline="")


def open_output_text(path: Path) -> Iterator[object]:
    if ".gz" in path.suffixes or path.name.endswith(".gz.tmp"):
        return gzip.open(path, "wt", newline="")
    return path.open("wt", newline="")


def numeric_sort_value(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def gene_sort_key(key: tuple[str, str]) -> tuple[tuple[int, int | str], tuple[int, int | str]]:
    return (numeric_sort_value(key[0]), numeric_sort_value(key[1]))


def first_present(fieldnames: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def parse_evidence_codes(raw_values: list[str]) -> set[str] | None:
    if not raw_values:
        return None

    codes: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip().upper()
            if part:
                codes.add(part)
    return codes


def evidence_is_retained(
    evidence: str,
    allowed_evidence: set[str] | None,
    *,
    include_iea: bool,
) -> bool:
    evidence = evidence.upper()
    if include_iea:
        return allowed_evidence is None or evidence in allowed_evidence
    if allowed_evidence is None:
        return evidence != "IEA"
    return evidence in allowed_evidence


def evidence_filter_label(allowed_evidence: set[str] | None, *, include_iea: bool) -> str:
    if allowed_evidence is not None:
        return ",".join(sorted(allowed_evidence))
    return "all_evidence_including_iea" if include_iea else "non_iea"


def load_selected_gene_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Selected gene table not found: {path}")

    keys: set[tuple[str, str]] = set()
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        tax_col = first_present(fieldnames, ("tax_id", "taxid", "TaxID"))
        gene_col = first_present(fieldnames, ("GeneID", "gene_id", "geneid"))
        if tax_col is None or gene_col is None:
            raise ValueError(
                f"{path} must contain tax_id and GeneID columns; observed {fieldnames}"
            )

        for row in reader:
            tax_id = clean_field(row[tax_col])
            gene_id = clean_field(row[gene_col])
            if tax_id and gene_id:
                keys.add((tax_id, gene_id))

    if not keys:
        raise ValueError(f"No selected (tax_id, GeneID) keys found in {path}")
    return keys


def is_obsolete(term: Any | None) -> bool:
    return bool(getattr(term, "is_obsolete", False)) if term is not None else False


def direct_true_path_parents(term: Any) -> list[str]:
    parents: list[str] = []
    for parent in getattr(term, "parents", []):
        parents.append(parent.id)

    relationships = getattr(term, "relationship", {}) or {}
    for part_of_target in relationships.get("part_of", []):
        parents.append(part_of_target.id)

    return parents


def collect_true_path_ancestors(
    go_id: str,
    go_dag: GODag,
    cache: dict[str, frozenset[str]],
    visiting: set[str] | None = None,
) -> frozenset[str]:
    """Return all is_a/part_of ancestors for go_id, excluding go_id itself."""

    if go_id in cache:
        return cache[go_id]

    term = go_dag.get(go_id)
    if term is None or is_obsolete(term):
        cache[go_id] = frozenset()
        return cache[go_id]

    if visiting is None:
        visiting = set()
    if go_id in visiting:
        cache[go_id] = frozenset()
        return cache[go_id]

    visiting.add(go_id)
    ancestors: set[str] = set()
    for parent_id in direct_true_path_parents(term):
        parent = go_dag.get(parent_id)
        if parent is None or is_obsolete(parent):
            continue
        ancestors.add(parent_id)
        ancestors.update(collect_true_path_ancestors(parent_id, go_dag, cache, visiting))
    visiting.remove(go_id)

    cache[go_id] = frozenset(ancestors)
    return cache[go_id]


def load_go_vocabulary(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"GO vocabulary not found: {path}")

    go_ids: list[str] = []
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "GO_ID" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required column GO_ID")
        for row in reader:
            go_id = clean_field(row["GO_ID"])
            if go_id:
                go_ids.append(go_id)

    if not go_ids:
        raise ValueError(f"No GO IDs loaded from {path}")
    return tuple(go_ids), {go_id: index for index, go_id in enumerate(go_ids)}


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "gene_labels": output_dir / "true_path_gene_go_labels.tsv.gz",
        "gene_metrics": output_dir / "true_path_gene_label_metrics.tsv.gz",
        "term_counts": output_dir / "true_path_go_term_counts.tsv",
        "summary": output_dir / "true_path_label_summary.tsv",
        "policy": output_dir / "true_path_label_policy.tsv",
        "missing_go": output_dir / "true_path_missing_go_ids.tsv",
        "label_cache": output_dir / "label_index.tsv.gz",
    }


def refuse_overwrite(paths: Iterable[Path], force: bool) -> None:
    if force:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing true-path outputs without --force:\n"
            f"{formatted}"
        )


def write_metric_table(path: Path, metrics: dict[str, object]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        for key in sorted(metrics):
            writer.writerow((key, metrics[key]))
    tmp_path.replace(path)


def write_missing_go_ids(path: Path, missing_counts: Counter[str]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("GO_ID", "explicit_gene_go_occurrences"))
        for go_id, count in sorted(missing_counts.items()):
            writer.writerow((go_id, count))
    tmp_path.replace(path)


def write_label_cache(
    path: Path,
    label_index: dict[tuple[str, str], tuple[int, ...]],
    vocabulary_go_ids: tuple[str, ...],
) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with gzip.open(tmp_path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("tax_id", "GeneID", "go_indices", "go_ids"))
        for key in sorted(label_index, key=gene_sort_key):
            indices = label_index[key]
            go_ids = [vocabulary_go_ids[index] for index in indices]
            writer.writerow((key[0], key[1], ",".join(map(str, indices)), ",".join(go_ids)))
    tmp_path.replace(path)


def write_outputs(
    *,
    paths: dict[str, Path],
    selected_keys: set[tuple[str, str]],
    explicit_labels: dict[tuple[str, str], set[str]],
    go_dag: GODag,
    ancestor_cache: dict[str, frozenset[str]],
    missing_go_counts: Counter[str],
    annotation_metrics: dict[str, object],
    policy_metrics: dict[str, object],
    vocabulary: tuple[tuple[str, ...], dict[str, int]] | None,
    write_gene_labels: bool,
) -> None:
    term_total_counts: Counter[str] = Counter()
    term_explicit_counts: Counter[str] = Counter()
    term_inferred_counts: Counter[str] = Counter()
    term_explicit_only_counts: Counter[str] = Counter()
    term_inferred_only_counts: Counter[str] = Counter()
    term_both_counts: Counter[str] = Counter()

    selected_genes_with_explicit_labels = 0
    selected_genes_with_true_path_labels = 0
    genes_with_added_ancestor_labels = 0
    explicit_gene_go_pairs = 0
    true_path_gene_go_pairs = 0
    inferred_ancestor_gene_go_pairs = 0
    explicit_only_gene_go_pairs = 0
    inferred_only_gene_go_pairs = 0
    explicit_and_inferred_gene_go_pairs = 0

    label_index: dict[tuple[str, str], tuple[int, ...]] = {}
    vocabulary_go_ids: tuple[str, ...] = ()
    go_to_index: dict[str, int] = {}
    label_cache_labels_dropped_outside_vocabulary = 0
    if vocabulary is not None:
        vocabulary_go_ids, go_to_index = vocabulary

    gene_label_tmp = paths["gene_labels"].with_name(paths["gene_labels"].name + ".tmp")
    gene_metrics_tmp = paths["gene_metrics"].with_name(paths["gene_metrics"].name + ".tmp")

    gene_label_handle = None
    gene_label_writer = None
    try:
        if write_gene_labels:
            gene_label_handle = open_output_text(gene_label_tmp)
            gene_label_writer = csv.writer(
                gene_label_handle,
                delimiter="\t",
                lineterminator="\n",
            )
            gene_label_writer.writerow(
                ("tax_id", "GeneID", "GO_ID", "label_value", "label_type", "label_policy")
            )

        with open_output_text(gene_metrics_tmp) as metrics_handle:
            gene_metrics_writer = csv.writer(
                metrics_handle,
                delimiter="\t",
                lineterminator="\n",
            )
            gene_metrics_writer.writerow(
                (
                    "tax_id",
                    "GeneID",
                    "explicit_go_count",
                    "true_path_go_count",
                    "inferred_ancestor_go_count",
                    "explicit_only_go_count",
                    "inferred_only_go_count",
                    "explicit_and_inferred_go_count",
                    "missing_explicit_go_count",
                )
            )

            for key in sorted(selected_keys, key=gene_sort_key):
                explicit = explicit_labels.get(key, set())
                missing_explicit = {go_id for go_id in explicit if go_dag.get(go_id) is None}
                inferred: set[str] = set()
                for go_id in explicit:
                    inferred.update(collect_true_path_ancestors(go_id, go_dag, ancestor_cache))

                true_path = explicit | inferred
                both = explicit & inferred
                explicit_only = explicit - inferred
                inferred_only = inferred - explicit

                selected_genes_with_explicit_labels += int(bool(explicit))
                selected_genes_with_true_path_labels += int(bool(true_path))
                genes_with_added_ancestor_labels += int(bool(inferred_only))
                explicit_gene_go_pairs += len(explicit)
                true_path_gene_go_pairs += len(true_path)
                inferred_ancestor_gene_go_pairs += len(inferred)
                explicit_only_gene_go_pairs += len(explicit_only)
                inferred_only_gene_go_pairs += len(inferred_only)
                explicit_and_inferred_gene_go_pairs += len(both)

                if vocabulary is not None:
                    indices: list[int] = []
                    for go_id in sorted(true_path):
                        index = go_to_index.get(go_id)
                        if index is None:
                            label_cache_labels_dropped_outside_vocabulary += 1
                        else:
                            indices.append(index)
                    if indices:
                        label_index[key] = tuple(sorted(set(indices)))

                gene_metrics_writer.writerow(
                    (
                        key[0],
                        key[1],
                        len(explicit),
                        len(true_path),
                        len(inferred),
                        len(explicit_only),
                        len(inferred_only),
                        len(both),
                        len(missing_explicit),
                    )
                )

                for go_id in true_path:
                    term_total_counts[go_id] += 1
                for go_id in explicit:
                    term_explicit_counts[go_id] += 1
                for go_id in inferred:
                    term_inferred_counts[go_id] += 1
                for go_id in explicit_only:
                    term_explicit_only_counts[go_id] += 1
                for go_id in inferred_only:
                    term_inferred_only_counts[go_id] += 1
                for go_id in both:
                    term_both_counts[go_id] += 1

                if gene_label_writer is not None:
                    for go_id in sorted(true_path):
                        if go_id in both:
                            label_type = "explicit_and_inferred"
                        elif go_id in explicit:
                            label_type = "explicit"
                        else:
                            label_type = "inferred_ancestor"
                        gene_label_writer.writerow(
                            (key[0], key[1], go_id, 1, label_type, policy_metrics["label_policy"])
                        )
    finally:
        if gene_label_handle is not None:
            gene_label_handle.close()

    if write_gene_labels:
        gene_label_tmp.replace(paths["gene_labels"])
    gene_metrics_tmp.replace(paths["gene_metrics"])

    if not term_total_counts:
        raise ValueError("No true-path GO labels were produced")

    term_counts_tmp = paths["term_counts"].with_name(paths["term_counts"].name + ".tmp")
    with term_counts_tmp.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "GO_ID",
                "trainable_gene_count",
                "explicit_gene_count",
                "inferred_ancestor_gene_count",
                "explicit_only_gene_count",
                "inferred_only_gene_count",
                "explicit_and_inferred_gene_count",
                "total_trainable_label_rows",
                "total_explicit_gene_go_pairs",
                "total_inferred_ancestor_gene_go_pairs",
                "total_trainable_genes",
                "selected_genes_with_true_path_labels",
                "selected_genes_with_added_ancestor_labels",
                "annotation_rows_seen",
                "evidence_filter",
                "label_policy",
            )
        )
        for go_id in sorted(term_total_counts):
            writer.writerow(
                (
                    go_id,
                    term_total_counts[go_id],
                    term_explicit_counts[go_id],
                    term_inferred_counts[go_id],
                    term_explicit_only_counts[go_id],
                    term_inferred_only_counts[go_id],
                    term_both_counts[go_id],
                    true_path_gene_go_pairs,
                    explicit_gene_go_pairs,
                    inferred_ancestor_gene_go_pairs,
                    len(selected_keys),
                    selected_genes_with_true_path_labels,
                    genes_with_added_ancestor_labels,
                    annotation_metrics["annotation_rows_seen"],
                    policy_metrics["evidence_filter"],
                    policy_metrics["label_policy"],
                )
            )
    term_counts_tmp.replace(paths["term_counts"])

    if vocabulary is not None:
        write_label_cache(paths["label_cache"], label_index, vocabulary_go_ids)

    summary = {
        **annotation_metrics,
        **policy_metrics,
        "selected_genes_total": len(selected_keys),
        "selected_genes_with_explicit_labels": selected_genes_with_explicit_labels,
        "selected_genes_with_true_path_labels": selected_genes_with_true_path_labels,
        "selected_genes_with_added_ancestor_labels": genes_with_added_ancestor_labels,
        "explicit_unique_gene_go_pairs": explicit_gene_go_pairs,
        "true_path_unique_gene_go_pairs": true_path_gene_go_pairs,
        "inferred_ancestor_gene_go_pairs": inferred_ancestor_gene_go_pairs,
        "explicit_only_gene_go_pairs": explicit_only_gene_go_pairs,
        "inferred_only_gene_go_pairs": inferred_only_gene_go_pairs,
        "explicit_and_inferred_gene_go_pairs": explicit_and_inferred_gene_go_pairs,
        "added_gene_go_pairs_after_true_path_closure": true_path_gene_go_pairs - explicit_gene_go_pairs,
        "go_terms_explicit": len(term_explicit_counts),
        "go_terms_true_path": len(term_total_counts),
        "missing_explicit_go_ids": len(missing_go_counts),
        "missing_explicit_gene_go_occurrences": sum(missing_go_counts.values()),
        "ancestor_cache_entries": len(ancestor_cache),
        "gene_label_table_written": int(write_gene_labels),
        "label_cache_written": int(vocabulary is not None),
        "label_cache_genes_written": len(label_index) if vocabulary is not None else 0,
        "label_cache_gene_go_labels": sum(len(indices) for indices in label_index.values())
        if vocabulary is not None
        else 0,
        "label_cache_labels_dropped_outside_vocabulary": label_cache_labels_dropped_outside_vocabulary,
    }
    write_metric_table(paths["summary"], summary)
    write_metric_table(paths["policy"], policy_metrics)
    write_missing_go_ids(paths["missing_go"], missing_go_counts)


def stream_explicit_labels(
    *,
    annotations: Path,
    selected_keys: set[tuple[str, str]],
    allowed_evidence: set[str] | None,
    include_iea: bool,
    progress_every: int,
) -> tuple[dict[tuple[str, str], set[str]], Counter[str], dict[str, object]]:
    if not annotations.is_file():
        raise FileNotFoundError(f"Annotation table not found: {annotations}")

    explicit_labels: dict[tuple[str, str], set[str]] = {}
    missing_go_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    annotation_rows_seen = 0
    retained_annotation_rows = 0
    retained_duplicate_gene_go_rows = 0
    started = time.time()

    with gzip.open(annotations, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"tax_id", "GeneID", "GO_ID", "Evidence"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{annotations} is missing required columns: {sorted(missing)}")

        for annotation_rows_seen, row in enumerate(reader, start=1):
            if progress_every > 0 and annotation_rows_seen % progress_every == 0:
                elapsed = time.time() - started
                explicit_pairs = sum(len(values) for values in explicit_labels.values())
                log_progress(
                    f"Scanned {annotation_rows_seen:,} annotation rows; "
                    f"retained {explicit_pairs:,} unique explicit selected gene-GO pairs; "
                    f"elapsed {elapsed / 60:.1f} min"
                )

            evidence = clean_field(row["Evidence"]).upper()
            if not evidence_is_retained(
                evidence,
                allowed_evidence,
                include_iea=include_iea,
            ):
                continue

            key = (clean_field(row["tax_id"]), clean_field(row["GeneID"]))
            if key not in selected_keys:
                continue

            go_id = clean_field(row["GO_ID"])
            if not go_id:
                continue

            evidence_counts[evidence] += 1
            retained_annotation_rows += 1
            values = explicit_labels.setdefault(key, set())
            before = len(values)
            values.add(go_id)
            retained_duplicate_gene_go_rows += int(len(values) == before)

    explicit_pairs = sum(len(values) for values in explicit_labels.values())
    metrics = {
        "annotation_rows_seen": annotation_rows_seen,
        "retained_selected_annotation_rows": retained_annotation_rows,
        "retained_duplicate_gene_go_rows": retained_duplicate_gene_go_rows,
        "retained_unique_explicit_gene_go_pairs": explicit_pairs,
        "selected_genes_with_any_retained_annotation": len(explicit_labels),
        "retained_evidence_codes": ",".join(
            f"{code}:{evidence_counts[code]}" for code in sorted(evidence_counts)
        ),
    }
    return explicit_labels, missing_go_counts, metrics


def count_missing_explicit_go_ids(
    explicit_labels: dict[tuple[str, str], set[str]],
    go_dag: GODag,
) -> Counter[str]:
    missing_counts: Counter[str] = Counter()
    for go_ids in explicit_labels.values():
        for go_id in go_ids:
            if go_dag.get(go_id) is None:
                missing_counts[go_id] += 1
    return missing_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-genes", type=Path, default=DEFAULT_SELECTED_GENES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--label-policy",
        default="true_path_closure",
        help="Short label-policy name written into output metrics and label rows.",
    )
    parser.add_argument(
        "--allowed-evidence",
        action="append",
        default=[],
        help=(
            "Optional comma-delimited evidence allow-list. "
            "Default keeps every evidence code except IEA."
        ),
    )
    parser.add_argument(
        "--include-iea",
        action="store_true",
        help="Include IEA annotations. Without this flag, IEA annotations are excluded.",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        help=(
            "Optional GO vocabulary TSV. When provided, also writes label_index.tsv.gz "
            "with true-path labels restricted to that vocabulary."
        ),
    )
    parser.add_argument(
        "--no-write-gene-labels",
        action="store_true",
        help="Skip writing the expanded gene-GO label table. Counts and metrics are still written.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.time()
    paths = output_paths(args.output_dir)
    requested_outputs = [
        paths["gene_metrics"],
        paths["term_counts"],
        paths["summary"],
        paths["policy"],
        paths["missing_go"],
    ]
    if not args.no_write_gene_labels:
        requested_outputs.append(paths["gene_labels"])
    if args.vocabulary is not None:
        requested_outputs.append(paths["label_cache"])
    refuse_overwrite(requested_outputs, args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowed_evidence = parse_evidence_codes(args.allowed_evidence)
    evidence_label = evidence_filter_label(allowed_evidence, include_iea=args.include_iea)
    policy_metrics = {
        "label_policy": args.label_policy,
        "label_semantics": "explicit_gene2go_terms_plus_is_a_and_part_of_ancestors",
        "label_type_values": "explicit,inferred_ancestor,explicit_and_inferred",
        "true_path_relationships": "is_a,part_of",
        "evidence_filter": evidence_label,
        "include_iea": int(args.include_iea),
        "selected_genes_path": args.selected_genes,
        "annotations_path": args.annotations,
        "go_basic_obo_path": args.go_basic_obo,
        "output_dir": args.output_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.vocabulary is not None:
        policy_metrics["vocabulary_path"] = args.vocabulary

    log_progress(f"Loading selected genes: {args.selected_genes}")
    selected_keys = load_selected_gene_keys(args.selected_genes)
    log_progress(f"Loaded {len(selected_keys):,} selected (tax_id, GeneID) keys")

    log_progress(f"Streaming annotations: {args.annotations}")
    explicit_labels, missing_go_counts, annotation_metrics = stream_explicit_labels(
        annotations=args.annotations,
        selected_keys=selected_keys,
        allowed_evidence=allowed_evidence,
        include_iea=args.include_iea,
        progress_every=args.progress_every,
    )

    log_progress(f"Loading go-basic ontology: {args.go_basic_obo}")
    from goatools.obo_parser import GODag

    go_dag = GODag(str(args.go_basic_obo), optional_attrs={"relationship"})
    missing_go_counts = count_missing_explicit_go_ids(explicit_labels, go_dag)

    vocabulary = None
    if args.vocabulary is not None:
        log_progress(f"Loading vocabulary for optional label cache: {args.vocabulary}")
        vocabulary = load_go_vocabulary(args.vocabulary)

    log_progress("Writing true-path labels, counts, and metrics")
    ancestor_cache: dict[str, frozenset[str]] = {}
    write_outputs(
        paths=paths,
        selected_keys=selected_keys,
        explicit_labels=explicit_labels,
        go_dag=go_dag,
        ancestor_cache=ancestor_cache,
        missing_go_counts=missing_go_counts,
        annotation_metrics=annotation_metrics,
        policy_metrics=policy_metrics,
        vocabulary=vocabulary,
        write_gene_labels=not args.no_write_gene_labels,
    )

    elapsed = time.time() - started
    log_progress(
        f"Wrote true-path label artifacts under {args.output_dir} in {elapsed / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
