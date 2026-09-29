#!/usr/bin/env python3
"""Build the GO vocabulary and normalized sparse adjacency for the GCN."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


DEFAULT_TERM_COUNTS = Path("outputs/model_vocab/go_term_counts_length_filtered.tsv")
DEFAULT_GO_BASIC_OBO = Path("data/go/go-basic.obo")
DEFAULT_OUTPUT_DIR = Path("outputs/model_vocab/go_adjacency_min_count_1")

VOCABULARY_FILENAME = "go_vocabulary.tsv"
ADJACENCY_FILENAME = "go_adjacency_sparse.npz"
SUMMARY_FILENAME = "go_adjacency_summary.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic GO vocabulary and D^-1/2 A D^-1/2 adjacency "
            "matrix for the Evo2-GO GCN."
        )
    )
    parser.add_argument("--term-counts", type=Path, default=DEFAULT_TERM_COUNTS)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def read_term_counts(term_counts_path: Path, min_count: int) -> list[dict[str, Any]]:
    """Read GO term counts and apply the selected vocabulary cutoff."""

    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if not term_counts_path.exists():
        raise FileNotFoundError(f"GO term count table not found: {term_counts_path}")

    terms: list[dict[str, Any]] = []
    with term_counts_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"GO_ID", "trainable_gene_count"}
        observed = set(reader.fieldnames or ())
        missing = expected - observed
        if missing:
            raise ValueError(
                f"{term_counts_path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            go_id = row["GO_ID"].strip()
            count = int(row["trainable_gene_count"])
            if go_id and count >= min_count:
                terms.append({"GO_ID": go_id, "trainable_gene_count": count})

    terms.sort(key=lambda item: item["GO_ID"])
    if not terms:
        raise ValueError(f"No GO terms passed min_count={min_count}")
    return terms


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "vocabulary": output_dir / VOCABULARY_FILENAME,
        "adjacency": output_dir / ADJACENCY_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
    }


def refuse_overwrite(paths: dict[str, Path], force: bool) -> None:
    if force:
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing GO adjacency outputs without --force:\n"
            f"{formatted}"
        )


def term_name(term: Any | None) -> str:
    return "" if term is None else str(getattr(term, "name", ""))


def term_namespace(term: Any | None) -> str:
    return "" if term is None else str(getattr(term, "namespace", ""))


def is_obsolete(term: Any | None) -> bool:
    return bool(getattr(term, "is_obsolete", False)) if term is not None else False


def write_vocabulary(
    vocabulary_path: Path,
    terms: list[dict[str, Any]],
    go_dag: GODag,
) -> tuple[dict[str, int], dict[str, int]]:
    """Write sorted GO vocabulary and return GO_ID -> row index."""

    go_to_index: dict[str, int] = {}
    status_counts = {
        "terms_in_go_basic": 0,
        "terms_missing_from_go_basic": 0,
        "obsolete_terms": 0,
    }

    with vocabulary_path.open("w", newline="") as handle:
        fieldnames = [
            "go_index",
            "GO_ID",
            "trainable_gene_count",
            "in_go_basic",
            "is_obsolete",
            "name",
            "namespace",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for index, row in enumerate(terms):
            go_id = row["GO_ID"]
            term = go_dag.get(go_id)
            in_go_basic = term is not None
            obsolete = is_obsolete(term)
            go_to_index[go_id] = index

            status_counts["terms_in_go_basic"] += int(in_go_basic)
            status_counts["terms_missing_from_go_basic"] += int(not in_go_basic)
            status_counts["obsolete_terms"] += int(obsolete)

            writer.writerow(
                {
                    "go_index": index,
                    "GO_ID": go_id,
                    "trainable_gene_count": row["trainable_gene_count"],
                    "in_go_basic": int(in_go_basic),
                    "is_obsolete": int(obsolete),
                    "name": term_name(term),
                    "namespace": term_namespace(term),
                }
            )

    return go_to_index, status_counts


def add_bidirectional_edge(
    edges: set[tuple[int, int]],
    source_index: int,
    target_index: int,
) -> None:
    edges.add((source_index, target_index))
    edges.add((target_index, source_index))


def build_binary_adjacency(
    terms: list[dict[str, Any]],
    go_to_index: dict[str, int],
    go_dag: GODag,
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """Build binary self-looped GO adjacency before symmetric normalization."""

    edges: set[tuple[int, int]] = set()
    edge_counts = {
        "is_a_relationships_in_vocab": 0,
        "part_of_relationships_in_vocab": 0,
        "self_loops": len(terms),
        "terms_with_no_ontology_edges": 0,
    }

    # Self-loops preserve each term's own learnable feature during aggregation.
    for index in range(len(terms)):
        edges.add((index, index))

    for row in terms:
        go_id = row["GO_ID"]
        source_index = go_to_index[go_id]
        term = go_dag.get(go_id)

        # Terms missing from go-basic, or obsolete there, stay in the output
        # vocabulary but receive only their self-loop. This preserves all
        # observed trainable labels while avoiding invalid ontology edges.
        if term is None or is_obsolete(term):
            edge_counts["terms_with_no_ontology_edges"] += 1
            continue

        ontology_edges_for_term = 0

        # is_a edges: child <-> parent, restricted to retained vocabulary terms.
        for parent in getattr(term, "parents", []):
            parent_id = parent.id
            if parent_id in go_to_index:
                add_bidirectional_edge(edges, source_index, go_to_index[parent_id])
                edge_counts["is_a_relationships_in_vocab"] += 1
                ontology_edges_for_term += 1

        # part_of edges loaded through goatools optional_attrs={"relationship"}.
        relationships = getattr(term, "relationship", {}) or {}
        for part_of_target in relationships.get("part_of", []):
            target_id = part_of_target.id
            if target_id in go_to_index:
                add_bidirectional_edge(edges, source_index, go_to_index[target_id])
                edge_counts["part_of_relationships_in_vocab"] += 1
                ontology_edges_for_term += 1

        edge_counts["terms_with_no_ontology_edges"] += int(ontology_edges_for_term == 0)

    edge_list = sorted(edges)
    rows = np.array([edge[0] for edge in edge_list], dtype=np.int64)
    cols = np.array([edge[1] for edge in edge_list], dtype=np.int64)
    values = np.ones(len(edge_list), dtype=np.float32)
    adjacency = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(terms), len(terms)),
        dtype=np.float32,
    ).tocsr()

    # Ensure duplicate edges, if any, are represented as a binary adjacency.
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return adjacency, edge_counts


def symmetric_normalize(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Compute D^-1/2 A D^-1/2 as float32 CSR."""

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        degree_inv_sqrt = np.where(degree > 0.0, np.power(degree, -0.5), 0.0)
    degree_matrix = sparse.diags(degree_inv_sqrt.astype(np.float32), format="csr")
    normalized = degree_matrix @ adjacency @ degree_matrix
    normalized = normalized.astype(np.float32).tocsr()
    normalized.eliminate_zeros()
    return normalized


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        for key in sorted(summary):
            writer.writerow([key, summary[key]])


def main() -> int:
    args = parse_args()
    paths = output_paths(args.output_dir)
    refuse_overwrite(paths, force=args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading GO term counts: {args.term_counts}", file=sys.stderr)
    terms = read_term_counts(args.term_counts, min_count=args.min_count)

    print(f"Loading go-basic ontology: {args.go_basic_obo}", file=sys.stderr)
    from goatools.obo_parser import GODag

    go_dag = GODag(str(args.go_basic_obo), optional_attrs={"relationship"})

    print(f"Writing vocabulary: {paths['vocabulary']}", file=sys.stderr)
    go_to_index, status_counts = write_vocabulary(paths["vocabulary"], terms, go_dag)

    print("Building binary GO adjacency", file=sys.stderr)
    binary_adjacency, edge_counts = build_binary_adjacency(terms, go_to_index, go_dag)

    print("Applying symmetric D^-1/2 A D^-1/2 normalization", file=sys.stderr)
    normalized_adjacency = symmetric_normalize(binary_adjacency)

    print(f"Writing sparse adjacency: {paths['adjacency']}", file=sys.stderr)
    sparse.save_npz(paths["adjacency"], normalized_adjacency)

    degree = np.asarray(binary_adjacency.sum(axis=1)).ravel()
    summary = {
        "min_count": args.min_count,
        "vocabulary_terms": len(terms),
        "binary_adjacency_nnz": int(binary_adjacency.nnz),
        "normalized_adjacency_nnz": int(normalized_adjacency.nnz),
        "min_binary_degree": float(degree.min()),
        "median_binary_degree": float(np.median(degree)),
        "mean_binary_degree": float(degree.mean()),
        "max_binary_degree": float(degree.max()),
        "term_counts_path": args.term_counts,
        "go_basic_obo_path": args.go_basic_obo,
        **status_counts,
        **edge_counts,
    }
    print(f"Writing summary: {paths['summary']}", file=sys.stderr)
    write_summary(paths["summary"], summary)

    print(
        f"Built GO adjacency for {len(terms):,} terms with {normalized_adjacency.nnz:,} "
        f"non-zero normalized edges.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
