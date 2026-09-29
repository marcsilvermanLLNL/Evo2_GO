#!/usr/bin/env python3
"""Count GO terms for a selected set of trainable genes.

This is the reproducible script form of the GO-counting step used before
building a GO adjacency. It reads a selected representative-CDS metadata table,
then streams the evidence-bearing annotation table and counts each GO term once
per selected (tax_id, GeneID).
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


DEFAULT_SELECTED_GENES = Path(
    "outputs/map_gene_cds/length_filtered_max_8192_non_iea/"
    "non_iea_length_filtered_representative_gene_cds.tsv.gz"
)
DEFAULT_ANNOTATIONS = Path(
    "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz"
)
DEFAULT_OUTPUT = Path("outputs/model_vocab/go_term_counts_length_filtered_non_iea.tsv")


def log_progress(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def clean_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_evidence_codes(raw_values: list[str]) -> set[str] | None:
    """Parse optional evidence-code filters.

    Returns None when all non-IEA codes should be retained.
    """

    if not raw_values:
        return None

    codes: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip().upper()
            if part:
                codes.add(part)
    return codes


def load_selected_gene_keys(path: Path) -> set[tuple[str, str]]:
    """Load selected trainable (tax_id, GeneID) keys from metadata."""

    if not path.is_file():
        raise FileNotFoundError(f"Selected gene metadata not found: {path}")

    keys: set[tuple[str, str]] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"tax_id", "GeneID"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            keys.add((clean_field(row["tax_id"]), clean_field(row["GeneID"])))

    if not keys:
        raise ValueError(f"No selected genes found in {path}")
    return keys


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


def write_counts(
    output_path: Path,
    counts: Counter[str],
    total_selected_genes: int,
    selected_genes_with_counted_labels: int,
    annotation_rows_seen: int,
    counted_unique_gene_go_pairs: int,
    evidence_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "GO_ID",
                "trainable_gene_count",
                "total_trainable_label_rows",
                "total_trainable_genes",
                "selected_genes_with_counted_labels",
                "annotation_rows_seen",
                "evidence_filter",
            ]
        )
        for go_id in sorted(counts):
            writer.writerow(
                [
                    go_id,
                    counts[go_id],
                    counted_unique_gene_go_pairs,
                    total_selected_genes,
                    selected_genes_with_counted_labels,
                    annotation_rows_seen,
                    evidence_label,
                ]
            )

    tmp_path.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-genes", type=Path, default=DEFAULT_SELECTED_GENES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        help="Include IEA annotations. If --allowed-evidence is also set, this only changes the label text.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output without --force: {args.output}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation table not found: {args.annotations}")

    started = time.time()
    allowed_evidence = parse_evidence_codes(args.allowed_evidence)
    if allowed_evidence is None:
        evidence_label = "all_evidence_including_iea" if args.include_iea else "non_iea"
    else:
        evidence_label = ",".join(sorted(allowed_evidence))

    log_progress(f"Loading selected genes: {args.selected_genes}")
    selected_keys = load_selected_gene_keys(args.selected_genes)
    log_progress(f"Loaded {len(selected_keys):,} selected (tax_id, GeneID) keys")

    counts: Counter[str] = Counter()
    seen_gene_go: set[tuple[str, str, str]] = set()
    genes_with_counted_labels: set[tuple[str, str]] = set()
    annotation_rows_seen = 0

    log_progress(f"Streaming annotations: {args.annotations}")
    with gzip.open(args.annotations, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"tax_id", "GeneID", "GO_ID", "Evidence"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{args.annotations} is missing required columns: {sorted(missing)}")

        for annotation_rows_seen, row in enumerate(reader, start=1):
            if annotation_rows_seen % 10_000_000 == 0:
                elapsed = time.time() - started
                log_progress(
                    f"Scanned {annotation_rows_seen:,} rows; "
                    f"counted {len(seen_gene_go):,} selected gene-GO pairs; "
                    f"elapsed {elapsed / 60:.1f} min"
                )

            evidence = clean_field(row["Evidence"])
            if not evidence_is_retained(
                evidence,
                allowed_evidence,
                include_iea=args.include_iea,
            ):
                continue

            tax_id = clean_field(row["tax_id"])
            gene_id = clean_field(row["GeneID"])
            key = (tax_id, gene_id)
            if key not in selected_keys:
                continue

            go_id = clean_field(row["GO_ID"])
            gene_go_key = (tax_id, gene_id, go_id)
            if gene_go_key in seen_gene_go:
                continue

            seen_gene_go.add(gene_go_key)
            genes_with_counted_labels.add(key)
            counts[go_id] += 1

    if not counts:
        raise ValueError("No GO term counts were produced")

    write_counts(
        output_path=args.output,
        counts=counts,
        total_selected_genes=len(selected_keys),
        selected_genes_with_counted_labels=len(genes_with_counted_labels),
        annotation_rows_seen=annotation_rows_seen,
        counted_unique_gene_go_pairs=len(seen_gene_go),
        evidence_label=evidence_label,
    )

    elapsed = time.time() - started
    log_progress(
        f"Wrote {len(counts):,} GO term counts and {len(seen_gene_go):,} "
        f"selected gene-GO pairs to {args.output} in {elapsed / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
