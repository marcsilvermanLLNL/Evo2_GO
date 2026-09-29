#!/usr/bin/env python3
"""Create an Evo2-GO taxon-holdout split manifest from embedding metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Sequence

from evo2_go.model.dataset import (
    DEFAULT_EMBEDDING_ROOT,
    complete_metadata_paths_from_manifest,
    load_embedding_records,
    load_label_index_cache,
    parse_tax_ids,
    tax_ids_from_file,
)


DEFAULT_OUTPUT = Path("outputs/model_training/splits/human_taxon_holdout_manifest.tsv")
DEFAULT_SUMMARY = Path("outputs/model_training/splits/human_taxon_holdout_summary.tsv")


def split_score(identity: tuple[str, str, str], seed: str) -> float:
    key = "\t".join((seed, *identity)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def tax_id_set(raw_tax_ids: Sequence[str] | None) -> set[str]:
    return parse_tax_ids(raw_tax_ids) or set()


def write_summary(path: Path, rows: list[tuple[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    tmp_path.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--holdout-tax-id",
        action="append",
        default=[],
        help=(
            "Taxon assigned only to validation/test in legacy split mode. "
            "Can be passed multiple times or as comma-delimited values."
        ),
    )
    parser.add_argument(
        "--val-tax-id",
        action="append",
        default=[],
        help="Taxon assigned entirely to validation in separate-taxon mode.",
    )
    parser.add_argument(
        "--test-tax-id",
        action="append",
        default=[],
        help="Taxon assigned entirely to test in separate-taxon mode.",
    )
    parser.add_argument(
        "--tax-id",
        action="append",
        default=[],
        help="Optional taxon filter applied to the embedding manifest.",
    )
    parser.add_argument(
        "--tax-id-file",
        action="append",
        default=[],
        help=(
            "Optional file of tax IDs used to filter the embedding manifest. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--expected-embedding-dim", type=int, default=4096)
    parser.add_argument("--holdout-val-fraction", type=float, default=0.5)
    parser.add_argument("--holdout-test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", default="evo2-go-human-taxon-holdout-v1")
    parser.add_argument(
        "--drop-examples-without-labels",
        action="store_true",
        help=(
            "When --label-cache is provided, omit records with no retained labels. "
            "By default every embedding record receives a split assignment."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing split manifest: {args.output}")
    if args.summary.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing split summary: {args.summary}")

    total_holdout_fraction = args.holdout_val_fraction + args.holdout_test_fraction
    if abs(total_holdout_fraction - 1.0) > 1e-8:
        raise ValueError("holdout validation/test fractions must sum to 1.0")
    if min(args.holdout_val_fraction, args.holdout_test_fraction) < 0.0:
        raise ValueError("holdout fractions must be non-negative")

    val_tax_ids = tax_id_set(args.val_tax_id)
    test_tax_ids = tax_id_set(args.test_tax_id)
    separate_val_test_taxa = bool(val_tax_ids or test_tax_ids)
    if separate_val_test_taxa:
        if not val_tax_ids:
            raise ValueError("--val-tax-id is required when --test-tax-id is used")
        if not test_tax_ids:
            raise ValueError("--test-tax-id is required when --val-tax-id is used")
        overlap = val_tax_ids.intersection(test_tax_ids)
        if overlap:
            raise ValueError(f"Taxon IDs cannot be both validation and test: {sorted(overlap)}")
        holdout_tax_ids = val_tax_ids.union(test_tax_ids)
    else:
        holdout_tax_ids = tax_id_set(args.holdout_tax_id) or {"9606"}

    selected_tax_id_values = list(args.tax_id)
    for tax_id_file in args.tax_id_file:
        selected_tax_id_values.extend(tax_ids_from_file(tax_id_file))
    selected_tax_ids = parse_tax_ids(selected_tax_id_values)
    metadata_paths = complete_metadata_paths_from_manifest(
        args.embedding_root / "embedding_dataset_manifest.tsv",
        tax_ids=selected_tax_ids,
    )
    records = load_embedding_records(
        metadata_paths,
        max_records=args.max_records,
        expected_embedding_dim=args.expected_embedding_dim,
    )
    records_before_label_filter = len(records)

    label_index = None
    if args.label_cache is not None:
        label_index = load_label_index_cache(args.label_cache)
        if args.drop_examples_without_labels:
            records = [record for record in records if label_index.get(record.gene_key)]

    split_counts: Counter[str] = Counter()
    labeled_split_counts: Counter[str] = Counter()
    tax_split_counts: Counter[tuple[str, str]] = Counter()
    tax_ids_seen: set[str] = set()
    holdout_seen = False
    val_seen = False
    test_seen = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    fieldnames = (
        "split",
        "tax_id",
        "GeneID",
        "cds_id",
        "embedding_id",
        "sequence_id",
        "sequence_length",
        "token_count",
        "split_strategy",
        "ortholog_safe",
        "seed",
    )
    split_strategy = "separate_taxon_val_test" if separate_val_test_taxa else "taxon_holdout"
    rows_written = 0
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            if separate_val_test_taxa:
                if record.tax_id in val_tax_ids:
                    holdout_seen = True
                    val_seen = True
                    split = "val"
                elif record.tax_id in test_tax_ids:
                    holdout_seen = True
                    test_seen = True
                    split = "test"
                else:
                    split = "train"
            elif record.tax_id in holdout_tax_ids:
                holdout_seen = True
                split = (
                    "val"
                    if split_score(record.identity, args.seed) < args.holdout_val_fraction
                    else "test"
                )
            else:
                split = "train"

            rows_written += 1
            tax_ids_seen.add(record.tax_id)
            split_counts[split] += 1
            tax_split_counts[(record.tax_id, split)] += 1
            if label_index is not None and label_index.get(record.gene_key):
                labeled_split_counts[split] += 1

            writer.writerow(
                {
                    "split": split,
                    "tax_id": record.tax_id,
                    "GeneID": record.gene_id,
                    "cds_id": record.cds_id,
                    "embedding_id": record.embedding_id,
                    "sequence_id": record.sequence_id,
                    "sequence_length": record.sequence_length,
                    "token_count": record.token_count,
                    "split_strategy": split_strategy,
                    "ortholog_safe": 0,
                    "seed": args.seed,
                }
            )

    if rows_written == 0:
        raise ValueError("No embedding records selected for split manifest")
    if not holdout_seen:
        raise ValueError(f"No records were found for held-out taxon IDs: {sorted(holdout_tax_ids)}")
    if separate_val_test_taxa and not val_seen:
        raise ValueError(f"No records were found for validation taxon IDs: {sorted(val_tax_ids)}")
    if separate_val_test_taxa and not test_seen:
        raise ValueError(f"No records were found for test taxon IDs: {sorted(test_tax_ids)}")

    tmp_path.replace(args.output)

    summary_rows: list[tuple[object, object]] = [
        ("split_strategy", split_strategy),
        ("ortholog_safe", 0),
        ("embedding_root", args.embedding_root),
        ("label_cache", args.label_cache or ""),
        ("holdout_tax_ids", ",".join(sorted(holdout_tax_ids))),
        ("val_tax_ids", ",".join(sorted(val_tax_ids))),
        ("test_tax_ids", ",".join(sorted(test_tax_ids))),
        ("holdout_val_fraction", args.holdout_val_fraction),
        ("holdout_test_fraction", args.holdout_test_fraction),
        ("records_before_label_filter", records_before_label_filter),
        ("records_written", rows_written),
        ("taxa", len(tax_ids_seen)),
        ("seed", args.seed),
        ("train_records", split_counts["train"]),
        ("val_records", split_counts["val"]),
        ("test_records", split_counts["test"]),
    ]
    if label_index is not None:
        summary_rows.extend(
            [
                ("train_records_with_retained_labels", labeled_split_counts["train"]),
                ("val_records_with_retained_labels", labeled_split_counts["val"]),
                ("test_records_with_retained_labels", labeled_split_counts["test"]),
            ]
        )
    for tax_id, split in sorted(tax_split_counts):
        summary_rows.append((f"tax_{tax_id}_{split}_records", tax_split_counts[(tax_id, split)]))

    write_summary(args.summary, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
