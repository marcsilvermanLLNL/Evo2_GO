#!/usr/bin/env python3
"""Validate Evo2-GO embedding metadata, shards, and optional label coverage."""

from __future__ import annotations

import argparse
import csv
import gzip
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from evo2_go.model.dataset import (
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_GO_VOCABULARY,
    EmbeddingRecord,
    complete_metadata_paths_from_manifest,
    load_embedding_records,
    load_go_vocabulary,
    load_label_index_cache,
    parse_tax_ids,
)


DEFAULT_INPUT_ROOT = Path("outputs/map_gene_cds/length_filtered_max_8192_non_iea")
DEFAULT_OUTPUT_DIR = Path("outputs/model_training/non_iea_min_count_10/dataset_validation")


def open_tsv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("rt", newline="")


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    tmp_path.replace(path)


def write_shard_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (
        "shard_path",
        "metadata_rows_selected",
        "tensor_rows",
        "tensor_width",
        "dtype",
        "bad_shard_rows",
        "nonfinite_selected_rows",
        "status",
        "error_message",
    )
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def load_kept_identity_set(input_root: Path, tax_ids: set[str]) -> set[tuple[str, str, str]]:
    identities: set[tuple[str, str, str]] = set()
    for tax_id in sorted(tax_ids, key=lambda value: int(value) if value.isdigit() else value):
        metadata_path = input_root / f"tax_{tax_id}" / "kept_representative_gene_cds.tsv.gz"
        if not metadata_path.is_file():
            continue
        with open_tsv(metadata_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"tax_id", "GeneID", "cds_id"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
            for row in reader:
                identities.add(
                    (
                        str(row["tax_id"]).strip(),
                        str(row["GeneID"]).strip(),
                        str(row["cds_id"]).strip(),
                    )
                )
    return identities


def group_records_by_shard(records: list[EmbeddingRecord]) -> dict[Path, list[EmbeddingRecord]]:
    by_shard: dict[Path, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        by_shard[record.shard_path].append(record)
    return dict(by_shard)


def validate_shards(
    records: list[EmbeddingRecord],
    *,
    expected_embedding_dim: int,
    max_shards: int | None,
    check_all_finite: bool,
) -> tuple[list[dict[str, object]], Counter[str]]:
    shard_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    by_shard = group_records_by_shard(records)
    for shard_number, (shard_path, shard_records) in enumerate(
        sorted(by_shard.items(), key=lambda item: str(item[0]))
    ):
        if max_shards is not None and shard_number >= max_shards:
            counts["shards_not_checked_due_to_limit"] += len(by_shard) - max_shards
            break

        row = {
            "shard_path": shard_path,
            "metadata_rows_selected": len(shard_records),
            "tensor_rows": "",
            "tensor_width": "",
            "dtype": "",
            "bad_shard_rows": 0,
            "nonfinite_selected_rows": 0,
            "status": "ok",
            "error_message": "",
        }
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            tensor = payload["pooled_embeddings"]
            row["tensor_rows"] = int(tensor.shape[0])
            row["tensor_width"] = int(tensor.shape[1]) if tensor.ndim == 2 else ""
            row["dtype"] = str(tensor.dtype)

            if tensor.ndim != 2 or tensor.shape[1] != expected_embedding_dim:
                row["status"] = "failed"
                row["error_message"] = f"unexpected tensor shape {tuple(tensor.shape)}"
            bad_rows = [
                record.shard_row
                for record in shard_records
                if record.shard_row < 0 or record.shard_row >= tensor.shape[0]
            ]
            row["bad_shard_rows"] = len(bad_rows)
            if bad_rows:
                row["status"] = "failed"
                row["error_message"] = "metadata shard_row out of tensor bounds"

            if check_all_finite and row["status"] == "ok":
                nonfinite = 0
                for record in shard_records:
                    embedding = tensor[record.shard_row].to(dtype=torch.float32)
                    if not torch.isfinite(embedding).all():
                        nonfinite += 1
                row["nonfinite_selected_rows"] = nonfinite
                if nonfinite:
                    row["status"] = "failed"
                    row["error_message"] = "non-finite selected embedding rows"
        except Exception as exc:  # noqa: BLE001 - validation should report all failures.
            row["status"] = "failed"
            row["error_message"] = repr(exc)

        counts[f"shards_{row['status']}"] += 1
        shard_rows.append(row)

    return shard_rows, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_GO_VOCABULARY)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tax-id", action="append", default=[])
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--expected-embedding-dim", type=int, default=4096)
    parser.add_argument("--check-all-finite", action="store_true")
    args = parser.parse_args()

    started = time.time()
    tax_ids = parse_tax_ids(args.tax_id)
    vocabulary = load_go_vocabulary(args.vocabulary)
    metadata_paths = complete_metadata_paths_from_manifest(
        args.embedding_root / "embedding_dataset_manifest.tsv",
        tax_ids=tax_ids,
    )
    records = load_embedding_records(
        metadata_paths,
        max_records=args.max_records,
        expected_embedding_dim=args.expected_embedding_dim,
    )

    record_tax_ids = {record.tax_id for record in records}
    kept_identities = load_kept_identity_set(args.input_root, record_tax_ids)
    metadata_join_failures = sum(
        1 for record in records if record.identity not in kept_identities
    )
    sequence_token_mismatches = sum(
        1 for record in records if record.sequence_length != record.token_count
    )
    over_context_records = sum(1 for record in records if record.token_count > 8192)

    label_cache_genes = 0
    examples_with_labels = ""
    examples_without_labels = ""
    if args.label_cache is not None:
        label_index = load_label_index_cache(args.label_cache, n_terms=vocabulary.n_terms)
        label_cache_genes = len(label_index)
        with_labels = sum(1 for record in records if label_index.get(record.gene_key))
        examples_with_labels = with_labels
        examples_without_labels = len(records) - with_labels

    shard_rows, shard_counts = validate_shards(
        records,
        expected_embedding_dim=args.expected_embedding_dim,
        max_shards=args.max_shards,
        check_all_finite=args.check_all_finite,
    )

    status = "pass"
    if (
        metadata_join_failures
        or sequence_token_mismatches
        or over_context_records
        or shard_counts.get("shards_failed", 0)
    ):
        status = "fail"

    summary_rows = [
        ("status", status),
        ("embedding_root", args.embedding_root),
        ("input_root", args.input_root),
        ("vocabulary", args.vocabulary),
        ("vocabulary_terms", vocabulary.n_terms),
        ("metadata_paths", len(metadata_paths)),
        ("records_loaded", len(records)),
        ("taxa_loaded", len(record_tax_ids)),
        ("unique_gene_keys", len({record.gene_key for record in records})),
        ("unique_identities", len({record.identity for record in records})),
        ("metadata_join_failures", metadata_join_failures),
        ("sequence_token_mismatches", sequence_token_mismatches),
        ("over_context_records", over_context_records),
        ("label_cache", args.label_cache or ""),
        ("label_cache_genes", label_cache_genes),
        ("examples_with_labels", examples_with_labels),
        ("examples_without_labels", examples_without_labels),
        ("shards_seen", len(group_records_by_shard(records))),
        ("shards_checked", len(shard_rows)),
        ("shards_ok", shard_counts.get("shards_ok", 0)),
        ("shards_failed", shard_counts.get("shards_failed", 0)),
        ("shards_not_checked_due_to_limit", shard_counts.get("shards_not_checked_due_to_limit", 0)),
        ("check_all_finite", args.check_all_finite),
        ("elapsed_seconds", f"{time.time() - started:.3f}"),
    ]

    write_key_value_tsv(args.output_dir / "validation_summary.tsv", summary_rows)
    write_shard_summary(args.output_dir / "shard_validation.tsv", shard_rows)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
