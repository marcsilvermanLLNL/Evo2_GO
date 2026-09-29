#!/usr/bin/env python3
"""Deduplicate Evo2 embedding metadata against current length-filtered inputs.

This repair is intended for the shared embedding root after overlapping resume
jobs have left duplicate metadata rows for the same stable example identity.
The length-filtered kept CDS tables are treated as the source of truth.

By default this script only audits and prints what it would change. Pass
``--execute`` to write cleaned per-taxon metadata plus rebuilt aggregate files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BUILD_ID = "all_labels_true_path_42taxa_human_test_min_count_100_v1"

METADATA_COLUMNS = [
    "embedding_id",
    "tax_id",
    "GeneID",
    "cds_id",
    "sequence_id",
    "sequence_length",
    "token_count",
    "bucket_id",
    "batch_id",
    "batch_size",
    "shard_path",
    "shard_row",
    "layer_name",
    "pooled_shape",
    "pooled_dtype",
    "model_name",
    "model_path",
    "worker_id",
    "gpu_id",
]

BATCH_COLUMNS = [
    "tax_id",
    "bucket_id",
    "batch_id",
    "worker_id",
    "gpu_id",
    "status",
    "records",
    "batch_size",
    "max_token_count",
    "sequence_tokens",
    "padded_tokens",
    "wall_start_epoch",
    "wall_end_epoch",
    "runtime_seconds",
    "tokenize_seconds",
    "forward_pool_seconds",
    "write_seconds",
    "sequence_tokens_per_second",
    "padded_tokens_per_second",
    "peak_gpu_memory_bytes",
    "peak_gpu_memory_gb",
    "shard_path",
    "error_message",
]

DATASET_MANIFEST_COLUMNS = [
    "tax_id",
    "status",
    "records_embedded",
    "batches_complete",
    "sequence_tokens",
    "embedding_metadata_path",
    "batch_metrics_path",
    "tax_output_dir",
    "embedding_shards_dir",
    "run_summary_path",
    "bucket_summary_path",
]

BUCKET_SUMMARY_COLUMNS = [
    "bucket_id",
    "batches",
    "records",
    "sequence_tokens",
    "padded_tokens",
    "runtime_seconds_sum",
    "sequence_tokens_per_second_overall",
    "padded_tokens_per_second_overall",
    "mean_batch_sequence_tokens_per_second",
    "mean_batch_padded_tokens_per_second",
    "mean_peak_gpu_memory_gb",
]


@dataclass(frozen=True)
class TaxonAudit:
    tax_id: str
    kept_count: int
    metadata_rows: int
    unique_identities: int
    duplicate_rows: int
    rows_not_in_kept: int
    missing_kept: int
    action: str


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def numeric_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("rt", newline="")


def open_gzip_writer(path: Path):
    return gzip.open(path, "wt", newline="")


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(clean(value))
    except ValueError:
        return default


def full_cds_id_from_sequence_id(sequence_id: str, fallback: str) -> str:
    marker = "|cds_id="
    start = sequence_id.find(marker)
    if start < 0 and sequence_id.startswith("cds_id="):
        start = -1
        marker = "cds_id="
    if start < 0:
        return fallback

    value_start = start + len(marker)
    length_marker = "|length="
    end = sequence_id.find(length_marker, value_start)
    if end < 0:
        return fallback
    cds_id = sequence_id[value_start:end].strip()
    return cds_id or fallback


def identity_from_metadata_row(row: dict[str, str]) -> tuple[str, str, str]:
    sequence_id = clean(row.get("sequence_id", ""))
    fallback_cds_id = clean(row.get("cds_id", ""))
    return (
        clean(row.get("tax_id", "")),
        clean(row.get("GeneID", "")),
        full_cds_id_from_sequence_id(sequence_id, fallback_cds_id),
    )


def read_kept_identities(path: Path) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str, str], int]]:
    identities: list[tuple[str, str, str]] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"tax_id", "GeneID", "cds_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            identities.append(
                (clean(row["tax_id"]), clean(row["GeneID"]), clean(row["cds_id"]))
            )
    return identities, {identity: i for i, identity in enumerate(identities)}


def read_metadata_rows(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(METADATA_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing metadata columns: {sorted(missing)}")
        return list(reader)


def row_choice_key(row: dict[str, str]) -> tuple[int, int, int, str, str]:
    shard_path = Path(clean(row.get("shard_path", "")))
    return (
        0 if shard_path.is_file() else 1,
        parse_int(row.get("batch_id")),
        parse_int(row.get("shard_row")),
        str(shard_path),
        clean(row.get("embedding_id", "")),
    )


def choose_deduped_rows(
    metadata_rows: list[dict[str, str]],
    kept_order: dict[tuple[str, str, str], int],
) -> tuple[list[dict[str, str]], int, int, int]:
    best_by_identity: dict[tuple[str, str, str], dict[str, str]] = {}
    duplicate_rows = 0
    rows_not_in_kept = 0

    for row in metadata_rows:
        identity = identity_from_metadata_row(row)
        if identity not in kept_order:
            rows_not_in_kept += 1
            continue
        current = best_by_identity.get(identity)
        if current is None:
            best_by_identity[identity] = row
        else:
            duplicate_rows += 1
            if row_choice_key(row) < row_choice_key(current):
                best_by_identity[identity] = row

    cleaned_rows = sorted(
        best_by_identity.values(),
        key=lambda row: kept_order[identity_from_metadata_row(row)],
    )
    missing_kept = len(kept_order) - len(best_by_identity)
    return cleaned_rows, duplicate_rows, rows_not_in_kept, missing_kept


def batch_group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        clean(row.get("bucket_id", "")),
        clean(row.get("batch_id", "")),
        clean(row.get("shard_path", "")),
    )


def synthesize_batch_rows(metadata_rows: Iterable[dict[str, str]], tax_id: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        grouped[batch_group_key(row)].append(row)

    batch_rows: list[dict[str, str]] = []
    for (bucket_id, batch_id, shard_path), rows in sorted(grouped.items()):
        token_counts = [parse_int(row.get("token_count")) for row in rows]
        max_token_count = max(token_counts) if token_counts else 0
        records = len(rows)
        sequence_tokens = sum(token_counts)
        padded_tokens = max_token_count * records
        first = rows[0]
        batch_rows.append(
            {
                "tax_id": tax_id,
                "bucket_id": bucket_id,
                "batch_id": batch_id,
                "worker_id": clean(first.get("worker_id", "")),
                "gpu_id": clean(first.get("gpu_id", "")),
                "status": "success",
                "records": str(records),
                "batch_size": str(records),
                "max_token_count": str(max_token_count),
                "sequence_tokens": str(sequence_tokens),
                "padded_tokens": str(padded_tokens),
                "wall_start_epoch": "",
                "wall_end_epoch": "",
                "runtime_seconds": "",
                "tokenize_seconds": "",
                "forward_pool_seconds": "",
                "write_seconds": "",
                "sequence_tokens_per_second": "",
                "padded_tokens_per_second": "",
                "peak_gpu_memory_bytes": "",
                "peak_gpu_memory_gb": "",
                "shard_path": shard_path,
                "error_message": "",
            }
        )
    return batch_rows


def write_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]], *, backup_suffix: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    replace_with_optional_backup(path, tmp_path, backup_suffix)


def write_gzip_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]], *, backup_suffix: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_gzip_writer(tmp_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    replace_with_optional_backup(path, tmp_path, backup_suffix)


def replace_with_optional_backup(path: Path, tmp_path: Path, backup_suffix: str) -> None:
    if backup_suffix and path.exists():
        backup_path = path.with_name(path.name + backup_suffix)
        if backup_path.exists():
            tmp_path.unlink(missing_ok=True)
            raise FileExistsError(f"Backup already exists: {backup_path}")
        path.rename(backup_path)
        tmp_path.rename(path)
    else:
        tmp_path.replace(path)


def write_run_summary(path: Path, rows: list[tuple[str, object]], *, backup_suffix: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    replace_with_optional_backup(path, tmp_path, backup_suffix)


def read_key_value_tsv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {clean(row.get("metric", "")): clean(row.get("value", "")) for row in reader}


def read_batch_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def bucket_summary_rows(batch_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in batch_rows:
        if clean(row.get("status")) == "success":
            grouped[clean(row.get("bucket_id"))].append(row)

    rows: list[dict[str, object]] = []
    for bucket_id in sorted(grouped):
        rows_for_bucket = grouped[bucket_id]
        sequence_tokens = sum(parse_int(row.get("sequence_tokens")) for row in rows_for_bucket)
        padded_tokens = sum(parse_int(row.get("padded_tokens")) for row in rows_for_bucket)
        runtime_values = [
            float(clean(row.get("runtime_seconds")))
            for row in rows_for_bucket
            if clean(row.get("runtime_seconds"))
        ]
        runtime_sum = sum(runtime_values)
        rows.append(
            {
                "bucket_id": bucket_id,
                "batches": len(rows_for_bucket),
                "records": sum(parse_int(row.get("records")) for row in rows_for_bucket),
                "sequence_tokens": sequence_tokens,
                "padded_tokens": padded_tokens,
                "runtime_seconds_sum": f"{runtime_sum:.6f}" if runtime_values else "",
                "sequence_tokens_per_second_overall": (
                    f"{sequence_tokens / runtime_sum:.3f}" if runtime_sum > 0 else ""
                ),
                "padded_tokens_per_second_overall": (
                    f"{padded_tokens / runtime_sum:.3f}" if runtime_sum > 0 else ""
                ),
                "mean_batch_sequence_tokens_per_second": "",
                "mean_batch_padded_tokens_per_second": "",
                "mean_peak_gpu_memory_gb": "",
            }
        )
    return rows


def tax_ids_from_args(raw_tax_ids: list[str]) -> set[str] | None:
    if not raw_tax_ids:
        return None
    tax_ids: set[str] = set()
    for raw in raw_tax_ids:
        for part in raw.split(","):
            part = part.strip()
            if part:
                tax_ids.add(part)
    return tax_ids


def discover_tax_ids(input_root: Path, embedding_root: Path, requested: set[str] | None) -> list[str]:
    if requested is not None:
        return sorted(requested, key=numeric_sort_key)
    tax_ids = {
        path.name.removeprefix("tax_")
        for path in embedding_root.glob("tax_*")
        if path.is_dir()
    }
    tax_ids.update(
        path.name.removeprefix("tax_")
        for path in input_root.glob("tax_*")
        if path.is_dir()
    )
    return sorted(tax_ids, key=numeric_sort_key)


def process_taxon(
    tax_id: str,
    *,
    input_root: Path,
    embedding_root: Path,
    execute: bool,
    backup_suffix: str,
) -> tuple[TaxonAudit, list[dict[str, str]] | None]:
    kept_path = input_root / f"tax_{tax_id}" / "kept_representative_gene_cds.tsv.gz"
    metadata_path = embedding_root / f"tax_{tax_id}" / "embedding_metadata.tsv.gz"

    if not kept_path.is_file() or not metadata_path.is_file():
        return (
            TaxonAudit(tax_id, 0, 0, 0, 0, 0, 0, "missing-input-or-metadata"),
            None,
        )

    kept_identities, kept_order = read_kept_identities(kept_path)
    metadata_rows = read_metadata_rows(metadata_path)
    cleaned_rows, duplicate_rows, rows_not_in_kept, missing_kept = choose_deduped_rows(
        metadata_rows,
        kept_order,
    )

    needs_rewrite = bool(duplicate_rows or rows_not_in_kept or missing_kept)
    action = "rewrite" if needs_rewrite else "unchanged"
    audit = TaxonAudit(
        tax_id=tax_id,
        kept_count=len(kept_identities),
        metadata_rows=len(metadata_rows),
        unique_identities=len(cleaned_rows),
        duplicate_rows=duplicate_rows,
        rows_not_in_kept=rows_not_in_kept,
        missing_kept=missing_kept,
        action=action,
    )

    if not execute or not needs_rewrite:
        return audit, None

    if missing_kept:
        raise RuntimeError(
            f"Refusing to rewrite tax_{tax_id}: {missing_kept} kept identities are missing embeddings"
        )

    tax_dir = embedding_root / f"tax_{tax_id}"
    batch_rows = synthesize_batch_rows(cleaned_rows, tax_id)
    sequence_tokens = sum(parse_int(row.get("token_count")) for row in cleaned_rows)
    padded_tokens = sum(parse_int(row.get("padded_tokens")) for row in batch_rows)

    write_gzip_tsv(metadata_path, METADATA_COLUMNS, cleaned_rows, backup_suffix=backup_suffix)
    write_tsv(tax_dir / "batch_metrics.tsv", BATCH_COLUMNS, batch_rows, backup_suffix=backup_suffix)
    write_tsv(
        tax_dir / "bucket_summary.tsv",
        BUCKET_SUMMARY_COLUMNS,
        bucket_summary_rows(batch_rows),
        backup_suffix=backup_suffix,
    )
    write_run_summary(
        tax_dir / "run_summary.tsv",
        [
            ("status", "complete" if cleaned_rows else "incomplete"),
            ("tax_id", tax_id),
            ("batches", len(batch_rows)),
            ("successful_batches", len(batch_rows)),
            ("records", len(cleaned_rows)),
            ("sequence_tokens", sequence_tokens),
            ("padded_tokens", padded_tokens),
        ],
        backup_suffix=backup_suffix,
    )
    return audit, batch_rows


def completed_taxon(embedding_root: Path, tax_id: str) -> bool:
    summary = read_key_value_tsv(embedding_root / f"tax_{tax_id}" / "run_summary.tsv")
    return summary.get("status") == "complete"


def rebuild_aggregate_files(
    *,
    embedding_root: Path,
    execute: bool,
    backup_suffix: str,
    audit_rows: list[TaxonAudit],
    elapsed_seconds: float,
) -> None:
    tax_dirs = sorted(
        [path for path in embedding_root.glob("tax_*") if path.is_dir()],
        key=lambda path: numeric_sort_key(path.name.removeprefix("tax_")),
    )

    if not execute:
        return

    root_metadata_rows = 0
    metadata_tmp = embedding_root / f".embedding_metadata.tsv.gz.tmp.{os.getpid()}"
    with open_gzip_writer(metadata_tmp) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METADATA_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for tax_dir in tax_dirs:
            tax_id = tax_dir.name.removeprefix("tax_")
            if not completed_taxon(embedding_root, tax_id):
                continue
            metadata_path = tax_dir / "embedding_metadata.tsv.gz"
            if not metadata_path.is_file():
                continue
            for row in read_metadata_rows(metadata_path):
                writer.writerow({field: row.get(field, "") for field in METADATA_COLUMNS})
                root_metadata_rows += 1
    replace_with_optional_backup(
        embedding_root / "embedding_metadata.tsv.gz",
        metadata_tmp,
        backup_suffix,
    )

    root_batch_rows: list[dict[str, str]] = []
    for tax_dir in tax_dirs:
        tax_id = tax_dir.name.removeprefix("tax_")
        if completed_taxon(embedding_root, tax_id):
            root_batch_rows.extend(read_batch_rows(tax_dir / "batch_metrics.tsv"))

    write_tsv(embedding_root / "batch_metrics.tsv", BATCH_COLUMNS, root_batch_rows, backup_suffix=backup_suffix)
    write_tsv(
        embedding_root / "bucket_summary.tsv",
        BUCKET_SUMMARY_COLUMNS,
        bucket_summary_rows(root_batch_rows),
        backup_suffix=backup_suffix,
    )

    manifest_rows = []
    manifest_status_counts: Counter[str] = Counter()
    for tax_dir in tax_dirs:
        tax_id = tax_dir.name.removeprefix("tax_")
        summary = read_key_value_tsv(tax_dir / "run_summary.tsv")
        status = summary.get("status", "missing")
        manifest_status_counts[status] += 1
        manifest_rows.append(
            {
                "tax_id": tax_id,
                "status": status,
                "records_embedded": summary.get("records", ""),
                "batches_complete": summary.get("batches", ""),
                "sequence_tokens": summary.get("sequence_tokens", ""),
                "embedding_metadata_path": str(tax_dir / "embedding_metadata.tsv.gz"),
                "batch_metrics_path": str(tax_dir / "batch_metrics.tsv"),
                "tax_output_dir": str(tax_dir),
                "embedding_shards_dir": str(tax_dir / "embedding_shards"),
                "run_summary_path": str(tax_dir / "run_summary.tsv"),
                "bucket_summary_path": str(tax_dir / "bucket_summary.tsv"),
            }
        )
    write_tsv(
        embedding_root / "embedding_dataset_manifest.tsv",
        DATASET_MANIFEST_COLUMNS,
        manifest_rows,
        backup_suffix=backup_suffix,
    )

    duplicate_rows_removed = sum(row.duplicate_rows + row.rows_not_in_kept for row in audit_rows)
    rewritten_taxa = sum(1 for row in audit_rows if row.action == "rewrite")
    write_run_summary(
        embedding_root / "run_summary.tsv",
        [
            ("status", "complete"),
            ("operation", "deduplicate_embedding_metadata"),
            ("taxa_scanned", len(audit_rows)),
            ("taxa_rewritten", rewritten_taxa),
            ("duplicate_or_stale_rows_removed", duplicate_rows_removed),
            ("aggregate_metadata_rows", root_metadata_rows),
            ("complete_taxa", manifest_status_counts.get("complete", 0)),
            ("incomplete_taxa", manifest_status_counts.get("incomplete", 0)),
            ("missing_taxa", manifest_status_counts.get("missing", 0)),
            ("elapsed_seconds", f"{elapsed_seconds:.3f}"),
        ],
        backup_suffix=backup_suffix,
    )

    report_path = embedding_root / "embedding_run_report.txt"
    report_tmp = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    with report_tmp.open("wt") as handle:
        handle.write("Evo2 Embedding Metadata Deduplication Report\n")
        handle.write("===========================================\n\n")
        handle.write(f"Status: complete\n")
        handle.write(f"Embedding root: {embedding_root}\n")
        handle.write(f"Taxa scanned: {len(audit_rows)}\n")
        handle.write(f"Taxa rewritten: {rewritten_taxa}\n")
        handle.write(f"Duplicate/stale rows removed: {duplicate_rows_removed}\n")
        handle.write(f"Aggregate metadata rows: {root_metadata_rows}\n")
        handle.write(f"Complete taxa: {manifest_status_counts.get('complete', 0)}\n")
        handle.write(f"Incomplete taxa: {manifest_status_counts.get('incomplete', 0)}\n")
        handle.write(f"Elapsed seconds: {elapsed_seconds:.3f}\n\n")
        handle.write("Source of truth: current length-filtered kept_representative_gene_cds.tsv.gz files.\n")
    replace_with_optional_backup(report_path, report_tmp, backup_suffix)


def write_audit_report(path: Path, rows: list[TaxonAudit], *, execute: bool, backup_suffix: str) -> None:
    fieldnames = [
        "tax_id",
        "kept_count",
        "metadata_rows",
        "unique_identities",
        "duplicate_rows",
        "rows_not_in_kept",
        "missing_kept",
        "action",
    ]
    dict_rows = [
        {
            "tax_id": row.tax_id,
            "kept_count": row.kept_count,
            "metadata_rows": row.metadata_rows,
            "unique_identities": row.unique_identities,
            "duplicate_rows": row.duplicate_rows,
            "rows_not_in_kept": row.rows_not_in_kept,
            "missing_kept": row.missing_kept,
            "action": row.action,
        }
        for row in rows
    ]
    if execute:
        write_tsv(path, fieldnames, dict_rows, backup_suffix=backup_suffix)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(dict_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--tax-id", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="Write repaired files. Default only audits.")
    parser.add_argument(
        "--backup-suffix",
        default="",
        help=(
            "Optional suffix for renamed backups before replacement. "
            "Use a timestamped suffix if you want rollback files."
        ),
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Audit report path. Defaults under the embedding root when --execute is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_root = args.input_root or project_root / "outputs/map_gene_cds/length_filtered_max_8192"
    embedding_root = args.embedding_root or project_root / "outputs/builds" / BUILD_ID / "data/embeddings"
    input_root = input_root.resolve()
    embedding_root = embedding_root.resolve()

    started = time.time()
    requested_tax_ids = tax_ids_from_args(args.tax_id)
    tax_ids = discover_tax_ids(input_root, embedding_root, requested_tax_ids)
    if not tax_ids:
        raise SystemExit("No taxa found to scan")

    log(f"Project root: {project_root}")
    log(f"Input root: {input_root}")
    log(f"Embedding root: {embedding_root}")
    log(f"Mode: {'execute' if args.execute else 'audit-only'}")
    log(f"Taxa to scan: {len(tax_ids)}")

    audit_rows: list[TaxonAudit] = []
    for i, tax_id in enumerate(tax_ids, start=1):
        audit, _ = process_taxon(
            tax_id,
            input_root=input_root,
            embedding_root=embedding_root,
            execute=args.execute,
            backup_suffix=args.backup_suffix,
        )
        audit_rows.append(audit)
        if i % 100 == 0 or audit.action == "rewrite":
            log(
                f"tax_{tax_id}: action={audit.action} kept={audit.kept_count} "
                f"metadata_rows={audit.metadata_rows} duplicates={audit.duplicate_rows} "
                f"missing_kept={audit.missing_kept}"
            )

    elapsed = time.time() - started
    counts = Counter(row.action for row in audit_rows)
    duplicate_rows = sum(row.duplicate_rows for row in audit_rows)
    rows_not_in_kept = sum(row.rows_not_in_kept for row in audit_rows)
    missing_kept = sum(row.missing_kept for row in audit_rows)

    audit_report = args.audit_report
    if audit_report is None and args.execute:
        audit_report = embedding_root / "deduplicate_embedding_metadata_report.tsv"

    if audit_report is not None:
        write_audit_report(audit_report, audit_rows, execute=args.execute, backup_suffix=args.backup_suffix)
    else:
        write_audit_report(Path("-"), audit_rows, execute=False, backup_suffix="")

    if args.execute:
        if missing_kept:
            raise RuntimeError(
                f"Refusing aggregate rebuild because {missing_kept} kept identities are missing embeddings"
            )
        rebuild_aggregate_files(
            embedding_root=embedding_root,
            execute=True,
            backup_suffix=args.backup_suffix,
            audit_rows=audit_rows,
            elapsed_seconds=time.time() - started,
        )

    log("Summary:")
    log(f"  unchanged taxa: {counts.get('unchanged', 0)}")
    log(f"  rewritten taxa: {counts.get('rewrite', 0)}")
    log(f"  missing input/metadata taxa: {counts.get('missing-input-or-metadata', 0)}")
    log(f"  duplicate rows: {duplicate_rows}")
    log(f"  rows not in current kept set: {rows_not_in_kept}")
    log(f"  missing kept identities: {missing_kept}")
    log(f"  elapsed seconds: {time.time() - started:.3f}")
    return 0 if missing_kept == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
