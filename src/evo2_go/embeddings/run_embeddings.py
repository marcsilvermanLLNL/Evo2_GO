#!/usr/bin/env python3
"""Embed length-filtered representative CDS records with bucketed batch sizes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterator


DEFAULT_INPUT_ROOT = Path("outputs/map_gene_cds/length_filtered_max_8192")
DEFAULT_OUTPUT_ROOT = Path("outputs/evo2_embeddings/length_filtered_max_8192/evo2_7b_base_32k")
DEFAULT_LAYER = "blocks.28.mlp.l3"
DEFAULT_MODEL_NAME = "evo2_7b_base"
DEFAULT_MODEL_PATH = Path("evo2_model/evo2_7b_base.pt")
DEFAULT_EXPECTED_MAX_SEQLEN = "32768"
DEFAULT_BUCKETS = (
    ("bucket_1", 1, 512, 8),
    ("bucket_2", 513, 1024, 8),
    ("bucket_3", 1025, 2048, 16),
    ("bucket_4", 2049, 4096, 8),
    ("bucket_5", 4097, 6144, 8),
    ("bucket_6", 6145, 8192, 6),
)

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


@dataclass(frozen=True)
class BucketSpec:
    bucket_id: str
    min_len: int
    max_len: int
    batch_size: int


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str
    fields: dict[str, str]

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class BatchTask:
    tax_id: str
    bucket_id: str
    batch_id: int
    records: tuple[FastaRecord, ...]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def parse_header(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in header.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def iter_fasta(path: Path) -> Iterator[FastaRecord]:
    header: str | None = None
    parts: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, sequence="".join(parts), fields=parse_header(header))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        yield FastaRecord(header=header, sequence="".join(parts), fields=parse_header(header))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "value"


def parse_bucket_specs(raw: str | None) -> list[BucketSpec]:
    specs = [BucketSpec(*item) for item in DEFAULT_BUCKETS]
    if not raw:
        return specs

    overrides: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, value = item.split(":", 1)
        overrides[name.strip()] = int(value)
    return [
        BucketSpec(spec.bucket_id, spec.min_len, spec.max_len, overrides.get(spec.bucket_id, spec.batch_size))
        for spec in specs
    ]


def bucket_for_length(length: int, specs: list[BucketSpec]) -> BucketSpec:
    for spec in specs:
        if spec.min_len <= length <= spec.max_len:
            return spec
    raise ValueError(f"No bucket configured for sequence length {length}")


def parse_tax_ids(values: list[str], tax_list: Path | None) -> list[str]:
    tax_ids: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                tax_ids.append(part)
    if tax_list:
        with tax_list.open("rt") as handle:
            for raw_line in handle:
                value = raw_line.strip()
                if value and not value.startswith("#"):
                    tax_ids.append(value.split()[0])
    return tax_ids


def parse_bucket_ids(values: list[str]) -> set[str]:
    bucket_ids: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                bucket_ids.add(part)
    return bucket_ids


def discover_taxa(input_root: Path, requested_taxa: list[str], limit_taxa: int) -> list[str]:
    if requested_taxa:
        taxa = requested_taxa
    else:
        taxa = sorted(
            (path.name.removeprefix("tax_") for path in input_root.glob("tax_*") if path.is_dir()),
            key=lambda value: int(value) if value.isdigit() else value,
        )
    if limit_taxa:
        taxa = taxa[:limit_taxa]
    return taxa


def input_fasta_for_taxon(input_root: Path, tax_id: str) -> Path:
    return input_root / f"tax_{tax_id}" / "kept_representative_gene_cds.fna.gz"


def completed_taxon(output_root: Path, tax_id: str) -> bool:
    summary = output_root / f"tax_{tax_id}" / "run_summary.tsv"
    if not summary.is_file():
        return False
    with summary.open("rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2 and row[0] == "status" and row[1] == "complete":
                return True
    return False


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tsv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_tsv(tmp_path, fieldnames, rows)
    tmp_path.replace(path)


def shard_path_for_task(output_root: Path, task: BatchTask) -> Path:
    return (
        output_root
        / f"tax_{task.tax_id}"
        / "embedding_shards"
        / task.bucket_id
        / f"batch_{task.batch_id:08d}.pt"
    )


def batch_metadata_path(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}.embedding_metadata.tsv")


def batch_metrics_path(shard_path: Path) -> Path:
    return shard_path.with_name(f"{shard_path.stem}.batch_metrics.tsv")


def metadata_row_for_record(
    task: BatchTask,
    record: FastaRecord,
    token_count: int,
    shard_path: Path,
    shard_row: int,
    layer_name: str,
    model_name: str,
    model_path: str | None,
    worker_id: int | str,
    gpu_id: str,
) -> dict[str, object]:
    gene_id = record.fields.get("GeneID", "")
    embedding_id = f"tax_{task.tax_id}|GeneID={gene_id}|bucket={task.bucket_id}|batch={task.batch_id}|row={shard_row}"
    return {
        "embedding_id": embedding_id,
        "tax_id": task.tax_id,
        "GeneID": gene_id,
        "cds_id": record.fields.get("cds_id", ""),
        "sequence_id": record.header,
        "sequence_length": record.length,
        "token_count": token_count,
        "bucket_id": task.bucket_id,
        "batch_id": task.batch_id,
        "batch_size": len(task.records),
        "shard_path": str(shard_path),
        "shard_row": shard_row,
        "layer_name": layer_name,
        "pooled_shape": json.dumps([4096]),
        "pooled_dtype": "torch.float16",
        "model_name": model_name,
        "model_path": model_path,
        "worker_id": worker_id,
        "gpu_id": gpu_id,
    }


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def batch_outputs_complete(
    output_root: Path,
    task: BatchTask,
    layer_name: str,
    model_name: str,
    model_path: str | None,
) -> bool:
    shard_path = shard_path_for_task(output_root, task)
    metadata_path = batch_metadata_path(shard_path)
    metrics_path = batch_metrics_path(shard_path)
    if not shard_path.is_file() or shard_path.stat().st_size == 0:
        return False
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        return False
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return False

    try:
        metadata_rows = read_tsv_rows(metadata_path)
        metrics_rows = read_tsv_rows(metrics_path)
    except (OSError, csv.Error, UnicodeDecodeError):
        return False

    if len(metrics_rows) != 1:
        return False
    metric_row = metrics_rows[0]
    if metric_row.get("status") != "success":
        return False
    if metric_row.get("tax_id") != task.tax_id or metric_row.get("bucket_id") != task.bucket_id:
        return False
    if metric_row.get("batch_id") != str(task.batch_id):
        return False
    if metric_row.get("records") != str(len(task.records)):
        return False
    if metric_row.get("shard_path") != str(shard_path):
        return False

    if len(metadata_rows) != len(task.records):
        return False
    for idx, (row, record) in enumerate(zip(metadata_rows, task.records, strict=True)):
        expected = metadata_row_for_record(
            task=task,
            record=record,
            token_count=record.length,
            shard_path=shard_path,
            shard_row=idx,
            layer_name=layer_name,
            model_name=model_name,
            model_path=model_path,
            worker_id=row.get("worker_id", ""),
            gpu_id=row.get("gpu_id", ""),
        )
        for key in (
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
        ):
            if str(row.get(key, "")) != str(expected[key]):
                return False
    return True


def synchronize_all_gpus(torch_module) -> None:
    if not torch_module.cuda.is_available():
        return
    for device_idx in range(torch_module.cuda.device_count()):
        torch_module.cuda.synchronize(device_idx)


def reset_peak_memory(torch_module) -> None:
    if not torch_module.cuda.is_available():
        return
    for device_idx in range(torch_module.cuda.device_count()):
        torch_module.cuda.reset_peak_memory_stats(device_idx)


def peak_memory_bytes(torch_module) -> int:
    if not torch_module.cuda.is_available():
        return 0
    peaks = [torch_module.cuda.max_memory_allocated(device_idx) for device_idx in range(torch_module.cuda.device_count())]
    return max(peaks) if peaks else 0


def build_input(records: tuple[FastaRecord, ...], tokenizer: object, device: object, torch_module) -> tuple[object, list[int]]:
    tokenized = [tokenizer.tokenize(record.sequence) for record in records]
    lengths = [len(tokens) for tokens in tokenized]
    max_length = max(lengths)
    pad_id = getattr(tokenizer, "pad_id", 0)
    rows = [tokens + [pad_id] * (max_length - len(tokens)) for tokens in tokenized]
    return torch_module.tensor(rows, dtype=torch_module.long, device=device), lengths


def metric_row_for_error(
    task: BatchTask,
    worker_id: int,
    gpu_id: str,
    status: str,
    error: str,
    start_epoch: float,
    end_epoch: float,
) -> dict[str, object]:
    records = len(task.records)
    sequence_tokens = sum(record.length for record in task.records)
    max_token_count = max((record.length for record in task.records), default=0)
    padded_tokens = records * max_token_count
    return {
        "tax_id": task.tax_id,
        "bucket_id": task.bucket_id,
        "batch_id": task.batch_id,
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "status": status,
        "records": records,
        "batch_size": records,
        "max_token_count": max_token_count,
        "sequence_tokens": sequence_tokens,
        "padded_tokens": padded_tokens,
        "wall_start_epoch": f"{start_epoch:.6f}",
        "wall_end_epoch": f"{end_epoch:.6f}",
        "runtime_seconds": f"{(end_epoch - start_epoch):.6f}",
        "tokenize_seconds": "",
        "forward_pool_seconds": "",
        "write_seconds": "",
        "sequence_tokens_per_second": "",
        "padded_tokens_per_second": "",
        "peak_gpu_memory_bytes": "",
        "peak_gpu_memory_gb": "",
        "shard_path": "",
        "error_message": error.replace("\n", " ")[:500],
    }


def worker_main(
    worker_id: int,
    gpu_id: str,
    gpu_visibility_env: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_root: str,
    layer_name: str,
    model_name: str,
    model_path: str | None,
    expected_max_seqlen: str | None,
) -> None:
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if name == gpu_visibility_env:
            os.environ[name] = gpu_id
        else:
            os.environ.pop(name, None)

    original_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    import torch
    if original_affinity is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, original_affinity)
    from evo2 import Evo2

    output_root_path = Path(output_root)
    worker_dir = output_root_path / "_worker_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = worker_dir / f"worker_{worker_id}.embedding_metadata.tsv"
    metrics_path = worker_dir / f"worker_{worker_id}.batch_metrics.tsv"

    with metadata_path.open("wt", newline="") as metadata_handle, metrics_path.open("wt", newline="") as metrics_handle:
        metadata_writer = csv.DictWriter(
            metadata_handle,
            fieldnames=METADATA_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        metrics_writer = csv.DictWriter(
            metrics_handle,
            fieldnames=BATCH_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        metadata_writer.writeheader()
        metrics_writer.writeheader()

        try:
            load_start = time.perf_counter()
            model = Evo2(model_name, local_path=model_path)
            model.model.eval()
            load_seconds = time.perf_counter() - load_start
            inner_config = getattr(getattr(model, "model", None), "config", None)
            actual_max_seqlen = getattr(inner_config, "max_seqlen", None)
            if expected_max_seqlen and actual_max_seqlen != int(expected_max_seqlen):
                raise RuntimeError(f"Expected max_seqlen={expected_max_seqlen}, got {actual_max_seqlen}")
            result_queue.put(
                {
                    "type": "worker_loaded",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "gpu_visibility_env": gpu_visibility_env,
                    "model_load_seconds": load_seconds,
                    "visible_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "max_seqlen": actual_max_seqlen,
                }
            )
        except BaseException as exc:
            result_queue.put(
                {
                    "type": "worker_failed",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "gpu_visibility_env": gpu_visibility_env,
                    "error": repr(exc),
                }
            )
            result_queue.put({"type": "worker_done", "worker_id": worker_id, "gpu_id": gpu_id})
            return

        device = next(model.model.parameters()).device
        while True:
            task = task_queue.get()
            if task is None:
                break

            start_epoch = time.time()
            start_perf = time.perf_counter()
            shard_path = shard_path_for_task(output_root_path, task)
            sidecar_metadata_path = batch_metadata_path(shard_path)
            sidecar_metrics_path = batch_metrics_path(shard_path)
            try:
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                synchronize_all_gpus(torch)
                reset_peak_memory(torch)

                tokenize_start = time.perf_counter()
                input_ids, lengths = build_input(task.records, model.tokenizer, device, torch)
                tokenize_seconds = time.perf_counter() - tokenize_start

                forward_start = time.perf_counter()
                with torch.inference_mode():
                    _, activations = model(
                        input_ids,
                        return_embeddings=True,
                        layer_names=[layer_name],
                    )
                    activation = activations[layer_name]
                    pooled = torch.stack(
                        [activation[row_idx, :length].mean(dim=0) for row_idx, length in enumerate(lengths)]
                    ).to(torch.float16)
                    pooled_cpu = pooled.cpu()
                synchronize_all_gpus(torch)
                forward_pool_seconds = time.perf_counter() - forward_start
                peak_bytes = peak_memory_bytes(torch)

                write_start = time.perf_counter()
                sequence_ids = [record.header for record in task.records]
                torch.save(
                    {
                        "tax_id": task.tax_id,
                        "bucket_id": task.bucket_id,
                        "batch_id": task.batch_id,
                        "sequence_ids": sequence_ids,
                        "header_fields": [record.fields for record in task.records],
                        "sequence_lengths": [record.length for record in task.records],
                        "token_counts": lengths,
                        "layer_name": layer_name,
                        "pooled_embeddings": pooled_cpu,
                        "pooled_dtype": str(pooled_cpu.dtype),
                        "model_name": model_name,
                        "model_path": model_path,
                        "max_seqlen": actual_max_seqlen,
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                    },
                    shard_path,
                )
                write_seconds = time.perf_counter() - write_start
                end_epoch = time.time()
                runtime_seconds = time.perf_counter() - start_perf
                sequence_tokens = sum(lengths)
                max_token_count = max(lengths)
                padded_tokens = len(lengths) * max_token_count

                batch_metadata_rows = []
                for shard_row, (record, token_count) in enumerate(zip(task.records, lengths, strict=True)):
                    metadata_row = metadata_row_for_record(
                        task=task,
                        record=record,
                        token_count=token_count,
                        shard_path=shard_path,
                        shard_row=shard_row,
                        layer_name=layer_name,
                        model_name=model_name,
                        model_path=model_path,
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                    )
                    metadata_row["pooled_shape"] = json.dumps([int(dim) for dim in pooled_cpu[shard_row].shape])
                    metadata_row["pooled_dtype"] = str(pooled_cpu.dtype)
                    batch_metadata_rows.append(metadata_row)
                    metadata_writer.writerow(metadata_row)

                metrics_row = {
                    "tax_id": task.tax_id,
                    "bucket_id": task.bucket_id,
                    "batch_id": task.batch_id,
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "success",
                    "records": len(task.records),
                    "batch_size": len(task.records),
                    "max_token_count": max_token_count,
                    "sequence_tokens": sequence_tokens,
                    "padded_tokens": padded_tokens,
                    "wall_start_epoch": f"{start_epoch:.6f}",
                    "wall_end_epoch": f"{end_epoch:.6f}",
                    "runtime_seconds": f"{runtime_seconds:.6f}",
                    "tokenize_seconds": f"{tokenize_seconds:.6f}",
                    "forward_pool_seconds": f"{forward_pool_seconds:.6f}",
                    "write_seconds": f"{write_seconds:.6f}",
                    "sequence_tokens_per_second": f"{(sequence_tokens / runtime_seconds):.3f}",
                    "padded_tokens_per_second": f"{(padded_tokens / runtime_seconds):.3f}",
                    "peak_gpu_memory_bytes": peak_bytes,
                    "peak_gpu_memory_gb": f"{peak_bytes / (1024**3):.3f}",
                    "shard_path": str(shard_path),
                    "error_message": "",
                }
                metrics_writer.writerow(metrics_row)
                write_tsv_atomic(sidecar_metadata_path, METADATA_COLUMNS, batch_metadata_rows)
                write_tsv_atomic(sidecar_metrics_path, BATCH_COLUMNS, [metrics_row])
                metadata_handle.flush()
                metrics_handle.flush()
                result_queue.put(
                    {
                        "type": "batch_done",
                        "status": "success",
                        "tax_id": task.tax_id,
                        "wall_start_epoch": start_epoch,
                        "wall_end_epoch": end_epoch,
                    }
                )
                del input_ids, activations, activation, pooled, pooled_cpu
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                end_epoch = time.time()
                status = "oom" if "out of memory" in str(exc).lower() else "error"
                metrics_writer.writerow(
                    metric_row_for_error(task, worker_id, gpu_id, status, str(exc), start_epoch, end_epoch)
                )
                metrics_handle.flush()
                torch.cuda.empty_cache()
                result_queue.put(
                    {
                        "type": "batch_done",
                        "status": status,
                        "tax_id": task.tax_id,
                        "error": str(exc),
                        "wall_start_epoch": start_epoch,
                        "wall_end_epoch": end_epoch,
                    }
                )

    result_queue.put({"type": "worker_done", "worker_id": worker_id, "gpu_id": gpu_id})


def write_dry_run(
    output_root: Path,
    selected_taxa: list[str],
    input_root: Path,
    bucket_specs: list[BucketSpec],
    selected_bucket_ids: set[str],
    limit_records_per_taxon: int,
    max_batches_per_taxon: int,
    max_batches_per_bucket: int,
) -> None:
    rows: list[dict[str, object]] = []
    summary_specs = [
        spec for spec in bucket_specs if not selected_bucket_ids or spec.bucket_id in selected_bucket_ids
    ]
    for tax_id in selected_taxa:
        fasta = input_fasta_for_taxon(input_root, tax_id)
        counts: dict[str, int] = defaultdict(int)
        records_seen = 0
        if fasta.is_file():
            for record in iter_fasta(fasta):
                if limit_records_per_taxon and records_seen >= limit_records_per_taxon:
                    break
                spec = bucket_for_length(record.length, bucket_specs)
                if not selected_bucket_ids or spec.bucket_id in selected_bucket_ids:
                    counts[spec.bucket_id] += 1
                records_seen += 1
        for spec in summary_specs:
            batches = (counts[spec.bucket_id] + spec.batch_size - 1) // spec.batch_size
            if max_batches_per_bucket:
                batches = min(batches, max_batches_per_bucket)
            elif max_batches_per_taxon:
                batches = min(batches, max_batches_per_taxon)
            rows.append(
                {
                    "tax_id": tax_id,
                    "bucket_id": spec.bucket_id,
                    "batch_size": spec.batch_size,
                    "records": counts[spec.bucket_id],
                    "batches": batches,
                    "input_fasta": fasta,
                    "input_exists": fasta.is_file(),
                }
            )
    write_tsv(output_root / "dry_run_task_summary.tsv", [
        "tax_id",
        "bucket_id",
        "batch_size",
        "records",
        "batches",
        "input_fasta",
        "input_exists",
    ], rows)


def batch_start_indices(total_records: int, batch_size: int, max_batches_per_bucket: int) -> list[int]:
    """Return batch starts, evenly sampling the bucket when capped."""

    starts = list(range(0, total_records, batch_size))
    if max_batches_per_bucket <= 0 or len(starts) <= max_batches_per_bucket:
        return starts
    if max_batches_per_bucket == 1:
        return [starts[len(starts) // 2]]

    last_index = len(starts) - 1
    selected = {
        starts[round((last_index * idx) / (max_batches_per_bucket - 1))]
        for idx in range(max_batches_per_bucket)
    }
    return sorted(selected)


def iter_taxon_tasks(
    tax_id: str,
    fasta: Path,
    bucket_specs: list[BucketSpec],
    selected_bucket_ids: set[str],
    limit_records_per_taxon: int,
    max_batches_per_taxon: int,
    max_batches_per_bucket: int,
) -> Iterator[BatchTask]:
    buckets: dict[str, list[FastaRecord]] = defaultdict(list)
    records_seen = 0
    for record in iter_fasta(fasta):
        if limit_records_per_taxon and records_seen >= limit_records_per_taxon:
            break
        spec = bucket_for_length(record.length, bucket_specs)
        if not selected_bucket_ids or spec.bucket_id in selected_bucket_ids:
            buckets[spec.bucket_id].append(record)
        records_seen += 1

    batch_id = 0
    batches_emitted = 0
    specs_by_id = {spec.bucket_id: spec for spec in bucket_specs}
    for bucket_id in [
        spec.bucket_id for spec in bucket_specs if not selected_bucket_ids or spec.bucket_id in selected_bucket_ids
    ]:
        records = sorted(buckets[bucket_id], key=lambda record: (record.length, record.header))
        batch_size = specs_by_id[bucket_id].batch_size
        for start in batch_start_indices(len(records), batch_size, max_batches_per_bucket):
            if max_batches_per_taxon and batches_emitted >= max_batches_per_taxon:
                return
            batch_records = tuple(records[start : start + batch_size])
            if batch_records:
                yield BatchTask(tax_id=tax_id, bucket_id=bucket_id, batch_id=batch_id, records=batch_records)
                batch_id += 1
                batches_emitted += 1


def aggregate_worker_outputs(
    output_root: Path,
    selected_taxa: list[str],
) -> tuple[list[dict[str, str]], int]:
    worker_dir = output_root / "_worker_logs"
    metadata_output = output_root / "embedding_metadata.tsv.gz"
    batch_output = output_root / "batch_metrics.tsv"

    sidecar_metadata_paths = sorted(
        path
        for tax_id in selected_taxa
        for path in (output_root / f"tax_{tax_id}" / "embedding_shards").glob(
            "bucket_*/*.embedding_metadata.tsv"
        )
    )
    sidecar_metrics_paths = sorted(
        path
        for tax_id in selected_taxa
        for path in (output_root / f"tax_{tax_id}" / "embedding_shards").glob(
            "bucket_*/*.batch_metrics.tsv"
        )
    )

    metadata_rows = 0
    per_tax_metadata_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata_sources = sidecar_metadata_paths or sorted(
        worker_dir.glob("worker_*.embedding_metadata.tsv")
    )
    for path in metadata_sources:
        with path.open("rt", newline="") as in_handle:
            reader = csv.DictReader(in_handle, delimiter="\t")
            for row in reader:
                per_tax_metadata_rows[row["tax_id"]].append(row)
                metadata_rows += 1

    batch_rows: list[dict[str, str]] = []
    metrics_sources = sidecar_metrics_paths or sorted(worker_dir.glob("worker_*.batch_metrics.tsv"))
    for path in metrics_sources:
        with path.open("rt", newline="") as in_handle:
            reader = csv.DictReader(in_handle, delimiter="\t")
            for row in reader:
                batch_rows.append(row)

    for tax_id in selected_taxa:
        tax_rows = [row for row in batch_rows if row["tax_id"] == tax_id]
        status = (
            "complete"
            if tax_rows and all(row["status"] == "success" for row in tax_rows)
            else "incomplete"
        )
        tax_dir = output_root / f"tax_{tax_id}"
        tax_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(tax_dir / "embedding_metadata.tsv.gz", "wt", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=METADATA_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in per_tax_metadata_rows.get(tax_id, []):
                writer.writerow(row)
        with (tax_dir / "batch_metrics.tsv").open("wt", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=BATCH_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in tax_rows:
                writer.writerow(row)
        with (tax_dir / "run_summary.tsv").open("wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("metric", "value"))
            writer.writerow(("status", status))
            writer.writerow(("tax_id", tax_id))
            writer.writerow(("batches", len(tax_rows)))
            successful_tax_rows = [row for row in tax_rows if row["status"] == "success"]
            writer.writerow(("successful_batches", len(successful_tax_rows)))
            writer.writerow(("records", sum(int(row["records"]) for row in successful_tax_rows)))
            writer.writerow(
                ("sequence_tokens", sum(int(row["sequence_tokens"]) for row in successful_tax_rows))
            )
            writer.writerow(
                ("padded_tokens", sum(int(row["padded_tokens"]) for row in successful_tax_rows))
            )

    with gzip.open(metadata_output, "wt", newline="") as out_handle:
        writer = csv.DictWriter(
            out_handle,
            fieldnames=METADATA_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for tax_dir in sorted(output_root.glob("tax_*"), key=lambda path: path.name):
            tax_id = tax_dir.name.removeprefix("tax_")
            if not completed_taxon(output_root, tax_id):
                continue
            tax_metadata = tax_dir / "embedding_metadata.tsv.gz"
            if not tax_metadata.is_file():
                continue
            with gzip.open(tax_metadata, "rt", newline="") as in_handle:
                reader = csv.DictReader(in_handle, delimiter="\t")
                for row in reader:
                    writer.writerow(row)

    with batch_output.open("wt", newline="") as out_handle:
        writer = csv.DictWriter(
            out_handle,
            fieldnames=BATCH_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for tax_dir in sorted(output_root.glob("tax_*"), key=lambda path: path.name):
            tax_id = tax_dir.name.removeprefix("tax_")
            if not completed_taxon(output_root, tax_id):
                continue
            tax_metrics = tax_dir / "batch_metrics.tsv"
            if not tax_metrics.is_file():
                continue
            with tax_metrics.open("rt", newline="") as in_handle:
                reader = csv.DictReader(in_handle, delimiter="\t")
                for row in reader:
                    writer.writerow(row)

    return batch_rows, metadata_rows


def write_bucket_summary_file(path: Path, batch_rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in batch_rows:
        if row["status"] == "success":
            grouped[row["bucket_id"]].append(row)

    rows = []
    for bucket_id in sorted(grouped):
        rows_for_bucket = grouped[bucket_id]
        sequence_tokens = sum(int(row["sequence_tokens"]) for row in rows_for_bucket)
        padded_tokens = sum(int(row["padded_tokens"]) for row in rows_for_bucket)
        runtime = sum(float(row["runtime_seconds"]) for row in rows_for_bucket)
        rows.append(
            {
                "bucket_id": bucket_id,
                "batches": len(rows_for_bucket),
                "records": sum(int(row["records"]) for row in rows_for_bucket),
                "sequence_tokens": sequence_tokens,
                "padded_tokens": padded_tokens,
                "runtime_seconds_sum": f"{runtime:.6f}",
                "sequence_tokens_per_second_overall": f"{(sequence_tokens / runtime):.3f}" if runtime > 0 else "",
                "padded_tokens_per_second_overall": f"{(padded_tokens / runtime):.3f}" if runtime > 0 else "",
                "mean_batch_sequence_tokens_per_second": f"{mean(float(row['sequence_tokens_per_second']) for row in rows_for_bucket):.3f}",
                "mean_batch_padded_tokens_per_second": f"{mean(float(row['padded_tokens_per_second']) for row in rows_for_bucket):.3f}",
                "mean_peak_gpu_memory_gb": f"{mean(float(row['peak_gpu_memory_gb']) for row in rows_for_bucket):.3f}",
            }
        )

    write_tsv(
        path,
        [
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
        ],
        rows,
    )


def write_bucket_summary(output_root: Path, batch_rows: list[dict[str, str]], selected_taxa: list[str]) -> None:
    write_bucket_summary_file(output_root / "bucket_summary.tsv", batch_rows)
    for tax_id in selected_taxa:
        tax_rows = [row for row in batch_rows if row["tax_id"] == tax_id]
        write_bucket_summary_file(output_root / f"tax_{tax_id}" / "bucket_summary.tsv", tax_rows)


def read_taxon_run_summary(output_root: Path, tax_id: str) -> dict[str, str]:
    path = output_root / f"tax_{tax_id}" / "run_summary.tsv"
    rows: dict[str, str] = {}
    if not path.is_file():
        return rows
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metric = row.get("metric", "")
            if metric:
                rows[metric] = row.get("value", "")
    return rows


def write_dataset_manifest(output_root: Path, selected_taxa: list[str]) -> None:
    rows = []
    manifest_taxa = set(selected_taxa)
    manifest_taxa.update(
        path.name.removeprefix("tax_")
        for path in output_root.glob("tax_*")
        if path.is_dir()
    )
    for tax_id in sorted(manifest_taxa, key=lambda value: int(value) if value.isdigit() else value):
        tax_dir = output_root / f"tax_{tax_id}"
        summary = read_taxon_run_summary(output_root, tax_id)
        rows.append(
            {
                "tax_id": tax_id,
                "status": summary.get("status", "missing"),
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
    write_tsv(output_root / "embedding_dataset_manifest.tsv", DATASET_MANIFEST_COLUMNS, rows)


def write_run_report(
    output_root: Path,
    status: str,
    selected_taxa: list[str],
    submitted_batches: int,
    completed_batches: int,
    failed_batches: int,
    loaded_workers: int,
    failed_workers: int,
    submitted_records: int,
    metadata_rows: int,
    submitted_tokens: int,
    elapsed_seconds_total: float,
    processing_elapsed_seconds: float,
    sum_batch_runtime_seconds: float,
    worker_loads: list[dict[str, object]],
    worker_failures: list[dict[str, object]],
) -> None:
    """Write a concise human-readable run report."""

    throughput_excluding_load = (
        submitted_tokens / processing_elapsed_seconds if processing_elapsed_seconds > 0 else 0.0
    )
    sum_batch_throughput = submitted_tokens / sum_batch_runtime_seconds if sum_batch_runtime_seconds > 0 else 0.0
    report_path = output_root / "embedding_run_report.txt"
    with report_path.open("wt") as handle:
        handle.write("Evo2 Length-Filtered Embedding Run Report\n")
        handle.write("=========================================\n\n")
        handle.write(f"Status: {status}\n")
        handle.write(f"Output root: {output_root}\n")
        handle.write(f"Taxa: {','.join(selected_taxa)}\n\n")
        handle.write("Counts\n")
        handle.write("------\n")
        handle.write(f"Submitted batches: {submitted_batches}\n")
        handle.write(f"Completed batches: {completed_batches}\n")
        handle.write(f"Failed batches: {failed_batches}\n")
        handle.write(f"Submitted records: {submitted_records}\n")
        handle.write(f"Metadata rows: {metadata_rows}\n")
        handle.write(f"Submitted sequence tokens: {submitted_tokens}\n\n")
        handle.write("Timing\n")
        handle.write("------\n")
        handle.write(f"Total elapsed seconds including model load: {elapsed_seconds_total:.3f}\n")
        handle.write(f"Processing elapsed seconds excluding model load: {processing_elapsed_seconds:.3f}\n")
        handle.write(f"Sum of batch runtime seconds: {sum_batch_runtime_seconds:.3f}\n")
        handle.write(
            f"Overall sequence tokens/sec excluding model load: {throughput_excluding_load:.3f}\n"
        )
        handle.write(f"Sum-batch sequence tokens/sec: {sum_batch_throughput:.3f}\n\n")
        handle.write("Workers\n")
        handle.write("-------\n")
        handle.write(f"Loaded workers: {loaded_workers}\n")
        handle.write(f"Failed workers: {failed_workers}\n")
        for worker in worker_loads:
            handle.write(
                "Loaded worker "
                f"{worker.get('worker_id')} gpu={worker.get('gpu_id')} "
                f"visible_gpus={worker.get('visible_gpus')} "
                f"model_load_seconds={float(worker.get('model_load_seconds', 0.0)):.3f}\n"
            )
        for worker in worker_failures:
            handle.write(
                "Failed worker "
                f"{worker.get('worker_id')} gpu={worker.get('gpu_id')} "
                f"error={worker.get('error')}\n"
            )
        handle.write("\nFiles\n")
        handle.write("-----\n")
        handle.write("run_summary.tsv\n")
        handle.write("bucket_summary.tsv\n")
        handle.write("batch_metrics.tsv\n")
        handle.write("embedding_metadata.tsv.gz\n")
        handle.write("tax_<tax_id>/embedding_shards/bucket_*/batch_*.pt\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tax-id", action="append", default=[])
    parser.add_argument("--tax-list", type=Path)
    parser.add_argument("--limit-taxa", type=int, default=0)
    parser.add_argument(
        "--incomplete-only",
        action="store_true",
        help="Run only taxa whose tax_<tax_id>/run_summary.tsv is missing or not complete.",
    )
    parser.add_argument(
        "--limit-runnable-taxa",
        type=int,
        default=0,
        help="Limit taxa after missing/completed filtering. Use 0 for no limit.",
    )
    parser.add_argument(
        "--bucket-id",
        action="append",
        default=[],
        help="Restrict work to one or more bucket IDs, for example --bucket-id bucket_1.",
    )
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EVO2_EMBED_MAX_WORKERS", "4")))
    parser.add_argument("--gpu-ids", default=os.environ.get("EVO2_EMBED_GPU_IDS", "0,1,2,3"))
    parser.add_argument(
        "--gpu-visibility-env",
        default=os.environ.get("EVO2_GPU_VISIBILITY_ENV", "CUDA_VISIBLE_DEVICES"),
        choices=("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"),
        help="Single environment variable used to isolate one GPU per worker.",
    )
    parser.add_argument("--bucket-batch-sizes", default=os.environ.get("EVO2_BUCKET_BATCH_SIZES"))
    parser.add_argument("--layer-name", default=os.environ.get("EVO2_LAYER_NAME", DEFAULT_LAYER))
    parser.add_argument(
        "--model-name",
        default=os.environ.get("EVO2_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Evo2 wrapper model name.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("EVO2_MODEL_PATH", DEFAULT_MODEL_PATH)),
        help=(
            "Local Evo2 checkpoint. Defaults to the remodel-local checkpoint path "
            "evo2_model/evo2_7b_base.pt."
        ),
    )
    parser.add_argument(
        "--expected-max-seqlen",
        default=os.environ.get("EVO2_EXPECTED_MAX_SEQLEN", DEFAULT_EXPECTED_MAX_SEQLEN),
        help="Expected max_seqlen reported by the loaded Evo2 model. Use an empty string to skip the check.",
    )
    parser.add_argument("--limit-records-per-taxon", type=int, default=0)
    parser.add_argument("--max-batches-per-taxon", type=int, default=0)
    parser.add_argument(
        "--max-batches-per-bucket",
        type=int,
        default=0,
        help="Limit batches sampled from each length bucket. Capped buckets are sampled evenly by length.",
    )
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument(
        "--no-progress-timeout-seconds",
        type=int,
        default=int(os.environ.get("EVO2_NO_PROGRESS_TIMEOUT_SECONDS", "0")),
        help="Abort if no successful batch or worker-load progress is observed for this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-failed-batches",
        type=int,
        default=int(os.environ.get("EVO2_MAX_FAILED_BATCHES", "0")),
        help="Abort after this many failed batches. Use 0 to allow all tasks to finish.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument(
        "--resume-completed-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip individual batch shards that already have matching .pt, metadata, and metrics sidecars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bucket_specs = parse_bucket_specs(args.bucket_batch_sizes)
    requested_taxa = parse_tax_ids(args.tax_id, args.tax_list)
    selected_bucket_ids = parse_bucket_ids(args.bucket_id)
    valid_bucket_ids = {spec.bucket_id for spec in bucket_specs}
    unknown_bucket_ids = sorted(selected_bucket_ids - valid_bucket_ids)
    if unknown_bucket_ids:
        raise ValueError(f"Unknown --bucket-id value(s): {','.join(unknown_bucket_ids)}")
    selected_taxa = discover_taxa(args.input_root, requested_taxa, args.limit_taxa)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if len(gpu_ids) < args.max_workers:
        raise ValueError(f"Need at least {args.max_workers} GPU ids, got {gpu_ids}")
    if args.limit_runnable_taxa < 0:
        raise ValueError("--limit-runnable-taxa must be >= 0")
    if args.no_progress_timeout_seconds < 0:
        raise ValueError("--no-progress-timeout-seconds must be >= 0")
    if args.max_failed_batches < 0:
        raise ValueError("--max-failed-batches must be >= 0")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_dry_run(
            args.output_root,
            selected_taxa,
            args.input_root,
            bucket_specs,
            selected_bucket_ids,
            args.limit_records_per_taxon,
            args.max_batches_per_taxon,
            args.max_batches_per_bucket,
        )
        log(f"Wrote dry-run task summary: {args.output_root / 'dry_run_task_summary.tsv'}")
        return 0

    runnable_taxa = []
    skipped_taxa = []
    missing_taxa = []
    deferred_taxa = []
    for tax_id in selected_taxa:
        fasta = input_fasta_for_taxon(args.input_root, tax_id)
        if not fasta.is_file():
            missing_taxa.append(tax_id)
            continue
        if completed_taxon(args.output_root, tax_id) and args.skip_completed and not args.overwrite:
            skipped_taxa.append(tax_id)
            continue
        runnable_taxa.append(tax_id)
    if args.incomplete_only:
        selected_taxa = runnable_taxa + skipped_taxa + missing_taxa
    if args.limit_runnable_taxa:
        deferred_taxa = runnable_taxa[args.limit_runnable_taxa :]
        runnable_taxa = runnable_taxa[: args.limit_runnable_taxa]

    if not runnable_taxa:
        with (args.output_root / "run_summary.tsv").open("wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("metric", "value"))
            writer.writerow(("status", "complete"))
            writer.writerow(("input_root", args.input_root))
            writer.writerow(("output_root", args.output_root))
            writer.writerow(("requested_taxa", len(selected_taxa)))
            writer.writerow(("runnable_taxa", 0))
            writer.writerow(("skipped_completed_taxa", len(skipped_taxa)))
            writer.writerow(("missing_input_taxa", len(missing_taxa)))
            writer.writerow(("deferred_taxa", len(deferred_taxa)))
            writer.writerow(("message", "No runnable taxa selected"))
        log("No runnable taxa selected; exiting without loading Evo2")
        return 0

    model_name = args.model_name
    model_path = str(args.model_path) if args.model_path else None
    expected_max_seqlen = args.expected_max_seqlen or None
    if model_path is not None and not Path(model_path).is_file():
        raise FileNotFoundError(
            "Missing Evo2 model checkpoint: "
            f"{model_path}. Copy or symlink the checkpoint into the remodel tree, "
            "or pass --model-path with an explicit site-local checkpoint path."
        )
    start_time = time.time()

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(args.queue_size, args.max_workers))
    result_queue = ctx.Queue()
    workers = []
    for worker_id in range(args.max_workers):
        process = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                args.gpu_visibility_env,
                task_queue,
                result_queue,
                str(args.output_root),
                args.layer_name,
                model_name,
                model_path,
                expected_max_seqlen,
            ),
        )
        process.start()
        workers.append(process)

    loaded_workers = 0
    finished_workers = 0
    failed_workers = 0
    completed_batches = 0
    failed_batches = 0
    worker_loads: list[dict[str, object]] = []
    worker_failures: list[dict[str, object]] = []
    planned_batches = 0
    planned_records = 0
    planned_tokens = 0
    new_batches_submitted = 0
    new_records_submitted = 0
    new_tokens_submitted = 0
    skipped_existing_batches = 0
    skipped_existing_records = 0
    skipped_existing_tokens = 0
    earliest_batch_start_epoch: float | None = None
    latest_batch_end_epoch: float | None = None
    last_progress_time = time.time()
    abort_reason = ""

    def check_no_progress_timeout() -> None:
        nonlocal abort_reason
        if abort_reason or args.no_progress_timeout_seconds <= 0:
            return
        if new_batches_submitted <= completed_batches:
            return
        idle_seconds = time.time() - last_progress_time
        if idle_seconds >= args.no_progress_timeout_seconds:
            abort_reason = (
                f"No successful batch or worker-load progress for {idle_seconds:.1f} seconds "
                f"while {completed_batches}/{new_batches_submitted} submitted batches were complete"
            )

    def handle_message(message: dict[str, object]) -> None:
        nonlocal loaded_workers
        nonlocal finished_workers
        nonlocal failed_workers
        nonlocal completed_batches
        nonlocal failed_batches
        nonlocal earliest_batch_start_epoch
        nonlocal latest_batch_end_epoch
        nonlocal last_progress_time
        nonlocal abort_reason

        try:
            message_type = str(message["type"])
        except KeyError:
            log(f"WARNING: Ignoring malformed worker message: {message}")
            return

        if message_type == "worker_loaded":
            loaded_workers += 1
            worker_loads.append(message)
            last_progress_time = time.time()
            log(
                f"Worker {message['worker_id']} loaded on gpu_id={message['gpu_id']} "
                f"via {message['gpu_visibility_env']} visible_gpus={message['visible_gpus']} "
                f"load_seconds={message['model_load_seconds']:.1f}"
            )
        elif message_type == "worker_failed":
            failed_workers += 1
            worker_failures.append(message)
            log(
                f"Worker {message['worker_id']} failed on gpu_id={message['gpu_id']} "
                f"via {message['gpu_visibility_env']}: {message['error']}"
            )
        elif message_type == "batch_done":
            completed_batches += 1
            if message["status"] != "success":
                failed_batches += 1
                if args.max_failed_batches and failed_batches >= args.max_failed_batches and not abort_reason:
                    abort_reason = f"Failed batch threshold reached: {failed_batches}/{args.max_failed_batches}"
            else:
                last_progress_time = time.time()
            start_epoch = message.get("wall_start_epoch")
            end_epoch = message.get("wall_end_epoch")
            if start_epoch is not None:
                earliest_batch_start_epoch = (
                    float(start_epoch)
                    if earliest_batch_start_epoch is None
                    else min(earliest_batch_start_epoch, float(start_epoch))
                )
            if end_epoch is not None:
                latest_batch_end_epoch = (
                    float(end_epoch)
                    if latest_batch_end_epoch is None
                    else max(latest_batch_end_epoch, float(end_epoch))
                )
            if completed_batches % 25 == 0 or completed_batches == new_batches_submitted:
                log(
                    f"Progress: completed_new_batches={completed_batches}/{new_batches_submitted} "
                    f"skipped_existing_batches={skipped_existing_batches}/{planned_batches} "
                    f"failed_batches={failed_batches}"
                )
        elif message_type == "worker_done":
            finished_workers += 1
            log(f"Worker {message['worker_id']} done")

    def drain_result_messages(block: bool = False, timeout: float = 0.0) -> bool:
        got_message = False
        while True:
            try:
                message = result_queue.get(timeout=timeout) if block and not got_message else result_queue.get_nowait()
            except queue.Empty:
                return got_message
            got_message = True
            handle_message(message)
            if abort_reason:
                return got_message

    def queue_put_with_progress(item: BatchTask | None) -> bool:
        nonlocal abort_reason
        while not abort_reason:
            drain_result_messages()
            if finished_workers >= len(workers):
                abort_reason = "All workers exited before all tasks were submitted"
                return False
            try:
                task_queue.put(item, timeout=1.0)
                return True
            except queue.Full:
                check_no_progress_timeout()
        return False

    for tax_id in runnable_taxa:
        if abort_reason:
            break
        fasta = input_fasta_for_taxon(args.input_root, tax_id)
        log(f"Submitting tax_id={tax_id} from {fasta}")
        for task in iter_taxon_tasks(
            tax_id,
            fasta,
            bucket_specs,
            selected_bucket_ids,
            args.limit_records_per_taxon,
            args.max_batches_per_taxon,
            args.max_batches_per_bucket,
        ):
            if abort_reason:
                break
            task_records = len(task.records)
            task_tokens = sum(record.length for record in task.records)
            planned_batches += 1
            planned_records += task_records
            planned_tokens += task_tokens
            if (
                args.resume_completed_batches
                and not args.overwrite
                and batch_outputs_complete(
                    output_root=args.output_root,
                    task=task,
                    layer_name=args.layer_name,
                    model_name=model_name,
                    model_path=model_path,
                )
            ):
                skipped_existing_batches += 1
                skipped_existing_records += task_records
                skipped_existing_tokens += task_tokens
                continue
            if queue_put_with_progress(task):
                new_batches_submitted += 1
                new_records_submitted += task_records
                new_tokens_submitted += task_tokens
        drain_result_messages()

    if not abort_reason:
        for _ in workers:
            if not queue_put_with_progress(None):
                break

    while not abort_reason and finished_workers < len(workers):
        if not drain_result_messages(block=True, timeout=30):
            check_no_progress_timeout()
            log(
                f"Progress: completed_new_batches={completed_batches}/{new_batches_submitted} "
                f"skipped_existing_batches={skipped_existing_batches}/{planned_batches} "
                f"failed_batches={failed_batches}"
            )

    if abort_reason:
        log(f"Aborting embedding run: {abort_reason}")
        for process in workers:
            if process.is_alive():
                process.terminate()

    for process in workers:
        process.join()
        if process.exitcode != 0 and not abort_reason:
            raise RuntimeError(f"Worker pid={process.pid} exited with code {process.exitcode}")

    batch_rows, metadata_rows = aggregate_worker_outputs(args.output_root, runnable_taxa)
    write_bucket_summary(args.output_root, batch_rows, runnable_taxa)
    write_dataset_manifest(args.output_root, runnable_taxa)
    elapsed_seconds_total = time.time() - start_time
    successful_batch_rows = [row for row in batch_rows if row["status"] == "success"]
    if successful_batch_rows:
        earliest_batch_start_epoch = min(float(row["wall_start_epoch"]) for row in successful_batch_rows)
        latest_batch_end_epoch = max(float(row["wall_end_epoch"]) for row in successful_batch_rows)
    processing_elapsed_seconds = (
        latest_batch_end_epoch - earliest_batch_start_epoch
        if earliest_batch_start_epoch is not None and latest_batch_end_epoch is not None
        else 0.0
    )
    sum_batch_runtime_seconds = sum(float(row["runtime_seconds"]) for row in successful_batch_rows)
    final_completed_batches = len(successful_batch_rows)
    final_records = sum(int(row["records"]) for row in successful_batch_rows)
    final_tokens = sum(int(row["sequence_tokens"]) for row in successful_batch_rows)
    status = (
        "complete"
        if (
            not abort_reason
            and failed_batches == 0
            and completed_batches == new_batches_submitted
            and failed_workers == 0
            and final_completed_batches == planned_batches
            and metadata_rows == planned_records
        )
        else "incomplete"
    )
    with (args.output_root / "run_summary.tsv").open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerow(("status", status))
        writer.writerow(("input_root", args.input_root))
        writer.writerow(("output_root", args.output_root))
        writer.writerow(("requested_taxa", len(selected_taxa)))
        writer.writerow(("runnable_taxa", len(runnable_taxa)))
        writer.writerow(("skipped_completed_taxa", len(skipped_taxa)))
        writer.writerow(("missing_input_taxa", len(missing_taxa)))
        writer.writerow(("deferred_taxa", len(deferred_taxa)))
        writer.writerow(("submitted_batches", planned_batches))
        writer.writerow(("completed_batches", final_completed_batches))
        writer.writerow(("failed_batches", failed_batches))
        writer.writerow(("abort_reason", abort_reason))
        writer.writerow(("new_batches_submitted", new_batches_submitted))
        writer.writerow(("new_batches_completed", completed_batches))
        writer.writerow(("skipped_existing_batches", skipped_existing_batches))
        writer.writerow(("loaded_workers", loaded_workers))
        writer.writerow(("failed_workers", failed_workers))
        writer.writerow(("submitted_records", planned_records))
        writer.writerow(("metadata_rows", metadata_rows))
        writer.writerow(("submitted_sequence_tokens", planned_tokens))
        writer.writerow(("new_records_submitted", new_records_submitted))
        writer.writerow(("skipped_existing_records", skipped_existing_records))
        writer.writerow(("embedded_records_from_sidecars", final_records))
        writer.writerow(("new_sequence_tokens_submitted", new_tokens_submitted))
        writer.writerow(("skipped_existing_sequence_tokens", skipped_existing_tokens))
        writer.writerow(("embedded_sequence_tokens_from_sidecars", final_tokens))
        writer.writerow(("elapsed_seconds_total_including_model_load", f"{elapsed_seconds_total:.3f}"))
        writer.writerow(("processing_elapsed_seconds_excluding_model_load", f"{processing_elapsed_seconds:.3f}"))
        writer.writerow(("sum_batch_runtime_seconds", f"{sum_batch_runtime_seconds:.3f}"))
        writer.writerow(("overall_sequence_tokens_per_second_excluding_model_load", f"{(final_tokens / processing_elapsed_seconds):.3f}" if processing_elapsed_seconds > 0 else ""))
        writer.writerow(("sum_batch_sequence_tokens_per_second", f"{(final_tokens / sum_batch_runtime_seconds):.3f}" if sum_batch_runtime_seconds > 0 else ""))
        writer.writerow(("model_name", model_name))
        writer.writerow(("model_path", model_path))
        writer.writerow(("layer_name", args.layer_name))
        writer.writerow(("max_workers", args.max_workers))
        writer.writerow(("gpu_ids", ",".join(gpu_ids[: args.max_workers])))
        writer.writerow(("gpu_visibility_env", args.gpu_visibility_env))
        writer.writerow(("no_progress_timeout_seconds", args.no_progress_timeout_seconds))
        writer.writerow(("max_failed_batches", args.max_failed_batches))
        writer.writerow(("bucket_batch_sizes", ",".join(f"{spec.bucket_id}:{spec.batch_size}" for spec in bucket_specs)))
        writer.writerow(("selected_bucket_ids", ",".join(sorted(selected_bucket_ids)) if selected_bucket_ids else "all"))
        writer.writerow(("resume_completed_batches", args.resume_completed_batches))
        writer.writerow(("max_batches_per_taxon", args.max_batches_per_taxon))
        writer.writerow(("max_batches_per_bucket", args.max_batches_per_bucket))
        writer.writerow(("worker_loads_json", json.dumps(worker_loads)))
        writer.writerow(("worker_failures_json", json.dumps(worker_failures)))

    write_run_report(
        args.output_root,
        status,
        runnable_taxa,
        planned_batches,
        final_completed_batches,
        failed_batches,
        loaded_workers,
        failed_workers,
        planned_records,
        metadata_rows,
        final_tokens,
        elapsed_seconds_total,
        processing_elapsed_seconds,
        sum_batch_runtime_seconds,
        worker_loads,
        worker_failures,
    )

    log(f"Wrote embeddings: {args.output_root}")
    log(f"Wrote batch metrics: {args.output_root / 'batch_metrics.tsv'}")
    log(f"Wrote bucket summary: {args.output_root / 'bucket_summary.tsv'}")
    log(f"Wrote run report: {args.output_root / 'embedding_run_report.txt'}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
