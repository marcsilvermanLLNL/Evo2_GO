#!/usr/bin/env python3
"""Train the Evo2-GO dual encoder on precomputed Evo2 embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from evo2_go.model.architecture import (
    AsymmetricLoss,
    AsymmetricLossConfig,
    DualEncoderConfig,
    DualEncoderModel,
    GoTermGCNConfig,
    SequenceEncoderConfig,
    load_sparse_adjacency_npz,
)
from evo2_go.model.dataset import (
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_GO_VOCABULARY,
    Evo2GoEmbeddingDataset,
    evo2_go_collate_fn,
    load_evo2_go_dataset,
    tax_ids_from_file,
)
from evo2_go.model.training_utils import (
    AdamWConfig,
    GradientClippingConfig,
    WarmupCosineLRConfig,
    WarmupCosineLRScheduler,
    build_adamw_optimizer,
    cafa_gene_averaged_fmax,
    clip_global_grad_norm,
    load_training_checkpoint,
    save_training_checkpoint,
)
from evo2_go.training.config import defaults_from_config, resolve_project_root


DEFAULT_ADJACENCY = Path(
    "outputs/model_vocab/go_adjacency_non_iea_min_count_10/go_adjacency_sparse.npz"
)
DEFAULT_OUTPUT_DIR = Path("outputs/model_training/non_iea_min_count_10/runs/smoke")
METRIC_FIELDS = (
    "event",
    "step",
    "examples_seen",
    "effective_epochs",
    "loss",
    "positive_loss",
    "negative_loss",
    "lr",
    "grad_norm",
    "positive_labels_per_batch",
    "negative_labels_per_batch",
    "positive_labels_per_example",
    "mean_positive_probability",
    "mean_negative_probability",
    "logit_scale",
    "fmax",
    "threshold",
    "precision",
    "recall",
    "validation_batches",
    "validation_examples",
    "elapsed_seconds",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cycle_loader(loader: DataLoader) -> Iterator[dict[str, object]]:
    while True:
        yield from loader


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    tmp_path.replace(path)


def write_metrics_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = METRIC_FIELDS
    if path.exists() and path.stat().st_size > 0:
        with path.open("rt", newline="") as handle:
            header = handle.readline().rstrip("\n").split("\t")
        if header and header != list(METRIC_FIELDS):
            fieldnames = tuple(header)
    with path.open("at", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def effective_epochs(examples_seen: int, train_records: int) -> float:
    if train_records <= 0:
        return 0.0
    return examples_seen / float(train_records)


def artifact_size(path: Path | None) -> int | str:
    if path is None:
        return ""
    try:
        if not path.is_file():
            return ""
        return path.stat().st_size
    except FileNotFoundError:
        return ""


def artifact_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_tax_ids(tax_ids: list[str] | None) -> str:
    if not tax_ids:
        return ""
    preview = ",".join(tax_ids[:10])
    suffix = ",..." if len(tax_ids) > 10 else ""
    return f"{len(tax_ids)} values: {preview}{suffix}"


def build_static_run_metadata(
    args: argparse.Namespace,
    *,
    project_root: Path,
    run_started_at_utc: str,
    device: torch.device,
    train_bundle,
    val_bundle,
    n_terms: int,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Collect reproducibility and data-exposure metadata for this run."""

    return {
        "run_started_at_utc": run_started_at_utc,
        "project_root": str(project_root),
        "config": str(args.config or ""),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": str(device),
        "embedding_root": str(args.embedding_root),
        "embedding_root_size_bytes": artifact_size(args.embedding_root),
        "vocabulary": str(args.vocabulary),
        "vocabulary_size_bytes": artifact_size(args.vocabulary),
        "adjacency": str(args.adjacency),
        "adjacency_size_bytes": artifact_size(args.adjacency),
        "label_cache": str(args.label_cache),
        "label_cache_size_bytes": artifact_size(args.label_cache),
        "split_manifest": str(args.split_manifest),
        "split_manifest_size_bytes": artifact_size(args.split_manifest),
        "validation_protocol": (
            "fixed_panel"
            if args.validation_panel is not None
            else "ordered_prefix"
            if args.validation_batches > 0
            else "full_split"
        ),
        "validation_panel": str(args.validation_panel or ""),
        "validation_panel_size_bytes": artifact_size(args.validation_panel),
        "validation_panel_sha256": artifact_sha256(args.validation_panel),
        "tax_ids_file": str(args.tax_ids_file or ""),
        "tax_ids_count": len(args.tax_ids or []),
        "output_dir": str(args.output_dir),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "records_before_label_filter_train": train_bundle.records_before_label_filter,
        "records_before_label_filter_val": val_bundle.records_before_label_filter,
        "train_records": len(train_bundle.dataset),
        "val_records": len(val_bundle.dataset),
        "label_index_size_train": train_bundle.label_index_size,
        "label_index_size_val": val_bundle.label_index_size,
        "n_terms": n_terms,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "validate_every": args.validate_every,
        "validation_batches": args.validation_batches,
        "reset_best_on_resume": args.reset_best_on_resume,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "peak_lr": args.peak_lr,
        "weight_decay": args.weight_decay,
        "loss_reduction": args.loss_reduction,
        "loss_positive_weight": args.loss_positive_weight,
        "loss_negative_weight": args.loss_negative_weight,
        "max_cached_shards": args.max_cached_shards,
        "max_records": args.max_records or "",
        "resume_checkpoint": str(args.resume_checkpoint or ""),
        "stop_after_seconds": args.stop_after_seconds or "",
        "model_class": type(model).__name__,
        "model_parameters": parameter_count(model),
        "model_trainable_parameters": trainable_parameter_count(model),
        "split_note": "temporary engineering split unless ortholog_safe=1 in manifest",
    }


@torch.no_grad()
def validate(
    model: DualEncoderModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    thresholds: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    batches_seen = 0
    for batch in loader:
        embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
        batch_targets = batch["targets"].to(device=device, dtype=torch.float32)
        probs = model.predict_proba(embeddings)
        probabilities.append(probs.cpu())
        targets.append(batch_targets.cpu())
        batches_seen += 1
        if max_batches > 0 and batches_seen >= max_batches:
            break

    if not probabilities:
        raise ValueError("Validation loader produced no batches")

    result = cafa_gene_averaged_fmax(
        torch.cat(probabilities, dim=0),
        torch.cat(targets, dim=0),
        thresholds=thresholds,
    )
    model.train()
    return {
        "fmax": result.fmax,
        "threshold": result.threshold,
        "precision": result.precision,
        "recall": result.recall,
        "validation_batches": batches_seen,
        "validation_examples": int(sum(tensor.shape[0] for tensor in targets)),
    }


def build_loader(
    dataset: Evo2GoEmbeddingDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=evo2_go_collate_fn,
    )


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    pre_args, _ = pre_parser.parse_known_args()
    project_root = resolve_project_root(pre_args.project_root)
    config_defaults = defaults_from_config(
        pre_args.config,
        project_root=project_root,
        section="training",
    )

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_GO_VOCABULARY)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument(
        "--validation-panel",
        type=Path,
        help=(
            "Saved split-manifest subset used as the entire validation dataset. "
            "Requires --validation-batches 0."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--loss-reduction",
        choices=("balanced", "mean"),
        default="balanced",
        help=(
            "ASL reduction. 'mean' is the old dense batch x term mean; "
            "'balanced' averages positive and negative label losses separately."
        ),
    )
    parser.add_argument("--loss-positive-weight", type=float, default=1.0)
    parser.add_argument("--loss-negative-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tax-id",
        dest="tax_ids",
        action="append",
        help="Restrict loaded embedding records to these tax IDs. Can be repeated or comma-delimited.",
    )
    parser.add_argument(
        "--tax-ids-file",
        "--tax-id-file",
        dest="tax_ids_file",
        type=Path,
        help="Restrict loaded embedding records to tax IDs listed in this file.",
    )
    parser.add_argument("--max-cached-shards", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--reset-best-on-resume",
        action="store_true",
        help=(
            "Reset best Fmax/step after loading a checkpoint. Use this when the "
            "validation protocol changed."
        ),
    )
    parser.add_argument(
        "--stop-after-seconds",
        type=float,
        help="Gracefully stop after this many wall-clock seconds by saving latest_checkpoint.pt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved training inputs without loading data or starting training.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if args.label_cache is None:
        parser.error("--label-cache is required unless supplied by --config")
    if args.split_manifest is None:
        parser.error("--split-manifest is required unless supplied by --config")
    if args.validation_panel is not None and args.validation_batches != 0:
        parser.error("--validation-panel requires --validation-batches 0")

    file_tax_ids = tax_ids_from_file(args.tax_ids_file)
    if file_tax_ids:
        args.tax_ids = [*(args.tax_ids or []), *file_tax_ids]

    if args.smoke:
        args.max_steps = min(args.max_steps, 20)
        args.warmup_steps = min(args.warmup_steps, args.max_steps)
        args.validate_every = min(args.validate_every, 10)
        args.validation_batches = min(args.validation_batches, 2)
        args.checkpoint_every = min(args.checkpoint_every, 20)
        args.log_every = min(args.log_every, 1)
        args.batch_size = min(args.batch_size, 8)
        args.num_workers = min(args.num_workers, 0)

    if args.dry_run:
        for key in (
            "config",
            "project_root",
            "embedding_root",
            "vocabulary",
            "adjacency",
            "label_cache",
            "split_manifest",
            "output_dir",
            "train_split",
            "val_split",
            "batch_size",
            "num_workers",
            "max_steps",
            "warmup_steps",
            "validate_every",
            "validation_batches",
            "validation_panel",
            "checkpoint_every",
            "log_every",
            "peak_lr",
            "weight_decay",
            "loss_reduction",
            "loss_positive_weight",
            "loss_negative_weight",
            "device",
            "tax_ids_file",
            "tax_ids",
            "tax_ids_count",
            "stop_after_seconds",
            "resume_checkpoint",
            "reset_best_on_resume",
        ):
            if key == "tax_ids_count":
                value = len(args.tax_ids or [])
            elif key == "tax_ids":
                value = summarize_tax_ids(args.tax_ids)
            else:
                value = getattr(args, key)
            print(f"{key}\t{value}")
        return 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_started_at_utc = utc_now_iso()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.tsv"
    run_metadata_path = args.output_dir / "training_run_metadata.tsv"
    run_summary_path = args.output_dir / "training_run_summary.tsv"

    train_bundle = load_evo2_go_dataset(
        embedding_root=args.embedding_root,
        vocabulary_path=args.vocabulary,
        label_index_cache_path=args.label_cache,
        split_manifest_path=args.split_manifest,
        split_name=args.train_split,
        tax_ids=args.tax_ids,
        max_records=args.max_records,
        max_cached_shards=args.max_cached_shards,
    )
    val_bundle = load_evo2_go_dataset(
        embedding_root=args.embedding_root,
        vocabulary_path=args.vocabulary,
        label_index_cache_path=args.label_cache,
        split_manifest_path=args.validation_panel or args.split_manifest,
        split_name=args.val_split,
        tax_ids=args.tax_ids,
        max_records=args.max_records,
        max_cached_shards=args.max_cached_shards,
        preselect_split_records=args.validation_panel is not None,
    )
    if train_bundle.vocabulary.go_ids != val_bundle.vocabulary.go_ids:
        raise ValueError("Train and validation vocabularies differ")

    train_loader = build_loader(
        train_bundle.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        val_bundle.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    adjacency = load_sparse_adjacency_npz(args.adjacency)
    n_terms = train_bundle.vocabulary.n_terms
    model = DualEncoderModel(
        adjacency,
        DualEncoderConfig(
            go_encoder=GoTermGCNConfig(n_terms=n_terms),
            sequence_encoder=SequenceEncoderConfig(),
        ),
    ).to(device)
    loss_fn = AsymmetricLoss(
        AsymmetricLossConfig(
            reduction=args.loss_reduction,
            positive_weight=args.loss_positive_weight,
            negative_weight=args.loss_negative_weight,
        )
    )
    optimizer = build_adamw_optimizer(
        model,
        AdamWConfig(lr=args.peak_lr, weight_decay=args.weight_decay),
    )
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        WarmupCosineLRConfig(
            peak_lr=args.peak_lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
        ),
    )
    thresholds = torch.linspace(0.01, 0.99, 50)

    static_metadata = build_static_run_metadata(
        args,
        project_root=project_root,
        run_started_at_utc=run_started_at_utc,
        device=device,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        n_terms=n_terms,
        model=model,
    )
    write_key_value_tsv(run_metadata_path, list(static_metadata.items()))

    best_fmax = -1.0
    best_step = 0
    start_step = 0
    examples_seen = 0
    last_step = 0
    stop_reason = "max_steps"
    last_validation: dict[str, float] = {}

    def dynamic_metadata(step: int) -> dict[str, Any]:
        return {
            **static_metadata,
            "checkpoint_step": step,
            "examples_seen": examples_seen,
            "effective_epochs": f"{effective_epochs(examples_seen, len(train_bundle.dataset)):.8f}",
            "best_fmax": best_fmax,
            "best_step": best_step,
        }

    if args.resume_checkpoint is not None:
        checkpoint = load_training_checkpoint(
            args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_step = int(checkpoint.get("step", 0))
        best_fmax = float(checkpoint.get("best_fmax", -1.0) or -1.0)
        checkpoint_metadata = checkpoint.get("metadata", {})
        if isinstance(checkpoint_metadata, dict):
            examples_seen = int(
                checkpoint_metadata.get("examples_seen", start_step * args.batch_size)
                or 0
            )
            best_step = int(checkpoint_metadata.get("best_step", 0) or 0)
        else:
            examples_seen = start_step * args.batch_size
        if args.reset_best_on_resume:
            best_fmax = -1.0
            best_step = 0
        write_metrics_row(
            metrics_path,
            {
                "event": "resume",
                "step": start_step,
                "examples_seen": examples_seen,
                "effective_epochs": f"{effective_epochs(examples_seen, len(train_bundle.dataset)):.8f}",
                "lr": f"{scheduler.get_lr():.12g}",
                "fmax": f"{best_fmax:.8f}",
                "elapsed_seconds": f"{time.time() - started:.3f}",
            },
        )

    train_iterator = cycle_loader(train_loader)
    model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        last_step = step
        batch = next(train_iterator)
        embeddings = batch["embeddings"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        examples_seen += int(embeddings.shape[0])

        scheduler.step(step)
        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings)
        loss_result = loss_fn.forward_with_diagnostics(logits, targets)
        loss = loss_result.loss
        loss.backward()
        grad_norm = clip_global_grad_norm(model, GradientClippingConfig())
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            write_metrics_row(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "examples_seen": examples_seen,
                    "effective_epochs": f"{effective_epochs(examples_seen, len(train_bundle.dataset)):.8f}",
                    "loss": f"{float(loss.item()):.8f}",
                    "positive_loss": f"{float(loss_result.positive_loss.detach().item()):.8f}",
                    "negative_loss": f"{float(loss_result.negative_loss.detach().item()):.8f}",
                    "lr": f"{scheduler.get_lr():.12g}",
                    "grad_norm": f"{float(grad_norm.item()):.8f}",
                    "positive_labels_per_batch": f"{float(loss_result.positive_labels_per_batch.detach().item()):.8f}",
                    "negative_labels_per_batch": f"{float(loss_result.negative_labels_per_batch.detach().item()):.8f}",
                    "positive_labels_per_example": f"{float(loss_result.positive_labels_per_example.detach().item()):.8f}",
                    "mean_positive_probability": f"{float(loss_result.mean_positive_probability.detach().item()):.8f}",
                    "mean_negative_probability": f"{float(loss_result.mean_negative_probability.detach().item()):.8f}",
                    "logit_scale": f"{float(model.logit_scale.detach().clamp(min=model.config.min_logit_scale, max=model.config.max_logit_scale).item()):.8f}",
                    "elapsed_seconds": f"{time.time() - started:.3f}",
                },
            )

        should_validate = step % args.validate_every == 0 or step == args.max_steps
        if should_validate:
            validation = validate(
                model,
                val_loader,
                device=device,
                max_batches=args.validation_batches,
                thresholds=thresholds,
            )
            last_validation = validation
            write_metrics_row(
                metrics_path,
                {
                    "event": "validation",
                    "step": step,
                    "examples_seen": examples_seen,
                    "effective_epochs": f"{effective_epochs(examples_seen, len(train_bundle.dataset)):.8f}",
                    "loss": "",
                    "lr": f"{scheduler.get_lr():.12g}",
                    "grad_norm": "",
                    "elapsed_seconds": f"{time.time() - started:.3f}",
                    **{key: f"{value:.8f}" for key, value in validation.items()},
                },
            )
            if validation["fmax"] > best_fmax:
                best_fmax = validation["fmax"]
                best_step = step
                save_training_checkpoint(
                    args.output_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_fmax=best_fmax,
                    metadata=dynamic_metadata(step),
                )

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            save_training_checkpoint(
                args.output_dir / f"checkpoint_step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_fmax=best_fmax,
                metadata=dynamic_metadata(step),
            )
            save_training_checkpoint(
                args.output_dir / "latest_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_fmax=best_fmax,
                metadata=dynamic_metadata(step),
            )

        if args.stop_after_seconds is not None and time.time() - started >= args.stop_after_seconds:
            stop_reason = "time_limit_stop"
            save_training_checkpoint(
                args.output_dir / f"checkpoint_step_{step:08d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_fmax=best_fmax,
                metadata=dynamic_metadata(step),
            )
            save_training_checkpoint(
                args.output_dir / "latest_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_fmax=best_fmax,
                metadata=dynamic_metadata(step),
            )
            write_metrics_row(
                metrics_path,
                {
                    "event": "time_limit_stop",
                    "step": step,
                    "examples_seen": examples_seen,
                    "effective_epochs": f"{effective_epochs(examples_seen, len(train_bundle.dataset)):.8f}",
                    "lr": f"{scheduler.get_lr():.12g}",
                    "fmax": f"{best_fmax:.8f}",
                    "elapsed_seconds": f"{time.time() - started:.3f}",
                },
            )
            break

    if last_step == 0:
        last_step = start_step
        if start_step >= args.max_steps:
            stop_reason = "already_at_or_past_max_steps"

    elapsed_seconds = time.time() - started
    summary_rows: list[tuple[object, object]] = [
        ("status", "complete"),
        ("stop_reason", stop_reason),
        ("run_started_at_utc", run_started_at_utc),
        ("run_finished_at_utc", utc_now_iso()),
        ("elapsed_seconds", f"{elapsed_seconds:.3f}"),
        ("elapsed_hours", f"{elapsed_seconds / 3600.0:.6f}"),
        ("start_step", start_step),
        ("final_step", last_step),
        ("max_steps", args.max_steps),
        ("steps_completed_this_run", max(0, last_step - start_step)),
        ("examples_seen", examples_seen),
        ("effective_epochs", f"{effective_epochs(examples_seen, len(train_bundle.dataset)):.8f}"),
        ("train_records", len(train_bundle.dataset)),
        ("val_records", len(val_bundle.dataset)),
        ("n_terms", n_terms),
        ("batch_size", args.batch_size),
        ("best_fmax", f"{best_fmax:.8f}"),
        ("best_step", best_step),
        ("latest_checkpoint", args.output_dir / "latest_checkpoint.pt"),
        ("best_checkpoint", args.output_dir / "best_checkpoint.pt"),
        ("training_metrics", metrics_path),
    ]
    for key in ("fmax", "threshold", "precision", "recall", "validation_batches", "validation_examples"):
        if key in last_validation:
            summary_rows.append((f"last_validation_{key}", f"{last_validation[key]:.8f}"))
    write_key_value_tsv(run_summary_path, summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
