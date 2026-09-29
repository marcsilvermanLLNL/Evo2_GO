"""Database-build orchestration for the remodeled Evo2-GO pipeline."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


STEP_ORDER: tuple[str, ...] = (
    "map_cds_by_tax",
    "length_filter",
    "run_embeddings",
    "true_path_counts",
    "go_adjacency",
    "label_cache",
    "validate_embeddings",
)


@dataclass(frozen=True)
class BuildPaths:
    """Resolved paths used by the database-build workflow."""

    project_root: Path
    build_dir: Path
    selected_genes: Path
    representative_cds_root: Path
    length_filtered_cds_root: Path
    embedding_root: Path
    annotations: Path
    go_basic_obo: Path
    labels_dir: Path
    adjacency_dir: Path
    term_counts: Path
    vocabulary: Path
    adjacency: Path
    label_cache: Path
    build_manifest: Path
    step_status: Path


@dataclass(frozen=True)
class StepPlan:
    """One executable workflow step."""

    name: str
    command: tuple[str, ...]
    expected_outputs: tuple[Path, ...]
    description: str


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def get_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Config field {key!r} must be boolean")


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def optional_project_path(
    config: dict[str, Any],
    key: str,
    project_root: Path,
    default: Path,
) -> Path:
    if key in config:
        return project_path(project_root, config[key])
    return default


def parse_step_list(raw_steps: str | None) -> tuple[str, ...] | None:
    if raw_steps is None:
        return None
    steps = tuple(part.strip() for part in raw_steps.split(",") if part.strip())
    unknown = [step for step in steps if step not in STEP_ORDER]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}; allowed={STEP_ORDER}")
    return steps


def selected_steps(
    *,
    config_steps: Sequence[str] | None,
    cli_steps: tuple[str, ...] | None,
    from_step: str | None,
    through_step: str | None,
) -> tuple[str, ...]:
    steps = tuple(cli_steps or config_steps or STEP_ORDER)
    unknown = [step for step in steps if step not in STEP_ORDER]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}; allowed={STEP_ORDER}")

    if from_step is not None:
        steps = tuple(
            step for step in steps if STEP_ORDER.index(step) >= STEP_ORDER.index(from_step)
        )
    if through_step is not None:
        steps = tuple(
            step
            for step in steps
            if STEP_ORDER.index(step) <= STEP_ORDER.index(through_step)
        )
    if not steps:
        raise ValueError("No workflow steps selected")
    return steps


def resolve_paths(config: dict[str, Any], project_root: Path) -> BuildPaths:
    build_id = str(config.get("build_id", "")).strip()
    if not build_id:
        raise ValueError("Config requires build_id")

    build_root = project_path(project_root, config.get("build_root", "outputs/builds"))
    build_dir = optional_project_path(config, "build_dir", project_root, build_root / build_id)
    labels_dir = optional_project_path(
        config,
        "labels_dir",
        project_root,
        build_dir / "data" / "labels",
    )
    adjacency_dir = optional_project_path(
        config,
        "adjacency_dir",
        project_root,
        build_dir / "adjacency",
    )

    selected_genes = optional_project_path(
        config,
        "selected_genes",
        project_root,
        build_dir / "data" / "embeddings" / "embedding_metadata.tsv.gz",
    )
    representative_cds_root = optional_project_path(
        config,
        "representative_cds_root",
        project_root,
        Path("outputs/map_gene_cds/by_tax"),
    )
    length_filtered_cds_root = optional_project_path(
        config,
        "length_filtered_cds_root",
        project_root,
        Path("outputs/map_gene_cds/length_filtered_max_8192"),
    )
    embedding_root = optional_project_path(
        config,
        "embedding_root",
        project_root,
        build_dir / "data" / "embeddings",
    )
    annotations = project_path(
        project_root,
        config.get(
            "annotations",
            "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz",
        ),
    )
    go_basic_obo = project_path(project_root, config.get("go_basic_obo", "data/go/go-basic.obo"))

    return BuildPaths(
        project_root=project_root,
        build_dir=build_dir,
        selected_genes=selected_genes,
        representative_cds_root=representative_cds_root,
        length_filtered_cds_root=length_filtered_cds_root,
        embedding_root=embedding_root,
        annotations=annotations,
        go_basic_obo=go_basic_obo,
        labels_dir=labels_dir,
        adjacency_dir=adjacency_dir,
        term_counts=labels_dir / "true_path_go_term_counts.tsv",
        vocabulary=adjacency_dir / "go_vocabulary.tsv",
        adjacency=adjacency_dir / "go_adjacency_sparse.npz",
        label_cache=labels_dir / "label_index.tsv.gz",
        build_manifest=build_dir / "build_database_manifest.tsv",
        step_status=build_dir / "build_database_steps.tsv",
    )


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not existing else f"{src_dir}{os.pathsep}{existing}"
    return env


def force_args(force: bool) -> list[str]:
    return ["--force"] if force else []


def include_iea_args(include_iea: bool) -> list[str]:
    return ["--include-iea"] if include_iea else []


def evidence_args(config: dict[str, Any]) -> list[str]:
    raw = config.get("allowed_evidence", [])
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = [str(value) for value in raw]
    else:
        raise ValueError("allowed_evidence must be a string or list of strings")

    args: list[str] = []
    for value in values:
        args.extend(["--allowed-evidence", value])
    return args


def build_step_plans(
    config: dict[str, Any],
    paths: BuildPaths,
    *,
    selected: Sequence[str],
    force: bool,
) -> dict[str, StepPlan]:
    include_iea = get_bool(config, "include_iea", False)
    write_gene_labels = get_bool(config, "write_gene_labels", False)
    label_policy = str(config.get("label_policy", "true_path_labels"))
    min_count = int(config.get("min_count", 1))
    length_filter_config = config.get("length_filter", {})
    if not isinstance(length_filter_config, dict):
        raise ValueError("[length_filter] must be a TOML table when provided")
    embeddings_config = config.get("embeddings", {})
    if not isinstance(embeddings_config, dict):
        raise ValueError("[embeddings] must be a TOML table when provided")

    map_config = config.get("cds_mapping", {})
    if not isinstance(map_config, dict):
        raise ValueError("[cds_mapping] must be a TOML table when provided")
    map_output_root = project_path(
        paths.project_root,
        map_config.get("output_root", "outputs/map_gene_cds/by_tax"),
    )
    map_command = [
        sys.executable,
        "-m",
        "evo2_go.database.map_gene_cds_by_tax",
        "--labels",
        str(
            project_path(
                paths.project_root,
                map_config.get(
                    "labels",
                    "outputs/process_Gene2GO/processed/final/positive_gene_go_labels.tsv.gz",
                ),
            )
        ),
        "--download-manifest",
        str(
            project_path(
                paths.project_root,
                map_config.get("download_manifest", "data/refseq_cds/downloaded_cds.tsv"),
            )
        ),
        "--output-root",
        str(map_output_root),
        "--max-workers",
        str(int(map_config.get("max_workers", 16))),
    ]
    for tax_id in map_config.get("tax_ids", []):
        map_command.extend(["--tax-id", str(tax_id)])
    if int(map_config.get("limit_taxa", 0)) > 0:
        map_command.extend(["--limit-taxa", str(int(map_config["limit_taxa"]))])
    if force:
        map_command.append("--force")

    length_filter_command = [
        sys.executable,
        "-m",
        "evo2_go.embeddings.length_filter",
        "--input-root",
        str(paths.representative_cds_root),
        "--output-root",
        str(paths.length_filtered_cds_root),
        "--max-tokens",
        str(int(length_filter_config.get("max_tokens", 8192))),
        "--max-workers",
        str(int(length_filter_config.get("max_workers", 16))),
    ]
    for tax_id in length_filter_config.get("tax_ids", []):
        length_filter_command.extend(["--tax-id", str(tax_id)])
    if int(length_filter_config.get("limit_taxa", 0)) > 0:
        length_filter_command.extend(["--limit-taxa", str(int(length_filter_config["limit_taxa"]))])
    if force:
        length_filter_command.append("--force")

    embedding_command = [
        sys.executable,
        "-m",
        "evo2_go.embeddings.run_embeddings",
        "--input-root",
        str(paths.length_filtered_cds_root),
        "--output-root",
        str(paths.embedding_root),
        "--max-workers",
        str(int(embeddings_config.get("max_workers", 4))),
        "--gpu-ids",
        str(embeddings_config.get("gpu_ids", "0,1,2,3")),
        "--incomplete-only",
    ]
    if embeddings_config.get("model_name"):
        embedding_command.extend(["--model-name", str(embeddings_config["model_name"])])
    if embeddings_config.get("model_path"):
        embedding_command.extend(
            ["--model-path", str(project_path(paths.project_root, embeddings_config["model_path"]))]
        )
    if embeddings_config.get("expected_max_seqlen"):
        embedding_command.extend(
            ["--expected-max-seqlen", str(embeddings_config["expected_max_seqlen"])]
        )
    for tax_id in embeddings_config.get("tax_ids", []):
        embedding_command.extend(["--tax-id", str(tax_id)])
    if embeddings_config.get("tax_list"):
        embedding_command.extend(
            ["--tax-list", str(project_path(paths.project_root, embeddings_config["tax_list"]))]
        )
    if int(embeddings_config.get("limit_taxa", 0)) > 0:
        embedding_command.extend(["--limit-taxa", str(int(embeddings_config["limit_taxa"]))])
    if int(embeddings_config.get("limit_runnable_taxa", 0)) > 0:
        embedding_command.extend(
            ["--limit-runnable-taxa", str(int(embeddings_config["limit_runnable_taxa"]))]
        )
    for bucket_id in embeddings_config.get("bucket_ids", []):
        embedding_command.extend(["--bucket-id", str(bucket_id)])
    bucket_batch_sizes = embeddings_config.get("bucket_batch_sizes", "")
    if bucket_batch_sizes:
        embedding_command.extend(["--bucket-batch-sizes", str(bucket_batch_sizes)])
    for key, flag in (
        ("limit_records_per_taxon", "--limit-records-per-taxon"),
        ("max_batches_per_taxon", "--max-batches-per-taxon"),
        ("max_batches_per_bucket", "--max-batches-per-bucket"),
        ("queue_size", "--queue-size"),
        ("no_progress_timeout_seconds", "--no-progress-timeout-seconds"),
        ("max_failed_batches", "--max-failed-batches"),
    ):
        if int(embeddings_config.get(key, 0)) > 0:
            embedding_command.extend([flag, str(int(embeddings_config[key]))])
    if embeddings_config.get("layer_name"):
        embedding_command.extend(["--layer-name", str(embeddings_config["layer_name"])])
    if embeddings_config.get("resume_completed_batches") is False:
        embedding_command.append("--no-resume-completed-batches")
    if embeddings_config.get("dry_run", False):
        embedding_command.append("--dry-run")
    if force:
        embedding_command.append("--overwrite")

    validate_command = [
        sys.executable,
        "-m",
        "evo2_go.embeddings.validate_embeddings",
        "--embedding-root",
        str(paths.embedding_root),
        "--input-root",
        str(paths.length_filtered_cds_root),
        "--vocabulary",
        str(paths.vocabulary),
        "--label-cache",
        str(paths.label_cache),
        "--output-dir",
        str(paths.build_dir / "embedding_validation"),
    ]
    for tax_id in embeddings_config.get("tax_ids", []):
        validate_command.extend(["--tax-id", str(tax_id)])
    if int(embeddings_config.get("validation_max_records", 0)) > 0:
        validate_command.extend(
            ["--max-records", str(int(embeddings_config["validation_max_records"]))]
        )
    if int(embeddings_config.get("validation_max_shards", 0)) > 0:
        validate_command.extend(
            ["--max-shards", str(int(embeddings_config["validation_max_shards"]))]
        )
    if int(embeddings_config.get("expected_embedding_dim", 0)) > 0:
        validate_command.extend(
            ["--expected-embedding-dim", str(int(embeddings_config["expected_embedding_dim"]))]
        )
    if embeddings_config.get("check_all_finite", False):
        validate_command.append("--check-all-finite")

    true_path_base = [
        sys.executable,
        "-m",
        "evo2_go.database.true_path_labels",
        "--selected-genes",
        str(paths.selected_genes),
        "--annotations",
        str(paths.annotations),
        "--go-basic-obo",
        str(paths.go_basic_obo),
        "--output-dir",
        str(paths.labels_dir),
        "--label-policy",
        label_policy,
        *include_iea_args(include_iea),
        *evidence_args(config),
    ]
    if not write_gene_labels:
        true_path_base.append("--no-write-gene-labels")

    label_cache_command = [
        *true_path_base,
        "--vocabulary",
        str(paths.vocabulary),
    ]
    if write_gene_labels:
        label_cache_command.append("--no-write-gene-labels")
    label_cache_command.extend(force_args(force or "true_path_counts" in selected))

    return {
        "map_cds_by_tax": StepPlan(
            name="map_cds_by_tax",
            command=tuple(map_command),
            expected_outputs=(
                map_output_root / "mapping_status.tsv",
                map_output_root / "run_summary.tsv",
            ),
            description="Run representative-CDS mapping across downloaded taxa.",
        ),
        "true_path_counts": StepPlan(
            name="true_path_counts",
            command=tuple([*true_path_base, *force_args(force)]),
            expected_outputs=(paths.term_counts, paths.labels_dir / "true_path_label_summary.tsv"),
            description="Build true-path term counts for the selected gene universe.",
        ),
        "length_filter": StepPlan(
            name="length_filter",
            command=tuple(length_filter_command),
            expected_outputs=(
                paths.length_filtered_cds_root / "dataset_length_filter_summary.tsv",
                paths.length_filtered_cds_root / "filter_status.tsv",
            ),
            description="Filter representative CDS records to the configured Evo2 context length.",
        ),
        "run_embeddings": StepPlan(
            name="run_embeddings",
            command=tuple(embedding_command),
            expected_outputs=(paths.embedding_root / "embedding_dataset_manifest.tsv",),
            description=(
                "Extract embeddings into one shared output root. Repeated runs skip complete taxa "
                "and resume complete batch shards."
            ),
        ),
        "validate_embeddings": StepPlan(
            name="validate_embeddings",
            command=tuple(validate_command),
            expected_outputs=(paths.build_dir / "embedding_validation" / "validation_summary.tsv",),
            description="Validate the shared embedding root before training.",
        ),
        "go_adjacency": StepPlan(
            name="go_adjacency",
            command=tuple(
                [
                    sys.executable,
                    "-m",
                    "evo2_go.ontology.go_adjacency",
                    "--term-counts",
                    str(paths.term_counts),
                    "--go-basic-obo",
                    str(paths.go_basic_obo),
                    "--output-dir",
                    str(paths.adjacency_dir),
                    "--min-count",
                    str(min_count),
                    *force_args(force),
                ]
            ),
            expected_outputs=(
                paths.vocabulary,
                paths.adjacency,
                paths.adjacency_dir / "go_adjacency_summary.tsv",
            ),
            description="Build GO vocabulary and normalized sparse adjacency.",
        ),
        "label_cache": StepPlan(
            name="label_cache",
            command=tuple(label_cache_command),
            expected_outputs=(paths.label_cache,),
            description="Build vocabulary-aligned true-path label cache.",
        ),
    }


def shell_quote(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def outputs_exist(paths: Iterable[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("field", "value"))
        writer.writerows(rows)
    tmp_path.replace(path)


def append_step_status(
    path: Path,
    *,
    step: str,
    status: str,
    command: Sequence[str],
    message: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("at", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if write_header:
            writer.writerow(("timestamp", "step", "status", "command", "message"))
        writer.writerow(
            (
                datetime.now().isoformat(timespec="seconds"),
                step,
                status,
                shell_quote(command),
                message,
            )
        )


def write_manifest(
    *,
    config_path: Path,
    config: dict[str, Any],
    paths: BuildPaths,
    steps: Sequence[str],
    dry_run: bool,
) -> None:
    rows = [
        ("build_id", config.get("build_id", "")),
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("config", config_path),
        ("dry_run", int(dry_run)),
        ("project_root", paths.project_root),
        ("build_dir", paths.build_dir),
        ("selected_genes", paths.selected_genes),
        ("representative_cds_root", paths.representative_cds_root),
        ("length_filtered_cds_root", paths.length_filtered_cds_root),
        ("embedding_root", paths.embedding_root),
        ("annotations", paths.annotations),
        ("go_basic_obo", paths.go_basic_obo),
        ("labels_dir", paths.labels_dir),
        ("adjacency_dir", paths.adjacency_dir),
        ("term_counts", paths.term_counts),
        ("vocabulary", paths.vocabulary),
        ("adjacency", paths.adjacency),
        ("label_cache", paths.label_cache),
        ("steps", ",".join(steps)),
        ("include_iea", int(get_bool(config, "include_iea", False))),
        ("label_policy", config.get("label_policy", "true_path_labels")),
        ("min_count", config.get("min_count", 1)),
        ("write_gene_labels", int(get_bool(config, "write_gene_labels", False))),
    ]
    embeddings_config = config.get("embeddings", {})
    if isinstance(embeddings_config, dict):
        for key in ("model_name", "model_path", "expected_max_seqlen"):
            if key in embeddings_config:
                rows.append((f"embedding_{key}", embeddings_config[key]))
    write_key_value_tsv(paths.build_manifest, rows)


def validate_inputs(paths: BuildPaths, selected: Sequence[str]) -> None:
    selected_set = set(selected)
    selected_genes_from_embeddings = (
        "run_embeddings" in selected_set
        and paths.selected_genes == paths.embedding_root / "embedding_metadata.tsv.gz"
    )
    if "true_path_counts" in selected or "label_cache" in selected:
        for path, label in (
            (paths.selected_genes, "selected genes"),
            (paths.annotations, "annotations"),
            (paths.go_basic_obo, "go-basic ontology"),
        ):
            if path == paths.selected_genes and selected_genes_from_embeddings:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
    if "length_filter" in selected and not paths.representative_cds_root.is_dir():
        raise FileNotFoundError(f"Missing representative CDS root: {paths.representative_cds_root}")
    if (
        "run_embeddings" in selected
        and "length_filter" not in selected_set
        and not paths.length_filtered_cds_root.is_dir()
    ):
        raise FileNotFoundError(f"Missing length-filtered CDS root: {paths.length_filtered_cds_root}")
    if "validate_embeddings" in selected:
        for path, label in (
            (paths.embedding_root / "embedding_dataset_manifest.tsv", "embedding manifest"),
            (paths.length_filtered_cds_root, "length-filtered CDS root"),
            (paths.vocabulary, "GO vocabulary"),
            (paths.label_cache, "label cache"),
        ):
            if label == "embedding manifest" and "run_embeddings" in selected_set:
                continue
            if label == "length-filtered CDS root" and "length_filter" in selected_set:
                continue
            if label == "GO vocabulary" and "go_adjacency" in selected_set:
                continue
            if label == "label cache" and "label_cache" in selected_set:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
    if (
        "go_adjacency" in selected
        and "true_path_counts" not in selected_set
        and not paths.term_counts.is_file()
    ):
        raise FileNotFoundError(f"Missing true-path term counts: {paths.term_counts}")
    if "go_adjacency" in selected and not paths.go_basic_obo.is_file():
        raise FileNotFoundError(f"Missing go-basic ontology: {paths.go_basic_obo}")
    if (
        "label_cache" in selected
        and "go_adjacency" not in selected_set
        and not paths.vocabulary.is_file()
    ):
        raise FileNotFoundError(f"Missing GO vocabulary: {paths.vocabulary}")


def run_plan(
    *,
    plans: dict[str, StepPlan],
    selected: Sequence[str],
    paths: BuildPaths,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    env = command_env()
    for step_name in selected:
        plan = plans[step_name]
        if skip_existing and outputs_exist(plan.expected_outputs):
            log(f"Skipping {step_name}: expected outputs already exist")
            if not dry_run:
                append_step_status(
                    paths.step_status,
                    step=step_name,
                    status="skipped_existing",
                    command=plan.command,
                )
            continue

        log(f"Step {step_name}: {plan.description}")
        log(shell_quote(plan.command))
        if dry_run:
            continue

        append_step_status(
            paths.step_status,
            step=step_name,
            status="started",
            command=plan.command,
        )
        subprocess.run(plan.command, cwd=paths.project_root, env=env, check=True)
        if not outputs_exist(plan.expected_outputs):
            missing = [str(path) for path in plan.expected_outputs if not path.is_file()]
            raise RuntimeError(
                f"Step {step_name} completed but expected outputs are missing: {missing}"
            )
        append_step_status(
            paths.step_status,
            step=step_name,
            status="complete",
            command=plan.command,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a documented Evo2-GO database build from a TOML config. "
            "The workflow calls migrated Python modules rather than hiding their behavior."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Directory relative paths are resolved against and subprocesses run from.",
    )
    parser.add_argument(
        "--steps",
        help=f"Comma-delimited step list. Allowed: {','.join(STEP_ORDER)}",
    )
    parser.add_argument("--from-step", choices=STEP_ORDER)
    parser.add_argument("--through-step", choices=STEP_ORDER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to steps that support it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_toml(config_path)
    project_root = args.project_root.resolve()
    paths = resolve_paths(config, project_root)
    config_steps = config.get("steps")
    if config_steps is not None and not isinstance(config_steps, list):
        raise ValueError("Config field steps must be a list of step names")

    steps = selected_steps(
        config_steps=tuple(config_steps) if config_steps is not None else None,
        cli_steps=parse_step_list(args.steps),
        from_step=args.from_step,
        through_step=args.through_step,
    )
    validate_inputs(paths, steps)

    if args.dry_run:
        log("Dry run: no files will be written.")
    else:
        paths.build_dir.mkdir(parents=True, exist_ok=True)
        paths.labels_dir.mkdir(parents=True, exist_ok=True)
        paths.adjacency_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            config_path=config_path,
            config=config,
            paths=paths,
            steps=steps,
            dry_run=args.dry_run,
        )

    plans = build_step_plans(config, paths, selected=steps, force=args.force)
    run_plan(
        plans=plans,
        selected=steps,
        paths=paths,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    log(f"Database build workflow finished: build_dir={paths.build_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
