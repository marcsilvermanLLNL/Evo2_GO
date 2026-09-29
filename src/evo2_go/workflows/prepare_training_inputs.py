"""Prepare rerunnable CPU-side Evo2-GO training inputs from embeddings."""

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

from evo2_go.model.dataset import tax_ids_from_file


STEP_ORDER: tuple[str, ...] = (
    "true_path_counts",
    "go_adjacency",
    "label_cache",
    "split_manifest",
    "validate_embeddings",
)


@dataclass(frozen=True)
class TrainingInputPaths:
    project_root: Path
    build_dir: Path
    embedding_root: Path
    selected_genes: Path
    annotations: Path
    go_basic_obo: Path
    length_filtered_cds_root: Path
    labels_dir: Path
    adjacency_dir: Path
    splits_dir: Path
    validation_dir: Path
    term_counts: Path
    vocabulary: Path
    adjacency: Path
    label_cache: Path
    split_manifest: Path
    split_summary: Path
    training_inputs_manifest: Path
    step_status: Path


@dataclass(frozen=True)
class StepPlan:
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
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
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


def require_table(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] must be a TOML table when provided")
    return value


def string_list(value: object, *, default: Sequence[str] = ()) -> list[str]:
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, str):
        return [value]
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"Expected a string or list of strings, got {type(value).__name__}")


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
    steps = tuple(cli_steps or config_steps or STEP_ORDER[:-1])
    unknown = [step for step in steps if step not in STEP_ORDER]
    if unknown:
        raise ValueError(f"Unknown step(s): {unknown}; allowed={STEP_ORDER}")

    if from_step is not None:
        steps = tuple(
            step for step in steps if STEP_ORDER.index(step) >= STEP_ORDER.index(from_step)
        )
    if through_step is not None:
        steps = tuple(
            step for step in steps if STEP_ORDER.index(step) <= STEP_ORDER.index(through_step)
        )
    if not steps:
        raise ValueError("No workflow steps selected")
    return steps


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
    values = string_list(config.get("allowed_evidence", []))
    args: list[str] = []
    for value in values:
        args.extend(["--allowed-evidence", value])
    return args


def add_repeated_args(command: list[str], flag: str, values: Sequence[str]) -> None:
    for value in values:
        command.extend([flag, str(value)])


def tax_ids_from_config_files(value: object, project_root: Path) -> list[str]:
    tax_ids: list[str] = []
    for raw_path in string_list(value):
        tax_ids.extend(tax_ids_from_file(project_path(project_root, raw_path)))
    return tax_ids


def tax_id_files_from_config(value: object, project_root: Path) -> list[str]:
    return [str(project_path(project_root, raw_path)) for raw_path in string_list(value)]


def split_tax_ids(
    config: dict[str, Any],
    split_config: dict[str, Any],
    project_root: Path,
    *,
    include_file_ids: bool = True,
) -> dict[str, list[str]]:
    val_ids = string_list(
        split_config.get("val_tax_ids", split_config.get("val_tax_id", config.get("validation_tax_id"))),
    )
    if include_file_ids:
        val_ids.extend(
            tax_ids_from_config_files(
                split_config.get("val_tax_ids_file", split_config.get("val_tax_id_file", None)),
                project_root,
            )
        )
    test_ids = string_list(
        split_config.get("test_tax_ids", split_config.get("test_tax_id", config.get("test_tax_id"))),
    )
    if include_file_ids:
        test_ids.extend(
            tax_ids_from_config_files(
                split_config.get("test_tax_ids_file", split_config.get("test_tax_id_file", None)),
                project_root,
            )
        )
    holdout_ids = string_list(
        split_config.get("holdout_tax_ids", split_config.get("holdout_tax_id", None)),
    )
    if include_file_ids:
        holdout_ids.extend(
            tax_ids_from_config_files(
                split_config.get("holdout_tax_ids_file", split_config.get("holdout_tax_id_file", None)),
                project_root,
            )
        )
    selected_ids = string_list(split_config.get("tax_ids", split_config.get("tax_id", None)))
    if include_file_ids:
        selected_ids.extend(
            tax_ids_from_config_files(
                split_config.get("tax_ids_file", split_config.get("tax_id_file", None)),
                project_root,
            )
        )
    return {
        "val": val_ids,
        "test": test_ids,
        "holdout": holdout_ids,
        "selected": selected_ids,
    }


def resolve_paths(config: dict[str, Any], project_root: Path) -> TrainingInputPaths:
    build_id = str(config.get("build_id", "")).strip()
    if not build_id:
        raise ValueError("Config requires build_id")

    build_root = project_path(project_root, config.get("build_root", "outputs/builds"))
    build_dir = optional_project_path(config, "build_dir", project_root, build_root / build_id)
    embedding_root = optional_project_path(
        config,
        "embedding_root",
        project_root,
        build_dir / "data" / "embeddings",
    )
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
    split_config = require_table(config, "split")
    split_name = str(split_config.get("name", "human_test_taxon_val"))
    splits_dir = optional_project_path(
        split_config,
        "splits_dir",
        project_root,
        build_dir / "data" / "splits",
    )
    validation_config = require_table(config, "validation")
    validation_dir = optional_project_path(
        validation_config,
        "output_dir",
        project_root,
        build_dir / "embedding_validation",
    )

    return TrainingInputPaths(
        project_root=project_root,
        build_dir=build_dir,
        embedding_root=embedding_root,
        selected_genes=optional_project_path(
            config,
            "selected_genes",
            project_root,
            embedding_root / "embedding_metadata.tsv.gz",
        ),
        annotations=project_path(
            project_root,
            config.get(
                "annotations",
                "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz",
            ),
        ),
        go_basic_obo=project_path(project_root, config.get("go_basic_obo", "data/go/go-basic.obo")),
        length_filtered_cds_root=optional_project_path(
            config,
            "length_filtered_cds_root",
            project_root,
            Path("outputs/map_gene_cds/length_filtered_max_8192"),
        ),
        labels_dir=labels_dir,
        adjacency_dir=adjacency_dir,
        splits_dir=splits_dir,
        validation_dir=validation_dir,
        term_counts=optional_project_path(
            config,
            "term_counts",
            project_root,
            labels_dir / "true_path_go_term_counts.tsv",
        ),
        vocabulary=optional_project_path(
            config,
            "vocabulary",
            project_root,
            adjacency_dir / "go_vocabulary.tsv",
        ),
        adjacency=optional_project_path(
            config,
            "adjacency",
            project_root,
            adjacency_dir / "go_adjacency_sparse.npz",
        ),
        label_cache=optional_project_path(
            config,
            "label_cache",
            project_root,
            labels_dir / "label_index.tsv.gz",
        ),
        split_manifest=optional_project_path(
            split_config,
            "manifest",
            project_root,
            splits_dir / f"{split_name}_manifest.tsv",
        ),
        split_summary=optional_project_path(
            split_config,
            "summary",
            project_root,
            splits_dir / f"{split_name}_summary.tsv",
        ),
        training_inputs_manifest=optional_project_path(
            config,
            "training_inputs_manifest",
            project_root,
            build_dir / "training_inputs_manifest.tsv",
        ),
        step_status=optional_project_path(
            config,
            "step_status",
            project_root,
            build_dir / "training_inputs_steps.tsv",
        ),
    )


def build_step_plans(
    config: dict[str, Any],
    paths: TrainingInputPaths,
    *,
    selected: Sequence[str],
    force: bool,
) -> dict[str, StepPlan]:
    include_iea = get_bool(config, "include_iea", False)
    write_gene_labels = get_bool(config, "write_gene_labels", False)
    label_policy = str(config.get("label_policy", "true_path_labels"))
    min_count = int(config.get("min_count", 1))
    split_config = require_table(config, "split")
    validation_config = require_table(config, "validation")

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

    split_ids = split_tax_ids(
        config,
        split_config,
        paths.project_root,
        include_file_ids=False,
    )
    split_strategy = str(split_config.get("strategy", "taxon_holdout")).strip()
    split_tax_id_files = tax_id_files_from_config(
        split_config.get("tax_ids_file", split_config.get("tax_id_file", None)),
        paths.project_root,
    )
    if split_strategy in {"taxon_holdout", "separate_taxon_val_test"}:
        split_command = [
            sys.executable,
            "-m",
            "evo2_go.workflows.taxon_holdout_split",
            "--embedding-root",
            str(paths.embedding_root),
            "--label-cache",
            str(paths.label_cache),
            "--output",
            str(paths.split_manifest),
            "--summary",
            str(paths.split_summary),
            "--seed",
            str(split_config.get("seed", "evo2-go-human-taxon-holdout-v1")),
            "--expected-embedding-dim",
            str(int(split_config.get("expected_embedding_dim", 4096))),
            "--holdout-val-fraction",
            str(float(split_config.get("holdout_val_fraction", 0.5))),
            "--holdout-test-fraction",
            str(float(split_config.get("holdout_test_fraction", 0.5))),
            *force_args(force),
        ]
        add_repeated_args(split_command, "--val-tax-id", split_ids["val"])
        add_repeated_args(split_command, "--test-tax-id", split_ids["test"])
        add_repeated_args(split_command, "--holdout-tax-id", split_ids["holdout"])
        add_repeated_args(split_command, "--tax-id", split_ids["selected"])
        add_repeated_args(split_command, "--tax-id-file", split_tax_id_files)
        if int(split_config.get("max_records", 0)) > 0:
            split_command.extend(["--max-records", str(int(split_config["max_records"]))])
        if get_bool(split_config, "drop_examples_without_labels", False):
            split_command.append("--drop-examples-without-labels")
    elif split_strategy in {"mmseqs_linclust", "mmseqs_cluster_split"}:
        split_cluster_dir = optional_project_path(
            split_config,
            "cluster_dir",
            paths.project_root,
            paths.splits_dir / f"{str(split_config.get('name', 'split'))}_mmseqs_linclust",
        )
        split_command = [
            sys.executable,
            "-m",
            "evo2_go.workflows.mmseqs_cluster_split",
            "--embedding-root",
            str(paths.embedding_root),
            "--input-root",
            str(paths.length_filtered_cds_root),
            "--label-cache",
            str(paths.label_cache),
            "--output",
            str(paths.split_manifest),
            "--summary",
            str(paths.split_summary),
            "--cluster-dir",
            str(split_cluster_dir),
            "--mmseqs-path",
            str(
                optional_project_path(
                    split_config,
                    "mmseqs_path",
                    paths.project_root,
                    paths.project_root / "envs" / "mmseqs" / "bin" / "mmseqs",
                )
            ),
            "--seed",
            str(split_config.get("seed", "evo2-go-mmseqs-linclust-split-v1")),
            "--expected-embedding-dim",
            str(int(split_config.get("expected_embedding_dim", 4096))),
            "--min-seq-id",
            str(float(split_config.get("min_seq_id", 0.70))),
            "--coverage",
            str(float(split_config.get("coverage", 0.50))),
            "--cov-mode",
            str(int(split_config.get("cov_mode", 1))),
            "--threads",
            str(int(split_config.get("threads", 16))),
            "--train-fraction",
            str(float(split_config.get("train_fraction", 0.80))),
            "--val-fraction",
            str(float(split_config.get("val_fraction", 0.10))),
            "--test-fraction",
            str(float(split_config.get("test_fraction", 0.10))),
            "--balance-by",
            str(split_config.get("balance_by", "records")),
            *force_args(force),
        ]
        add_repeated_args(split_command, "--tax-id", split_ids["selected"])
        add_repeated_args(split_command, "--tax-id-file", split_tax_id_files)
        if int(split_config.get("limit_taxa", 0)) > 0:
            split_command.extend(["--limit-taxa", str(int(split_config["limit_taxa"]))])
        if int(split_config.get("max_records", 0)) > 0:
            split_command.extend(["--max-records", str(int(split_config["max_records"]))])
        if "cluster_mode" in split_config:
            split_command.extend(["--cluster-mode", str(int(split_config["cluster_mode"]))])
        if "kmer_per_seq_scale" in split_config:
            split_command.extend(["--kmer-per-seq-scale", str(split_config["kmer_per_seq_scale"])])
        if "linclust_version" in split_config:
            split_command.extend(["--linclust-version", str(int(split_config["linclust_version"]))])
        if "include_adjacency" in split_config:
            split_command.extend(["--include-adjacency", str(int(split_config["include_adjacency"]))])
        if "num_adjacency" in split_config:
            split_command.extend(["--num-adjacency", str(int(split_config["num_adjacency"]))])
        if get_bool(split_config, "reuse_clusters", False):
            split_command.append("--reuse-clusters")
    else:
        raise ValueError(
            "Unsupported split strategy "
            f"{split_strategy!r}; expected taxon_holdout or mmseqs_linclust"
        )

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
        str(paths.validation_dir),
    ]
    validation_tax_ids = string_list(validation_config.get("tax_ids", []))
    validation_tax_ids.extend(
        tax_ids_from_config_files(
            validation_config.get("tax_ids_file", validation_config.get("tax_id_file", None)),
            paths.project_root,
        )
    )
    add_repeated_args(validate_command, "--tax-id", validation_tax_ids)
    for key, flag in (
        ("max_records", "--max-records"),
        ("max_shards", "--max-shards"),
        ("expected_embedding_dim", "--expected-embedding-dim"),
    ):
        if int(validation_config.get(key, 0)) > 0:
            validate_command.extend([flag, str(int(validation_config[key]))])
    if get_bool(validation_config, "check_all_finite", False):
        validate_command.append("--check-all-finite")

    return {
        "true_path_counts": StepPlan(
            name="true_path_counts",
            command=tuple([*true_path_base, *force_args(force)]),
            expected_outputs=(paths.term_counts, paths.labels_dir / "true_path_label_summary.tsv"),
            description="Build true-path labels and GO term counts for embedded genes.",
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
            description="Build deterministic GO vocabulary and normalized sparse adjacency.",
        ),
        "label_cache": StepPlan(
            name="label_cache",
            command=tuple(label_cache_command),
            expected_outputs=(paths.label_cache,),
            description="Build vocabulary-aligned true-path label cache.",
        ),
        "split_manifest": StepPlan(
            name="split_manifest",
            command=tuple(split_command),
            expected_outputs=(paths.split_manifest, paths.split_summary),
            description="Create the training/validation/test split manifest from embeddings.",
        ),
        "validate_embeddings": StepPlan(
            name="validate_embeddings",
            command=tuple(validate_command),
            expected_outputs=(paths.validation_dir / "validation_summary.tsv",),
            description="Validate embedding metadata, shards, and label-cache coverage.",
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
    paths: TrainingInputPaths,
    steps: Sequence[str],
    dry_run: bool,
) -> None:
    split_config = require_table(config, "split")
    split_ids = split_tax_ids(config, split_config, paths.project_root)
    rows = [
        ("build_id", config.get("build_id", "")),
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("config", config_path),
        ("dry_run", int(dry_run)),
        ("project_root", paths.project_root),
        ("build_dir", paths.build_dir),
        ("embedding_root", paths.embedding_root),
        ("embedding_dataset_manifest", paths.embedding_root / "embedding_dataset_manifest.tsv"),
        ("selected_genes", paths.selected_genes),
        ("annotations", paths.annotations),
        ("go_basic_obo", paths.go_basic_obo),
        ("labels_dir", paths.labels_dir),
        ("adjacency_dir", paths.adjacency_dir),
        ("splits_dir", paths.splits_dir),
        ("term_counts", paths.term_counts),
        ("vocabulary", paths.vocabulary),
        ("adjacency", paths.adjacency),
        ("label_cache", paths.label_cache),
        ("split_manifest", paths.split_manifest),
        ("split_summary", paths.split_summary),
        ("steps", ",".join(steps)),
        ("include_iea", int(get_bool(config, "include_iea", False))),
        ("label_policy", config.get("label_policy", "true_path_labels")),
        ("min_count", config.get("min_count", 1)),
        ("write_gene_labels", int(get_bool(config, "write_gene_labels", False))),
        ("split_strategy", split_config.get("strategy", "taxon_holdout")),
        ("split_val_tax_ids", ",".join(split_ids["val"])),
        ("split_test_tax_ids", ",".join(split_ids["test"])),
        ("split_holdout_tax_ids", ",".join(split_ids["holdout"])),
        ("split_selected_taxa", len(split_ids["selected"])),
        (
            "split_tax_ids_file",
            split_config.get("tax_ids_file", split_config.get("tax_id_file", "")),
        ),
        ("split_limit_taxa", split_config.get("limit_taxa", "")),
        ("split_mmseqs_path", split_config.get("mmseqs_path", "")),
        ("split_min_seq_id", split_config.get("min_seq_id", "")),
        ("split_coverage", split_config.get("coverage", "")),
        ("split_cov_mode", split_config.get("cov_mode", "")),
        ("split_cluster_mode", split_config.get("cluster_mode", "")),
        ("split_train_fraction", split_config.get("train_fraction", "")),
        ("split_val_fraction", split_config.get("val_fraction", "")),
        ("split_test_fraction", split_config.get("test_fraction", "")),
        ("split_balance_by", split_config.get("balance_by", "")),
        ("ortholog_safe", 0),
    ]
    write_key_value_tsv(paths.training_inputs_manifest, rows)


def validate_inputs(
    paths: TrainingInputPaths,
    selected: Sequence[str],
    config: dict[str, Any],
) -> None:
    selected_set = set(selected)
    if "true_path_counts" in selected_set or "label_cache" in selected_set:
        for path, label in (
            (paths.selected_genes, "selected genes"),
            (paths.annotations, "annotations"),
            (paths.go_basic_obo, "go-basic ontology"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
    if (
        "go_adjacency" in selected_set
        and "true_path_counts" not in selected_set
        and not paths.term_counts.is_file()
    ):
        raise FileNotFoundError(f"Missing true-path term counts: {paths.term_counts}")
    if "go_adjacency" in selected_set and not paths.go_basic_obo.is_file():
        raise FileNotFoundError(f"Missing go-basic ontology: {paths.go_basic_obo}")
    if (
        "label_cache" in selected_set
        and "go_adjacency" not in selected_set
        and not paths.vocabulary.is_file()
    ):
        raise FileNotFoundError(f"Missing GO vocabulary: {paths.vocabulary}")
    if "split_manifest" in selected_set or "validate_embeddings" in selected_set:
        manifest = paths.embedding_root / "embedding_dataset_manifest.tsv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing embedding dataset manifest: {manifest}")
    if (
        "split_manifest" in selected_set
        and "label_cache" not in selected_set
        and not paths.label_cache.is_file()
    ):
        raise FileNotFoundError(f"Missing label cache: {paths.label_cache}")
    if "split_manifest" in selected_set:
        split_config = require_table(config, "split")
        split_strategy = str(split_config.get("strategy", "taxon_holdout")).strip()
        if split_strategy in {"mmseqs_linclust", "mmseqs_cluster_split"}:
            if not paths.length_filtered_cds_root.is_dir():
                raise FileNotFoundError(
                    f"Missing length-filtered CDS root for MMseqs split: {paths.length_filtered_cds_root}"
                )
            mmseqs_path = optional_project_path(
                split_config,
                "mmseqs_path",
                paths.project_root,
                paths.project_root / "envs" / "mmseqs" / "bin" / "mmseqs",
            )
            if not mmseqs_path.is_file():
                raise FileNotFoundError(f"Missing MMseqs binary for cluster split: {mmseqs_path}")
    if "validate_embeddings" in selected_set:
        for path, label in (
            (paths.length_filtered_cds_root, "length-filtered CDS root"),
            (paths.vocabulary, "GO vocabulary"),
            (paths.label_cache, "label cache"),
        ):
            if label == "GO vocabulary" and "go_adjacency" in selected_set:
                continue
            if label == "label cache" and "label_cache" in selected_set:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")


def run_plan(
    *,
    plans: dict[str, StepPlan],
    selected: Sequence[str],
    paths: TrainingInputPaths,
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

        append_step_status(paths.step_status, step=step_name, status="started", command=plan.command)
        subprocess.run(plan.command, cwd=paths.project_root, env=env, check=True)
        if not outputs_exist(plan.expected_outputs):
            missing = [str(path) for path in plan.expected_outputs if not path.is_file()]
            raise RuntimeError(
                f"Step {step_name} completed but expected outputs are missing: {missing}"
            )
        append_step_status(paths.step_status, step=step_name, status="complete", command=plan.command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rerunnable CPU-side Evo2-GO training input preparation from an "
            "existing embedding build."
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
    parser.add_argument("--force", action="store_true", help="Pass --force to steps that support it.")
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
    validate_inputs(paths, steps, config)

    if args.dry_run:
        log("Dry run: no files will be written.")
    else:
        paths.build_dir.mkdir(parents=True, exist_ok=True)
        paths.labels_dir.mkdir(parents=True, exist_ok=True)
        paths.adjacency_dir.mkdir(parents=True, exist_ok=True)
        paths.splits_dir.mkdir(parents=True, exist_ok=True)
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
    log(f"Training input preparation finished: build_dir={paths.build_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
