"""Configuration helpers for remodel training and evaluation commands."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "embedding_root",
    "vocabulary",
    "adjacency",
    "label_cache",
    "split_manifest",
    "validation_panel",
    "output_dir",
    "resume_checkpoint",
    "checkpoint",
    "score_matrix_output",
    "publication_summary_output",
    "publication_threshold_output",
    "publication_gene_output",
    "publication_go_term_output",
    "publication_calibration_output",
    "tax_ids_file",
}

LIST_KEYS = {
    "tax_ids",
}


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_project_root(project_root: Path | None) -> Path:
    return (project_root or Path.cwd()).expanduser().resolve()


def resolve_path(value: Any, project_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def build_dir_from_config(config: dict[str, Any], project_root: Path) -> Path | None:
    build_id = config.get("build_id")
    if not build_id:
        return None
    build_root = Path(str(config.get("build_root", "outputs/builds")))
    if not build_root.is_absolute():
        build_root = project_root / build_root
    return build_root / str(build_id)


def defaults_from_config(
    config_path: Path | None,
    *,
    project_root: Path | None = None,
    section: str,
) -> dict[str, Any]:
    """Return argparse defaults from a training/evaluation TOML config."""

    if config_path is None:
        return {}

    root = resolve_project_root(project_root)
    config = read_toml(config_path)
    defaults: dict[str, Any] = {}

    build_dir = build_dir_from_config(config, root)
    run_name = config.get("run_name")
    if build_dir is not None:
        defaults.setdefault("embedding_root", build_dir / "data" / "embeddings")
        defaults.setdefault("vocabulary", build_dir / "adjacency" / "go_vocabulary.tsv")
        defaults.setdefault("adjacency", build_dir / "adjacency" / "go_adjacency_sparse.npz")
        defaults.setdefault("label_cache", build_dir / "data" / "labels" / "label_index.tsv.gz")
        defaults.setdefault(
            "split_manifest",
            build_dir / "data" / "splits" / "human_test_taxon_val_manifest.tsv",
        )
        if run_name:
            defaults.setdefault("output_dir", build_dir / "runs" / str(run_name))

    for key, value in config.items():
        if isinstance(value, dict):
            continue
        if key in {"build_id", "build_root", "run_name"}:
            continue
        defaults[key] = [str(item) for item in value] if key in LIST_KEYS else (
            resolve_path(value, root) if key in PATH_KEYS else value
        )

    section_values = config.get(section, {})
    if section_values:
        if not isinstance(section_values, dict):
            raise ValueError(f"[{section}] must be a TOML table in {config_path}")
        for key, value in section_values.items():
            defaults[key] = [str(item) for item in value] if key in LIST_KEYS else (
                resolve_path(value, root) if key in PATH_KEYS else value
            )

    if section == "evaluation":
        checkpoint = defaults.get("checkpoint")
        if checkpoint is None and defaults.get("output_dir") is not None:
            defaults["checkpoint"] = Path(defaults["output_dir"]) / "best_checkpoint.pt"
        eval_output = section_values.get("output_dir") if isinstance(section_values, dict) else None
        if eval_output is None and defaults.get("output_dir") is not None:
            split_name = str(defaults.get("split_name", "test"))
            defaults["output_dir"] = Path(defaults["output_dir"]) / f"{split_name}_eval"

    return defaults
