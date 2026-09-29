"""Top-level command dispatcher for the remodeled Evo2-GO pipeline."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS: dict[str, str] = {
    "build-database": "evo2_go.workflows.build_database",
    "prepare-training-inputs": "evo2_go.workflows.prepare_training_inputs",
    "filter-cds-length": "evo2_go.embeddings.length_filter",
    "run-embeddings": "evo2_go.embeddings.run_embeddings",
    "validate-embeddings": "evo2_go.embeddings.validate_embeddings",
    "map-gene-cds": "evo2_go.database.map_gene_cds",
    "map-gene-cds-by-tax": "evo2_go.database.map_gene_cds_by_tax",
    "go-term-counts": "evo2_go.database.go_term_counts",
    "true-path-labels": "evo2_go.database.true_path_labels",
    "go-adjacency": "evo2_go.ontology.go_adjacency",
    "taxon-holdout-split": "evo2_go.workflows.taxon_holdout_split",
    "mmseqs-cluster-split": "evo2_go.workflows.mmseqs_cluster_split",
    "validation-panel": "evo2_go.training.validation_panel",
    "train-model": "evo2_go.training.train",
    "evaluate-model": "evo2_go.training.evaluate",
    "visualize-results": "evo2_go.training.visualize",
}


def print_help() -> None:
    print(
        "Usage: evo2-go <command> [args]\n"
        "\n"
        "Migrated commands:\n"
        "  build-database        Run a config-driven database build workflow.\n"
        "  prepare-training-inputs\n"
        "                        Refresh labels, adjacency, label cache, and split manifest.\n"
        "  filter-cds-length     Filter representative CDS records by max Evo2 context length.\n"
        "  run-embeddings        Extract Evo2 embeddings into one shared embedding root.\n"
        "  validate-embeddings   Validate embedding metadata, shards, and label coverage.\n"
        "  map-gene-cds          Select representative CDS records for one taxon or pilot run.\n"
        "  map-gene-cds-by-tax   Run representative-CDS mapping across taxa.\n"
        "  go-term-counts        Count GO support for selected trainable genes.\n"
        "  true-path-labels      Expand explicit labels to true-path ancestor labels.\n"
        "  go-adjacency          Build deterministic GO vocabulary and sparse adjacency.\n"
        "  taxon-holdout-split   Create a taxon holdout split from embedding metadata.\n"
        "  mmseqs-cluster-split  Create a cluster split from MMseqs linclust clusters.\n"
        "  validation-panel     Create a fixed, seeded sample of a validation split.\n"
        "  train-model           Train the Evo2-GO dual encoder from a build/config.\n"
        "  evaluate-model        Evaluate a checkpoint on one split.\n"
        "  visualize-results     Write training/evaluation result figures.\n"
        "\n"
        "Run `evo2-go <command> --help` for command-specific options.",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print_help()
        return 0

    command = args[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_help()
        return 2

    module = importlib.import_module(COMMANDS[command])
    command_main = getattr(module, "main", None)
    if command_main is None:
        raise RuntimeError(f"Command module has no main(): {COMMANDS[command]}")

    old_argv = sys.argv
    try:
        sys.argv = [f"evo2-go {command}", *args[1:]]
        return int(command_main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
