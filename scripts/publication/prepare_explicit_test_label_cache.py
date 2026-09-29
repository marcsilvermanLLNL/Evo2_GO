#!/usr/bin/env python3
"""Build a vocabulary-aligned label cache from direct Gene2GO annotations.

Unlike the training label cache, this cache does not add ``is_a`` or
``part_of`` ancestors.  It is intended for a secondary evaluation of the
trained model against the original positive Gene2GO gene--term pairs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "mmseqs_linclust_full_85_5_10_min_count_15000_bs256_mean_loss_fixed_val50k_v2"
DEFAULT_BUILD_ID = "all_labels_true_path_full_cleaned_min_count_15000_v1"
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/builds"
    / DEFAULT_BUILD_ID
    / "runs"
    / DEFAULT_RUN_NAME
    / "test_eval_val_threshold_061"
    / "test_split_manifest_compact.tsv"
)
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz"
)
DEFAULT_VOCABULARY = (
    PROJECT_ROOT
    / "outputs/builds"
    / DEFAULT_BUILD_ID
    / "adjacency/go_vocabulary.tsv"
)
DEFAULT_GO_BASIC_OBO = PROJECT_ROOT / "data/go/go-basic.obo"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/builds"
    / DEFAULT_BUILD_ID
    / "data/labels/explicit_test_label_index.tsv.gz"
)


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def packed_identity(tax_id: str, gene_id: str) -> int:
    tax_value = int(tax_id)
    gene_value = int(gene_id)
    if not 0 <= tax_value < 2**32 or not 0 <= gene_value < 2**32:
        raise ValueError(f"Identity cannot be packed into uint64: {tax_id!r}, {gene_id!r}")
    return (tax_value << 32) | gene_value


def load_selected_genes(path: Path) -> set[int]:
    selected: set[int] = set()
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"tax_id", "GeneID"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            selected.add(packed_identity(row["tax_id"], row["GeneID"]))
    if not selected:
        raise ValueError(f"No gene identities found in {path}")
    return selected


def load_vocabulary(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    go_ids: list[str] = []
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "GO_ID" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing GO_ID")
        indexed: dict[int, str] = {}
        for fallback_index, row in enumerate(reader):
            index = int(row.get("go_index", fallback_index))
            indexed[index] = row["GO_ID"].strip()
    expected = set(range(len(indexed)))
    if set(indexed) != expected:
        raise ValueError(f"Non-contiguous go_index values in {path}")
    go_ids = [indexed[index] for index in range(len(indexed))]
    return tuple(go_ids), {go_id: index for index, go_id in enumerate(go_ids)}


def load_alt_id_map(path: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    current_id: str | None = None
    current_alt_ids: list[str] = []

    def finish() -> None:
        if current_id is not None:
            for alt_id in current_alt_ids:
                aliases[alt_id] = current_id

    with path.open("rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finish()
                current_id = None
                current_alt_ids = []
            elif line.startswith("["):
                finish()
                current_id = None
                current_alt_ids = []
            elif line.startswith("id: "):
                current_id = line.removeprefix("id: ").strip()
            elif line.startswith("alt_id: "):
                current_alt_ids.append(line.removeprefix("alt_id: ").strip())
        finish()
    return aliases


def write_summary(path: Path, rows: list[tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    tmp.replace(path)


def build_cache(
    *,
    selected: set[int],
    annotations: Path,
    vocabulary_ids: tuple[str, ...],
    go_to_index: dict[str, int],
    alt_to_primary: dict[str, str],
    vocabulary_path: Path,
    output: Path,
    progress_every: int,
) -> list[tuple[str, object]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    rows_seen = 0
    selected_annotation_rows = 0
    selected_genes_seen = 0
    genes_written = 0
    direct_pairs_written = 0
    direct_pairs_outside_vocabulary = 0
    current_identity: int | None = None
    current_tax_id = ""
    current_gene_id = ""
    current_indices: set[int] = set()
    current_outside: set[str] = set()
    started = time.time()

    def flush(writer: csv.writer) -> None:
        nonlocal selected_genes_seen, genes_written, direct_pairs_written
        nonlocal direct_pairs_outside_vocabulary
        if current_identity is None:
            return
        selected_genes_seen += 1
        direct_pairs_outside_vocabulary += len(current_outside)
        if not current_indices:
            return
        indices = sorted(current_indices)
        writer.writerow(
            (
                current_tax_id,
                current_gene_id,
                ",".join(str(index) for index in indices),
                ",".join(vocabulary_ids[index] for index in indices),
            )
        )
        genes_written += 1
        direct_pairs_written += len(indices)

    try:
        with gzip.open(tmp, "wt", newline="") as output_handle:
            writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("tax_id", "GeneID", "go_indices", "go_ids"))
            with gzip.open(annotations, "rt", newline="") as annotation_handle:
                reader = csv.DictReader(annotation_handle, delimiter="\t")
                required = {"tax_id", "GeneID", "GO_ID", "Evidence"}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{annotations} is missing columns: {sorted(missing)}")
                for rows_seen, row in enumerate(reader, start=1):
                    if progress_every > 0 and rows_seen % progress_every == 0:
                        elapsed = (time.time() - started) / 60.0
                        log(
                            f"Scanned {rows_seen:,} annotation rows; retained "
                            f"{direct_pairs_written + len(current_indices):,} direct pairs "
                            f"for {genes_written + int(bool(current_indices)):,} genes; "
                            f"elapsed {elapsed:.1f} min"
                        )
                    tax_id = row["tax_id"].strip()
                    gene_id = row["GeneID"].strip()
                    identity = packed_identity(tax_id, gene_id)
                    if identity not in selected:
                        continue
                    if identity != current_identity:
                        flush(writer)
                        current_identity = identity
                        current_tax_id = tax_id
                        current_gene_id = gene_id
                        current_indices = set()
                        current_outside = set()
                    selected_annotation_rows += 1
                    raw_go_id = row["GO_ID"].strip()
                    go_id = alt_to_primary.get(raw_go_id, raw_go_id)
                    go_index = go_to_index.get(go_id)
                    if go_index is None:
                        current_outside.add(go_id)
                    else:
                        current_indices.add(go_index)
                flush(writer)
        tmp.replace(output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return [
        ("label_semantics", "explicit_gene2go_terms_without_true_path_propagation"),
        ("evidence_filter", "all_evidence_including_iea"),
        ("selected_test_genes", len(selected)),
        ("annotation_rows_seen", rows_seen),
        ("selected_annotation_rows", selected_annotation_rows),
        ("selected_genes_with_any_annotation", selected_genes_seen),
        ("genes_with_retained_explicit_labels", genes_written),
        ("genes_without_retained_explicit_labels", len(selected) - genes_written),
        ("retained_unique_explicit_gene_go_pairs", direct_pairs_written),
        ("unique_explicit_gene_go_pairs_outside_model_vocabulary", direct_pairs_outside_vocabulary),
        ("model_vocabulary_terms", len(vocabulary_ids)),
        ("annotations", annotations),
        ("vocabulary", vocabulary_path),
        ("output", output),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--go-basic-obo", type=Path, default=DEFAULT_GO_BASIC_OBO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = args.summary or args.output.with_name("explicit_test_label_summary.tsv")
    existing = [path for path in (args.output, summary) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing explicit-label outputs without --force: "
            + ", ".join(str(path) for path in existing)
        )
    log(f"Loading held-out identities from {args.split_manifest}")
    selected = load_selected_genes(args.split_manifest)
    log(f"Loaded {len(selected):,} held-out gene identities")
    vocabulary_ids, go_to_index = load_vocabulary(args.vocabulary)
    alt_to_primary = load_alt_id_map(args.go_basic_obo)
    log(f"Streaming direct annotations from {args.annotations}")
    summary_rows = build_cache(
        selected=selected,
        annotations=args.annotations,
        vocabulary_ids=vocabulary_ids,
        go_to_index=go_to_index,
        alt_to_primary=alt_to_primary,
        vocabulary_path=args.vocabulary,
        output=args.output,
        progress_every=args.progress_every,
    )
    write_summary(summary, summary_rows)
    log(f"Wrote {args.output}")
    log(f"Wrote {summary}")


if __name__ == "__main__":
    main()
