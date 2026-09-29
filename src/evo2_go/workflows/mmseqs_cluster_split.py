#!/usr/bin/env python3
"""Create an Evo2-GO split by clustering CDS sequences with MMseqs linclust."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

from evo2_go.model.dataset import (
    DEFAULT_EMBEDDING_ROOT,
    EmbeddingRecord,
    cds_id_from_sequence_id,
    complete_metadata_paths_from_manifest,
    load_embedding_records,
    load_label_index_cache,
    parse_tax_ids,
    tax_ids_from_file,
)


DEFAULT_INPUT_ROOT = Path("outputs/map_gene_cds/length_filtered_max_8192")
DEFAULT_MMSEQS = Path("envs/mmseqs/bin/mmseqs")
DEFAULT_OUTPUT = Path("outputs/model_training/splits/mmseqs_linclust_manifest.tsv")
DEFAULT_SUMMARY = Path("outputs/model_training/splits/mmseqs_linclust_summary.tsv")
DEFAULT_CLUSTER_DIR = Path("outputs/model_training/splits/mmseqs_linclust")

SplitName = Literal["train", "val", "test"]
BalanceBy = Literal["records", "labeled_records"]


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


@dataclass
class ClusterStats:
    representative: str
    members: list[str]
    records: list[EmbeddingRecord]
    labeled_records: int

    def weight(self, balance_by: BalanceBy) -> int:
        if balance_by == "records":
            return len(self.records)
        if balance_by == "labeled_records":
            return self.labeled_records
        raise ValueError(f"Unsupported balance unit: {balance_by}")


def clean_mmseqs_env() -> dict[str, str]:
    """Return an environment that avoids inherited OpenMP binding surprises."""

    env = os.environ.copy()
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    return env


def open_text(path: Path) -> Iterator[object]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def iter_fasta(path: Path) -> Iterator[FastaRecord]:
    header: str | None = None
    parts: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = str(raw_line).strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, sequence="".join(parts))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        yield FastaRecord(header=header, sequence="".join(parts))


def wrap_sequence(sequence: str, width: int = 80) -> Iterator[str]:
    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def input_fasta_for_taxon(input_root: Path, tax_id: str) -> Path:
    return input_root / f"tax_{tax_id}" / "kept_representative_gene_cds.fna.gz"


def stable_score(value: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\t{value}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def natural_id_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def validate_fractions(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    if min(train, val, test) < 0.0:
        raise ValueError("split fractions must be non-negative")
    if train == 0.0 or test == 0.0:
        raise ValueError("train and test fractions must be positive")


def load_tax_ids(raw_tax_ids: Sequence[str], tax_id_files: Sequence[Path], limit_taxa: int) -> set[str] | None:
    values = list(raw_tax_ids)
    for tax_id_file in tax_id_files:
        values.extend(tax_ids_from_file(tax_id_file))
    if limit_taxa > 0:
        values = values[:limit_taxa]
    return parse_tax_ids(values)


def records_by_tax(records: Sequence[EmbeddingRecord]) -> dict[str, list[EmbeddingRecord]]:
    grouped: dict[str, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tax_id].append(record)
    return dict(grouped)


def write_cluster_input_fasta(
    path: Path,
    *,
    input_root: Path,
    records: Sequence[EmbeddingRecord],
) -> tuple[int, int]:
    """Write one FASTA containing exactly the selected embedded records."""

    wanted_headers = {record.sequence_id for record in records}
    written_headers: set[str] = set()
    grouped_records = records_by_tax(records)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt") as output:
        for tax_id in sorted(grouped_records, key=natural_id_sort_key):
            fasta_path = input_fasta_for_taxon(input_root, tax_id)
            if not fasta_path.is_file():
                raise FileNotFoundError(f"Missing length-filtered FASTA for taxon {tax_id}: {fasta_path}")
            for fasta_record in iter_fasta(fasta_path):
                if fasta_record.header not in wanted_headers:
                    continue
                output.write(f">{fasta_record.header}\n")
                for line in wrap_sequence(fasta_record.sequence):
                    output.write(f"{line}\n")
                written_headers.add(fasta_record.header)

    missing = len(wanted_headers.difference(written_headers))
    if missing:
        examples = sorted(wanted_headers.difference(written_headers))[:5]
        raise ValueError(f"{missing:,} selected embedding records were not found in FASTA; examples={examples}")
    tmp_path.replace(path)
    return len(written_headers), len(grouped_records)


def run_mmseqs_linclust(
    input_fasta: Path,
    out_dir: Path,
    *,
    min_identity: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int | None,
    kmer_per_seq_scale: str | None,
    linclust_version: int | None,
    include_adjacency: int | None,
    num_adjacency: int | None,
    threads: int,
    mmseqs_path: Path,
) -> tuple[Path, Path]:
    """Run MMseqs linclust and return cluster TSV and representatives FASTA."""

    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "seqdb"
    clu = out_dir / "clu"
    tmp = out_dir / "tmp"
    repseq_db = out_dir / "repseq"
    rep_fasta = out_dir / "representatives.fasta"
    tsv = out_dir / "clusters.tsv"

    env = clean_mmseqs_env()
    subprocess.run(
        [str(mmseqs_path), "createdb", str(input_fasta), str(db), "--threads", str(threads)],
        check=True,
        env=env,
    )

    env = clean_mmseqs_env()
    cmd = [
        str(mmseqs_path),
        "linclust",
        str(db),
        str(clu),
        str(tmp),
        "--min-seq-id",
        str(min_identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    if cluster_mode is not None:
        cmd.extend(["--cluster-mode", str(cluster_mode)])
    if kmer_per_seq_scale is not None:
        cmd.extend(["--kmer-per-seq-scale", kmer_per_seq_scale])
    if linclust_version is not None:
        cmd.extend(["--linclust-version", str(linclust_version)])
    if include_adjacency is not None:
        cmd.extend(["--include-adjacency", str(include_adjacency)])
    if num_adjacency is not None:
        cmd.extend(["--num-adjacency", str(num_adjacency)])
    subprocess.run(cmd, check=True, env=env)

    env = clean_mmseqs_env()
    subprocess.run(
        [
            str(mmseqs_path),
            "createtsv",
            str(db),
            str(db),
            str(clu),
            str(tsv),
            "--threads",
            str(threads),
        ],
        check=True,
        env=env,
    )

    env = clean_mmseqs_env()
    subprocess.run(
        [
            str(mmseqs_path),
            "result2repseq",
            str(db),
            str(clu),
            str(repseq_db),
            "--threads",
            str(threads),
        ],
        check=True,
        env=env,
    )

    env = clean_mmseqs_env()
    subprocess.run(
        [str(mmseqs_path), "result2flat", str(db), str(db), str(repseq_db), str(rep_fasta)],
        check=True,
        env=env,
    )

    return tsv, rep_fasta


def parse_mmseqs_tsv(tsv_path: Path) -> dict[str, list[str]]:
    cluster_to_members: dict[str, list[str]] = defaultdict(list)
    with tsv_path.open("rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed MMseqs cluster row {line_number} in {tsv_path}")
            rep, member = fields[:2]
            cluster_to_members[rep].append(member)
    if not cluster_to_members:
        raise ValueError(f"No MMseqs clusters found in {tsv_path}")
    return dict(cluster_to_members)


def build_cluster_stats(
    cluster_to_members: dict[str, list[str]],
    header_to_record: dict[str, EmbeddingRecord],
    label_index: dict[tuple[str, str], tuple[int, ...]] | None,
) -> list[ClusterStats]:
    seen_members: set[str] = set()
    clusters: list[ClusterStats] = []
    for representative, members in sorted(cluster_to_members.items()):
        records: list[EmbeddingRecord] = []
        for member in members:
            record = header_to_record.get(member)
            if record is None:
                raise ValueError(f"MMseqs member is not in selected embedding records: {member!r}")
            if member in seen_members:
                raise ValueError(f"MMseqs member appears in multiple clusters: {member!r}")
            seen_members.add(member)
            records.append(record)
        labeled_records = (
            sum(1 for record in records if label_index and label_index.get(record.gene_key))
            if label_index is not None
            else 0
        )
        clusters.append(
            ClusterStats(
                representative=representative,
                members=members,
                records=records,
                labeled_records=labeled_records,
            )
        )

    missing = set(header_to_record).difference(seen_members)
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(f"{len(missing):,} selected records are missing from MMseqs clusters; examples={examples}")
    return clusters


def assign_clusters(
    clusters: Sequence[ClusterStats],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    balance_by: BalanceBy,
    seed: str,
) -> dict[str, SplitName]:
    """Deterministically assign whole clusters while approximately matching target sizes."""

    fractions: dict[SplitName, float] = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    active_fractions = {split: fraction for split, fraction in fractions.items() if fraction > 0.0}
    total_weight = sum(cluster.weight(balance_by) for cluster in clusters)
    if total_weight <= 0:
        raise ValueError(f"Cannot balance split by {balance_by}; total weight is zero")

    target = {split: fraction * total_weight for split, fraction in active_fractions.items()}
    current = {split: 0.0 for split in active_fractions}
    assignments: dict[str, SplitName] = {}
    sorted_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -cluster.weight(balance_by),
            stable_score(cluster.representative, seed),
            cluster.representative,
        ),
    )

    for cluster in sorted_clusters:
        weight = float(cluster.weight(balance_by))

        def penalty(split: SplitName) -> tuple[float, float, str]:
            score = 0.0
            for candidate_split, target_value in target.items():
                new_value = current[candidate_split]
                if candidate_split == split:
                    new_value += weight
                if target_value > 0.0:
                    score += ((new_value - target_value) / target_value) ** 2
            return (score, stable_score(f"{cluster.representative}\t{split}", seed), split)

        selected_split = min(active_fractions, key=penalty)
        assignments[cluster.representative] = selected_split
        current[selected_split] += weight

    return assignments


def write_split_manifest(
    path: Path,
    clusters: Sequence[ClusterStats],
    cluster_to_split: dict[str, SplitName],
    *,
    seed: str,
    strategy: str,
) -> tuple[Counter[str], Counter[tuple[str, str]], Counter[str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    split_counts: Counter[str] = Counter()
    tax_split_counts: Counter[tuple[str, str]] = Counter()
    cluster_split_counts: Counter[str] = Counter()

    fieldnames = (
        "split",
        "tax_id",
        "GeneID",
        "cds_id",
        "embedding_id",
        "sequence_id",
        "sequence_length",
        "token_count",
        "mmseqs_cluster_id",
        "mmseqs_cluster_representative",
        "mmseqs_cluster_size",
        "split_strategy",
        "ortholog_safe",
        "seed",
    )
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for cluster_index, cluster in enumerate(
            sorted(clusters, key=lambda item: item.representative),
            start=1,
        ):
            split = cluster_to_split[cluster.representative]
            cluster_id = f"mmseqs_cluster_{cluster_index:09d}"
            cluster_split_counts[split] += 1
            for record in sorted(
                cluster.records,
                key=lambda item: (
                    natural_id_sort_key(item.tax_id),
                    natural_id_sort_key(item.gene_id),
                    item.cds_id,
                ),
            ):
                split_counts[split] += 1
                tax_split_counts[(record.tax_id, split)] += 1
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
                        "mmseqs_cluster_id": cluster_id,
                        "mmseqs_cluster_representative": cluster.representative,
                        "mmseqs_cluster_size": len(cluster.records),
                        "split_strategy": strategy,
                        "ortholog_safe": 0,
                        "seed": seed,
                    }
                )

    tmp_path.replace(path)
    return split_counts, tax_split_counts, cluster_split_counts


def write_key_value_tsv(path: Path, rows: list[tuple[object, object]]) -> None:
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
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--mmseqs-path", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--tax-id", action="append", default=[])
    parser.add_argument("--tax-id-file", action="append", type=Path, default=[])
    parser.add_argument("--limit-taxa", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--expected-embedding-dim", type=int, default=4096)
    parser.add_argument("--min-seq-id", type=float, default=0.70)
    parser.add_argument("--coverage", type=float, default=0.50)
    parser.add_argument("--cov-mode", type=int, default=1)
    parser.add_argument(
        "--cluster-mode",
        type=int,
        help="Optional MMseqs linclust cluster mode override. Omit to match linclust defaults.",
    )
    parser.add_argument(
        "--kmer-per-seq-scale",
        help="Optional MMseqs linclust k-mer scaling override, e.g. 0.3 for nucleotide sequences.",
    )
    parser.add_argument(
        "--linclust-version",
        type=int,
        choices=(1, 2),
        help="Optional MMseqs linclust implementation override.",
    )
    parser.add_argument(
        "--include-adjacency",
        type=int,
        choices=(0, 1),
        help="Optional MMseqs linclust adjacency center-swapping override.",
    )
    parser.add_argument(
        "--num-adjacency",
        type=int,
        help="Optional MMseqs linclust number of adjacency center-swapping passes.",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--balance-by", choices=("records", "labeled_records"), default="records")
    parser.add_argument("--seed", default="evo2-go-mmseqs-linclust-split-v1")
    parser.add_argument(
        "--reuse-clusters",
        action="store_true",
        help="Reuse existing cluster_dir/clusters.tsv instead of rerunning MMseqs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_fractions(args.train_fraction, args.val_fraction, args.test_fraction)
    if args.limit_taxa < 0:
        raise ValueError("--limit-taxa must be >= 0")
    if not args.mmseqs_path.is_file():
        raise FileNotFoundError(f"MMseqs binary not found: {args.mmseqs_path}")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing split manifest: {args.output}")
    if args.summary.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing split summary: {args.summary}")

    selected_tax_ids = load_tax_ids(args.tax_id, args.tax_id_file, args.limit_taxa)
    metadata_paths = complete_metadata_paths_from_manifest(
        args.embedding_root / "embedding_dataset_manifest.tsv",
        tax_ids=selected_tax_ids,
    )
    if args.limit_taxa > 0 and selected_tax_ids is None:
        metadata_paths = metadata_paths[: args.limit_taxa]
    records = load_embedding_records(
        metadata_paths,
        max_records=args.max_records,
        expected_embedding_dim=args.expected_embedding_dim,
    )
    records_before_label_filter = len(records)

    label_index = None
    if args.label_cache is not None:
        label_index = load_label_index_cache(args.label_cache)

    header_to_record = {record.sequence_id: record for record in records}
    if len(header_to_record) != len(records):
        raise ValueError("Duplicate FASTA sequence_id values found in selected embedding records")

    cluster_input_fasta = args.cluster_dir / "input_sequences.fna"
    sequences_written, taxa_written = write_cluster_input_fasta(
        cluster_input_fasta,
        input_root=args.input_root,
        records=records,
    )

    cluster_tsv = args.cluster_dir / "clusters.tsv"
    representatives_fasta = args.cluster_dir / "representatives.fasta"
    if args.reuse_clusters:
        if not cluster_tsv.is_file():
            raise FileNotFoundError(f"--reuse-clusters requested but missing {cluster_tsv}")
    else:
        cluster_tsv, representatives_fasta = run_mmseqs_linclust(
            cluster_input_fasta,
            args.cluster_dir,
            min_identity=args.min_seq_id,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            cluster_mode=args.cluster_mode,
            kmer_per_seq_scale=args.kmer_per_seq_scale,
            linclust_version=args.linclust_version,
            include_adjacency=args.include_adjacency,
            num_adjacency=args.num_adjacency,
            threads=args.threads,
            mmseqs_path=args.mmseqs_path,
        )

    cluster_to_members = parse_mmseqs_tsv(cluster_tsv)
    clusters = build_cluster_stats(cluster_to_members, header_to_record, label_index)
    cluster_to_split = assign_clusters(
        clusters,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        balance_by=args.balance_by,
        seed=args.seed,
    )
    strategy = "mmseqs_linclust_cluster_split"
    split_counts, tax_split_counts, cluster_split_counts = write_split_manifest(
        args.output,
        clusters,
        cluster_to_split,
        seed=args.seed,
        strategy=strategy,
    )

    labeled_split_counts: Counter[str] = Counter()
    if label_index is not None:
        for cluster in clusters:
            split = cluster_to_split[cluster.representative]
            labeled_split_counts[split] += cluster.labeled_records

    cluster_sizes = [len(cluster.records) for cluster in clusters]
    singleton_clusters = sum(1 for size in cluster_sizes if size == 1)
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    total_balance_weight = sum(cluster.weight(args.balance_by) for cluster in clusters)
    summary_rows: list[tuple[object, object]] = [
        ("split_strategy", strategy),
        ("ortholog_safe", 0),
        ("embedding_root", args.embedding_root),
        ("input_root", args.input_root),
        ("label_cache", args.label_cache or ""),
        ("cluster_dir", args.cluster_dir),
        ("cluster_input_fasta", cluster_input_fasta),
        ("cluster_tsv", cluster_tsv),
        ("representatives_fasta", representatives_fasta if representatives_fasta.is_file() else ""),
        ("mmseqs_path", args.mmseqs_path),
        ("min_seq_id", args.min_seq_id),
        ("coverage", args.coverage),
        ("cov_mode", args.cov_mode),
        ("cluster_mode", args.cluster_mode if args.cluster_mode is not None else ""),
        ("kmer_per_seq_scale", args.kmer_per_seq_scale or ""),
        ("linclust_version", args.linclust_version if args.linclust_version is not None else ""),
        ("include_adjacency", args.include_adjacency if args.include_adjacency is not None else ""),
        ("num_adjacency", args.num_adjacency if args.num_adjacency is not None else ""),
        ("threads", args.threads),
        ("seed", args.seed),
        ("balance_by", args.balance_by),
        ("train_fraction", args.train_fraction),
        ("val_fraction", args.val_fraction),
        ("test_fraction", args.test_fraction),
        ("records_before_label_filter", records_before_label_filter),
        ("records_written", sum(split_counts.values())),
        ("sequences_written_to_cluster_fasta", sequences_written),
        ("taxa", taxa_written),
        ("clusters", len(clusters)),
        ("singleton_clusters", singleton_clusters),
        ("max_cluster_size", max_cluster_size),
        ("balance_weight_total", total_balance_weight),
        ("train_records", split_counts["train"]),
        ("val_records", split_counts["val"]),
        ("test_records", split_counts["test"]),
        ("train_clusters", cluster_split_counts["train"]),
        ("val_clusters", cluster_split_counts["val"]),
        ("test_clusters", cluster_split_counts["test"]),
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

    write_key_value_tsv(args.summary, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
