#!/usr/bin/env python3
"""Map labeled GeneIDs to representative CDS sequences.

This script is intended as the first pilot for:

    GeneID -> CDS FASTA candidates -> longest CDS

It uses the GO label table to choose a target set of labeled genes, then scans
local CDS FASTA files such as ``*_cds_from_genomic.fna.gz``. The selected
representative CDS per
``(tax_id, GeneID)`` is deterministic:

1. longest sequence
2. lowest IUPAC ambiguity count
3. lexicographically smallest CDS ID

CDS FASTA headers usually carry ``GeneID`` and often ``protein_id``. Therefore
the default matcher accepts a CDS when its FASTA header GeneID matches a target
labeled gene. Optionally pass ``--gene2refseq`` to load RefSeq provenance, or
use ``--require-refseq-accession-match`` for a stricter pilot.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from typing import Iterator


DNA_ALPHABET = frozenset("ACGTNRYSWKMBDHV")
GENE_ID_PATTERNS = (
    re.compile(r"(?:GeneID:|GeneID=)(\d+)"),
    re.compile(r"(?:geneid:|geneid=)(\d+)", re.IGNORECASE),
)
ACCESSION_PATTERN = re.compile(r"\b[A-Z]{1,4}_?\d+(?:\.\d+)?\b")
BRACKET_FIELD_PATTERN = re.compile(r"\[([^=\]]+)=([^\]]*)\]")

GENE2REFSEQ_COLUMNS = (
    "#tax_id",
    "GeneID",
    "status",
    "RNA_nucleotide_accession.version",
    "RNA_nucleotide_gi",
    "protein_accession.version",
    "protein_gi",
    "genomic_nucleotide_accession.version",
    "genomic_nucleotide_gi",
    "start_position_on_the_genomic_accession",
    "end_position_on_the_genomic_accession",
    "orientation",
    "assembly",
    "mature_peptide_accession.version",
    "mature_peptide_gi",
    "Symbol",
)

MATCHING_ACCESSION_COLUMNS = (
    "RNA_nucleotide_accession.version",
    "protein_accession.version",
    "mature_peptide_accession.version",
)

PROVENANCE_ACCESSION_COLUMNS = (
    "RNA_nucleotide_accession.version",
    "protein_accession.version",
    "genomic_nucleotide_accession.version",
    "mature_peptide_accession.version",
)

PROGRESS_RECORD_INTERVAL = 50000
PROGRESS_GENE_INTERVAL = 100


def log_progress(message: str) -> None:
    """Write a timestamped progress message to stderr for Flux logs."""

    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class RefSeqRow:
    """One gene2refseq row relevant to a target labeled gene."""

    tax_id: int
    gene_id: int
    status: str
    rna_accession: str
    protein_accession: str
    genomic_accession: str
    start: str
    end: str
    orientation: str
    assembly: str
    mature_peptide_accession: str
    symbol: str


@dataclass(frozen=True)
class CdsCandidate:
    """One CDS FASTA record matched to a labeled gene."""

    tax_id: int
    gene_id: int
    cds_id: str
    length: int
    ambiguity_count: int
    matched_refseq_accessions: tuple[str, ...]
    source_fasta: str
    header: str
    sequence: str


def open_text(path: Path):
    """Open a plain text or gzip-compressed file for reading text."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("rt", newline="")


def open_write_text(path: Path):
    """Open a plain text or gzip-compressed file for writing text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="")
    return path.open("wt", newline="")


def clean_field(value: object) -> str:
    """Normalize a table field."""

    if value is None:
        return ""
    return str(value).strip()


def accession_without_version(accession: str) -> str:
    """Drop a trailing accession version if one exists."""

    return accession.split(".", 1)[0]


def accession_variants(accession: str) -> set[str]:
    """Return versioned and unversioned forms for matching."""

    accession = clean_field(accession)
    if not accession or accession == "-":
        return set()
    return {accession, accession_without_version(accession)}


def row_accessions(row: dict[str, str]) -> set[str]:
    """Collect gene-specific RefSeq accession values for CDS matching."""

    accessions: set[str] = set()
    for column in MATCHING_ACCESSION_COLUMNS:
        accessions.update(accession_variants(row.get(column, "")))
    return accessions


def parse_int_field(row: dict[str, str], column: str) -> int:
    """Parse a required integer column from a TSV row."""

    value = clean_field(row.get(column))
    if not value:
        raise ValueError(f"Missing required integer field {column!r}")
    return int(value)


def validate_gene2refseq_header(fieldnames: list[str] | None, path: Path) -> None:
    """Fail early when gene2refseq columns do not match NCBI's schema."""

    if fieldnames is None:
        raise ValueError(f"{path} is missing a header")
    observed = tuple(fieldnames)
    if observed != GENE2REFSEQ_COLUMNS:
        raise ValueError(
            f"Unexpected gene2refseq header in {path}: expected "
            f"{GENE2REFSEQ_COLUMNS!r}, observed {observed!r}"
        )


def validate_label_header(fieldnames: list[str] | None, path: Path) -> None:
    """Validate the minimal positive label table header."""

    expected = ("tax_id", "GeneID", "GO_ID")
    if tuple(fieldnames or ()) != expected:
        raise ValueError(
            f"Unexpected label header in {path}: expected {expected!r}, "
            f"observed {tuple(fieldnames or ())!r}"
        )


def iter_label_rows(path: Path) -> Iterator[tuple[int, int, str]]:
    """Stream rows from positive_gene_go_labels.tsv.gz."""

    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_label_header(reader.fieldnames, path)
        for row in reader:
            yield (
                parse_int_field(row, "tax_id"),
                parse_int_field(row, "GeneID"),
                clean_field(row.get("GO_ID")),
            )


def choose_target_genes(
    labels_path: Path,
    limit_genes: int,
    tax_ids: set[int] | None,
) -> list[tuple[int, int]]:
    """Choose the first N unique labeled genes from the label table."""

    targets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for tax_id, gene_id, _go_id in iter_label_rows(labels_path):
        if tax_ids is not None and tax_id not in tax_ids:
            continue

        key = (tax_id, gene_id)
        if key in seen:
            continue

        seen.add(key)
        targets.append(key)

        if limit_genes > 0 and len(targets) >= limit_genes:
            break

    return targets


def write_target_labels(
    labels_path: Path,
    target_keys: set[tuple[int, int]],
    output_path: Path,
) -> dict[tuple[int, int], set[str]]:
    """Write all GO labels for target genes and return labels by gene."""

    labels_by_gene: dict[tuple[int, int], set[str]] = {key: set() for key in target_keys}

    with open_write_text(output_path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("tax_id", "GeneID", "GO_ID"))

        for tax_id, gene_id, go_id in iter_label_rows(labels_path):
            key = (tax_id, gene_id)
            if key not in target_keys:
                continue

            writer.writerow((tax_id, gene_id, go_id))
            labels_by_gene[key].add(go_id)

    return labels_by_gene


def tax_label_cache_path(labels_path: Path, tax_id: int) -> Path:
    """Return the stable per-taxon label cache path for a global label table."""

    return labels_path.parent / "by_tax" / f"tax_{tax_id}.positive_gene_go_labels.tsv.gz"


def write_tax_label_cache(labels_path: Path, tax_id: int, cache_path: Path) -> None:
    """Write labels for one taxon so repeated pilots avoid global label scans."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open_write_text(cache_path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("tax_id", "GeneID", "GO_ID"))
        for row_tax_id, gene_id, go_id in iter_label_rows(labels_path):
            if row_tax_id == tax_id:
                writer.writerow((row_tax_id, gene_id, go_id))


def labels_for_mapping(labels_path: Path, tax_ids: set[int] | None) -> tuple[Path, set[int] | None]:
    """Use a per-taxon label cache when a single taxon is requested."""

    if tax_ids is None or len(tax_ids) != 1:
        return labels_path, tax_ids

    tax_id = next(iter(tax_ids))
    cache_path = tax_label_cache_path(labels_path, tax_id)
    cache_is_current = (
        cache_path.is_file()
        and cache_path.stat().st_size > 0
        and cache_path.stat().st_mtime >= labels_path.stat().st_mtime
    )
    if cache_is_current:
        log_progress(f"Using cached taxon label table: {cache_path}")
    else:
        log_progress(f"Building cached taxon label table: {cache_path}")
        cache_start = datetime.now()
        write_tax_label_cache(labels_path=labels_path, tax_id=tax_id, cache_path=cache_path)
        log_progress(
            "Finished cached taxon label table: "
            f"{cache_path} elapsed_seconds={(datetime.now() - cache_start).total_seconds():.1f}"
        )

    return cache_path, None


def load_gene2refseq(
    path: Path,
    target_keys: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], list[RefSeqRow]], dict[tuple[int, int], set[str]], dict[str, set[tuple[int, int]]]]:
    """Load gene2refseq rows and accession indexes for target genes."""

    rows_by_gene: dict[tuple[int, int], list[RefSeqRow]] = {key: [] for key in target_keys}
    accessions_by_gene: dict[tuple[int, int], set[str]] = {key: set() for key in target_keys}
    genes_by_accession: dict[str, set[tuple[int, int]]] = {}

    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_gene2refseq_header(reader.fieldnames, path)

        for row in reader:
            tax_id = parse_int_field(row, "#tax_id")
            gene_id = parse_int_field(row, "GeneID")
            key = (tax_id, gene_id)
            if key not in target_keys:
                continue

            refseq_row = RefSeqRow(
                tax_id=tax_id,
                gene_id=gene_id,
                status=clean_field(row.get("status")),
                rna_accession=clean_field(row.get("RNA_nucleotide_accession.version")),
                protein_accession=clean_field(row.get("protein_accession.version")),
                genomic_accession=clean_field(row.get("genomic_nucleotide_accession.version")),
                start=clean_field(row.get("start_position_on_the_genomic_accession")),
                end=clean_field(row.get("end_position_on_the_genomic_accession")),
                orientation=clean_field(row.get("orientation")),
                assembly=clean_field(row.get("assembly")),
                mature_peptide_accession=clean_field(row.get("mature_peptide_accession.version")),
                symbol=clean_field(row.get("Symbol")),
            )
            rows_by_gene[key].append(refseq_row)

            for accession in row_accessions(row):
                accessions_by_gene[key].add(accession)
                genes_by_accession.setdefault(accession, set()).add(key)

    return rows_by_gene, accessions_by_gene, genes_by_accession


def write_gene2refseq_matches(
    path: Path,
    rows_by_gene: dict[tuple[int, int], list[RefSeqRow]],
) -> None:
    """Write the target subset of gene2refseq rows."""

    with open_write_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "tax_id",
                "GeneID",
                "status",
                "RNA_nucleotide_accession.version",
                "protein_accession.version",
                "genomic_nucleotide_accession.version",
                "start_position_on_the_genomic_accession",
                "end_position_on_the_genomic_accession",
                "orientation",
                "assembly",
                "mature_peptide_accession.version",
                "Symbol",
            )
        )

        for key in sorted(rows_by_gene):
            for row in rows_by_gene[key]:
                writer.writerow(
                    (
                        row.tax_id,
                        row.gene_id,
                        row.status,
                        row.rna_accession,
                        row.protein_accession,
                        row.genomic_accession,
                        row.start,
                        row.end,
                        row.orientation,
                        row.assembly,
                        row.mature_peptide_accession,
                        row.symbol,
                    )
                )


def discover_fasta_paths(explicit_paths: Iterable[Path], glob_patterns: Iterable[str]) -> list[Path]:
    """Resolve explicit FASTA paths and glob patterns into existing files."""

    paths: list[Path] = []
    seen: set[Path] = set()

    for path in explicit_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"CDS FASTA file not found: {path}")
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    for pattern in glob_patterns:
        for match_text in sorted(glob.glob(pattern, recursive=True)):
            match = Path(match_text)
            resolved = match.resolve()
            if not resolved.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)

    return paths


def parse_header_fields(header: str) -> dict[str, list[str]]:
    """Parse NCBI-style bracket fields from a FASTA header."""

    fields: dict[str, list[str]] = {}
    for key, value in BRACKET_FIELD_PATTERN.findall(header):
        fields.setdefault(key, []).append(value)
    return fields


def extract_gene_ids(header: str) -> set[int]:
    """Extract GeneID values from a FASTA header."""

    gene_ids: set[int] = set()
    for pattern in GENE_ID_PATTERNS:
        for match in pattern.findall(header):
            gene_ids.add(int(match))
    return gene_ids


def extract_accessions(header: str) -> set[str]:
    """Extract accession-like tokens from a FASTA header."""

    accessions: set[str] = set()
    fields = parse_header_fields(header)

    for field_name in ("protein_id", "transcript_id", "db_xref"):
        for value in fields.get(field_name, []):
            if value.startswith("GeneID:"):
                continue
            for match in ACCESSION_PATTERN.findall(value):
                accessions.update(accession_variants(match))

    for match in ACCESSION_PATTERN.findall(header):
        accessions.update(accession_variants(match))

    return accessions


def normalize_sequence(sequence_parts: list[str], header: str) -> tuple[str, int]:
    """Normalize a FASTA sequence and count IUPAC ambiguity symbols."""

    sequence = "".join(sequence_parts).replace(" ", "").replace("\t", "").upper()
    if not sequence:
        raise ValueError(f"Empty sequence for FASTA record: {header}")

    invalid_symbols = sorted(set(sequence) - DNA_ALPHABET)
    if invalid_symbols:
        invalid_text = ", ".join(invalid_symbols)
        raise ValueError(f"Invalid DNA symbols in FASTA record {header!r}: {invalid_text}")

    ambiguity_count = sum(1 for base in sequence if base not in {"A", "C", "G", "T"})
    return sequence, ambiguity_count


def fasta_records(path: Path) -> Iterator[tuple[str, str, int]]:
    """Yield ``(header, sequence, ambiguity_count)`` from a FASTA file."""

    header: str | None = None
    sequence_parts: list[str] = []

    with open_text(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    sequence, ambiguity_count = normalize_sequence(sequence_parts, header)
                    yield header, sequence, ambiguity_count

                header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line.strip())

    if header is not None:
        sequence, ambiguity_count = normalize_sequence(sequence_parts, header)
        yield header, sequence, ambiguity_count


def candidate_sort_key(candidate: CdsCandidate) -> tuple[int, int, str]:
    """Sort key where the smallest tuple is the best candidate."""

    return (-candidate.length, candidate.ambiguity_count, candidate.cds_id)


def candidate_output_row(candidate: CdsCandidate, include_sequence: bool = False) -> tuple[object, ...]:
    """Convert a candidate to a TSV row."""

    base = (
        candidate.tax_id,
        candidate.gene_id,
        candidate.cds_id,
        candidate.length,
        candidate.ambiguity_count,
        "|".join(candidate.matched_refseq_accessions),
        candidate.source_fasta,
        candidate.header,
    )

    if include_sequence:
        return base + (candidate.sequence,)
    return base


def make_candidate(
    key: tuple[int, int],
    header: str,
    sequence: str,
    ambiguity_count: int,
    source_fasta: Path,
    matched_refseq_accessions: set[str],
) -> CdsCandidate:
    """Build a candidate object from one FASTA record."""

    cds_id = header.split(maxsplit=1)[0]
    return CdsCandidate(
        tax_id=key[0],
        gene_id=key[1],
        cds_id=cds_id,
        length=len(sequence),
        ambiguity_count=ambiguity_count,
        matched_refseq_accessions=tuple(sorted(matched_refseq_accessions)),
        source_fasta=str(source_fasta),
        header=header,
        sequence=sequence,
    )


def match_fasta_record_to_targets(
    header: str,
    target_keys: set[tuple[int, int]],
    target_keys_by_gene_id: dict[int, set[tuple[int, int]]],
    accessions_by_gene: dict[tuple[int, int], set[str]],
    genes_by_accession: dict[str, set[tuple[int, int]]],
    require_refseq_accession_match: bool,
) -> list[tuple[tuple[int, int], set[str]]]:
    """Return matching target genes and their matched RefSeq accessions."""

    header_gene_ids = extract_gene_ids(header)
    header_accessions = extract_accessions(header)

    matched: dict[tuple[int, int], set[str]] = {}

    for gene_id in header_gene_ids:
        for key in target_keys_by_gene_id.get(gene_id, set()):
            refseq_matches = header_accessions & accessions_by_gene.get(key, set())
            if require_refseq_accession_match and not refseq_matches:
                continue

            matched[key] = set(refseq_matches)

    for accession in header_accessions:
        for key in genes_by_accession.get(accession, set()):
            if key not in target_keys:
                continue

            if require_refseq_accession_match:
                matched.setdefault(key, set()).add(accession)
            elif key not in matched:
                matched[key] = {accession}
            else:
                matched[key].add(accession)

    return sorted(matched.items())


def scan_cds_fastas(
    fasta_paths: list[Path],
    target_keys: set[tuple[int, int]],
    target_keys_by_gene_id: dict[int, set[tuple[int, int]]],
    accessions_by_gene: dict[tuple[int, int], set[str]],
    genes_by_accession: dict[str, set[tuple[int, int]]],
    output_path: Path,
    require_refseq_accession_match: bool,
) -> tuple[dict[tuple[int, int], CdsCandidate], int, int]:
    """Scan CDS FASTA files, write candidates, and keep the best per gene."""

    best_by_gene: dict[tuple[int, int], CdsCandidate] = {}
    records_seen = 0
    candidates_seen = 0
    target_count = len(target_keys)
    next_gene_progress = PROGRESS_GENE_INTERVAL

    scan_start = datetime.now()
    log_progress(f"Starting CDS FASTA scan: files={len(fasta_paths)} target_genes={target_count}")

    with open_write_text(output_path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "tax_id",
                "GeneID",
                "cds_id",
                "length",
                "ambiguity_count",
                "matched_refseq_accessions",
                "source_fasta",
                "header",
            )
        )

        for fasta_path in fasta_paths:
            file_start = datetime.now()
            file_records_seen = 0
            log_progress(f"Scanning CDS FASTA: {fasta_path}")

            for header, sequence, ambiguity_count in fasta_records(fasta_path):
                records_seen += 1
                file_records_seen += 1
                if records_seen % PROGRESS_RECORD_INTERVAL == 0:
                    elapsed_seconds = (datetime.now() - scan_start).total_seconds()
                    log_progress(
                        "CDS scan progress: "
                        f"records_seen={records_seen} "
                        f"candidates_seen={candidates_seen} "
                        f"target_genes_with_candidate={len(best_by_gene)}/{target_count} "
                        f"elapsed_seconds={elapsed_seconds:.1f}",
                    )

                matches = match_fasta_record_to_targets(
                    header=header,
                    target_keys=target_keys,
                    target_keys_by_gene_id=target_keys_by_gene_id,
                    accessions_by_gene=accessions_by_gene,
                    genes_by_accession=genes_by_accession,
                    require_refseq_accession_match=require_refseq_accession_match,
                )
                if not matches:
                    continue

                previous_matched_gene_count = len(best_by_gene)
                for key, matched_refseq_accessions in matches:
                    candidate = make_candidate(
                        key=key,
                        header=header,
                        sequence=sequence,
                        ambiguity_count=ambiguity_count,
                        source_fasta=fasta_path,
                        matched_refseq_accessions=matched_refseq_accessions,
                    )
                    writer.writerow(candidate_output_row(candidate))
                    candidates_seen += 1

                    current_best = best_by_gene.get(key)
                    if current_best is None or candidate_sort_key(candidate) < candidate_sort_key(current_best):
                        best_by_gene[key] = candidate

                matched_gene_count = len(best_by_gene)
                if matched_gene_count > previous_matched_gene_count and (
                    matched_gene_count >= next_gene_progress or matched_gene_count == target_count
                ):
                    elapsed_seconds = (datetime.now() - scan_start).total_seconds()
                    log_progress(
                        "CDS target-gene progress: "
                        f"target_genes_with_candidate={matched_gene_count}/{target_count} "
                        f"records_seen={records_seen} "
                        f"candidates_seen={candidates_seen} "
                        f"elapsed_seconds={elapsed_seconds:.1f}",
                    )
                    while next_gene_progress <= matched_gene_count:
                        next_gene_progress += PROGRESS_GENE_INTERVAL

            file_elapsed_seconds = (datetime.now() - file_start).total_seconds()
            log_progress(
                "Finished CDS FASTA: "
                f"{fasta_path} records_in_file={file_records_seen} "
                f"total_records_seen={records_seen} "
                f"target_genes_with_candidate={len(best_by_gene)}/{target_count} "
                f"file_elapsed_seconds={file_elapsed_seconds:.1f}",
            )

    scan_elapsed_seconds = (datetime.now() - scan_start).total_seconds()
    log_progress(
        "Finished CDS FASTA scan: "
        f"records_seen={records_seen} "
        f"candidates_seen={candidates_seen} "
        f"target_genes_with_candidate={len(best_by_gene)}/{target_count} "
        f"elapsed_seconds={scan_elapsed_seconds:.1f}",
    )

    return best_by_gene, records_seen, candidates_seen


def write_representatives(path: Path, best_by_gene: dict[tuple[int, int], CdsCandidate]) -> None:
    """Write representative CDS metadata without sequence."""

    with open_write_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "tax_id",
                "GeneID",
                "cds_id",
                "length",
                "ambiguity_count",
                "matched_refseq_accessions",
                "source_fasta",
                "header",
            )
        )

        for key in sorted(best_by_gene):
            writer.writerow(candidate_output_row(best_by_gene[key]))


def wrap_sequence(sequence: str, width: int = 80) -> Iterator[str]:
    """Yield wrapped FASTA sequence lines."""

    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def write_representative_fasta(path: Path, best_by_gene: dict[tuple[int, int], CdsCandidate]) -> None:
    """Write representative CDS sequences as FASTA."""

    with open_write_text(path) as handle:
        for key in sorted(best_by_gene):
            candidate = best_by_gene[key]
            header = (
                f"tax_id={candidate.tax_id}|GeneID={candidate.gene_id}|"
                f"cds_id={candidate.cds_id}|length={candidate.length}"
            )
            handle.write(f">{header}\n")
            for line in wrap_sequence(candidate.sequence):
                handle.write(f"{line}\n")


def write_unmatched_genes(
    path: Path,
    target_keys: list[tuple[int, int]],
    rows_by_gene: dict[tuple[int, int], list[RefSeqRow]],
    best_by_gene: dict[tuple[int, int], CdsCandidate],
    labels_by_gene: dict[tuple[int, int], set[str]],
    gene2refseq_loaded: bool,
) -> None:
    """Write target genes with no representative CDS."""

    with open_write_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("tax_id", "GeneID", "label_count", "gene2refseq_rows", "reason"))

        for key in target_keys:
            if key in best_by_gene:
                continue

            refseq_count = len(rows_by_gene.get(key, []))
            if not gene2refseq_loaded:
                reason = "no_cds_fasta_match"
            else:
                reason = "no_cds_fasta_match" if refseq_count else "no_gene2refseq_rows"
            writer.writerow((key[0], key[1], len(labels_by_gene.get(key, set())), refseq_count, reason))


def write_tax_coverage(
    path: Path,
    target_keys: list[tuple[int, int]],
    rows_by_gene: dict[tuple[int, int], list[RefSeqRow]],
    best_by_gene: dict[tuple[int, int], CdsCandidate],
    gene2refseq_loaded: bool,
) -> None:
    """Write per-tax_id gene2refseq and representative-CDS coverage."""

    target_by_tax: dict[int, int] = {}
    refseq_by_tax: dict[int, int] = {}
    representative_by_tax: dict[int, int] = {}

    for key in target_keys:
        tax_id = key[0]
        target_by_tax[tax_id] = target_by_tax.get(tax_id, 0) + 1

        if rows_by_gene.get(key):
            refseq_by_tax[tax_id] = refseq_by_tax.get(tax_id, 0) + 1

        if key in best_by_gene:
            representative_by_tax[tax_id] = representative_by_tax.get(tax_id, 0) + 1

    with open_write_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "tax_id",
                "target_labeled_genes",
                "genes_with_gene2refseq",
                "gene2refseq_coverage_percent",
                "genes_with_representative_cds",
                "representative_cds_coverage_percent",
            )
        )

        for tax_id in sorted(target_by_tax):
            target_count = target_by_tax[tax_id]
            refseq_count: int | str = refseq_by_tax.get(tax_id, 0) if gene2refseq_loaded else "NA"
            representative_count = representative_by_tax.get(tax_id, 0)
            refseq_percent = f"{(refseq_count / target_count) * 100:.2f}" if gene2refseq_loaded else "NA"
            representative_percent = (representative_count / target_count) * 100
            writer.writerow(
                (
                    tax_id,
                    target_count,
                    refseq_count,
                    refseq_percent,
                    representative_count,
                    f"{representative_percent:.2f}",
                )
            )


def write_summary(path: Path, rows: Iterable[tuple[str, object]]) -> None:
    """Write a small key-value summary table."""

    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        for metric, value in rows:
            writer.writerow((metric, value))


def parse_tax_ids(raw_tax_ids: list[str]) -> set[int] | None:
    """Parse optional repeated/comma-delimited tax ID filters."""

    if not raw_tax_ids:
        return None

    tax_ids: set[int] = set()
    for raw in raw_tax_ids:
        for part in raw.split(","):
            part = part.strip()
            if part:
                tax_ids.add(int(part))

    return tax_ids


def default_cds_globs_for_tax_ids(tax_ids: set[int] | None) -> list[str]:
    """Use downloaded per-taxon CDS directories when a taxon filter is set."""

    if not tax_ids:
        return ["data/**/*_cds_from_genomic.fna.gz", "data/**/*_cds_from_genomic.fna"]

    patterns: list[str] = []
    for tax_id in sorted(tax_ids):
        patterns.append(f"data/refseq_cds/{tax_id}/*_cds_from_genomic.fna.gz")
        patterns.append(f"data/refseq_cds/{tax_id}/*_cds_from_genomic.fna")
    return patterns


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Select representative CDS sequences for labeled GeneIDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("outputs/process_Gene2GO/processed/final/positive_gene_go_labels.tsv.gz"),
        help="Positive GO label table with columns tax_id, GeneID, GO_ID.",
    )
    parser.add_argument(
        "--gene2refseq",
        type=Path,
        default=None,
        help="Optional NCBI gene2refseq.gz mapping table for provenance or strict accession matching.",
    )
    parser.add_argument(
        "--cds-fasta",
        type=Path,
        action="append",
        default=[],
        help="CDS FASTA file to scan. May be repeated.",
    )
    parser.add_argument(
        "--cds-glob",
        action="append",
        default=[],
        help=(
            "Glob pattern for CDS FASTA files, relative to the current directory. "
            "When omitted with --tax-id, defaults to data/refseq_cds/<tax_id>/*_cds_from_genomic.fna*."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/map_gene_cds/pilot_1000"),
        help="Directory for output tables and representative FASTA.",
    )
    parser.add_argument(
        "--limit-genes",
        type=int,
        default=1000,
        help="Number of unique labeled genes to target. Use 0 for no limit.",
    )
    parser.add_argument(
        "--tax-id",
        action="append",
        default=[],
        help="Optional tax ID filter. May be repeated or comma-delimited.",
    )
    parser.add_argument(
        "--require-refseq-accession-match",
        action="store_true",
        help="Require a CDS FASTA header accession to match --gene2refseq for the gene.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mapping workflow."""

    args = build_parser().parse_args(argv)
    run_start = datetime.now()
    log_progress("Starting map_gene_cds workflow")
    tax_ids = parse_tax_ids(args.tax_id)

    if args.limit_genes < 0:
        raise ValueError("--limit-genes must be >= 0")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Label table not found: {args.labels}")
    if args.require_refseq_accession_match and args.gene2refseq is None:
        raise ValueError("--require-refseq-accession-match requires --gene2refseq")
    if args.gene2refseq is not None and not args.gene2refseq.is_file():
        raise FileNotFoundError(f"gene2refseq file not found: {args.gene2refseq}")

    cds_globs = args.cds_glob if args.cds_glob else default_cds_globs_for_tax_ids(tax_ids)
    log_progress("Discovering CDS FASTA files")
    fasta_paths = discover_fasta_paths(args.cds_fasta, cds_globs)
    if not fasta_paths:
        raise FileNotFoundError(
            "No CDS FASTA files found. Provide --cds-fasta or adjust --cds-glob "
            "for local *_cds_from_genomic.fna.gz files."
        )
    log_progress(f"CDS FASTA files selected: {len(fasta_paths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_source, label_tax_ids = labels_for_mapping(args.labels, tax_ids)

    phase_start = datetime.now()
    log_progress("Choosing target labeled genes")
    target_genes = choose_target_genes(
        labels_path=label_source,
        limit_genes=args.limit_genes,
        tax_ids=label_tax_ids,
    )
    if not target_genes:
        raise ValueError("No target labeled genes found with the requested filters")
    log_progress(
        "Target labeled genes selected: "
        f"{len(target_genes)} elapsed_seconds={(datetime.now() - phase_start).total_seconds():.1f}"
    )

    target_keys = set(target_genes)
    target_tax_ids = {tax_id for tax_id, _gene_id in target_keys}
    if len(target_tax_ids) > 1 and not args.require_refseq_accession_match:
        raise ValueError(
            "Permissive GeneID-based CDS matching is only safe for a single target tax_id. "
            "Add --tax-id for a per-species run or use --require-refseq-accession-match."
        )

    target_keys_by_gene_id: dict[int, set[tuple[int, int]]] = {}
    for key in target_keys:
        target_keys_by_gene_id.setdefault(key[1], set()).add(key)

    phase_start = datetime.now()
    log_progress("Writing target GO labels")
    labels_by_gene = write_target_labels(
        labels_path=label_source,
        target_keys=target_keys,
        output_path=args.output_dir / "target_gene_go_labels.tsv.gz",
    )
    log_progress(f"Finished writing target GO labels elapsed_seconds={(datetime.now() - phase_start).total_seconds():.1f}")

    gene2refseq_loaded = args.gene2refseq is not None
    if gene2refseq_loaded:
        phase_start = datetime.now()
        log_progress("Loading target gene2refseq rows")
        rows_by_gene, accessions_by_gene, genes_by_accession = load_gene2refseq(
            path=args.gene2refseq,
            target_keys=target_keys,
        )
        log_progress(f"Finished loading target gene2refseq rows elapsed_seconds={(datetime.now() - phase_start).total_seconds():.1f}")

        phase_start = datetime.now()
        log_progress("Writing target gene2refseq rows")
        write_gene2refseq_matches(
            path=args.output_dir / "target_gene2refseq_rows.tsv.gz",
            rows_by_gene=rows_by_gene,
        )
        log_progress(f"Finished writing target gene2refseq rows elapsed_seconds={(datetime.now() - phase_start).total_seconds():.1f}")
    else:
        log_progress("Skipping gene2refseq loading; matching CDS records by FASTA header GeneID")
        rows_by_gene = {key: [] for key in target_keys}
        accessions_by_gene = {key: set() for key in target_keys}
        genes_by_accession = {}

    best_by_gene, fasta_records_seen, cds_candidates_seen = scan_cds_fastas(
        fasta_paths=fasta_paths,
        target_keys=target_keys,
        target_keys_by_gene_id=target_keys_by_gene_id,
        accessions_by_gene=accessions_by_gene,
        genes_by_accession=genes_by_accession,
        output_path=args.output_dir / "gene_cds_candidates.tsv.gz",
        require_refseq_accession_match=args.require_refseq_accession_match,
    )

    phase_start = datetime.now()
    log_progress("Writing representative and summary outputs")
    write_representatives(
        path=args.output_dir / "representative_gene_cds.tsv.gz",
        best_by_gene=best_by_gene,
    )
    write_representative_fasta(
        path=args.output_dir / "representative_gene_cds.fna.gz",
        best_by_gene=best_by_gene,
    )
    write_unmatched_genes(
        path=args.output_dir / "unmatched_labeled_genes.tsv.gz",
        target_keys=target_genes,
        rows_by_gene=rows_by_gene,
        best_by_gene=best_by_gene,
        labels_by_gene=labels_by_gene,
        gene2refseq_loaded=gene2refseq_loaded,
    )
    write_tax_coverage(
        path=args.output_dir / "tax_gene2refseq_coverage.tsv",
        target_keys=target_genes,
        rows_by_gene=rows_by_gene,
        best_by_gene=best_by_gene,
        gene2refseq_loaded=gene2refseq_loaded,
    )
    log_progress(f"Finished writing representative and summary outputs elapsed_seconds={(datetime.now() - phase_start).total_seconds():.1f}")

    target_genes_with_refseq = sum(1 for key in target_keys if rows_by_gene.get(key))
    write_summary(
        path=args.output_dir / "run_summary.tsv",
        rows=(
            ("labels", args.labels),
            ("label_source_used", label_source),
            ("gene2refseq", args.gene2refseq if args.gene2refseq is not None else "not_loaded"),
            ("cds_fasta_files", len(fasta_paths)),
            ("limit_genes", args.limit_genes),
            ("target_labeled_genes", len(target_genes)),
            (
                "target_genes_with_gene2refseq_rows",
                target_genes_with_refseq if gene2refseq_loaded else "NA",
            ),
            ("cds_fasta_records_seen", fasta_records_seen),
            ("cds_candidates_seen", cds_candidates_seen),
            ("representative_cds_written", len(best_by_gene)),
            ("unmatched_labeled_genes", len(target_genes) - len(best_by_gene)),
            ("require_refseq_accession_match", args.require_refseq_accession_match),
        ),
    )

    total_elapsed_seconds = (datetime.now() - run_start).total_seconds()
    log_progress(f"Wrote outputs to {args.output_dir}")
    log_progress(f"Target labeled genes: {len(target_genes)}")
    log_progress(f"Representative CDS records: {len(best_by_gene)}")
    log_progress(f"Unmatched labeled genes: {len(target_genes) - len(best_by_gene)}")
    log_progress(f"Finished map_gene_cds workflow elapsed_seconds={total_elapsed_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
