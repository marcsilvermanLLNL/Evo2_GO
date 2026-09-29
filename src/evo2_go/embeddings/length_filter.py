#!/usr/bin/env python3
"""Split representative CDS records by maximum sequence length.

This stage leaves the upstream representative-CDS mapping outputs untouched.
For each taxon it writes a length-filtered kept FASTA/TSV pair, a dropped
FASTA/TSV pair, and a small per-taxon summary. A driver-level summary aggregates
the same dropped-count and dropped-percent statistics across all taxa.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_ROOT = Path("outputs/map_gene_cds/by_tax")
DEFAULT_OUTPUT_ROOT = Path("outputs/map_gene_cds/length_filtered_max_8192")
DEFAULT_MAX_TOKENS = 8192
SUMMARY_COLUMNS = (
    "tax_id",
    "max_tokens",
    "input_fasta",
    "input_metadata",
    "kept_fasta",
    "dropped_fasta",
    "kept_metadata",
    "dropped_metadata",
    "total_cds",
    "kept_cds",
    "dropped_cds",
    "dropped_cds_percent",
    "total_tokens",
    "kept_tokens",
    "dropped_tokens",
    "dropped_tokens_percent",
    "max_cds_tokens",
    "metadata_rows",
    "kept_metadata_rows",
    "dropped_metadata_rows",
    "fasta_metadata_count_match",
)
STATUS_COLUMNS = (
    "tax_id",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "input_dir",
    "output_dir",
    "total_cds",
    "dropped_cds",
    "dropped_cds_percent",
    "error_log",
    "message",
)
DATASET_SUMMARY_COLUMNS = (
    "max_tokens",
    "requested_taxa",
    "completed_taxa",
    "successful_taxa",
    "skipped_completed_taxa",
    "failed_taxa",
    "total_cds",
    "kept_cds",
    "dropped_cds",
    "dropped_cds_percent",
    "total_tokens",
    "kept_tokens",
    "dropped_tokens",
    "dropped_tokens_percent",
    "elapsed_seconds",
)
MANAGED_OUTPUTS = (
    "kept_representative_gene_cds.fna.gz",
    "dropped_representative_gene_cds.fna.gz",
    "kept_representative_gene_cds.tsv.gz",
    "dropped_representative_gene_cds.tsv.gz",
    "length_filter_summary.tsv",
)


def log_progress(message: str) -> None:
    """Write a timestamped progress message to stderr for Flux logs."""

    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def clean_field(value: object) -> str:
    """Normalize a table field to a stripped string."""

    if value is None:
        return ""
    return str(value).strip()


def parse_tax_ids(raw_tax_ids: list[str]) -> set[int] | None:
    """Parse optional repeated or comma-delimited tax ID filters."""

    if not raw_tax_ids:
        return None

    tax_ids: set[int] = set()
    for raw in raw_tax_ids:
        for part in raw.split(","):
            part = part.strip()
            if part:
                tax_ids.add(int(part))
    return tax_ids


def percent(part: int, whole: int) -> str:
    """Format a percentage with two decimal places."""

    if whole == 0:
        return "0.00"
    return f"{(part / whole) * 100.0:.2f}"


def write_tsv(path: Path, header: Iterable[object], rows: Iterable[Iterable[object]]) -> None:
    """Write a small uncompressed TSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(tuple(header))
        for row in rows:
            writer.writerow(tuple(row))
    tmp_path.replace(path)


def read_summary(path: Path) -> dict[str, str]:
    """Read one per-taxon length_filter_summary.tsv."""

    if not path.is_file():
        return {}

    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            return {key: clean_field(value) for key, value in row.items()}
    return {}


@dataclass(frozen=True)
class TaxonJob:
    """One taxon's length-filtering job."""

    tax_id: int
    input_dir: Path
    output_dir: Path

    @property
    def input_fasta(self) -> Path:
        return self.input_dir / "representative_gene_cds.fna.gz"

    @property
    def input_metadata(self) -> Path:
        return self.input_dir / "representative_gene_cds.tsv.gz"


@dataclass
class TaxonStatus:
    """Driver-level status for one taxon."""

    tax_id: int
    status: str
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: str = ""
    input_dir: str = ""
    output_dir: str = ""
    total_cds: str = ""
    dropped_cds: str = ""
    dropped_cds_percent: str = ""
    error_log: str = ""
    message: str = ""

    def row(self) -> tuple[object, ...]:
        """Return a row matching STATUS_COLUMNS."""

        return (
            self.tax_id,
            self.status,
            self.started_at,
            self.finished_at,
            self.elapsed_seconds,
            self.input_dir,
            self.output_dir,
            self.total_cds,
            self.dropped_cds,
            self.dropped_cds_percent,
            self.error_log,
            self.message,
        )


@dataclass
class TaxonCounts:
    """Counters collected while filtering one taxon."""

    total_cds: int = 0
    kept_cds: int = 0
    dropped_cds: int = 0
    total_tokens: int = 0
    kept_tokens: int = 0
    dropped_tokens: int = 0
    max_cds_tokens: int = 0
    metadata_rows: int = 0
    kept_metadata_rows: int = 0
    dropped_metadata_rows: int = 0


def load_jobs(
    input_root: Path,
    output_root: Path,
    tax_filter: set[int] | None,
    limit_taxa: int,
) -> list[TaxonJob]:
    """Find per-taxon representative-CDS mapping outputs."""

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    jobs: list[TaxonJob] = []
    for input_dir in sorted(input_root.glob("tax_*"), key=lambda path: path.name):
        if not input_dir.is_dir():
            continue

        raw_tax_id = input_dir.name.removeprefix("tax_")
        if not raw_tax_id.isdigit():
            continue
        tax_id = int(raw_tax_id)
        if tax_filter is not None and tax_id not in tax_filter:
            continue

        job = TaxonJob(tax_id=tax_id, input_dir=input_dir, output_dir=output_root / input_dir.name)
        if not job.input_fasta.is_file() or not job.input_metadata.is_file():
            log_progress(f"Skipping tax_id={tax_id}: representative FASTA or TSV is missing")
            continue

        jobs.append(job)
        if limit_taxa > 0 and len(jobs) >= limit_taxa:
            break

    return jobs


def outputs_complete(output_dir: Path) -> bool:
    """Return True when all expected per-taxon outputs exist."""

    for filename in MANAGED_OUTPUTS:
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False
    summary = read_summary(output_dir / "length_filter_summary.tsv")
    return bool(summary)


def remove_managed_outputs(output_dir: Path) -> None:
    """Remove files generated by this stage before a forced rerun."""

    for filename in MANAGED_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def status_from_summary(
    job: TaxonJob,
    status: str,
    started_at: str = "",
    finished_at: str = "",
    elapsed_seconds: str = "",
    message: str = "",
) -> TaxonStatus:
    """Build a driver status row from an existing per-taxon summary."""

    summary = read_summary(job.output_dir / "length_filter_summary.tsv")
    return TaxonStatus(
        tax_id=job.tax_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed_seconds,
        input_dir=str(job.input_dir),
        output_dir=str(job.output_dir),
        total_cds=summary.get("total_cds", ""),
        dropped_cds=summary.get("dropped_cds", ""),
        dropped_cds_percent=summary.get("dropped_cds_percent", ""),
        message=message,
    )


def write_fasta_record(handle, header: str, sequence: str, line_width: int) -> None:
    """Write one FASTA record to an open text handle."""

    handle.write(header)
    handle.write("\n")
    if line_width <= 0:
        handle.write(sequence)
        handle.write("\n")
        return
    for start in range(0, len(sequence), line_width):
        handle.write(sequence[start : start + line_width])
        handle.write("\n")


def filter_fasta(job: TaxonJob, max_tokens: int, line_width: int, counts: TaxonCounts) -> None:
    """Split one representative CDS FASTA into kept and dropped records."""

    kept_path = job.output_dir / "kept_representative_gene_cds.fna.gz"
    dropped_path = job.output_dir / "dropped_representative_gene_cds.fna.gz"

    with gzip.open(job.input_fasta, "rt") as input_handle, gzip.open(
        kept_path, "wt"
    ) as kept_handle, gzip.open(dropped_path, "wt") as dropped_handle:
        header: str | None = None
        sequence_parts: list[str] = []

        def flush_record() -> None:
            if header is None:
                return

            sequence = "".join(sequence_parts).replace(" ", "").replace("\t", "")
            length = len(sequence)
            counts.total_cds += 1
            counts.total_tokens += length
            counts.max_cds_tokens = max(counts.max_cds_tokens, length)

            if length > max_tokens:
                counts.dropped_cds += 1
                counts.dropped_tokens += length
                write_fasta_record(dropped_handle, header, sequence, line_width)
            else:
                counts.kept_cds += 1
                counts.kept_tokens += length
                write_fasta_record(kept_handle, header, sequence, line_width)

        for raw_line in input_handle:
            line = raw_line.rstrip("\n\r")
            if line.startswith(">"):
                flush_record()
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line.strip())
        flush_record()


def filter_metadata(job: TaxonJob, max_tokens: int, counts: TaxonCounts) -> None:
    """Split the representative CDS metadata TSV using its length column."""

    kept_path = job.output_dir / "kept_representative_gene_cds.tsv.gz"
    dropped_path = job.output_dir / "dropped_representative_gene_cds.tsv.gz"

    with gzip.open(job.input_metadata, "rt", newline="") as input_handle, gzip.open(
        kept_path, "wt", newline=""
    ) as kept_handle, gzip.open(dropped_path, "wt", newline="") as dropped_handle:
        reader = csv.DictReader(input_handle, delimiter="\t")
        if not reader.fieldnames or "length" not in reader.fieldnames:
            raise ValueError(f"Expected a length column in {job.input_metadata}")

        kept_writer = csv.DictWriter(
            kept_handle,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        dropped_writer = csv.DictWriter(
            dropped_handle,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        kept_writer.writeheader()
        dropped_writer.writeheader()

        for row in reader:
            length = int(clean_field(row.get("length")))
            counts.metadata_rows += 1
            if length > max_tokens:
                counts.dropped_metadata_rows += 1
                dropped_writer.writerow(row)
            else:
                counts.kept_metadata_rows += 1
                kept_writer.writerow(row)


def write_taxon_summary(job: TaxonJob, max_tokens: int, counts: TaxonCounts) -> None:
    """Write one taxon's length-filter summary."""

    fasta_metadata_count_match = (
        counts.total_cds == counts.metadata_rows
        and counts.kept_cds == counts.kept_metadata_rows
        and counts.dropped_cds == counts.dropped_metadata_rows
    )
    row = (
        job.tax_id,
        max_tokens,
        job.input_fasta,
        job.input_metadata,
        job.output_dir / "kept_representative_gene_cds.fna.gz",
        job.output_dir / "dropped_representative_gene_cds.fna.gz",
        job.output_dir / "kept_representative_gene_cds.tsv.gz",
        job.output_dir / "dropped_representative_gene_cds.tsv.gz",
        counts.total_cds,
        counts.kept_cds,
        counts.dropped_cds,
        percent(counts.dropped_cds, counts.total_cds),
        counts.total_tokens,
        counts.kept_tokens,
        counts.dropped_tokens,
        percent(counts.dropped_tokens, counts.total_tokens),
        counts.max_cds_tokens,
        counts.metadata_rows,
        counts.kept_metadata_rows,
        counts.dropped_metadata_rows,
        fasta_metadata_count_match,
    )
    write_tsv(job.output_dir / "length_filter_summary.tsv", SUMMARY_COLUMNS, (row,))


def run_one_job(job: TaxonJob, max_tokens: int, line_width: int, force: bool) -> TaxonStatus:
    """Filter one taxon and return its status."""

    if outputs_complete(job.output_dir) and not force:
        return status_from_summary(
            job,
            status="skipped_completed",
            message="Existing complete length-filter outputs found",
        )

    if force:
        remove_managed_outputs(job.output_dir)

    job.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    counts = TaxonCounts()
    try:
        filter_fasta(job, max_tokens=max_tokens, line_width=line_width, counts=counts)
        filter_metadata(job, max_tokens=max_tokens, counts=counts)
        write_taxon_summary(job, max_tokens=max_tokens, counts=counts)
    except Exception as exc:
        finished = datetime.now()
        elapsed_seconds = (finished - started).total_seconds()
        return TaxonStatus(
            tax_id=job.tax_id,
            status="failed",
            started_at=started.isoformat(timespec="seconds"),
            finished_at=finished.isoformat(timespec="seconds"),
            elapsed_seconds=f"{elapsed_seconds:.1f}",
            input_dir=str(job.input_dir),
            output_dir=str(job.output_dir),
            message=str(exc),
        )

    finished = datetime.now()
    elapsed_seconds = (finished - started).total_seconds()
    if counts.total_cds != counts.metadata_rows:
        message = "FASTA and metadata record counts differ"
        status = "failed"
    elif counts.kept_cds != counts.kept_metadata_rows or counts.dropped_cds != counts.dropped_metadata_rows:
        message = "FASTA and metadata kept/dropped counts differ"
        status = "failed"
    else:
        message = ""
        status = "success"

    return status_from_summary(
        job,
        status=status,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        elapsed_seconds=f"{elapsed_seconds:.1f}",
        message=message,
    )


def write_status_table(path: Path, statuses: dict[int, TaxonStatus]) -> None:
    """Write the current driver status table."""

    write_tsv(path, STATUS_COLUMNS, (statuses[tax_id].row() for tax_id in sorted(statuses)))


def aggregate_summaries(output_root: Path, statuses: dict[int, TaxonStatus]) -> dict[str, int]:
    """Aggregate per-taxon summaries for successful or skipped taxa."""

    totals = {
        "total_cds": 0,
        "kept_cds": 0,
        "dropped_cds": 0,
        "total_tokens": 0,
        "kept_tokens": 0,
        "dropped_tokens": 0,
    }
    for status in statuses.values():
        if status.status not in {"success", "skipped_completed"}:
            continue
        summary = read_summary(Path(status.output_dir) / "length_filter_summary.tsv")
        for key in totals:
            totals[key] += int(summary.get(key, "0") or "0")
    return totals


def write_dataset_summary(
    path: Path,
    max_tokens: int,
    requested_taxa: int,
    statuses: dict[int, TaxonStatus],
    elapsed_seconds: float,
) -> None:
    """Write aggregate dropped-count and dropped-percent statistics."""

    status_counts: dict[str, int] = {}
    for status in statuses.values():
        status_counts[status.status] = status_counts.get(status.status, 0) + 1

    totals = aggregate_summaries(path.parent, statuses)
    row = (
        max_tokens,
        requested_taxa,
        len(statuses),
        status_counts.get("success", 0),
        status_counts.get("skipped_completed", 0),
        status_counts.get("failed", 0),
        totals["total_cds"],
        totals["kept_cds"],
        totals["dropped_cds"],
        percent(totals["dropped_cds"], totals["total_cds"]),
        totals["total_tokens"],
        totals["kept_tokens"],
        totals["dropped_tokens"],
        percent(totals["dropped_tokens"], totals["total_tokens"]),
        f"{elapsed_seconds:.1f}",
    )
    write_tsv(path, DATASET_SUMMARY_COLUMNS, (row,))


def write_run_summary(
    path: Path,
    args: argparse.Namespace,
    requested_taxa: int,
    statuses: dict[int, TaxonStatus],
    elapsed_seconds: float,
) -> None:
    """Write a small key-value run summary."""

    status_counts: dict[str, int] = {}
    for status in statuses.values():
        status_counts[status.status] = status_counts.get(status.status, 0) + 1

    rows = (
        ("input_root", args.input_root),
        ("output_root", args.output_root),
        ("max_tokens", args.max_tokens),
        ("requested_taxa", requested_taxa),
        ("completed_taxa", len(statuses)),
        ("successful_taxa", status_counts.get("success", 0)),
        ("skipped_completed_taxa", status_counts.get("skipped_completed", 0)),
        ("failed_taxa", status_counts.get("failed", 0)),
        ("dry_run", args.dry_run),
        ("incomplete_only", args.incomplete_only),
        ("max_workers", args.max_workers),
        ("force", args.force),
        ("elapsed_seconds", f"{elapsed_seconds:.1f}"),
    )
    write_tsv(path, ("metric", "value"), rows)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Split representative CDS records into <= max tokens and > max tokens outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing tax_<tax_id>/representative_gene_cds.{fna,tsv}.gz outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for length-filtered per-taxon outputs and aggregate summaries.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum allowed CDS length. Records longer than this are dropped.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of taxa to filter concurrently.",
    )
    parser.add_argument(
        "--tax-id",
        action="append",
        default=[],
        help="Optional tax ID filter. May be repeated or comma-delimited.",
    )
    parser.add_argument(
        "--limit-taxa",
        type=int,
        default=0,
        help="Limit the number of taxa to process. Use 0 for no limit.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=80,
        help="Line width for output FASTA sequence lines. Use 0 for unwrapped sequence lines.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun taxa even when complete length-filtered outputs already exist.",
    )
    parser.add_argument(
        "--incomplete-only",
        action="store_true",
        help="Only process taxa with missing or incomplete length-filter outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned jobs to the status table without filtering files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the all-taxa length filter."""

    run_start = datetime.now()
    args = build_parser().parse_args(argv)
    tax_filter = parse_tax_ids(args.tax_id)

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.limit_taxa < 0:
        raise ValueError("--limit-taxa must be >= 0")
    if args.line_width < 0:
        raise ValueError("--line-width must be >= 0")
    if shutil.which("gzip") is None:
        log_progress("gzip executable not found in PATH; Python gzip support will still be used")

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "filter_status.tsv"
    dataset_summary_path = args.output_root / "dataset_length_filter_summary.tsv"
    run_summary_path = args.output_root / "run_summary.tsv"

    jobs = load_jobs(
        input_root=args.input_root,
        output_root=args.output_root,
        tax_filter=tax_filter,
        limit_taxa=args.limit_taxa,
    )
    if not jobs:
        raise ValueError("No representative-CDS taxa found for the requested filters")

    if args.incomplete_only:
        requested_taxa = len(jobs)
        jobs = [job for job in jobs if not outputs_complete(job.output_dir)]
        log_progress(
            "Filtered to incomplete length-filter jobs: "
            f"incomplete_jobs={len(jobs)} requested_taxa={requested_taxa}"
        )
        if not jobs:
            log_progress("No incomplete length-filter outputs found")
            return 0

    statuses: dict[int, TaxonStatus] = {}
    if args.dry_run:
        for job in jobs:
            statuses[job.tax_id] = TaxonStatus(
                tax_id=job.tax_id,
                status="dry_run",
                input_dir=str(job.input_dir),
                output_dir=str(job.output_dir),
                message=f"Would split CDS records longer than {args.max_tokens} tokens",
            )
        write_status_table(status_path, statuses)
        write_dataset_summary(dataset_summary_path, args.max_tokens, len(jobs), statuses, 0.0)
        write_run_summary(run_summary_path, args, len(jobs), statuses, 0.0)
        log_progress(f"Dry run wrote planned jobs to {status_path}")
        return 0

    log_progress(
        "Starting representative CDS length filter: "
        f"jobs={len(jobs)} max_tokens={args.max_tokens} "
        f"max_workers={args.max_workers} output_root={args.output_root}"
    )

    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        pending_jobs = iter(jobs)
        futures: dict[Future[TaxonStatus], TaxonJob] = {}

        def submit_until_full() -> None:
            while len(futures) < args.max_workers:
                try:
                    job = next(pending_jobs)
                except StopIteration:
                    return
                futures[
                    executor.submit(
                        run_one_job,
                        job,
                        args.max_tokens,
                        args.line_width,
                        args.force,
                    )
                ] = job

        submit_until_full()
        while futures:
            done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = TaxonStatus(
                        tax_id=job.tax_id,
                        status="failed",
                        input_dir=str(job.input_dir),
                        output_dir=str(job.output_dir),
                        message=f"Driver exception: {exc}",
                    )

                statuses[status.tax_id] = status
                completed += 1
                if status.status == "failed":
                    failed += 1

                write_status_table(status_path, statuses)
                elapsed_seconds = (datetime.now() - run_start).total_seconds()
                write_dataset_summary(
                    dataset_summary_path,
                    args.max_tokens,
                    len(jobs),
                    statuses,
                    elapsed_seconds,
                )
                log_progress(
                    "Taxon length filter finished: "
                    f"tax_id={status.tax_id} status={status.status} "
                    f"dropped={status.dropped_cds}/{status.total_cds} "
                    f"completed={completed}/{len(jobs)} failed={failed}"
                )
            submit_until_full()

    elapsed_seconds = (datetime.now() - run_start).total_seconds()
    write_status_table(status_path, statuses)
    write_dataset_summary(dataset_summary_path, args.max_tokens, len(jobs), statuses, elapsed_seconds)
    write_run_summary(run_summary_path, args, len(jobs), statuses, elapsed_seconds)

    log_progress(f"Wrote status table: {status_path}")
    log_progress(f"Wrote dataset summary: {dataset_summary_path}")
    log_progress(f"Wrote run summary: {run_summary_path}")
    log_progress(f"Finished representative CDS length filter elapsed_seconds={elapsed_seconds:.1f} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
