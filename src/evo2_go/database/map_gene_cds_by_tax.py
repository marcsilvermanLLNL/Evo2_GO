#!/usr/bin/env python3
"""Run representative-CDS mapping independently for many taxa.

The underlying mapper intentionally requires one taxon at a time for the fast
GeneID-in-CDS-header match. This driver reads the downloaded RefSeq CDS
manifest, launches one ``map_gene_cds.py --tax-id`` process per taxon with
bounded concurrency, and writes a resumable status table.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_LABELS = Path("outputs/process_Gene2GO/processed/final/positive_gene_go_labels.tsv.gz")
DEFAULT_DOWNLOAD_MANIFEST = Path("data/refseq_cds/downloaded_cds.tsv")
DEFAULT_OUTPUT_ROOT = Path("outputs/map_gene_cds/by_tax")
REQUIRED_MAPPER_OUTPUTS = (
    "target_gene_go_labels.tsv.gz",
    "gene_cds_candidates.tsv.gz",
    "representative_gene_cds.tsv.gz",
    "representative_gene_cds.fna.gz",
    "unmatched_labeled_genes.tsv.gz",
    "tax_gene2refseq_coverage.tsv",
    "run_summary.tsv",
)
MANAGED_OUTPUTS = REQUIRED_MAPPER_OUTPUTS + ("target_gene2refseq_rows.tsv.gz",)
STATUS_COLUMNS = (
    "tax_id",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "output_dir",
    "representative_cds_written",
    "unmatched_labeled_genes",
    "representative_cds_coverage_percent",
    "error_log",
    "message",
)


def log_progress(message: str) -> None:
    """Write a timestamped progress message to stderr for batch logs."""

    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def clean_field(value: object) -> str:
    """Normalize one table field."""

    if value is None:
        return ""
    return str(value).strip()


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


def read_key_value_tsv(path: Path) -> dict[str, str]:
    """Read a two-column metric/value TSV file."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values

    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("metric", "value"):
            return values
        for row in reader:
            metric = clean_field(row.get("metric"))
            if metric:
                values[metric] = clean_field(row.get("value"))
    return values


def read_coverage_percent(path: Path, tax_id: int) -> str:
    """Read representative-CDS coverage for one taxon."""

    if not path.is_file():
        return ""

    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if clean_field(row.get("tax_id")) == str(tax_id):
                return clean_field(row.get("representative_cds_coverage_percent"))
    return ""


@dataclass(frozen=True)
class TaxonJob:
    """One taxon's mapping job."""

    tax_id: int
    output_dir: Path
    cds_path: Path


@dataclass
class TaxonStatus:
    """Driver-level status for one taxon."""

    tax_id: int
    status: str
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: str = ""
    output_dir: str = ""
    representative_cds_written: str = ""
    unmatched_labeled_genes: str = ""
    representative_cds_coverage_percent: str = ""
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
            self.output_dir,
            self.representative_cds_written,
            self.unmatched_labeled_genes,
            self.representative_cds_coverage_percent,
            self.error_log,
            self.message,
        )


def load_jobs(
    download_manifest: Path,
    output_root: Path,
    tax_filter: set[int] | None,
    limit_taxa: int,
) -> list[TaxonJob]:
    """Read downloaded_cds.tsv and return taxa with present CDS files."""

    if not download_manifest.is_file():
        raise FileNotFoundError(f"Download manifest not found: {download_manifest}")

    jobs: list[TaxonJob] = []
    seen_tax_ids: set[int] = set()
    with download_manifest.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"requested_tax_id", "local_path", "status"}
        observed = set(reader.fieldnames or ())
        if not required <= observed:
            raise ValueError(
                f"Unexpected download manifest header in {download_manifest}: "
                f"required columns {sorted(required)!r}, observed {reader.fieldnames!r}"
            )

        for row in reader:
            if clean_field(row.get("status")) != "present":
                continue

            tax_id = int(clean_field(row.get("requested_tax_id")))
            if tax_filter is not None and tax_id not in tax_filter:
                continue
            if tax_id in seen_tax_ids:
                continue

            cds_path = Path(clean_field(row.get("local_path")))
            if not cds_path.is_file():
                log_progress(f"Skipping tax_id={tax_id}: manifest CDS path is missing: {cds_path}")
                continue

            seen_tax_ids.add(tax_id)
            jobs.append(
                TaxonJob(
                    tax_id=tax_id,
                    output_dir=output_root / f"tax_{tax_id}",
                    cds_path=cds_path,
                )
            )

            if limit_taxa > 0 and len(jobs) >= limit_taxa:
                break

    return jobs


def mapper_outputs_complete(output_dir: Path) -> bool:
    """Return True when the expected mapper outputs are present and non-empty."""

    for filename in REQUIRED_MAPPER_OUTPUTS:
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def status_from_outputs(tax_id: int, output_dir: Path, status: str, message: str = "") -> TaxonStatus:
    """Build a driver status row from existing mapper outputs."""

    summary = read_key_value_tsv(output_dir / "run_summary.tsv")
    coverage_percent = read_coverage_percent(output_dir / "tax_gene2refseq_coverage.tsv", tax_id)
    return TaxonStatus(
        tax_id=tax_id,
        status=status,
        output_dir=str(output_dir),
        representative_cds_written=summary.get("representative_cds_written", ""),
        unmatched_labeled_genes=summary.get("unmatched_labeled_genes", ""),
        representative_cds_coverage_percent=coverage_percent,
        message=message,
    )


def remove_managed_outputs(output_dir: Path) -> None:
    """Remove files created by map_gene_cds.py before a forced rerun."""

    for filename in MANAGED_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def write_status_table(path: Path, statuses: dict[int, TaxonStatus]) -> None:
    """Write the current driver status table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(STATUS_COLUMNS)
        for tax_id in sorted(statuses):
            writer.writerow(statuses[tax_id].row())
    tmp_path.replace(path)


def write_run_summary(path: Path, rows: Iterable[tuple[str, object]]) -> None:
    """Write a small key-value summary table for the driver run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        for metric, value in rows:
            writer.writerow((metric, value))


def run_one_job(
    job: TaxonJob,
    labels: Path,
    mapper_script: Path | None,
    python_executable: str,
    logs_dir: Path,
    force: bool,
    extra_mapper_args: list[str],
) -> TaxonStatus:
    """Run map_gene_cds.py for one taxon and return its status."""

    if mapper_outputs_complete(job.output_dir) and not force:
        return status_from_outputs(
            tax_id=job.tax_id,
            output_dir=job.output_dir,
            status="skipped_completed",
            message="Existing complete mapper outputs found",
        )

    if force:
        remove_managed_outputs(job.output_dir)

    job.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"tax_{job.tax_id}.out"
    stderr_path = logs_dir / f"tax_{job.tax_id}.err"
    started = datetime.now()

    if mapper_script is None:
        cmd = [
            python_executable,
            "-m",
            "evo2_go.database.map_gene_cds",
        ]
    else:
        cmd = [
            python_executable,
            str(mapper_script),
        ]

    cmd.extend(
        [
            "--labels",
            str(labels),
            "--tax-id",
            str(job.tax_id),
            "--limit-genes",
            "0",
            "--output-dir",
            str(job.output_dir),
            "--cds-fasta",
            str(job.cds_path),
            "--cds-glob",
            "__NO_DEFAULT_CDS_GLOB__",
        ]
    )
    cmd.extend(extra_mapper_args)

    with stdout_path.open("wt") as stdout_handle, stderr_path.open("wt") as stderr_handle:
        result = subprocess.run(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )

    finished = datetime.now()
    elapsed_seconds = (finished - started).total_seconds()
    if result.returncode != 0:
        return TaxonStatus(
            tax_id=job.tax_id,
            status="failed",
            started_at=started.isoformat(timespec="seconds"),
            finished_at=finished.isoformat(timespec="seconds"),
            elapsed_seconds=f"{elapsed_seconds:.1f}",
            output_dir=str(job.output_dir),
            error_log=str(stderr_path),
            message=f"map_gene_cds.py exited with status {result.returncode}",
        )

    if not mapper_outputs_complete(job.output_dir):
        return TaxonStatus(
            tax_id=job.tax_id,
            status="failed",
            started_at=started.isoformat(timespec="seconds"),
            finished_at=finished.isoformat(timespec="seconds"),
            elapsed_seconds=f"{elapsed_seconds:.1f}",
            output_dir=str(job.output_dir),
            error_log=str(stderr_path),
            message="Mapper exited successfully but expected outputs are missing",
        )

    status = status_from_outputs(tax_id=job.tax_id, output_dir=job.output_dir, status="success")
    status.started_at = started.isoformat(timespec="seconds")
    status.finished_at = finished.isoformat(timespec="seconds")
    status.elapsed_seconds = f"{elapsed_seconds:.1f}"
    status.error_log = str(stderr_path)
    return status


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run map_gene_cds.py independently for all downloaded RefSeq CDS taxa.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Global positive GO label table. Per-taxon caches next to this file will be reused by map_gene_cds.py.",
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=DEFAULT_DOWNLOAD_MANIFEST,
        help="downloaded_cds.tsv produced by download_refseq_cds_for_taxa.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for per-taxon mapper outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of taxa to map concurrently.",
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
        help="Limit the number of manifest taxa to run. Use 0 for no limit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun taxa even when complete outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned jobs to the status table without launching mapper processes.",
    )
    parser.add_argument(
        "--mapper-script",
        type=Path,
        default=None,
        help=(
            "Optional path to a legacy map_gene_cds.py script. "
            "By default the packaged evo2_go.database.map_gene_cds module is used."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch mapper subprocesses.",
    )
    parser.add_argument(
        "mapper_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to each map_gene_cds.py invocation after a literal -- separator.",
    )
    return parser


def normalize_extra_mapper_args(args: list[str]) -> list[str]:
    """Drop argparse's optional REMAINDER separator."""

    if args and args[0] == "--":
        return args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the all-taxa mapping driver."""

    run_start = datetime.now()
    args = build_parser().parse_args(argv)
    tax_filter = parse_tax_ids(args.tax_id)
    extra_mapper_args = normalize_extra_mapper_args(args.mapper_args)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.limit_taxa < 0:
        raise ValueError("--limit-taxa must be >= 0")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Label table not found: {args.labels}")
    if args.mapper_script is not None and not args.mapper_script.is_file():
        raise FileNotFoundError(f"Mapper script not found: {args.mapper_script}")
    if shutil.which(args.python_executable) is None and not Path(args.python_executable).is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python_executable}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "mapping_status.tsv"
    summary_path = args.output_root / "run_summary.tsv"
    logs_dir = args.output_root / "logs"

    jobs = load_jobs(
        download_manifest=args.download_manifest,
        output_root=args.output_root,
        tax_filter=tax_filter,
        limit_taxa=args.limit_taxa,
    )
    if not jobs:
        raise ValueError("No present CDS taxa found for the requested filters")

    statuses: dict[int, TaxonStatus] = {}
    if args.dry_run:
        for job in jobs:
            statuses[job.tax_id] = TaxonStatus(
                tax_id=job.tax_id,
                status="dry_run",
                output_dir=str(job.output_dir),
                message=f"Would map CDS FASTA {job.cds_path}",
            )
        write_status_table(status_path, statuses)
        write_run_summary(
            summary_path,
            (
                ("download_manifest", args.download_manifest),
                ("labels", args.labels),
                ("output_root", args.output_root),
                ("requested_taxa", len(jobs)),
                ("dry_run", True),
                ("max_workers", args.max_workers),
                ("elapsed_seconds", "0.0"),
            ),
        )
        log_progress(f"Dry run wrote planned jobs to {status_path}")
        return 0

    log_progress(
        "Starting all-taxa CDS mapping: "
        f"jobs={len(jobs)} max_workers={args.max_workers} output_root={args.output_root}"
    )

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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
                        args.labels,
                        args.mapper_script,
                        args.python_executable,
                        logs_dir,
                        args.force,
                        extra_mapper_args,
                    )
                ] = job

        submit_until_full()
        while futures:
            done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    status = future.result()
                except Exception as exc:  # Defensive: keep the driver status table useful.
                    finished = datetime.now().isoformat(timespec="seconds")
                    status = TaxonStatus(
                        tax_id=job.tax_id,
                        status="failed",
                        finished_at=finished,
                        output_dir=str(job.output_dir),
                        error_log=str(logs_dir / f"tax_{job.tax_id}.err"),
                        message=f"Driver exception: {exc}",
                    )

                statuses[status.tax_id] = status
                completed += 1
                if status.status == "failed":
                    failed += 1

                write_status_table(status_path, statuses)
                log_progress(
                    "Taxon mapping finished: "
                    f"tax_id={status.tax_id} status={status.status} "
                    f"completed={completed}/{len(jobs)} failed={failed}"
                )
            submit_until_full()

    elapsed_seconds = (datetime.now() - run_start).total_seconds()
    status_counts: dict[str, int] = {}
    for status in statuses.values():
        status_counts[status.status] = status_counts.get(status.status, 0) + 1

    write_status_table(status_path, statuses)
    write_run_summary(
        summary_path,
        (
            ("download_manifest", args.download_manifest),
            ("labels", args.labels),
            ("output_root", args.output_root),
            ("requested_taxa", len(jobs)),
            ("completed_taxa", len(statuses)),
            ("successful_taxa", status_counts.get("success", 0)),
            ("skipped_completed_taxa", status_counts.get("skipped_completed", 0)),
            ("failed_taxa", status_counts.get("failed", 0)),
            ("dry_run", False),
            ("max_workers", args.max_workers),
            ("force", args.force),
            ("elapsed_seconds", f"{elapsed_seconds:.1f}"),
        ),
    )

    log_progress(f"Wrote status table: {status_path}")
    log_progress(f"Wrote run summary: {summary_path}")
    log_progress(f"Finished all-taxa CDS mapping elapsed_seconds={elapsed_seconds:.1f} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
