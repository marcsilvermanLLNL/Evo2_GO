#!/usr/bin/env python3
"""Create a fixed, deterministic validation panel from a split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import sys
import time
from pathlib import Path
from typing import Sequence


IDENTITY_FIELDS = ("tax_id", "GeneID", "cds_id")
PANEL_EXTRA_FIELDS = ("validation_panel_priority_u64", "validation_panel_seed")
SELECTION_METHOD = "smallest_sha256_u64_by_identity"


def clean_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def identity_from_row(row: dict[str, str]) -> tuple[str, str, str]:
    identity = tuple(clean_field(row.get(field, "")) for field in IDENTITY_FIELDS)
    if any(not value for value in identity):
        raise ValueError(f"Validation-panel row has an incomplete identity: {identity}")
    return identity  # type: ignore[return-value]


def identity_priority(identity: tuple[str, str, str], seed: str) -> int:
    payload = "\t".join((seed, *identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(path: Path, rows: list[tuple[object, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerows(rows)
    tmp_path.replace(path)


def create_validation_panel(
    split_manifest: Path,
    output: Path,
    *,
    split_name: str,
    panel_size: int,
    seed: str,
    progress_every: int = 5_000_000,
) -> dict[str, object]:
    """Select the identities with the smallest seeded hashes using O(panel_size) memory."""

    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")

    # The root of this min-heap is the selected row with the largest priority,
    # because priorities are negated. A lower priority replaces it.
    selected: list[tuple[int, int, tuple[str, ...], int]] = []
    eligible_rows = 0
    rows_seen = 0
    started = time.time()
    with split_manifest.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required = {"split", *IDENTITY_FIELDS}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"{split_manifest} is missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            rows_seen = row_number - 1
            if progress_every > 0 and rows_seen % progress_every == 0:
                print(
                    f"Scanned {rows_seen:,} split rows; "
                    f"eligible {eligible_rows:,}; elapsed {(time.time() - started) / 60:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
            if clean_field(row.get("split", "")) != split_name:
                continue
            identity = identity_from_row(row)
            priority = identity_priority(identity, seed)
            values = tuple(clean_field(row.get(field, "")) for field in fieldnames)
            entry = (-priority, row_number, values, priority)
            eligible_rows += 1
            if len(selected) < panel_size:
                heapq.heappush(selected, entry)
            elif -priority > selected[0][0]:
                heapq.heapreplace(selected, entry)

    if eligible_rows == 0:
        raise ValueError(f"No rows found for split {split_name!r} in {split_manifest}")

    selected.sort(key=lambda entry: entry[1])
    selected_identities: set[tuple[str, str, str]] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    with tmp_path.open("wt", newline="") as handle:
        output_fields = [*fieldnames, *PANEL_EXTRA_FIELDS]
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for _, row_number, values, priority in selected:
            row = dict(zip(fieldnames, values, strict=True))
            identity = identity_from_row(row)
            if identity in selected_identities:
                raise ValueError(
                    f"Duplicate identity selected from source row {row_number}: {identity}"
                )
            selected_identities.add(identity)
            row["validation_panel_priority_u64"] = str(priority)
            row["validation_panel_seed"] = seed
            writer.writerow(row)
    tmp_path.replace(output)

    return {
        "status": "complete",
        "source_split_manifest": split_manifest,
        "source_split": split_name,
        "eligible_records": eligible_rows,
        "source_rows_scanned": rows_seen,
        "requested_panel_records": panel_size,
        "panel_records": len(selected),
        "seed": seed,
        "selection_method": SELECTION_METHOD,
        "output": output,
        "output_size_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--panel-size", type=int, default=50_000)
    parser.add_argument("--seed", default="evo2-go-fixed-validation-panel-v1")
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = args.summary or args.output.with_name(
        f"{args.output.stem}_summary.tsv"
    )
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite validation panel: {args.output}")
    if summary.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite validation-panel summary: {summary}")

    result = create_validation_panel(
        args.split_manifest,
        args.output,
        split_name=args.split_name,
        panel_size=args.panel_size,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    write_summary(summary, [*result.items(), ("summary", summary)])
    for key, value in result.items():
        print(f"{key}\t{value}")
    print(f"summary\t{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
