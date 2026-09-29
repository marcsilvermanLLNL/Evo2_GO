"""Dataset utilities for Evo2 embedding to GO term prediction.

The current training input is a set of precomputed Evo2 embedding shards.  The
storage dtype can be FP16 or a later production dtype, but the model-facing
contract is always float32 embeddings with dense targets aligned to a
deterministic GO vocabulary.
"""

from __future__ import annotations

import csv
import gzip
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Fallback so metadata-only helpers do not require torch."""

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)

        @classmethod
        def __class_getitem__(cls, item: object) -> type["Dataset"]:
            return cls


DEFAULT_EMBEDDING_ROOT = Path(
    "outputs/evo2_embeddings/length_filtered_max_8192_non_iea/"
    "evo2_7b_base_32k_mean_blocks28"
)
DEFAULT_GO_VOCABULARY = Path(
    "outputs/model_vocab/go_adjacency_non_iea_min_count_10/go_vocabulary.tsv"
)
DEFAULT_ANNOTATIONS = Path(
    "outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz"
)


@dataclass(frozen=True)
class GoVocabulary:
    """Deterministic GO vocabulary aligned to the adjacency matrix."""

    go_ids: tuple[str, ...]
    go_to_index: dict[str, int]

    @property
    def n_terms(self) -> int:
        return len(self.go_ids)


@dataclass(frozen=True)
class EmbeddingRecord:
    """One embedding metadata row and its stable training identity."""

    embedding_id: str
    tax_id: str
    gene_id: str
    cds_id: str
    sequence_id: str
    sequence_length: int
    token_count: int
    shard_path: Path
    shard_row: int
    layer_name: str
    pooled_dtype: str

    @property
    def gene_key(self) -> tuple[str, str]:
        return (self.tax_id, self.gene_id)

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.tax_id, self.gene_id, self.cds_id)


@dataclass(frozen=True)
class Evo2GoDatasetBundle:
    """Convenience return value for building the dataset and its vocabulary."""

    dataset: "Evo2GoEmbeddingDataset"
    vocabulary: GoVocabulary
    records_before_label_filter: int
    label_index_size: int


SplitKeyMode = Literal["identity", "embedding_id"]


def log_progress(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def clean_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def open_text(path: Path) -> Iterator[object]:
    """Open plain or gzip-compressed text and close it at context exit."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("rt", newline="")


def parse_int_field(row: dict[str, str], field: str, source: Path) -> int:
    value = clean_field(row.get(field, ""))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer in {source} column {field}: {value!r}") from exc


def parse_tax_ids(raw_tax_ids: Sequence[str] | None) -> set[str] | None:
    if not raw_tax_ids:
        return None
    tax_ids: set[str] = set()
    for raw in raw_tax_ids:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                tax_ids.add(part)
    return tax_ids


def tax_ids_from_file(path: str | Path | None) -> list[str]:
    """Read a taxon filter file, preserving order and dropping duplicates."""

    if path in (None, ""):
        return []

    tax_ids: list[str] = []
    seen: set[str] = set()
    with open_text(Path(path)) as handle:
        for line in handle:
            stripped = str(line).split("#", maxsplit=1)[0].strip()
            if not stripped:
                continue
            token = stripped.replace(",", " ").split()[0]
            if token.lower() in {"tax_id", "taxid"}:
                continue
            if token not in seen:
                tax_ids.append(token)
                seen.add(token)
    return tax_ids


def cds_id_from_sequence_id(sequence_id: str, fallback: str) -> str:
    """Recover the full CDS id from the embedding FASTA header.

    Existing embedding sidecars may contain a truncated ``cds_id`` column
    because FASTA headers use pipe separators inside the CDS id itself, e.g.::

        tax_id=8355|GeneID=...|cds_id=lcl|NC_...|length=1116

    The stable identity should use the full value between ``cds_id=`` and the
    following ``|length=`` marker.
    """

    marker = "|cds_id="
    start = sequence_id.find(marker)
    if start < 0 and sequence_id.startswith("cds_id="):
        start = -1
        marker = "cds_id="
    if start < 0:
        return fallback

    value_start = start + len(marker)
    length_marker = "|length="
    end = sequence_id.find(length_marker, value_start)
    if end < 0:
        return fallback
    cds_id = sequence_id[value_start:end].strip()
    return cds_id or fallback


def load_go_vocabulary(path: str | Path) -> GoVocabulary:
    """Load a GO vocabulary table with contiguous deterministic indices."""

    vocab_path = Path(path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"GO vocabulary not found: {vocab_path}")

    with vocab_path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if "GO_ID" not in fieldnames:
            raise ValueError(f"{vocab_path} is missing required column GO_ID")
        has_go_index = "go_index" in fieldnames

        indexed_terms: dict[int, str] = {}
        ordered_terms: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            go_id = clean_field(row["GO_ID"])
            if not go_id:
                raise ValueError(f"Empty GO_ID in {vocab_path} row {row_number}")
            if has_go_index:
                go_index = parse_int_field(row, "go_index", vocab_path)
                if go_index in indexed_terms:
                    raise ValueError(f"Duplicate go_index {go_index} in {vocab_path}")
                indexed_terms[go_index] = go_id
            else:
                ordered_terms.append(go_id)

    if indexed_terms:
        expected_indices = set(range(len(indexed_terms)))
        observed_indices = set(indexed_terms)
        if observed_indices != expected_indices:
            missing = sorted(expected_indices.difference(observed_indices))[:10]
            extra = sorted(observed_indices.difference(expected_indices))[:10]
            raise ValueError(
                f"GO vocabulary indices are not contiguous from 0 in {vocab_path}; "
                f"missing={missing}, extra={extra}"
            )
        go_ids = tuple(indexed_terms[idx] for idx in range(len(indexed_terms)))
    else:
        go_ids = tuple(ordered_terms)

    if not go_ids:
        raise ValueError(f"No GO terms found in {vocab_path}")
    if len(go_ids) != len(set(go_ids)):
        raise ValueError(f"Duplicate GO_ID values found in {vocab_path}")

    return GoVocabulary(
        go_ids=go_ids,
        go_to_index={go_id: idx for idx, go_id in enumerate(go_ids)},
    )


def complete_metadata_paths_from_manifest(
    manifest_path: str | Path,
    *,
    tax_ids: set[str] | None = None,
    require_status: str = "complete",
) -> list[Path]:
    """Return per-taxon embedding metadata files from a dataset manifest."""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding dataset manifest not found: {path}")

    metadata_paths: list[Path] = []
    with path.open("rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"tax_id", "status", "embedding_metadata_path"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            tax_id = clean_field(row["tax_id"])
            if tax_ids is not None and tax_id not in tax_ids:
                continue
            if require_status and clean_field(row["status"]) != require_status:
                continue
            metadata_path = Path(clean_field(row["embedding_metadata_path"]))
            if not metadata_path.is_absolute():
                metadata_path = (path.parent / metadata_path).resolve()
            metadata_paths.append(metadata_path)

    if not metadata_paths:
        raise ValueError(f"No metadata paths selected from {path}")
    return metadata_paths


def load_embedding_records(
    metadata_paths: Iterable[str | Path],
    *,
    max_records: int | None = None,
    expected_embedding_dim: int = 4096,
    include_split_keys: set[str] | None = None,
    include_key_mode: SplitKeyMode = "identity",
) -> list[EmbeddingRecord]:
    """Load embedding metadata rows without opening shard tensors.

    ``include_split_keys`` permits a small saved panel to be applied while the
    metadata is streamed, avoiding materialization of every embedding record.
    """

    records: list[EmbeddingRecord] = []
    seen_identities: set[tuple[str, str, str]] = set()
    required_columns = {
        "embedding_id",
        "tax_id",
        "GeneID",
        "cds_id",
        "sequence_id",
        "sequence_length",
        "token_count",
        "shard_path",
        "shard_row",
        "layer_name",
        "pooled_shape",
        "pooled_dtype",
    }

    for raw_path in metadata_paths:
        metadata_path = Path(raw_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Embedding metadata not found: {metadata_path}")

        with open_text(metadata_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = required_columns.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{metadata_path} is missing required columns: {sorted(missing)}"
                )

            for row_number, row in enumerate(reader, start=2):
                sequence_id = clean_field(row["sequence_id"])
                fallback_cds_id = clean_field(row["cds_id"])
                cds_id = cds_id_from_sequence_id(sequence_id, fallback=fallback_cds_id)
                identity = (
                    clean_field(row["tax_id"]),
                    clean_field(row["GeneID"]),
                    cds_id,
                )
                embedding_id = clean_field(row["embedding_id"])
                if include_split_keys is not None:
                    if include_key_mode == "identity":
                        selection_key = "\t".join(identity)
                    elif include_key_mode == "embedding_id":
                        selection_key = embedding_id
                    else:
                        raise ValueError(
                            f"Unsupported split key mode: {include_key_mode}"
                        )
                    if selection_key not in include_split_keys:
                        continue
                record = EmbeddingRecord(
                    embedding_id=embedding_id,
                    tax_id=identity[0],
                    gene_id=identity[1],
                    cds_id=cds_id,
                    sequence_id=sequence_id,
                    sequence_length=parse_int_field(row, "sequence_length", metadata_path),
                    token_count=parse_int_field(row, "token_count", metadata_path),
                    shard_path=Path(clean_field(row["shard_path"])),
                    shard_row=parse_int_field(row, "shard_row", metadata_path),
                    layer_name=clean_field(row["layer_name"]),
                    pooled_dtype=clean_field(row["pooled_dtype"]),
                )
                if not record.tax_id or not record.gene_id or not record.cds_id:
                    raise ValueError(
                        f"Missing training identity field in {metadata_path} row {row_number}"
                    )
                if record.sequence_length != record.token_count:
                    raise ValueError(
                        f"sequence_length != token_count in {metadata_path} row {row_number}: "
                        f"{record.sequence_length} != {record.token_count}"
                    )

                shape = clean_field(row["pooled_shape"])
                if str(expected_embedding_dim) not in shape:
                    raise ValueError(
                        f"Unexpected pooled_shape in {metadata_path} row {row_number}: {shape}"
                    )
                if record.identity in seen_identities:
                    raise ValueError(f"Duplicate training identity found: {record.identity}")
                seen_identities.add(record.identity)
                records.append(record)

                if max_records is not None and len(records) >= max_records:
                    return records

    if not records:
        raise ValueError("No embedding records loaded")
    return records


def split_key_for_record(record: EmbeddingRecord, mode: SplitKeyMode = "identity") -> str:
    """Return the stable key used to join a record to a split manifest."""

    if mode == "identity":
        return "\t".join(record.identity)
    if mode == "embedding_id":
        return record.embedding_id
    raise ValueError(f"Unsupported split key mode: {mode}")


def load_split_manifest(
    path: str | Path,
    *,
    split_name: str | None = None,
    key_mode: SplitKeyMode = "identity",
) -> dict[str, str]:
    """Load split assignments from a manifest.

    The expected manifest columns for ``key_mode='identity'`` are
    ``tax_id``, ``GeneID``, ``cds_id``, and ``split``.  For
    ``key_mode='embedding_id'`` the expected columns are ``embedding_id`` and
    ``split``.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    assignments: dict[str, str] = {}
    with open_text(manifest_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"split"}
        if key_mode == "identity":
            required.update({"tax_id", "GeneID", "cds_id"})
        elif key_mode == "embedding_id":
            required.add("embedding_id")
        else:
            raise ValueError(f"Unsupported split key mode: {key_mode}")

        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            split = clean_field(row["split"])
            if split_name is not None and split != split_name:
                continue
            if key_mode == "identity":
                key = "\t".join(
                    (
                        clean_field(row["tax_id"]),
                        clean_field(row["GeneID"]),
                        clean_field(row["cds_id"]),
                    )
                )
            else:
                key = clean_field(row["embedding_id"])
            if not key or not split:
                raise ValueError(f"Empty split key or split in {manifest_path} row {row_number}")
            if key in assignments and assignments[key] != split:
                raise ValueError(f"Conflicting split assignments for {key!r}")
            assignments[key] = split

    if not assignments:
        detail = f" for split {split_name!r}" if split_name is not None else ""
        raise ValueError(f"No split assignments loaded from {manifest_path}{detail}")
    return assignments


def filter_records_by_split(
    records: Sequence[EmbeddingRecord],
    split_assignments: dict[str, str],
    *,
    split_name: str,
    key_mode: SplitKeyMode = "identity",
    require_all_records_assigned: bool = True,
) -> list[EmbeddingRecord]:
    """Filter embedding records to one split."""

    filtered: list[EmbeddingRecord] = []
    missing_assignments = 0
    for record in records:
        key = split_key_for_record(record, key_mode)
        assigned_split = split_assignments.get(key)
        if assigned_split is None:
            missing_assignments += 1
            continue
        if assigned_split == split_name:
            filtered.append(record)

    if require_all_records_assigned and missing_assignments:
        raise ValueError(
            f"{missing_assignments:,} embedding records are missing split assignments"
        )
    if not filtered:
        raise ValueError(f"No embedding records selected for split {split_name!r}")
    return filtered


def load_label_index_cache(
    path: str | Path,
    *,
    n_terms: int | None = None,
    selected_gene_keys: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Load cached ``(tax_id, GeneID) -> GO index`` labels."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Label index cache not found: {cache_path}")

    label_index: dict[tuple[str, str], tuple[int, ...]] = {}
    with open_text(cache_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"tax_id", "GeneID", "go_indices"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{cache_path} is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            tax_id = clean_field(row["tax_id"])
            gene_id = clean_field(row["GeneID"])
            gene_key = (tax_id, gene_id)
            if selected_gene_keys is not None and gene_key not in selected_gene_keys:
                continue
            raw_indices = clean_field(row["go_indices"])
            indices = tuple(
                sorted({int(part) for part in raw_indices.split(",") if part.strip()})
            )
            if n_terms is not None:
                bad = [idx for idx in indices if idx < 0 or idx >= n_terms]
                if bad:
                    raise ValueError(
                        f"Out-of-range GO indices in {cache_path} row {row_number}: {bad[:10]}"
                    )
            label_index[gene_key] = indices

    return label_index


def write_label_index_cache(
    path: str | Path,
    label_index: dict[tuple[str, str], tuple[int, ...]],
    vocabulary: GoVocabulary,
) -> None:
    """Write a compact label-index cache for future dataset construction."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    opener = gzip.open if cache_path.suffix == ".gz" else open

    with opener(tmp_path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("tax_id", "GeneID", "go_indices", "go_ids"))
        for tax_id, gene_id in sorted(label_index, key=lambda item: (int(item[0]), int(item[1]))):
            indices = label_index[(tax_id, gene_id)]
            go_ids = [vocabulary.go_ids[idx] for idx in indices]
            writer.writerow(
                (
                    tax_id,
                    gene_id,
                    ",".join(str(idx) for idx in indices),
                    ",".join(go_ids),
                )
            )
    tmp_path.replace(cache_path)


def build_label_index_from_annotations(
    annotations_path: str | Path,
    vocabulary: GoVocabulary,
    *,
    selected_gene_keys: set[tuple[str, str]] | None = None,
    excluded_evidence: set[str] | None = None,
    progress_every: int = 10_000_000,
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Stream the evidence-bearing annotation table into GO index labels."""

    annotations = Path(annotations_path)
    if not annotations.is_file():
        raise FileNotFoundError(f"Annotation table not found: {annotations}")
    excluded = (
        {"IEA"}
        if excluded_evidence is None
        else {value.upper() for value in excluded_evidence}
    )

    started = time.time()
    labels: dict[tuple[str, str], set[int]] = {}
    rows_seen = 0
    retained_pairs = 0
    with gzip.open(annotations, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"tax_id", "GeneID", "GO_ID", "Evidence"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{annotations} is missing required columns: {sorted(missing)}")

        for rows_seen, row in enumerate(reader, start=1):
            if progress_every > 0 and rows_seen % progress_every == 0:
                elapsed = time.time() - started
                log_progress(
                    f"Scanned {rows_seen:,} annotation rows; "
                    f"retained {retained_pairs:,} selected gene-GO labels; "
                    f"elapsed {elapsed / 60:.1f} min"
                )

            evidence = clean_field(row["Evidence"]).upper()
            if evidence in excluded:
                continue
            go_index = vocabulary.go_to_index.get(clean_field(row["GO_ID"]))
            if go_index is None:
                continue

            key = (clean_field(row["tax_id"]), clean_field(row["GeneID"]))
            if selected_gene_keys is not None and key not in selected_gene_keys:
                continue
            before = len(labels.setdefault(key, set()))
            labels[key].add(go_index)
            if len(labels[key]) != before:
                retained_pairs += 1

    log_progress(
        f"Built labels for {len(labels):,} genes from {rows_seen:,} annotation rows; "
        f"retained {retained_pairs:,} vocabulary-aligned gene-GO labels"
    )
    return {key: tuple(sorted(indices)) for key, indices in labels.items()}


class Evo2GoEmbeddingDataset(Dataset[dict[str, object]]):
    """PyTorch dataset over precomputed Evo2 embedding shards."""

    def __init__(
        self,
        records: Sequence[EmbeddingRecord],
        label_index: dict[tuple[str, str], tuple[int, ...]],
        *,
        n_terms: int,
        embedding_dim: int = 4096,
        max_cached_shards: int = 2,
        drop_examples_without_labels: bool = True,
        validate_embeddings: bool = False,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError(
                "torch is required to instantiate Evo2GoEmbeddingDataset. "
                "Install the training extra or run metadata-only commands."
            )
        if n_terms <= 0:
            raise ValueError("n_terms must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if max_cached_shards < 0:
            raise ValueError("max_cached_shards must be non-negative")

        filtered_records = list(records)
        if drop_examples_without_labels:
            filtered_records = [
                record for record in filtered_records if label_index.get(record.gene_key)
            ]
        if not filtered_records:
            raise ValueError("No embedding records remain after label filtering")

        self.records = tuple(filtered_records)
        self.label_index = label_index
        self.n_terms = n_terms
        self.embedding_dim = embedding_dim
        self.max_cached_shards = max_cached_shards
        self.drop_examples_without_labels = drop_examples_without_labels
        self.validate_embeddings = validate_embeddings
        self._shard_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state

    def _load_shard_tensor(self, shard_path: Path) -> torch.Tensor:
        if self.max_cached_shards > 0 and shard_path in self._shard_cache:
            tensor = self._shard_cache.pop(shard_path)
            self._shard_cache[shard_path] = tensor
            return tensor

        if not shard_path.is_file():
            raise FileNotFoundError(f"Embedding shard not found: {shard_path}")
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(shard_path, map_location="cpu")
        if "pooled_embeddings" not in payload:
            raise ValueError(f"Shard is missing pooled_embeddings: {shard_path}")
        tensor = payload["pooled_embeddings"]
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.ndim != 2 or tensor.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Unexpected pooled_embeddings shape in {shard_path}: {tuple(tensor.shape)}"
            )

        if self.max_cached_shards > 0:
            self._shard_cache[shard_path] = tensor
            while len(self._shard_cache) > self.max_cached_shards:
                self._shard_cache.popitem(last=False)
        return tensor

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        shard_tensor = self._load_shard_tensor(record.shard_path)
        if record.shard_row < 0 or record.shard_row >= shard_tensor.shape[0]:
            raise IndexError(
                f"shard_row {record.shard_row} is out of bounds for {record.shard_path} "
                f"with {shard_tensor.shape[0]} rows"
            )

        embedding = shard_tensor[record.shard_row].detach().to(dtype=torch.float32).clone()
        if self.validate_embeddings and not torch.isfinite(embedding).all():
            raise ValueError(f"Non-finite embedding values for {record.identity}")

        label_indices = self.label_index.get(record.gene_key, ())
        if not label_indices and self.drop_examples_without_labels:
            raise ValueError(f"No labels available for {record.gene_key}")

        return {
            "embedding": embedding,
            "label_indices": torch.tensor(label_indices, dtype=torch.long),
            "identity": {
                "tax_id": record.tax_id,
                "GeneID": record.gene_id,
                "cds_id": record.cds_id,
                "embedding_id": record.embedding_id,
                "sequence_id": record.sequence_id,
                "sequence_length": record.sequence_length,
                "token_count": record.token_count,
                "shard_path": str(record.shard_path),
                "shard_row": record.shard_row,
            },
            "n_terms": self.n_terms,
        }


def evo2_go_collate_fn(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Collate dataset samples into model inputs and dense multi-label targets."""

    if torch is None:
        raise ModuleNotFoundError("torch is required for evo2_go_collate_fn")
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    n_terms = int(samples[0]["n_terms"])
    embeddings = torch.stack([sample["embedding"] for sample in samples])  # type: ignore[list-item]
    targets = torch.zeros((len(samples), n_terms), dtype=torch.float32)
    label_indices_out: list[torch.Tensor] = []
    identities: list[dict[str, object]] = []

    for row_idx, sample in enumerate(samples):
        if int(sample["n_terms"]) != n_terms:
            raise ValueError("All samples in a batch must use the same GO vocabulary")
        label_indices = sample["label_indices"]
        if not torch.is_tensor(label_indices):
            label_indices = torch.as_tensor(label_indices, dtype=torch.long)
        if label_indices.numel() > 0:
            targets[row_idx, label_indices.to(dtype=torch.long)] = 1.0
        label_indices_out.append(label_indices.to(dtype=torch.long))
        identities.append(sample["identity"])  # type: ignore[arg-type]

    return {
        "embeddings": embeddings,
        "targets": targets,
        "identities": identities,
        "label_indices": label_indices_out,
    }


def load_evo2_go_dataset(
    *,
    embedding_root: str | Path = DEFAULT_EMBEDDING_ROOT,
    vocabulary_path: str | Path = DEFAULT_GO_VOCABULARY,
    annotations_path: str | Path | None = DEFAULT_ANNOTATIONS,
    label_index_cache_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    split_name: str | None = None,
    split_key_mode: SplitKeyMode = "identity",
    require_all_records_assigned_to_split: bool = True,
    tax_ids: Sequence[str] | None = None,
    max_records: int | None = None,
    max_cached_shards: int = 2,
    drop_examples_without_labels: bool = True,
    validate_embeddings: bool = False,
    preselect_split_records: bool = False,
) -> Evo2GoDatasetBundle:
    """Build the current Evo2-GO training dataset from manifest and labels.

    If ``label_index_cache_path`` exists it is reused. Otherwise
    ``annotations_path`` is streamed to build labels for the selected embedded
    genes, and the cache is written when a cache path is provided.
    """

    root = Path(embedding_root)
    manifest_path = root / "embedding_dataset_manifest.tsv"
    selected_tax_ids = parse_tax_ids(tax_ids)

    vocabulary = load_go_vocabulary(vocabulary_path)
    metadata_paths = complete_metadata_paths_from_manifest(
        manifest_path,
        tax_ids=selected_tax_ids,
    )
    split_assignments: dict[str, str] | None = None
    if preselect_split_records:
        if split_manifest_path is None or split_name is None:
            raise ValueError(
                "preselect_split_records requires split_manifest_path and split_name"
            )
        split_assignments = load_split_manifest(
            split_manifest_path,
            split_name=split_name,
            key_mode=split_key_mode,
        )
        records = load_embedding_records(
            metadata_paths,
            max_records=max_records,
            include_split_keys=set(split_assignments),
            include_key_mode=split_key_mode,
        )
        if (
            require_all_records_assigned_to_split
            and max_records is None
            and selected_tax_ids is None
        ):
            found_keys = {
                split_key_for_record(record, split_key_mode) for record in records
            }
            missing_keys = set(split_assignments).difference(found_keys)
            if missing_keys:
                preview = sorted(missing_keys)[:3]
                raise ValueError(
                    f"{len(missing_keys):,} selected split records have no embedding; "
                    f"examples={preview}"
                )
    else:
        records = load_embedding_records(metadata_paths, max_records=max_records)
    if split_manifest_path is not None:
        if split_name is None:
            raise ValueError("split_name is required when split_manifest_path is provided")
        if split_assignments is None:
            split_assignments = load_split_manifest(
                split_manifest_path,
                split_name=None,
                key_mode=split_key_mode,
            )
            records = filter_records_by_split(
                records,
                split_assignments,
                split_name=split_name,
                key_mode=split_key_mode,
                require_all_records_assigned=require_all_records_assigned_to_split,
            )
    selected_gene_keys = {record.gene_key for record in records}

    label_index: dict[tuple[str, str], tuple[int, ...]]
    if label_index_cache_path is not None and Path(label_index_cache_path).is_file():
        label_index = load_label_index_cache(
            label_index_cache_path,
            n_terms=vocabulary.n_terms,
            selected_gene_keys=selected_gene_keys,
        )
    else:
        if annotations_path is None:
            raise ValueError(
                "annotations_path is required when label_index_cache_path is absent"
            )
        label_index = build_label_index_from_annotations(
            annotations_path,
            vocabulary,
            selected_gene_keys=selected_gene_keys,
        )
        if label_index_cache_path is not None:
            write_label_index_cache(label_index_cache_path, label_index, vocabulary)

    dataset = Evo2GoEmbeddingDataset(
        records,
        label_index,
        n_terms=vocabulary.n_terms,
        max_cached_shards=max_cached_shards,
        drop_examples_without_labels=drop_examples_without_labels,
        validate_embeddings=validate_embeddings,
    )
    return Evo2GoDatasetBundle(
        dataset=dataset,
        vocabulary=vocabulary,
        records_before_label_filter=len(records),
        label_index_size=len(label_index),
    )
