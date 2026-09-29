"""Object model for the Evo2-to-GO data pipeline.

This module keeps three biological concepts separate:

1. ``GOTerm``: the ontology term itself.
2. ``GOAnnotation``: one GeneID-to-GO relationship from ``gene2go.gz``.
3. ``DNASequenceRecord``: one DNA sequence example that can carry annotations.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Mapping


# Standard DNA bases plus IUPAC DNA ambiguity codes.
DNA_ALPHABET: frozenset[str] = frozenset("ACGTNRYSWKMBDHV")

# The column names used by NCBI's gene2go.gz file.
GENE2GO_COLUMNS: tuple[str, ...] = (
    "#tax_id",
    "GeneID",
    "GO_ID",
    "Evidence",
    "Qualifier",
    "GO_term",
    "PubMed",
    "Category",
)


def _clean_text(value: Any) -> str:
    """Convert a raw field value into a stripped string."""

    # Treat missing values as the empty string so parsers can be simple.
    if value is None:
        return ""

    # Convert integers or other simple values to strings and trim whitespace.
    return str(value).strip()


def _parse_required_int(row: Mapping[str, Any], key: str) -> int:
    """Read one required integer column from a parsed table row."""

    # Pull the raw value from the input mapping.
    raw_value = row.get(key)

    # Normalize the raw value before type conversion.
    text_value = _clean_text(raw_value)

    # Refuse blank required integer fields early with a useful error.
    if not text_value:
        raise ValueError(f"Missing required integer field: {key}")

    # Convert the normalized text to an integer.
    return int(text_value)


def parse_pubmed_ids(pubmed_field: Any) -> tuple[str, ...]:
    """Parse the pipe-delimited PubMed field from gene2go.gz."""

    # Already-parsed tuple/list values should stay structured.
    if isinstance(pubmed_field, (tuple, list)):
        return tuple(_clean_text(part) for part in pubmed_field if _clean_text(part))

    # Normalize the field so None, integers, and strings behave consistently.
    text_value = _clean_text(pubmed_field)

    # NCBI uses '-' when no PubMed identifier is present.
    if not text_value or text_value == "-":
        return ()

    # Split pipe-delimited PubMed IDs and drop accidental blank pieces.
    return tuple(part.strip() for part in text_value.split("|") if part.strip())


def validate_gene2go_columns(fieldnames: Iterable[str] | None) -> None:
    """Validate that a table header matches the expected gene2go columns."""

    # A missing header means the input is not a usable gene2go-style table.
    if fieldnames is None:
        raise ValueError("gene2go file is missing a header row")

    # Normalize the incoming column names into an immutable tuple.
    observed_columns = tuple(fieldnames)

    # Allow cleaned tables to use "tax_id" instead of NCBI's "#tax_id".
    cleaned_columns = ("tax_id",) + GENE2GO_COLUMNS[1:]

    # Compare the observed columns with the accepted gene2go schemas.
    if observed_columns not in (GENE2GO_COLUMNS, cleaned_columns):
        raise ValueError(
            "Unexpected gene2go columns: "
            f"expected {GENE2GO_COLUMNS!r} or {cleaned_columns!r}, observed {observed_columns!r}"
        )


def qualifier_has_not(qualifier: Any) -> bool:
    """Return True when a GO qualifier contains the negating NOT flag."""

    # Normalize the qualifier before checking for pipe-delimited flags.
    text_value = _clean_text(qualifier)

    # Empty qualifiers are not negated.
    if not text_value:
        return False

    # gene2go qualifiers may contain values like "NOT|enables".
    qualifier_parts = (part.strip().upper() for part in text_value.split("|"))

    # A row is negated if any qualifier component is exactly NOT.
    return any(part == "NOT" for part in qualifier_parts)


def normalize_dna_sequence(sequence: str) -> str:
    """Normalize and validate a DNA sequence string."""

    # Remove all whitespace so wrapped FASTA sequence text is accepted.
    compact_sequence = "".join(str(sequence).split()).upper()

    # Refuse empty DNA sequences because they cannot produce useful embeddings.
    if not compact_sequence:
        raise ValueError("DNA sequence cannot be empty")

    # Compute characters that are not valid DNA/IUPAC DNA symbols.
    invalid_symbols = sorted(set(compact_sequence) - DNA_ALPHABET)

    # Report invalid characters explicitly; U is intentionally rejected as RNA.
    if invalid_symbols:
        invalid_text = ", ".join(invalid_symbols)
        raise ValueError(f"Invalid DNA symbol(s): {invalid_text}")

    # Return the cleaned uppercase DNA sequence.
    return compact_sequence


@dataclass(frozen=True)
class GOTerm:
    """A Gene Ontology term independent of any gene annotation event."""

    # Stable GO accession, for example "GO:0004449".
    go_id: str

    # Human-readable GO term name from the annotation source.
    name: str

    # GO namespace/category as supplied by gene2go: Function, Process, Component.
    category: str

    def __post_init__(self) -> None:
        """Validate and normalize immutable term fields after construction."""

        # Normalize GO ID text before validation.
        go_id = _clean_text(self.go_id)

        # Normalize term name text before validation.
        name = _clean_text(self.name)

        # Normalize GO category text before validation.
        category = _clean_text(self.category)

        # GO accessions in this project should use the canonical GO: prefix.
        if not go_id.startswith("GO:"):
            raise ValueError(f"GO ID must start with 'GO:': {go_id!r}")

        # A term without a name is not useful for summaries or debugging.
        if not name:
            raise ValueError("GO term name cannot be empty")

        # A category is required because the three GO namespaces differ.
        if not category:
            raise ValueError("GO term category cannot be empty")

        # Store the normalized GO ID on the frozen dataclass.
        object.__setattr__(self, "go_id", go_id)

        # Store the normalized term name on the frozen dataclass.
        object.__setattr__(self, "name", name)

        # Store the normalized category on the frozen dataclass.
        object.__setattr__(self, "category", category)


@dataclass(frozen=True)
class GOAnnotation:
    """One GeneID-to-GO annotation relationship from gene2go.gz."""

    # NCBI taxonomy identifier for the annotated gene.
    tax_id: int

    # NCBI GeneID for the annotated gene.
    gene_id: int

    # Stable GO accession, for example "GO:0004449".
    go_id: str

    # GO evidence code, for example IEA, IDA, IMP, or ISS.
    evidence: str

    # GO qualifier text, for example enables, involved_in, located_in, or NOT.
    qualifier: str

    # Human-readable GO term name copied from gene2go.
    go_term: str

    # PubMed identifiers supporting the annotation row, when present.
    pubmed_ids: tuple[str, ...] = ()

    # GO namespace/category from gene2go: Function, Process, or Component.
    category: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize immutable annotation fields."""

        # Normalize the GO accession.
        go_id = _clean_text(self.go_id)

        # Normalize the evidence code.
        evidence = _clean_text(self.evidence)

        # Normalize the qualifier text.
        qualifier = _clean_text(self.qualifier)

        # Normalize the GO term name.
        go_term = _clean_text(self.go_term)

        # Normalize the category text.
        category = _clean_text(self.category)

        # Normalize PubMed IDs whether the caller provided text or a sequence.
        pubmed_ids = parse_pubmed_ids(self.pubmed_ids)

        # Validate the tax ID as a positive integer.
        if int(self.tax_id) <= 0:
            raise ValueError(f"tax_id must be positive: {self.tax_id!r}")

        # Validate the GeneID as a positive integer.
        if int(self.gene_id) <= 0:
            raise ValueError(f"gene_id must be positive: {self.gene_id!r}")

        # GO accessions in this project should use the canonical GO: prefix.
        if not go_id.startswith("GO:"):
            raise ValueError(f"GO ID must start with 'GO:': {go_id!r}")

        # Evidence is central to later filtering, so it must be present.
        if not evidence:
            raise ValueError("GO evidence code cannot be empty")

        # A term name is required for readable summaries and provenance.
        if not go_term:
            raise ValueError("GO term name cannot be empty")

        # A category is required because GO labels are usually modeled by namespace.
        if not category:
            raise ValueError("GO category cannot be empty")

        # Store the normalized tax ID.
        object.__setattr__(self, "tax_id", int(self.tax_id))

        # Store the normalized GeneID.
        object.__setattr__(self, "gene_id", int(self.gene_id))

        # Store the normalized GO accession.
        object.__setattr__(self, "go_id", go_id)

        # Store the normalized evidence code.
        object.__setattr__(self, "evidence", evidence)

        # Store the normalized qualifier text.
        object.__setattr__(self, "qualifier", qualifier)

        # Store the normalized GO term name.
        object.__setattr__(self, "go_term", go_term)

        # Store normalized PubMed IDs as an immutable tuple.
        object.__setattr__(self, "pubmed_ids", pubmed_ids)

        # Store the normalized GO category.
        object.__setattr__(self, "category", category)

    @property
    def term(self) -> GOTerm:
        """Return this annotation's GO term as a standalone GOTerm object."""

        # Build a GOTerm from the annotation's term-level fields.
        return GOTerm(go_id=self.go_id, name=self.go_term, category=self.category)

    @property
    def is_negated(self) -> bool:
        """Return True if this annotation uses the GO NOT qualifier."""

        # Delegate qualifier parsing to the shared helper.
        return qualifier_has_not(self.qualifier)

    @classmethod
    def from_gene2go_row(cls, row: Mapping[str, Any]) -> "GOAnnotation":
        """Create one GOAnnotation from a gene2go DictReader row."""

        # NCBI's header uses "#tax_id"; tolerate "tax_id" for cleaned tables.
        tax_key = "#tax_id" if "#tax_id" in row else "tax_id"

        # Construct the annotation from the eight gene2go columns.
        return cls(
            tax_id=_parse_required_int(row, tax_key),
            gene_id=_parse_required_int(row, "GeneID"),
            go_id=_clean_text(row.get("GO_ID")),
            evidence=_clean_text(row.get("Evidence")),
            qualifier=_clean_text(row.get("Qualifier")),
            go_term=_clean_text(row.get("GO_term")),
            pubmed_ids=parse_pubmed_ids(row.get("PubMed")),
            category=_clean_text(row.get("Category")),
        )


@dataclass
class DNASequenceRecord:
    """One DNA sequence example with optional Evo2 embedding and GO labels."""

    # Sequence identifier from FASTA, manifest, or another upstream source.
    sequence_id: str

    # NCBI GeneID associated with this representative DNA sequence.
    gene_id: int

    # NCBI taxonomy identifier for the organism that owns the sequence.
    tax_id: int

    # DNA sequence that will be passed to Evo2.
    sequence: str

    # Optional embedding object; often a numpy array, torch tensor, or list.
    embedding: Any | None = None

    # Optional identifier or path for a separately stored embedding.
    embedding_id: str | None = None

    # GO annotations attached to this GeneID-level sequence example.
    annotations: list[GOAnnotation] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and normalize sequence-record fields after construction."""

        # Normalize the sequence identifier.
        self.sequence_id = _clean_text(self.sequence_id)

        # Validate that the sequence identifier is present.
        if not self.sequence_id:
            raise ValueError("sequence_id cannot be empty")

        # Normalize and validate the DNA sequence.
        self.sequence = normalize_dna_sequence(self.sequence)

        # Normalize the GeneID to an integer.
        self.gene_id = int(self.gene_id)

        # Normalize the tax ID to an integer.
        self.tax_id = int(self.tax_id)

        # Validate the GeneID as a positive integer.
        if self.gene_id <= 0:
            raise ValueError(f"gene_id must be positive: {self.gene_id!r}")

        # Validate the tax ID as a positive integer.
        if self.tax_id <= 0:
            raise ValueError(f"tax_id must be positive: {self.tax_id!r}")

        # Normalize the optional embedding identifier when present.
        if self.embedding_id is not None:
            self.embedding_id = _clean_text(self.embedding_id)

        # Ensure the annotations container is a mutable list.
        self.annotations = list(self.annotations)

        # Validate all provided annotations against this sequence record.
        for annotation in self.annotations:
            self._validate_annotation_matches_record(annotation)

    def _validate_annotation_matches_record(self, annotation: GOAnnotation) -> None:
        """Ensure an annotation belongs to this record's GeneID and tax ID."""

        # Check the GeneID because gene2go labels are GeneID-level labels.
        if annotation.gene_id != self.gene_id:
            raise ValueError(
                f"Annotation GeneID {annotation.gene_id} does not match record GeneID {self.gene_id}"
            )

        # Check the tax ID to avoid mixing same-numbered genes across taxa.
        if annotation.tax_id != self.tax_id:
            raise ValueError(
                f"Annotation tax_id {annotation.tax_id} does not match record tax_id {self.tax_id}"
            )

    def add_annotation(self, annotation: GOAnnotation) -> None:
        """Attach one GO annotation to this sequence record."""

        # Validate the annotation before mutating the record.
        self._validate_annotation_matches_record(annotation)

        # Append the validated annotation to this record.
        self.annotations.append(annotation)

    @property
    def length(self) -> int:
        """Return the DNA sequence length in bases."""

        # The normalized sequence has no whitespace, so len is the base count.
        return len(self.sequence)

    @property
    def go_ids(self) -> set[str]:
        """Return all GO IDs attached to this record, including NOT rows."""

        # Collect every GO accession from every attached annotation.
        return {annotation.go_id for annotation in self.annotations}

    @property
    def positive_go_ids(self) -> set[str]:
        """Return GO IDs that should be treated as positive training labels."""

        # Exclude annotations with a NOT qualifier from positive labels.
        return {annotation.go_id for annotation in self.annotations if not annotation.is_negated}

    def label_set(
        self,
        allowed_evidence: set[str] | None = None,
        categories: set[str] | None = None,
        include_negated: bool = False,
    ) -> set[str]:
        """Return GO IDs after applying common label-filtering choices."""

        # Normalize evidence filters when the caller provides them.
        normalized_evidence = (
            {_clean_text(evidence).upper() for evidence in allowed_evidence}
            if allowed_evidence is not None
            else None
        )

        # Normalize category filters when the caller provides them.
        normalized_categories = (
            {_clean_text(category).lower() for category in categories}
            if categories is not None
            else None
        )

        # Create the output set of GO labels.
        labels: set[str] = set()

        # Inspect each annotation attached to this record.
        for annotation in self.annotations:

            # Skip negated annotations unless the caller explicitly includes them.
            if annotation.is_negated and not include_negated:
                continue

            # Skip evidence codes outside the requested evidence set.
            if normalized_evidence is not None and annotation.evidence.upper() not in normalized_evidence:
                continue

            # Skip GO categories outside the requested category set.
            if normalized_categories is not None and annotation.category.lower() not in normalized_categories:
                continue

            # Add the annotation's GO accession after all filters pass.
            labels.add(annotation.go_id)

        # Return the filtered GO label set.
        return labels

    @property
    def terms(self) -> dict[str, GOTerm]:
        """Return unique GO terms attached to this record by GO ID."""

        # Build a GO ID to GOTerm mapping from attached annotations.
        return {annotation.go_id: annotation.term for annotation in self.annotations}


def iter_gene2go_annotations(path: str | Path) -> Iterator[GOAnnotation]:
    """Stream GOAnnotation objects from a gene2go.gz or plain TSV file."""

    # Normalize the filesystem path.
    path = Path(path)

    # Choose gzip.open for .gz files and open for plain text files.
    open_function = gzip.open if path.suffix == ".gz" else open

    # Open the annotation table in text mode.
    with open_function(path, mode="rt", newline="") as handle:

        # gene2go.gz is tab-delimited and includes a header row.
        reader = csv.DictReader(handle, delimiter="\t")

        # Validate the header before yielding any rows.
        validate_gene2go_columns(reader.fieldnames)

        # Stream rows one at a time so the 100M+ row file is not loaded at once.
        for row in reader:

            # Convert each parsed row into a typed annotation object.
            yield GOAnnotation.from_gene2go_row(row)


def group_annotations_by_gene(
    annotations: Iterable[GOAnnotation],
) -> dict[tuple[int, int], list[GOAnnotation]]:
    """Group annotations by (tax_id, gene_id)."""

    # Create the output mapping keyed by tax ID and GeneID.
    grouped: dict[tuple[int, int], list[GOAnnotation]] = {}

    # Iterate over the incoming annotations.
    for annotation in annotations:

        # Build the stable GeneID-level grouping key.
        key = (annotation.tax_id, annotation.gene_id)

        # Add the annotation to the matching group.
        grouped.setdefault(key, []).append(annotation)

    # Return the completed grouping.
    return grouped
