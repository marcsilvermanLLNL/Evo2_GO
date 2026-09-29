from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evo2_go.model.dataset import load_embedding_records, load_evo2_go_dataset
from evo2_go.training.validation_panel import create_validation_panel


class ValidationPanelTests(unittest.TestCase):
    def write_split(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("wt", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("split", "tax_id", "GeneID", "cds_id", "embedding_id"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def panel_identities(self, path: Path) -> set[tuple[str, str, str]]:
        with path.open("rt", newline="") as handle:
            return {
                (row["tax_id"], row["GeneID"], row["cds_id"])
                for row in csv.DictReader(handle, delimiter="\t")
            }

    def test_selection_is_fixed_and_independent_of_source_order(self) -> None:
        rows = [
            {
                "split": "val" if index < 100 else "train",
                "tax_id": str(index % 7 + 1),
                "GeneID": str(10_000 + index),
                "cds_id": f"cds_{index}",
                "embedding_id": f"embedding_{index}",
            }
            for index in range(110)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_source = root / "first.tsv"
            reversed_source = root / "reversed.tsv"
            first_output = root / "first_panel.tsv"
            reversed_output = root / "reversed_panel.tsv"
            self.write_split(first_source, rows)
            self.write_split(reversed_source, list(reversed(rows)))

            first = create_validation_panel(
                first_source,
                first_output,
                split_name="val",
                panel_size=25,
                seed="fixed-test-seed",
            )
            second = create_validation_panel(
                reversed_source,
                reversed_output,
                split_name="val",
                panel_size=25,
                seed="fixed-test-seed",
            )

            self.assertEqual(first["eligible_records"], 100)
            self.assertEqual(first["panel_records"], 25)
            self.assertEqual(second["panel_records"], 25)
            self.assertEqual(
                self.panel_identities(first_output),
                self.panel_identities(reversed_output),
            )

    def test_embedding_metadata_can_be_preselected_by_panel_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "embedding_metadata.tsv"
            fieldnames = (
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
            )
            with metadata.open("wt", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                for index in range(4):
                    writer.writerow(
                        {
                            "embedding_id": f"embedding_{index}",
                            "tax_id": "1",
                            "GeneID": str(100 + index),
                            "cds_id": f"cds_{index}",
                            "sequence_id": (
                                f"tax_id=1|GeneID={100 + index}|cds_id=cds_{index}|length=12"
                            ),
                            "sequence_length": "12",
                            "token_count": "12",
                            "shard_path": f"shard_{index}.pt",
                            "shard_row": "0",
                            "layer_name": "test",
                            "pooled_shape": "[4096]",
                            "pooled_dtype": "float16",
                        }
                    )

            selected = {"1\t101\tcds_1", "1\t103\tcds_3"}
            records = load_embedding_records(
                [metadata],
                include_split_keys=selected,
                include_key_mode="identity",
            )
            self.assertEqual([record.gene_id for record in records], ["101", "103"])

    def test_dataset_loader_uses_panel_and_filters_label_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedding_root = root / "embeddings"
            embedding_root.mkdir()
            metadata = embedding_root / "metadata.tsv"
            manifest = embedding_root / "embedding_dataset_manifest.tsv"
            vocabulary = root / "vocabulary.tsv"
            labels = root / "labels.tsv"
            panel = root / "panel.tsv"

            with manifest.open("wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("tax_id", "status", "embedding_metadata_path"))
                writer.writerow(("1", "complete", "metadata.tsv"))

            metadata_fields = (
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
            )
            with metadata.open("wt", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=metadata_fields,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                for index in range(4):
                    writer.writerow(
                        {
                            "embedding_id": f"embedding_{index}",
                            "tax_id": "1",
                            "GeneID": str(100 + index),
                            "cds_id": f"cds_{index}",
                            "sequence_id": (
                                f"tax_id=1|GeneID={100 + index}|cds_id=cds_{index}|length=12"
                            ),
                            "sequence_length": "12",
                            "token_count": "12",
                            "shard_path": f"shard_{index}.pt",
                            "shard_row": "0",
                            "layer_name": "test",
                            "pooled_shape": "[4096]",
                            "pooled_dtype": "float16",
                        }
                    )

            with vocabulary.open("wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("go_index", "GO_ID"))
                writer.writerow(("0", "GO:0000001"))
                writer.writerow(("1", "GO:0000002"))

            with labels.open("wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("tax_id", "GeneID", "go_indices"))
                for index in range(4):
                    writer.writerow(("1", str(100 + index), str(index % 2)))

            with panel.open("wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("split", "tax_id", "GeneID", "cds_id"))
                writer.writerow(("val", "1", "101", "cds_1"))
                writer.writerow(("val", "1", "103", "cds_3"))

            bundle = load_evo2_go_dataset(
                embedding_root=embedding_root,
                vocabulary_path=vocabulary,
                label_index_cache_path=labels,
                split_manifest_path=panel,
                split_name="val",
                preselect_split_records=True,
            )
            self.assertEqual(len(bundle.dataset), 2)
            self.assertEqual(bundle.label_index_size, 2)
            self.assertEqual(
                [record.gene_id for record in bundle.dataset.records],
                ["101", "103"],
            )


if __name__ == "__main__":
    unittest.main()
