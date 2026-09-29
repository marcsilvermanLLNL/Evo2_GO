# Database Pipeline

This document records the intended database workflow for the remodeled tree.
Run commands from the `Evo2-GO-Remodel` directory. Relative paths resolve
inside this tree.

## 1. Gene2GO Processing

Current status: required processed outputs are present in the remodel tree.

Source:

```text
data/gene2go.gz
```

Important outputs:

```text
outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz
outputs/process_Gene2GO/processed/final/positive_gene_go_labels.tsv.gz
```

The annotation table keeps evidence codes and should be used whenever the label
policy depends on evidence, such as excluding or including IEA.

## 2. Representative CDS Mapping

Current status: migrated command surface.

Policy:

```text
For each (tax_id, GeneID), select the longest CDS.
Tie break by lowest ambiguity count, then lexicographically smallest CDS id.
Drop labeled genes with no representative CDS.
```

Single-taxon command:

```bash
evo2-go map-gene-cds \
  --tax-id 9606 \
  --labels outputs/process_Gene2GO/processed/final/positive_gene_go_labels.tsv.gz \
  --output-dir outputs/map_gene_cds/tax_9606_full
```

Multi-taxon driver:

```bash
evo2-go map-gene-cds-by-tax \
  --download-manifest data/refseq_cds/downloaded_cds.tsv \
  --output-root outputs/map_gene_cds/by_tax
```

## 3. Length Filtering

Current status: migrated command surface.

The existing 8k Evo2 context pipeline filters representative CDS records to
`token_count <= 8192`.

Example:

```bash
evo2-go filter-cds-length \
  --input-root outputs/map_gene_cds/by_tax \
  --output-root outputs/map_gene_cds/length_filtered_max_8192 \
  --max-tokens 8192 \
  --incomplete-only
```

Important outputs:

```text
outputs/map_gene_cds/length_filtered_max_8192/filter_status.tsv
outputs/map_gene_cds/length_filtered_max_8192/dataset_length_filter_summary.tsv
outputs/map_gene_cds/length_filtered_max_8192/tax_<tax_id>/kept_representative_gene_cds.fna.gz
```

## 4. Evo2 Embeddings

Current status: migrated command surface.

All embedding jobs for one dataset should write into the same
`embedding_root`. The extractor stores durable per-taxon artifacts under
`tax_<tax_id>/`, skips completed taxa by default, and resumes completed batch
shards by default.

The default local checkpoint path is:

```text
evo2_model/evo2_7b_base.pt
```

Copy or symlink the checkpoint there before running embeddings, or pass
`--model-path` with an explicit site-local checkpoint path.

Shared output layout:

```text
outputs/builds/<build_id>/data/embeddings/
  embedding_dataset_manifest.tsv
  embedding_metadata.tsv.gz
  batch_metrics.tsv
  tax_<tax_id>/
    embedding_metadata.tsv.gz
    batch_metrics.tsv
    run_summary.tsv
    embedding_shards/bucket_*/batch_*.pt
```

Run explicitly listed taxa:

```bash
evo2-go run-embeddings \
  --input-root outputs/map_gene_cds/length_filtered_max_8192 \
  --output-root outputs/builds/<build_id>/data/embeddings \
  --model-name evo2_7b_base \
  --model-path evo2_model/evo2_7b_base.pt \
  --expected-max-seqlen 32768 \
  --tax-id 9606 \
  --tax-id 10090
```

Run the next runnable taxa in repeated jobs:

```bash
evo2-go run-embeddings \
  --input-root outputs/map_gene_cds/length_filtered_max_8192 \
  --output-root outputs/builds/<build_id>/data/embeddings \
  --model-name evo2_7b_base \
  --model-path evo2_model/evo2_7b_base.pt \
  --expected-max-seqlen 32768 \
  --incomplete-only \
  --limit-runnable-taxa 2
```

`embedding_dataset_manifest.tsv` is the preferred dataset index for model
loading. `embedding_metadata.tsv.gz` is rebuilt from every complete taxon in the
shared root for commands that need a single selected-gene table.

## 5. Training Input Preparation

Current status: migrated command surface.

After embeddings exist, refresh the CPU-side training inputs separately from
database preprocessing:

```bash
python3 prepare_training_inputs.py \
  --config configs/training_inputs.example.toml \
  --project-root . \
  --dry-run
```

Remove `--dry-run` when the printed commands and paths are correct. The default
example config runs:

```text
true_path_counts -> go_adjacency -> label_cache -> split_manifest
```

This writes:

```text
outputs/builds/<build_id>/data/labels/true_path_go_term_counts.tsv
outputs/builds/<build_id>/adjacency/go_vocabulary.tsv
outputs/builds/<build_id>/adjacency/go_adjacency_sparse.npz
outputs/builds/<build_id>/data/labels/label_index.tsv.gz
outputs/builds/<build_id>/data/splits/<split_name>_manifest.tsv
outputs/builds/<build_id>/training_inputs_manifest.tsv
outputs/builds/<build_id>/training_inputs_steps.tsv
```

The active example split uses taxon `10116` for validation and taxon `9606`
for test. It is not ortholog-safe and records `ortholog_safe = 0`.

## 6. GO Term Counts

Current status: migrated command surface.

The orchestrated path is:

```bash
python3 build_database.py \
  --config configs/build.example.toml \
  --project-root . \
  --dry-run
```

Remove `--dry-run` when the printed commands and paths are correct.

Example:

```bash
evo2-go go-term-counts \
  --selected-genes outputs/builds/<build_id>/data/embeddings/embedding_metadata.tsv.gz \
  --annotations outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz \
  --output outputs/builds/<build_id>/data/labels/explicit_go_term_counts.tsv \
  --include-iea \
  --force
```

## 7. True-Path Labels

Current status: migrated command surface.

True-path expansion adds `is_a` and `part_of` ancestors from `go-basic.obo`.

Example:

```bash
evo2-go true-path-labels \
  --selected-genes outputs/builds/<build_id>/data/embeddings/embedding_metadata.tsv.gz \
  --annotations outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz \
  --go-basic-obo data/go/go-basic.obo \
  --output-dir outputs/builds/<build_id>/data/labels \
  --include-iea \
  --force
```

## 8. GO Vocabulary And Adjacency

Current status: migrated command surface.

The vocabulary is deterministic and sorted by GO ID. The sparse adjacency uses
self-loops plus bidirectional `is_a` and `part_of` edges from `go-basic.obo`.

Example:

```bash
evo2-go go-adjacency \
  --term-counts outputs/builds/<build_id>/data/labels/true_path_go_term_counts.tsv \
  --output-dir outputs/builds/<build_id>/adjacency \
  --min-count 100 \
  --force
```

## 9. Embedding Validation

Current status: migrated command surface.

Validation checks the shared embedding manifest, per-taxon metadata, shard
tensors, the length-filtered source metadata join, and optional label-cache
coverage.

Example:

```bash
evo2-go validate-embeddings \
  --embedding-root outputs/builds/<build_id>/data/embeddings \
  --input-root outputs/map_gene_cds/length_filtered_max_8192 \
  --vocabulary outputs/builds/<build_id>/adjacency/go_vocabulary.tsv \
  --label-cache outputs/builds/<build_id>/data/labels/label_index.tsv.gz
```

## 10. Model Training

Current status: migrated command surface with config-driven training,
evaluation, checkpointing, metrics, and visualization. The architecture itself
is still mostly hardcoded and should be promoted into validated config tables.

See [codebase_vertical_summary.md](codebase_vertical_summary.md) for the full
data-to-model-to-analysis trace and current vulnerability review.
