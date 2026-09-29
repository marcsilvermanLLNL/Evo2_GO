# Command Reference

Use one top-level command:

```bash
evo2-go <command> --help
```

Without installing:

```bash
PYTHONPATH=src python3 -m evo2_go.cli <command> --help
```

## Migrated Commands

### `build-database`

Run a config-driven database build by calling migrated Python modules in order.
Use this for database assembly steps. Use `prepare-training-inputs` below to
refresh post-embedding labels, adjacency, label cache, and splits without
rerunning database preprocessing.

```bash
python3 build_database.py --config configs/build.example.toml --project-root . --dry-run
PYTHONPATH=src python3 -m evo2_go.cli build-database --config configs/build.example.toml --project-root . --dry-run
```

Useful controls:

```bash
python3 build_database.py --config configs/build.example.toml --project-root . --from-step go_adjacency
python3 build_database.py --config configs/build.example.toml --project-root . --steps true_path_counts,go_adjacency
python3 build_database.py --config configs/build.example.toml --project-root . --skip-existing
```

### `prepare-training-inputs`

Run the rerunnable CPU-side preparation after embeddings exist. The default
example config refreshes true-path counts, GO vocabulary/adjacency,
vocabulary-aligned label cache, and the human-test/rat-validation split
manifest for the active build.

```bash
python3 prepare_training_inputs.py --config configs/training_inputs.example.toml --project-root . --dry-run
PYTHONPATH=src python3 -m evo2_go.cli prepare-training-inputs --config configs/training_inputs.example.toml --project-root . --dry-run
```

Useful controls:

```bash
python3 prepare_training_inputs.py --config configs/training_inputs.example.toml --project-root . --skip-existing
python3 prepare_training_inputs.py --config configs/training_inputs.example.toml --project-root . --steps go_adjacency,label_cache,split_manifest --force
python3 prepare_training_inputs.py --config configs/training_inputs.example.toml --project-root . --from-step label_cache --force
```

SLURM wrappers for the same workflow are available when these CPU steps should
run through `sbatch`:

```bash
sbatch scripts/prepare_go_adjacency.slurm.sh --force
sbatch scripts/prepare_training_split.slurm.sh --force
```

The first job runs `true_path_counts,go_adjacency,label_cache`. The second job
runs only `split_manifest`.

The GO adjacency job requires the ontology dependencies from this remodel tree.
Set them up locally, not from the legacy `Evo2-GO` directory:

```bash
python3 -m venv envs/ontology
envs/ontology/bin/python -m pip install --upgrade pip
envs/ontology/bin/python -m pip install -e '.[ontology]'
```

After `envs/ontology/bin/python` exists, the SLURM wrappers use it by default.

### `map-gene-cds`

Select representative CDS records for labeled genes. This is the original
single-taxon/pilot mapper, copied into the package as the first stable database
component.

```bash
evo2-go map-gene-cds --help
```

### `map-gene-cds-by-tax`

Run `map-gene-cds` independently for many taxa from a downloaded CDS manifest.
This is the preferred command for resumable database assembly across taxa.

```bash
evo2-go map-gene-cds-by-tax --help
```

### `filter-cds-length`

Filter per-taxon representative CDS outputs to the configured Evo2 context
window. By default this writes one output directory per taxon under the shared
length-filter root.

```bash
evo2-go filter-cds-length \
  --input-root outputs/map_gene_cds/by_tax \
  --output-root outputs/map_gene_cds/length_filtered_max_8192 \
  --max-tokens 8192 \
  --incomplete-only
```

### `run-embeddings`

Extract Evo2 embeddings from the length-filtered CDS root. Use one shared
`--output-root` for all embedding jobs; each taxon is stored under
`tax_<tax_id>/`, and the root manifest indexes every taxon in that shared root.

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

Run the next runnable taxa in repeated queue jobs:

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

Completed taxa are skipped by default. Completed batch shards are also resumed
by default, so reusing the same output root is the intended workflow.

### `validate-embeddings`

Validate the shared embedding root, per-taxon metadata, shard tensor shapes, and
optional label-cache coverage.

```bash
evo2-go validate-embeddings \
  --embedding-root outputs/builds/<build_id>/data/embeddings \
  --input-root outputs/map_gene_cds/length_filtered_max_8192 \
  --vocabulary outputs/builds/<build_id>/adjacency/go_vocabulary.tsv \
  --label-cache outputs/builds/<build_id>/data/labels/label_index.tsv.gz
```

### `go-term-counts`

Count distinct `(tax_id, GeneID, GO_ID)` labels for a selected trainable gene
universe.

```bash
evo2-go go-term-counts --help
```

### `true-path-labels`

Build true-path labels by expanding explicit positive GO labels to `is_a` and
`part_of` ancestors. Can also write a vocabulary-aligned label cache when a
vocabulary is supplied.

```bash
evo2-go true-path-labels --help
```

### `go-adjacency`

Build the deterministic GO vocabulary and normalized sparse adjacency used by
the GO graph encoder.

```bash
evo2-go go-adjacency --help
```

### `taxon-holdout-split`

Create a split manifest from an embedding root. Separate-taxon mode assigns
one or more taxa entirely to validation and one or more taxa entirely to test;
all other taxa become train. By default every embedding record receives a split
assignment, and retained-label counts are reported when `--label-cache` is
provided.

```bash
evo2-go taxon-holdout-split \
  --embedding-root outputs/builds/<build_id>/data/embeddings \
  --label-cache outputs/builds/<build_id>/data/labels/label_index.tsv.gz \
  --output outputs/builds/<build_id>/data/splits/human_test_taxon_val_manifest.tsv \
  --summary outputs/builds/<build_id>/data/splits/human_test_taxon_val_summary.tsv \
  --val-tax-id 10116 \
  --test-tax-id 9606 \
    --force
```

### `validation-panel`

Create a persistent, seeded sample from a validation split. Selection uses the
smallest seeded SHA-256 identity hashes, so the chosen identities do not depend
on source row order. The output is itself a valid split manifest.

```bash
evo2-go validation-panel \
  --split-manifest outputs/builds/<split_build>/data/splits/<split>_manifest.tsv \
  --split-name val \
  --panel-size 50000 \
  --seed evo2-go-mmseqs-val-panel-50000-v1 \
  --output outputs/builds/<split_build>/data/splits/<split>_validation_panel_50000_v1.tsv
```

Training should consume the entire panel rather than truncate it:

```toml
[training]
validation_panel = "outputs/builds/<split_build>/data/splits/<split>_validation_panel_50000_v1.tsv"
validation_batches = 0
```

When resuming a model whose previous best checkpoint used a different
validation protocol, use a new run directory and set:

```toml
resume_checkpoint = "outputs/builds/<build>/runs/<old_run>/latest_checkpoint.pt"
reset_best_on_resume = true
```

## Remaining Migration Gaps

- Raw Gene2GO chunk processing.
- RefSeq CDS download wrappers.
- Run matrix submission for architecture/loss experiments.

Model training, evaluation, and visualization are available through
`train-model`, `evaluate-model`, and `visualize-results`. See
[codebase_vertical_summary.md](codebase_vertical_summary.md) for their code and
artifact contracts.
