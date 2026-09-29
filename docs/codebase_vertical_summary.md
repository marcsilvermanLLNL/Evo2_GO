# Evo2-GO Remodel: Vertical Codebase Summary

This document describes the code path from raw biological data to publication
figures. It is based on a repository-wide review performed on 2026-08-18. It
also records correctness, reproducibility, operational, methodological, and
security risks visible in the code and in the current training run.

## Scope And Code Surface

The repository contains roughly 21,000 lines of Python, shell, TOML, and test
documentation across these main areas:

| Area | Primary location | Responsibility |
|---|---|---|
| Command dispatcher | `src/evo2_go/cli.py` | Maps the `evo2-go` command to package modules. |
| Database integration | `src/evo2_go/database/` | Gene2GO parsing, GeneID-to-CDS mapping, GO counts, and true-path labels. |
| Embedding preparation | `src/evo2_go/embeddings/` | CDS length filtering, Evo2 extraction, resume metadata, and validation. |
| Ontology graph | `src/evo2_go/ontology/` | Deterministic GO vocabulary and normalized sparse adjacency. |
| Dataset and model | `src/evo2_go/model/` | Shard-backed dataset, dual encoder, loss, checkpoints, and Fmax. |
| Training and analysis | `src/evo2_go/training/` | Training, evaluation, general visualization, and run configuration. |
| Workflows | `src/evo2_go/workflows/` | Config-driven orchestration and split generation. |
| Site launchers | `scripts/` | Flux/Slurm wrappers, embedding repair, evaluation, and figures. |
| Experiment specifications | `configs/` | Build, training-input, split, training, and evaluation parameters. |
| Publication figures | `scripts/publication/` | Publication-specific figures and supporting tables/captions. |
| Tests | `tests/` | Initial deterministic validation-panel and metadata-selection tests. |

The package entry point is declared in `pyproject.toml`:

```text
evo2-go = evo2_go.cli:main
```

## Vertical System Map

```text
NCBI Gene2GO + Gene2RefSeq + RefSeq CDS + go-basic.obo
                         |
                         v
       processed positive GeneID-GO annotation tables
                         |
                         v
       (tax_id, GeneID) -> representative longest CDS
                         |
                         v
            CDS length filter (<= 8,192 bases/tokens)
                         |
                         v
       Evo2 layer activations -> mean-pooled 4,096-d vectors
                         |
                         v
        shared embedding shards + composite-key metadata
                         |
              +----------+-----------+
              |                      |
              v                      v
     GO true-path labels       nucleotide MMseqs linclust
     and GO graph/vocab        cluster-level 85/5/10 split
              |                      |
              +----------+-----------+
                         v
          shard-backed PyTorch dataset + dense GO targets
                         |
                         v
          sequence projection <-> two-layer GO GCN
                         |
                         v
             asymmetric multi-label model training
                         |
                         v
       checkpoint evaluation, Fmax, AUROC/AP, calibration
                         |
                         v
          general diagnostics + publication-ready figures
```

## Cross-Stage Data Contracts

The stable example identity is:

```text
(tax_id, GeneID, cds_id)
```

Bare `GeneID` is not globally unique. GO annotations are GeneID-level rather
than transcript- or protein-isoform-level, so the biological sequence policy
is:

```text
(tax_id, GeneID) -> longest representative CDS -> Evo2 embedding -> GO labels
```

The major joins are:

| Join | Key |
|---|---|
| Gene2GO to representative sequence | `(tax_id, GeneID)` |
| Length-filtered record to embedding metadata | `(tax_id, GeneID, cds_id)` |
| Embedding record to split manifest | `(tax_id, GeneID, cds_id)` by default |
| Embedding record to label cache | `(tax_id, GeneID)` |
| Dense target column to graph node | contiguous `go_index` in `go_vocabulary.tsv` |

## Stage 1: Raw Data And Upstream Processing

Important raw or site-local sources include:

```text
data/gene2go.gz
data/gene2refseq.gz
data/gene_group.gz
data/refseq_cds/
data/go/go-basic.obo
```

`src/evo2_go/database/records.py` provides typed objects and validation for
Gene2GO annotations, GO terms, and DNA records. It rejects invalid required
IDs, invalid DNA/IUPAC symbols, and negated `NOT` annotations when positive
labels are requested.

The active pipeline consumes already-processed annotation tables:

```text
outputs/process_Gene2GO/processed/final/positive_gene_go_annotations.tsv.gz
outputs/process_Gene2GO/processed/final/positive_gene_go_labels.tsv.gz
```

The code that creates those processed tables and the RefSeq download manifest
has not been migrated into this repository. Therefore the remodel is not yet a
complete fresh-start pipeline from remote NCBI sources. `build_database.py`
begins from processed annotations and a local `downloaded_cds.tsv` manifest.

## Stage 2: Representative CDS Integration

Primary code:

```text
src/evo2_go/database/map_gene_cds.py
src/evo2_go/database/map_gene_cds_by_tax.py
```

The mapper selects labeled genes, optionally uses Gene2RefSeq provenance, and
scans local CDS FASTA headers for GeneIDs/accessions. Candidate CDS sequences
are ranked deterministically by:

1. longest sequence;
2. lowest IUPAC ambiguity count;
3. lexicographically smallest CDS ID.

The per-taxon driver launches mapper subprocesses concurrently and records a
status table, per-taxon logs, coverage, unmatched genes, and summaries. It uses
argument arrays rather than a shell command string.

Primary outputs per taxon are representative FASTA/TSV files, candidate and
unmatched-gene tables, coverage, and `run_summary.tsv`.

## Stage 3: CDS Length Filtering

Primary code:

```text
src/evo2_go/embeddings/length_filter.py
```

The filter keeps representative CDS sequences with length at most 8,192 and
writes separate kept/dropped FASTA and metadata tables. It checks that FASTA
and metadata counts agree and aggregates per-taxon summaries.

Current full dataset result:

| Metric | Value |
|---|---:|
| Input representative CDS records | 31,180,045 |
| Kept records | 30,921,763 |
| Dropped records | 258,282 (0.83%) |
| Dropped sequence bases/tokens | 3,176,404,801 (5.89%) |
| Taxa processed | 2,342 |

The implementation treats sequence length as token count. That matches the
current character-level Evo2 tokenizer behavior but is an implicit contract
that should be tested rather than assumed.

## Stage 4: Evo2 Embedding Extraction

Primary code and launcher:

```text
src/evo2_go/embeddings/run_embeddings.py
scripts/run_embeddings_flux.sh
```

Each spawned worker is isolated to one GPU and loads one Evo2 model instance.
The current defaults use:

```text
model: evo2_7b_base
checkpoint: evo2_model/evo2_7b_base.pt
activation layer: blocks.28.mlp.l3
expected model max sequence length: 32,768
pipeline input cutoff: 8,192
```

Sequences are bucketed by length and padded within each batch. The selected
activation is mean-pooled across each unpadded sequence and stored as a
4,096-dimensional FP16 vector. Each shard is a Torch payload containing pooled
vectors and record provenance.

All embedding jobs for one dataset reuse a shared root:

```text
embedding_root/
  embedding_dataset_manifest.tsv
  embedding_metadata.tsv.gz
  batch_metrics.tsv
  tax_<tax_id>/
    embedding_metadata.tsv.gz
    batch_metrics.tsv
    run_summary.tsv
    embedding_shards/bucket_*/batch_*.pt
```

Per-batch sidecars support resume. Completed shards are skipped, while root and
per-taxon metadata are rebuilt from sidecars. Current cleaned state is 2,336
complete taxa and 30,921,763 embedded records; six taxa have zero retained
input records.

`src/evo2_go/embeddings/validate_embeddings.py` checks metadata identities,
shard dimensions and row indices, optional finiteness, the kept-CDS join, and
optional label-cache coverage.

## Stage 5: Embedding Integration And Repair

Primary repair code:

```text
scripts/cleanup_embedding_duplicate_metadata.py
```

The repair utility treats the current length-filtered kept-CDS tables as the
source of truth. It reconstructs full CDS IDs, removes duplicate/stale metadata
rows, refuses a rewrite when a kept record lacks an embedding, and can rebuild
root metadata and manifests. It is audit-only unless `--execute` is supplied.

This repair was necessary because concurrent embedding jobs write disjoint
per-taxon shards but share root aggregate files. Root aggregates are therefore
coordination artifacts, not safe concurrent transaction logs.

## Stage 6: GO Labels, Vocabulary, And Graph

Primary code:

```text
src/evo2_go/database/go_term_counts.py
src/evo2_go/database/true_path_labels.py
src/evo2_go/ontology/go_adjacency.py
src/evo2_go/workflows/prepare_training_inputs.py
```

The final cleaned build uses all positive Gene2GO evidence codes, including
IEA. Explicit labels are expanded through `is_a` and `part_of` ancestors in
`go-basic.obo`. A deterministic GO vocabulary retains terms observed on at
least 15,000 selected genes. The label cache maps `(tax_id, GeneID)` to sorted
vocabulary indices.

The graph builder adds self-loops and bidirectional `is_a`/`part_of` edges,
then computes `D^-1/2 A D^-1/2` as a SciPy sparse matrix.

Current final artifacts contain:

| Metric | Value |
|---|---:|
| Selected genes | 30,921,763 |
| Explicit unique gene-GO pairs | 116,854,737 |
| True-path unique gene-GO pairs | 931,265,931 |
| Vocabulary-aligned cached pairs | 894,696,809 |
| Vocabulary terms | 3,384 |
| Normalized adjacency nonzeros | 14,238 |
| Vocabulary terms missing from current OBO | 1 (`GO:0008139`) |

## Stage 7: Sequence-Similarity Split

Primary code and config:

```text
src/evo2_go/workflows/mmseqs_cluster_split.py
configs/training_inputs.mmseqs_linclust_full_85_5_10_kmer03_noadj.toml
```

The splitter reconstructs one combined nucleotide FASTA containing exactly the
embedded records, calls MMseqs `createdb`, `linclust`, `createtsv`,
`result2repseq`, and `result2flat`, then validates that every selected record
belongs to exactly one returned cluster.

Current settings are:

```text
minimum sequence identity: 0.70
coverage: 0.50
coverage mode: 1
kmer_per_seq_scale: 0.3
include_adjacency: 0
balance target: records
train/validation/test: 0.85/0.05/0.10
```

Whole clusters are assigned by a deterministic greedy record-balancing
algorithm. The full split contains 6,881,446 clusters and 30,921,763 records:

| Split | Records | Clusters |
|---|---:|---:|
| Train | 26,283,499 | 6,755,537 |
| Validation | 1,546,088 | 25,630 |
| Test | 3,092,176 | 100,279 |

The split proves that one MMseqs cluster is never divided across splits. It
does not prove that no cross-split pair exceeds the configured similarity
threshold; `linclust` is heuristic and adjacency was disabled to avoid an
MMseqs crash. The summary correctly records `ortholog_safe = 0`.

The older `taxon_holdout_split.py` remains available for frozen historical
experiments, but the final run uses the MMseqs cluster split.

## Stage 8: Dataset Assembly

Primary code:

```text
src/evo2_go/model/dataset.py
```

Dataset construction loads complete per-taxon embedding metadata, validates
unique composite identities, joins the split manifest, loads the GO label
cache, and drops examples without retained labels. Shards are loaded lazily
with a small per-process LRU cache. The collator produces:

```text
embeddings: float32 [batch, 4096]
targets:    dense float32 [batch, n_terms]
```

For the active run, 26,283,451 training records remain after label filtering,
along with all 1,546,088 validation records and 3,384 target terms.

## Stage 9: ML Model

Primary code:

```text
src/evo2_go/model/architecture.py
src/evo2_go/model/training_utils.py
```

The sequence branch is:

```text
4096 -> LayerNorm -> Linear(2048) -> GELU -> Dropout(0.1)
     -> LayerNorm -> Linear(1024) -> Dropout(0.1) -> L2 normalization
```

The GO branch begins with a learned 256-dimensional vector per GO term and
performs two sparse graph-convolution/message-passing steps:

```text
3384 x 256 -> sparse adjacency -> Linear(512) -> LayerNorm -> GELU
           -> sparse adjacency -> Linear(1024) -> LayerNorm -> L2 normalization
```

The model scores every sequence against every GO term using normalized dot
products multiplied by a learned, clamped logit scale. The current model has
12,025,857 trainable parameters.

Training uses asymmetric multi-label loss. The final config selects dense mean
reduction, AdamW, global gradient clipping at 1.0, 2,000 warmup steps, and
cosine learning-rate decay from a peak of `3e-4`.

## Stage 10: Training And Run Tracking

Primary code, launcher, and current config:

```text
src/evo2_go/training/train.py
scripts/train_model_flux.sh
configs/training_run.full_cleaned_min_count_15000_mmseqs_85_5_10_16h.toml
```

Training writes:

```text
training_run_metadata.tsv
training_metrics.tsv
training_run_summary.tsv
checkpoint_step_*.pt
latest_checkpoint.pt
best_checkpoint.pt
train_model.out
train_model.err
```

Metrics include train loss and its positive/negative components, learning
rate, gradient norm, positive-label density, positive/negative probabilities,
logit scale, examples seen, effective epochs, and validation Fmax/threshold/
precision/recall.

`evo2-go validation-panel` can now create a persistent, seeded validation
sample by selecting the smallest SHA-256 identity hashes. Training accepts the
saved panel with `validation_batches = 0`, records its SHA-256 in run metadata,
and can reset inherited best-checkpoint state when the validation protocol
changes. Panel records are selected while embedding metadata is streamed, and
the label cache is filtered to selected genes, avoiding a second full
validation dataset materialization.

Checkpoints include model, optimizer, scheduler, step, best Fmax, and metadata.
They are loaded with `weights_only=True` and a narrowly allowed Torch version
metadata type.

The code is single-device. Although the Flux launcher currently requests four
GPUs, `device=auto` selects one CUDA/ROCm device and there is no DDP or
`DataParallel` implementation.

## Stage 11: Evaluation

Primary code:

```text
src/evo2_go/training/evaluate.py
```

Evaluation defaults to `best_checkpoint.pt`, loads the configured split, and
computes CAFA-style gene-averaged Fmax over a threshold grid. Optional
publication outputs include:

- gene-averaged and micro threshold curves;
- per-gene precision, recall, F1, AUROC, and average precision;
- per-GO-term global-threshold and per-term-optimal metrics;
- probability calibration summaries;
- a reusable probability/target score matrix.

The reported Fmax is the maximum over thresholds using the evaluated split.
That is appropriate as a descriptive CAFA-style Fmax statistic, but the
resulting threshold must not be presented as an independently selected
deployment threshold when the evaluated split is the test set.

## Stage 12: Analysis And Figures

General analysis is implemented in `src/evo2_go/training/visualize.py`. It
reads saved run artifacts rather than rerunning inference and produces:

- split-size, training, loss, and validation diagnostics;
- test threshold curves;
- per-term frequency, namespace, and GO-depth views;
- gene annotation-count performance;
- GO benchmark tables and comparison plots.

Publication-specific scripts under `scripts/publication/` reproduce the final
1919-run style:

| Script | Intended outputs |
|---|---|
| `figure_01_threshold_curve.py` | Figure 1 threshold sweep. |
| `figure_02_go_term_frequency.py` | Figure 2 frequency/performance panels. |
| `create_publication_benchmark_figures.py` | Figures 3, 4, 5, 6, 8, and 9A-C. |
| `figure_10_dataset_label_composition.py` | Figure 10 label composition. |
| `figure_11_top_predicted_semantic_rollup.py` | Experimental semantic treemap; not accepted as a final figure. |

Figure 7 and the older single Figure 9 are retired but stale files remain in
the historical publication directory. Current publication scripts contain
1919-taxon and human-test wording that must be parameterized before applying
them to the MMseqs cluster test set.

## Workflow Orchestration

`build_database.py` / `workflows/build_database.py` orchestrate:

```text
map_cds_by_tax -> length_filter -> run_embeddings
               -> true_path_counts -> go_adjacency -> label_cache
               -> validate_embeddings
```

`prepare_training_inputs.py` / `workflows/prepare_training_inputs.py` rerun the
post-embedding portion and add split generation:

```text
true_path_counts -> go_adjacency -> label_cache
                 -> split_manifest -> validate_embeddings
```

Both workflows resolve TOML paths against a project root, log commands and
step status, validate prerequisites, and execute subprocesses without
`shell=True`. Individual commands remain independently runnable.

## Vulnerability And Risk Review

This is a research-code audit, not a formal penetration test. No embedded
passwords, API keys, access tokens, or network client code were found. Most
risks are scientific validity, data integrity, memory scaling, and
reproducibility rather than remote exploitation.

### High Priority

| Risk | Evidence and impact | Recommended action |
|---|---|---|
| GO vocabulary selection sees validation/test labels | `training_inputs.full_cleaned_min_count_15000.toml` builds term counts from all 30,921,763 embeddings before applying the MMseqs split. Test labels can influence which output heads survive `min_count=15000` and the recorded training frequencies. | Build vocabulary counts from training-split genes only, then create label caches for all splits using that fixed training vocabulary. Quantify whether the vocabulary changes. |
| Full-test evaluation is dense and in-memory | The test matrix has `3,092,176 x 3,384 = 10,463,923,584` scores: about 39 GiB as FP32. The evaluator simultaneously collects FP32 targets, creates large boolean matrices for each threshold, computes row/column rankings, and may write a very large compressed NPZ. | Refactor evaluation to streaming/chunked accumulators and disk-backed score storage. Compute per-gene rows by batch and per-term statistics in chunks. |
| Training dataset construction duplicates huge Python structures | Train and validation are loaded independently, each scanning embedding metadata and the full label cache. Persistent DataLoader workers replicate/cache substantial state. Two current attempts were killed near 476.87 GB. | Build one shared indexed metadata/label store, use memory mapping or compact arrays, avoid Python dict/list duplication, and size workers from measured RSS. |
| Validation protocol changed during the historical run | Steps 5k-20k used all 1,546,088 validation examples; resumed attempts used the same first 60 non-shuffled batches (15,360 examples, about 0.99%). Its `best_checkpoint.pt` therefore compares incompatible protocols. | The new `validation-panel` command and `reset_best_on_resume` option address future comparisons. Resume into a new run directory with the saved panel and re-evaluate candidate checkpoints consistently. |
| MMseqs split has not received a cross-split search audit | Whole clusters do not cross splits, but `linclust` is heuristic and `include_adjacency=0` can reduce sensitivity. | Run a post hoc train-vs-validation/test MMseqs search at the claimed threshold before making a no-similarity-leakage claim. Keep `ortholog_safe=0` until validated. |
| Minimal automated tests | Deterministic validation-panel selection and metadata preselection now have focused tests, but most parsing, resume behavior, split assignment, metrics, and figure semantics remain uncovered. | Continue adding small fixture-based unit and integration tests plus CI for compile, CLI smoke tests, and deterministic artifact checks. |

### Data Integrity And Operational Risks

| Risk | Evidence and impact | Recommended action |
|---|---|---|
| Shared embedding aggregates are last-writer-wins | Concurrent jobs write common root manifests, metadata, summaries, reports, and worker logs. This already produced duplicate/stale metadata requiring repair. | Make aggregate rebuild a separately locked/finalization step. Keep job-scoped worker logs and write only per-taxon artifacts during extraction. |
| A partial taxon can be marked complete | Extraction declares a taxon complete when discovered sidecars are successful, without comparing embedded identity/count coverage to the full kept-CDS source. Record/bucket limits can therefore produce a successful partial taxon. | Require an expected identity/count manifest and exact coverage before `status=complete`. |
| Resume metrics contain overlapping steps | After an OOM, the run resumes from the latest 5,000-step checkpoint while post-checkpoint metric rows remain. `training_metrics.tsv` can therefore contain duplicated step ranges and reset elapsed times. | Add `attempt_id`, wall-clock timestamps, checkpoint source, and monotonic global elapsed time; either retain attempt-specific files or deduplicate the consolidated view. |
| Retry logs and metadata are overwritten | `train_model.out`, `train_model.err`, and `training_run_metadata.tsv` are rewritten in one run directory. Flux did not expose `FLUX_JOB_ID`, so multiple bootstrap logs used `manual.log`. | Use an explicit attempt directory or detected scheduler/job identifier and keep an immutable run-attempt manifest. |
| Training asks for four GPUs but uses one | The current model is moved to `cuda` without distributed wrapping. Three allocated GPUs are idle. | Request one GPU until DDP is implemented, or implement and validate distributed training. |
| Split summary reports stale label coverage | The full MMseqs split config passed the old 1919-taxon label cache for reporting. Split membership is correct, but `*_records_with_retained_labels` in its summary does not describe the final label cache. | Regenerate only the labeled-count summary with the final cache or clearly mark those fields stale. |
| Cleanup writes without backups by default | The deduplication utility is audit-only by default, but `--execute` replaces many metadata files and its default backup suffix is empty. | Require or strongly default to a timestamped backup suffix for full-root repair. |

### Reproducibility And Scientific Interpretation Risks

| Risk | Evidence and impact | Recommended action |
|---|---|---|
| Training RNG is not controlled | No Python, NumPy, Torch, DataLoader generator, or deterministic-algorithm seed is recorded or set. | Add a required run seed, seed every library/worker, and record deterministic settings and backend flags. |
| Environment and inputs are not cryptographically pinned | Dependencies have no exact versions; model, OBO, annotation, split, and embedding artifacts have paths/sizes but generally no hashes. | Record package lock files, module versions, Git revision, and SHA-256 hashes for scientific inputs and checkpoints. |
| Architecture is mostly hardcoded | Sequence/GCN dimensions, dropout, loss gamma/clip, scheduler final fraction, and gradient clip are not exposed through the run TOML. | Promote architecture and optimizer/loss settings into validated config tables and record the resolved values. |
| IEA dominates the active explicit annotation stream | About 115.98 million of 117.61 million retained annotation rows are IEA. Results may largely reproduce computational annotation patterns rather than experimentally supported biology. | Report evidence composition prominently and compare against a non-IEA or evidence-stratified evaluation. |
| Test-set threshold optimization can be misread | Fmax and per-term optimal Fmax use test labels to select thresholds. These are descriptive oracle metrics, not deployment-threshold estimates. | Select any operational threshold on validation; label test-swept Fmax and per-term Fmax explicitly. |
| Historical names and captions can mislabel the final experiment | Active shared roots still contain `42taxa` in their name, and publication scripts hardcode `human`, `1919`, and taxon-holdout language. | Parameterize dataset/split labels and generate captions from run metadata. |
| One retained GO term is absent from the current OBO | `GO:0008139` receives no ontology metadata or graph edges beyond its self-loop. | Resolve replacement/obsolescence from a versioned ontology or disclose the isolated output head. |

### Security-Specific Notes

- Subprocesses are invoked with argument arrays and no `shell=True`; this
  reduces command-injection exposure.
- Checkpoints use `torch.load(..., weights_only=True)` with a narrow safe type.
- Embedding shards prefer `weights_only=True`, but the compatibility fallback
  calls unrestricted `torch.load` on older Torch versions. Never load an
  untrusted `.pt` shard in that environment.
- Publication NPZ files are loaded with `allow_pickle=False`.
- Workflow paths and executable/venv environment overrides assume trusted
  local users. An untrusted config or environment can select arbitrary local
  executables and output paths.
- Most table outputs use temporary files followed by atomic replacement, which
  reduces partial-file exposure but is not a substitute for inter-job locking.

## Recommended Order Of Remediation

1. Preserve the current run and its immutable artifacts; do not rewrite them
   while training is active.
2. Implement streaming/chunked evaluation before full MMseqs test evaluation.
3. Rebuild and compare a training-only GO vocabulary to quantify label-space
   leakage.
4. Use the fixed validation panel and recompute full validation for candidate
   checkpoints.
5. Run post hoc cross-split similarity validation.
6. Fix attempt-level logging, RNG capture, and dependency/input hashing.
7. Add fixture-based tests around each stable data contract.
8. Parameterize the publication scripts for the MMseqs test set and remove
   stale human/1919 wording.
