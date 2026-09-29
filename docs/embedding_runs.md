# Embedding Runs

Run embedding jobs from a Flux allocation or login environment that can submit
to Flux. The API shell cannot submit Flux jobs in the current environment.

## Required Local Inputs

The remodel tree should contain:

```text
data/go/
outputs/process_Gene2GO/processed/final/
outputs/map_gene_cds/by_tax/
outputs/map_gene_cds/length_filtered_max_8192/
outputs/map_gene_cds/length_filtered_max_8192_non_iea/
outputs/builds/all_labels_true_path_42taxa_human_test_min_count_100_v1/
evo2_model/evo2_7b_base.pt
```

Raw RefSeq CDS under `data/refseq_cds/` is only required when rerunning
representative CDS mapping from downloaded FASTA files.

## Shared Output Root

All embedding jobs for one build write to:

```text
outputs/builds/<build_id>/data/embeddings
```

Do not create per-run embedding artifact roots. Reusing the same root is how
completed taxa and completed batch shards are skipped.

## Flux Wrapper

Use the reusable Flux payload script:

```bash
scripts/run_embeddings_flux.sh --help
```

Default behavior:

- Positional tax IDs or repeated `--tax-id` values run only those taxa.
- `--tax-list <file>` runs taxa from a file.
- No tax IDs and no `--tax-list` enters queue mode: the script repeatedly
  selects the next incomplete taxon, one taxon per Python process by default,
  until no runnable taxa remain or the Flux allocation ends.
- Defaults point at the active remodel build, shared embedding root, local Evo2
  checkpoint, four GPUs, and the current bucket batch sizes.

## Single-Taxon Probe

The current small probe taxon is `881290`. It is not in the 42 complete taxa
and has length-filtered input in the remodel tree.

```bash
cd /usr/workspace/silverman2/LLNLagent/Evo2-GO-Remodel

flux batch -q pdebug -t 30m -N 1 -n 1 -c 16 -g 4 \
  --job-name evo2-tax-881290 \
  scripts/run_embeddings_flux.sh 881290
```

Suggested Flux resources for a probe:

```text
queue: pdebug
nodes: 1
tasks: 1
cores: 16
gpus: 4
time: 30m
```

The wrapper loads the default ROCm module and activates the default Evo2 venv:

```text
rocm/6.4
/p/vast1/ncpbr/evo2_tuo/venv/evo2_tuo
```

Override those only if the site environment changes:

```bash
EVO2_ROCM_MODULE=rocm/6.4 \
EVO2_VENV=/p/vast1/ncpbr/evo2_tuo/venv/evo2_tuo \
flux batch -q pdebug -t 30m -N 1 -n 1 -c 16 -g 4 \
  --job-name evo2-tax-881290 \
  scripts/run_embeddings_flux.sh 881290
```

## Queue Mode

To keep taking incomplete taxa until the allocation ends:

```bash
cd /usr/workspace/silverman2/LLNLagent/Evo2-GO-Remodel

flux batch -q pbatch -t 12h -N 1 -n 1 -c 16 -g 4 \
  --job-name evo2-embed-queue \
  scripts/run_embeddings_flux.sh
```

The queue wrapper defaults to one taxon per Python process. Change that with
`--taxa-per-process N` only after the small probe behaves as expected.
