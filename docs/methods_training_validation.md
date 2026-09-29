# Training Validation Protocol: Methods Record

This is the reproducibility record for validation and checkpoint selection in
the final full-cleaned MMseqs experiment. It was added on 2026-08-18 while the
fixed-panel continuation was starting. It distinguishes the final selection
protocol from earlier engineering validation runs so the paper does not imply
that one validation protocol was used throughout.

## Final Checkpoint-Selection Protocol

The source validation split contains 1,546,088 CDS records in 25,630 MMseqs2
clusters. A fixed panel of 50,000 records (3.23% of the validation records) was
sampled without replacement. Each record was assigned a deterministic priority
from the first unsigned 64 bits of:

```text
SHA-256(seed + "\t" + tax_id + "\t" + GeneID + "\t" + cds_id)
```

The 50,000 records with the smallest priorities were retained. This procedure
is uniform with respect to record identity, deterministic, and independent of
the source manifest's row order. The saved panel contains 50,000 unique
identities spanning 2,069 taxa and 4,355 MMseqs clusters.

Reproducibility values:

| Item | Value |
|---|---|
| Source validation records | 1,546,088 |
| Saved panel records | 50,000 |
| Sampling seed | `evo2-go-mmseqs-val-panel-50000-v1` |
| Selection method | Smallest seeded SHA-256 64-bit identity priorities |
| Panel SHA-256 | `4f20d0a090b7b7cb8e4e6a41eca466efbcacd1f834cd004fc30e7c881d9682a3` |
| Validation frequency | Every 5,000 optimizer steps |
| Validation batch size | 256 |
| Batches per panel evaluation | 196, including one final partial batch |
| Training Fmax thresholds | 50 values from 0.01 through 0.99 |

The panel is stored as a normal split manifest and is consumed in full with
`validation_batches = 0`. It is never reshuffled, rotated, or resampled between
checkpoints. Its path and SHA-256 are recorded in training-run metadata.

## Resume Boundary And Historical Validation

The final fixed-panel run resumes model, optimizer, and learning-rate scheduler
state from the old run's `latest_checkpoint.pt` at optimizer step 120,000. The
old checkpoint's best-Fmax state is explicitly discarded with
`reset_best_on_resume = true`, and the continuation writes to a new run
directory. Consequently, final checkpoint ranking begins anew under the fixed
panel.

Earlier validation measurements must not be pooled with the fixed-panel
measurements:

| Optimizer steps | Historical validation protocol |
|---|---|
| 5,000-20,000 | Complete 1,546,088-record validation split |
| 25,000-120,000 | Same first 60 non-shuffled batches: 15,360 records |
| After the step-120,000 resume | Complete saved 50,000-record fixed panel |

The validation protocol did not contribute to the loss or gradient updates.
Therefore the step-120,000 optimization state remains usable, but historical
best-checkpoint decisions are not comparable with the new fixed-panel ranking.
Before publication, the leading fixed-panel checkpoint or small set of
candidate checkpoints should be evaluated once on the complete validation
split using streaming metrics. Any operational probability threshold should be
chosen from validation, and the test split should remain reserved for final
evaluation.

## Paper-Ready Draft Language

> For final checkpoint selection, we constructed a fixed validation panel of
> 50,000 CDS records sampled without replacement from the 1,546,088-record
> MMseqs2 validation split. Sampling was deterministic and independent of file
> order: records were ranked by a seeded SHA-256 hash of the composite
> `(tax_id, GeneID, cds_id)` identity, and the 50,000 lowest-ranking records
> were retained. The same complete panel was evaluated every 5,000 optimizer
> steps using CAFA-style gene-averaged Fmax over 50 probability thresholds from
> 0.01 to 0.99. Training was resumed from optimizer step 120,000 after an
> engineering validation-protocol revision; model, optimizer, and scheduler
> state were retained, whereas previous best-checkpoint statistics were reset.
> Validation predictions were used only for monitoring and checkpoint
> selection and did not contribute to gradient updates. Final candidate
> checkpoints were subsequently assessed on the complete validation split
> before held-out test evaluation.

The last sentence describes the required final full-validation confirmation;
do not retain it in the manuscript until that evaluation has actually run.

## Planned Checkpoint Sensitivity Analysis

On 2026-08-19, evaluate historical checkpoints from steps 25,000, 50,000,
75,000, 100,000, and 120,000 on this same fixed panel. Include every 5,000-step
checkpoint if practical. This will determine whether the step-120,000 resume
state was already overfit. An older winning checkpoint can be selected or used
as a resume point without retraining from scratch because validation was not
part of gradient computation.

Full-validation confirmation requires a streaming evaluator before publication
because the current dense implementation is not memory-safe at this scale.

## Artifacts

```text
outputs/builds/mmseqs_linclust_full_85_5_10_kmer03_noadj_v1/data/splits/mmseqs_linclust_full_85_5_10_70id_50cov_kmer03_noadj_validation_panel_50000_v1.tsv
outputs/builds/mmseqs_linclust_full_85_5_10_kmer03_noadj_v1/data/splits/mmseqs_linclust_full_85_5_10_70id_50cov_kmer03_noadj_validation_panel_50000_v1_summary.tsv
configs/training_run.full_cleaned_min_count_15000_mmseqs_85_5_10_16h_fixed_val50k_v2.toml
```
