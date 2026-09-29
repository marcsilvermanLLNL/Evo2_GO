#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p pbatch
#SBATCH -J e2go-final-figures
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/create_final_publication_figures.slurm.sh
  bash scripts/create_final_publication_figures.slurm.sh --dry-run

Creates publication Figures 1-6, 9A-C, and 10 from the completed full-MMseqs
streaming evaluation. Figures 8 and 11 are intentionally omitted because they
require the monolithic score matrix that streaming evaluation does not write.

The three GO ontology roots are excluded throughout the performance figures
and from the Figure 10 label-composition scan:
  GO:0003674  molecular function (MF)
  GO:0005575  cellular component (CC)
  GO:0008150  biological process (BP)

Environment overrides:
  PROJECT_ROOT                 Remodel project root.
  EVO2_GO_FIGURE_PYTHON        Python executable. Default: python3.
  EVO2_GO_FIGURE_OUTPUT_DIR    Final figure directory.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_parent="$(cd "${script_dir}/.." && pwd)"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_root="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${submit_dir}/src/evo2_go/cli.py" ]]; then
  project_root="${submit_dir}"
elif [[ -f "${script_parent}/src/evo2_go/cli.py" ]]; then
  project_root="${script_parent}"
else
  echo "Could not resolve the Evo2-GO-Remodel project root." >&2
  echo "Submit from the repository root or set PROJECT_ROOT explicitly." >&2
  exit 1
fi

cd "${project_root}"

python_bin="${EVO2_GO_FIGURE_PYTHON:-python3}"
build_id="all_labels_true_path_full_cleaned_min_count_15000_v1"
run_name="mmseqs_linclust_full_85_5_10_min_count_15000_bs256_mean_loss_fixed_val50k_v2"
run_dir="outputs/builds/${build_id}/runs/${run_name}"
eval_dir="${run_dir}/test_eval_cpu"
output_dir="${EVO2_GO_FIGURE_OUTPUT_DIR:-figures/${run_name}/publication_ready}"
vocabulary="outputs/builds/${build_id}/adjacency/go_vocabulary.tsv"
label_index="outputs/builds/${build_id}/data/labels/label_index.tsv.gz"

required_inputs=(
  "${eval_dir}/test_evaluation_summary.tsv"
  "${eval_dir}/test_threshold_curve.tsv"
  "${eval_dir}/test_publication_summary.tsv"
  "${eval_dir}/test_publication_threshold_metrics.tsv"
  "${eval_dir}/test_publication_gene_metrics.tsv"
  "${eval_dir}/test_publication_go_term_metrics.tsv"
  "${eval_dir}/test_publication_calibration_metrics.tsv"
  "${vocabulary}"
  "${label_index}"
  "data/go/go-basic.obo"
)
for path in "${required_inputs[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing or empty required input: ${path}" >&2
    exit 1
  fi
done

if [[ "${dry_run}" == "1" ]]; then
  echo "Project root: ${project_root}"
  echo "Python: ${python_bin}"
  echo "Evaluation directory: ${eval_dir}"
  echo "Figure output directory: ${output_dir}"
  echo "Required inputs: present"
  echo "Figures: 1-6, 9A-C, and 10"
  echo "GO roots excluded: GO:0003674,GO:0005575,GO:0008150"
  exit 0
fi

mkdir -p "${output_dir}"
log_dir="outputs/logs/publication_figures"
mkdir -p "${log_dir}"
job_id="${SLURM_JOB_ID:-manual}"
exec > "${log_dir}/create_final_publication_figures_${job_id}.out"
exec 2> "${log_dir}/create_final_publication_figures_${job_id}.err"

mpl_root="${TMPDIR:-/tmp}/evo2_go_matplotlib_${job_id}"
mkdir -p "${mpl_root}"
export MPLCONFIGDIR="${mpl_root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Project root: ${project_root}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Python: ${python_bin}"
echo "Evaluation directory: ${eval_dir}"
echo "Figure output directory: ${output_dir}"
echo "GO roots excluded: GO:0003674,GO:0005575,GO:0008150"

"${python_bin}" scripts/publication/figure_01_threshold_curve.py \
  --eval-dir "${eval_dir}" \
  --output-dir "${output_dir}"

"${python_bin}" scripts/publication/figure_02_go_term_frequency.py \
  --eval-dir "${eval_dir}" \
  --output-dir "${output_dir}" \
  --training-universe-label "the full cleaned 30,921,763-sequence dataset used for vocabulary construction"

"${python_bin}" scripts/publication/create_publication_benchmark_figures.py \
  --eval-dir "${eval_dir}" \
  --output-dir "${output_dir}" \
  --skip-namespace-gene-annotation-count

"${python_bin}" scripts/publication/figure_10_dataset_label_composition.py \
  --build-id "${build_id}" \
  --run-name "${run_name}" \
  --go-vocabulary "${vocabulary}" \
  --label-index "${label_index}" \
  --output-dir "${output_dir}" \
  --dataset-label "the full cleaned 30,921,763-sequence dataset across 2,336 taxa" \
  --exclude-go-roots

expected_stems=(
  figure_01_mmseqs_test_threshold_sweep
  figure_02_go_term_frequency_performance
  figure_03_mmseqs_test_roc_pr_curves
  figure_04_probability_calibration
  figure_05_gene_level_discrimination
  figure_06_go_namespace_depth_performance
  figure_09a_molecular_function_depth_occurrence_f1_heatmap
  figure_09b_biological_process_depth_occurrence_f1_heatmap
  figure_09c_cellular_component_depth_occurrence_f1_heatmap
  figure_10_dataset_label_composition
)
for stem in "${expected_stems[@]}"; do
  for extension in png pdf; do
    path="${output_dir}/${stem}.${extension}"
    if [[ ! -s "${path}" ]]; then
      echo "Expected figure output is missing or empty: ${path}" >&2
      exit 1
    fi
  done
  caption="${output_dir}/${stem}_caption.txt"
  if [[ ! -s "${caption}" ]]; then
    echo "Expected figure caption is missing or empty: ${caption}" >&2
    exit 1
  fi
done

echo "Finished: $(date --iso-8601=seconds)"
echo "Figures: ${output_dir}"
