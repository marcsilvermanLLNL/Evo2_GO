#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p pdebug
#SBATCH -J evo2-go-viz-live12000

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/visualize_live12000.slurm.sh [visualize-results args]

Generates figures for the live 1342-taxon min_count=12000 human-test run.

Environment overrides:
  PROJECT_ROOT        Remodel project root. Default: submit dir if it is the remodel tree.
  EVO2_GO_VIZ_PYTHON  Python executable. Default: python3
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

submit_dir="$(pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_parent="$(cd "${script_dir}/.." && pwd)"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${submit_dir}/src/evo2_go/cli.py" ]]; then
  PROJECT_ROOT="${submit_dir}"
else
  PROJECT_ROOT="${script_parent}"
fi

cd "${PROJECT_ROOT}"

viz_python="${EVO2_GO_VIZ_PYTHON:-python3}"
run_dir="outputs/builds/all_labels_true_path_live_human_test_min_count_12000_v1/runs/human_test_taxon_val_live_1342taxa_min_count_12000_bs256_mean_loss_3h"
figure_folder="human_test_taxon_val_live_1342taxa_min_count_12000_bs256_mean_loss_3h"
output_root="figures"

if [[ ! -d "${run_dir}" ]]; then
  echo "Missing run directory: ${run_dir}" >&2
  exit 1
fi
if [[ ! -f "${run_dir}/test_eval/test_evaluation_summary.tsv" ]]; then
  echo "Missing test evaluation summary: ${run_dir}/test_eval/test_evaluation_summary.tsv" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

log_dir="${PROJECT_ROOT}/outputs/logs/visualization"
mkdir -p "${log_dir}"
log_file="${log_dir}/visualize_live12000_${SLURM_JOB_ID:-manual}.log"

exec > "${log_file}"
exec 2>&1

echo "Project root: ${PROJECT_ROOT}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Python: ${viz_python}"
echo "Run dir: ${run_dir}"
echo "Figure folder: ${figure_folder}"
echo "Output root: ${output_root}"

"${viz_python}" -m evo2_go.cli visualize-results \
  --run-dir "${run_dir}" \
  --figure-folder "${figure_folder}" \
  --output-root "${output_root}" \
  "$@"

echo "Finished: $(date --iso-8601=seconds)"
