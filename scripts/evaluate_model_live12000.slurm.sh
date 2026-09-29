#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p pdebug
#SBATCH -J evo2-go-eval-live12000

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/evaluate_model_live12000.slurm.sh [evaluate-model args]

Evaluates the live 1342-taxon min_count=12000 run on the configured test split.
The script restricts evaluation to taxa present in the frozen split manifest so
newer embeddings in the shared live embedding root do not trigger missing split
assignment errors.

Environment overrides:
  PROJECT_ROOT          Remodel project root. Default: submit dir if it is the remodel tree.
  EVO2_GO_EVAL_CONFIG   Config path. Default: configs/training_run.live_1342taxa_min_count_12000_3h.toml
  EVO2_GO_EVAL_SPLIT    Split name. Default: test
  EVO2_GO_EVAL_PYTHON   Python executable. Default: python3
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

eval_python="${EVO2_GO_EVAL_PYTHON:-python3}"
config="${EVO2_GO_EVAL_CONFIG:-configs/training_run.live_1342taxa_min_count_12000_3h.toml}"
split_name="${EVO2_GO_EVAL_SPLIT:-test}"
split_manifest="outputs/builds/all_labels_true_path_live_human_test_min_count_12000_v1/data/splits/human_test_taxon_val_manifest.tsv"

if [[ ! -f "${config}" ]]; then
  echo "Missing evaluation config: ${config}" >&2
  exit 1
fi
if [[ ! -f "${split_manifest}" ]]; then
  echo "Missing split manifest: ${split_manifest}" >&2
  exit 1
fi

tax_ids="$(
  awk -F '\t' 'NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next } { print $h["tax_id"] }' \
    "${split_manifest}" \
    | sort -u \
    | paste -sd, -
)"
if [[ -z "${tax_ids}" ]]; then
  echo "Could not derive tax IDs from ${split_manifest}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --config "${config}"
  --project-root "${PROJECT_ROOT}"
  --split-name "${split_name}"
  --tax-id "${tax_ids}"
)
args+=("$@")

bootstrap_log_dir="${PROJECT_ROOT}/outputs/logs/evaluation_bootstrap"
mkdir -p "${bootstrap_log_dir}"
bootstrap_log="${bootstrap_log_dir}/evaluate_model_live12000_${SLURM_JOB_ID:-manual}.log"

{
  echo "Project root: ${PROJECT_ROOT}"
  echo "Script path: ${BASH_SOURCE[0]}"
  echo "Host: $(hostname)"
  echo "Started bootstrap: $(date --iso-8601=seconds)"
  echo "Python: ${eval_python}"
  echo "PYTHONPATH: ${PYTHONPATH}"
  echo "Config: ${config}"
  echo "Split manifest: ${split_manifest}"
  echo "Taxa selected: $(tr ',' '\n' <<< "${tax_ids}" | wc -l)"
  echo "Args: ${args[*]}"
} > "${bootstrap_log}"

dry_run_output="$(
  "${eval_python}" -m evo2_go.cli evaluate-model \
    --dry-run \
    "${args[@]}" \
    2>> "${bootstrap_log}"
)"
printf '%s\n' "${dry_run_output}" >> "${bootstrap_log}"

output_dir="$(printf '%s\n' "${dry_run_output}" | awk -F '\t' '$1 == "output_dir" {print $2}')"
if [[ -z "${output_dir}" ]]; then
  echo "Could not resolve evaluation output_dir from config; see ${bootstrap_log}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
exec > "${output_dir}/evaluate_model.out"
exec 2> "${output_dir}/evaluate_model.err"

echo "Project root: ${PROJECT_ROOT}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Bootstrap log: ${bootstrap_log}"
echo "Taxa selected: $(tr ',' '\n' <<< "${tax_ids}" | wc -l)"
echo "Command: ${eval_python} -m evo2_go.cli evaluate-model ${args[*]}"

"${eval_python}" - <<'PY'
import torch
print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
PY

"${eval_python}" -m evo2_go.cli evaluate-model "${args[@]}"

echo "Finished: $(date --iso-8601=seconds)"
