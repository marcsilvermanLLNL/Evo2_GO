#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p pdebug
#SBATCH -J evo2-go-eval-10000

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/evaluate_model_10000.slurm.sh [evaluate-model args]

Evaluates the frozen 813-taxon min_count=10000 run on the configured test split.

Common examples:
  sbatch scripts/evaluate_model_10000.slurm.sh
  sbatch -p pbatch -t 02:00:00 scripts/evaluate_model_10000.slurm.sh
  sbatch scripts/evaluate_model_10000.slurm.sh --device cpu

Environment overrides:
  PROJECT_ROOT          Remodel project root. Default: submit dir if it is the remodel tree.
  EVO2_GO_EVAL_CONFIG   Config path. Default: configs/training_run.813taxa_min_count_10000_frozen813.toml
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
config="${EVO2_GO_EVAL_CONFIG:-configs/training_run.813taxa_min_count_10000_frozen813.toml}"
split_name="${EVO2_GO_EVAL_SPLIT:-test}"

if [[ ! -f "${config}" ]]; then
  echo "Missing evaluation config: ${config}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --config "${config}"
  --project-root "${PROJECT_ROOT}"
  --split-name "${split_name}"
)
args+=("$@")

bootstrap_log_dir="${PROJECT_ROOT}/outputs/logs/evaluation_bootstrap"
mkdir -p "${bootstrap_log_dir}"
bootstrap_log="${bootstrap_log_dir}/evaluate_model_10000_${SLURM_JOB_ID:-manual}.log"

{
  echo "Project root: ${PROJECT_ROOT}"
  echo "Script path: ${BASH_SOURCE[0]}"
  echo "Host: $(hostname)"
  echo "Started bootstrap: $(date --iso-8601=seconds)"
  echo "Python: ${eval_python}"
  echo "PYTHONPATH: ${PYTHONPATH}"
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
echo "Command: ${eval_python} -m evo2_go.cli evaluate-model ${args[*]}"

"${eval_python}" - <<'PY'
import torch
print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
PY

"${eval_python}" -m evo2_go.cli evaluate-model "${args[@]}"

echo "Finished: $(date --iso-8601=seconds)"
