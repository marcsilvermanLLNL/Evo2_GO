#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p pbatch
#SBATCH -J evo2-go-test-stream
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/evaluate_full_test_streaming.slurm.sh [evaluate-model args]

Runs bounded-memory, two-pass inference on the full MMseqs test split using the
best checkpoint from the completed fixed-validation-panel run. The evaluator
does not write the monolithic score matrix that would exceed available memory.

Environment overrides:
  PROJECT_ROOT          Remodel project root. Default: submit dir or script parent.
  EVO2_GO_EVAL_CONFIG   Evaluation config. Default: the fixed-val50k-v2 config.
  EVO2_GO_EVAL_PYTHON   Python executable. Default: python3.

Extra evaluate-model arguments are appended last and can override defaults.
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
  project_root="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${submit_dir}/src/evo2_go/cli.py" ]]; then
  project_root="${submit_dir}"
else
  project_root="${script_parent}"
fi
cd "${project_root}"

eval_python="${EVO2_GO_EVAL_PYTHON:-python3}"
config="${EVO2_GO_EVAL_CONFIG:-configs/training_run.full_cleaned_min_count_15000_mmseqs_85_5_10_16h_fixed_val50k_v2.toml}"

if [[ ! -f "${config}" ]]; then
  echo "Missing evaluation config: ${config}" >&2
  exit 1
fi

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

args=(
  --config "${config}"
  --project-root "${project_root}"
  --split-name test
  --streaming
  --exclude-go-roots
  --no-save-score-matrix
  --publication-metrics
  --num-workers 2
  --progress-every-batches 250
)
args+=("$@")

bootstrap_log_dir="${project_root}/outputs/logs/evaluation_bootstrap"
mkdir -p "${bootstrap_log_dir}"
bootstrap_log="${bootstrap_log_dir}/evaluate_full_test_streaming_${SLURM_JOB_ID:-manual}.log"

{
  echo "Project root: ${project_root}"
  echo "Script path: ${BASH_SOURCE[0]}"
  echo "Host: $(hostname)"
  echo "Started bootstrap: $(date --iso-8601=seconds)"
  echo "Python: ${eval_python}"
  echo "PYTHONPATH: ${PYTHONPATH}"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
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
checkpoint="$(printf '%s\n' "${dry_run_output}" | awk -F '\t' '$1 == "checkpoint" {print $2}')"
if [[ -z "${output_dir}" ]]; then
  echo "Could not resolve evaluation output_dir; see ${bootstrap_log}" >&2
  exit 1
fi
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "Missing resolved checkpoint: ${checkpoint:-<empty>}; see ${bootstrap_log}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
exec > "${output_dir}/evaluate_model.out"
exec 2> "${output_dir}/evaluate_model.err"

echo "Project root: ${project_root}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Checkpoint: ${checkpoint}"
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
