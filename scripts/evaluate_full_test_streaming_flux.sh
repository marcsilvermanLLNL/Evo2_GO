#!/usr/bin/env bash
# flux: -q pbatch
# flux: -N 1
# flux: -n 1
# flux: -c 16
# flux: -g 1
# flux: -t 12h
# flux: --job-name=evo2-go-test-gpu

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  flux batch -t 12h scripts/evaluate_full_test_streaming_flux.sh [evaluate-model args]

Runs bounded-memory, two-pass inference on the full MMseqs test split using one
GPU and the completed fixed-validation-panel run's best checkpoint. CUDA is
required: this launcher exits instead of silently falling back to CPU.

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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_root="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${PWD}/src/evo2_go/cli.py" ]]; then
  project_root="$(pwd)"
elif [[ -f "${script_dir}/../src/evo2_go/cli.py" ]]; then
  project_root="$(cd "${script_dir}/.." && pwd)"
else
  echo "Could not infer project root. Submit from Evo2-GO-Remodel or set PROJECT_ROOT." >&2
  exit 1
fi
cd "${project_root}"

eval_python="${EVO2_GO_EVAL_PYTHON:-python3}"
config="${EVO2_GO_EVAL_CONFIG:-configs/training_run.full_cleaned_min_count_15000_mmseqs_85_5_10_16h_fixed_val50k_v2.toml}"

if [[ ! -f "${config}" ]]; then
  echo "Missing evaluation config: ${config}" >&2
  exit 1
fi

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

args=(
  --config "${config}"
  --project-root "${project_root}"
  --split-name test
  --streaming
  --exclude-go-roots
  --no-save-score-matrix
  --publication-metrics
  --device cuda
  --batch-size 1024
  --num-workers 4
  --progress-every-batches 100
)
args+=("$@")

bootstrap_log_dir="${project_root}/outputs/logs/evaluation_bootstrap"
mkdir -p "${bootstrap_log_dir}"
bootstrap_log="${bootstrap_log_dir}/evaluate_full_test_streaming_flux_${FLUX_JOB_ID:-manual}.log"

{
  echo "Project root: ${project_root}"
  echo "Script path: ${BASH_SOURCE[0]}"
  echo "Host: $(hostname)"
  echo "Started bootstrap: $(date --iso-8601=seconds)"
  echo "Python: ${eval_python}"
  echo "PYTHONPATH: ${PYTHONPATH}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
  echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-}"
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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-}"
echo "Bootstrap log: ${bootstrap_log}"
echo "Command: ${eval_python} -m evo2_go.cli evaluate-model ${args[*]}"

"${eval_python}" - <<'PY'
import torch

print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("Flux allocation did not expose a GPU; refusing CPU fallback")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}", flush=True)
PY

"${eval_python}" -m evo2_go.cli evaluate-model "${args[@]}"

echo "Finished: $(date --iso-8601=seconds)"
