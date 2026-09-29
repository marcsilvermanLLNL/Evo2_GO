#!/usr/bin/env bash
# flux: -q pbatch
# flux: -N 1
# flux: -n 1
# flux: -c 16
# flux: -g 4
# flux: -t 2h
# flux: --job-name=evo2-go-train

set -euo pipefail

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

train_python="${EVO2_GO_TRAIN_PYTHON:-python3}"
bootstrap_log_dir="${project_root}/outputs/logs/training_bootstrap"
mkdir -p "${bootstrap_log_dir}"
bootstrap_log="${bootstrap_log_dir}/train_model_bootstrap_${FLUX_JOB_ID:-manual}.log"

args=("$@")
has_config=0
for arg in "${args[@]}"; do
  if [[ "${arg}" == "--config" ]]; then
    has_config=1
    break
  fi
done
if [[ "${has_config}" == "0" ]]; then
  args=(--config "${EVO2_GO_TRAIN_CONFIG:-configs/training_run.813taxa.toml}" "${args[@]}")
fi

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

{
  echo "Project root: ${project_root}"
  echo "Script path: ${BASH_SOURCE[0]}"
  echo "Host: $(hostname)"
  echo "Started bootstrap: $(date --iso-8601=seconds)"
  echo "Python: ${train_python}"
  echo "PYTHONPATH: ${PYTHONPATH}"
  echo "Args: ${args[*]}"
} > "${bootstrap_log}"

dry_run_output="$(
  "${train_python}" -m evo2_go.cli train-model \
    --project-root "${project_root}" \
    --dry-run \
    "${args[@]}" \
    2>> "${bootstrap_log}"
)"
printf '%s\n' "${dry_run_output}" >> "${bootstrap_log}"
output_dir="$(printf '%s\n' "${dry_run_output}" | awk -F '\t' '$1 == "output_dir" {print $2}')"
if [[ -z "${output_dir}" ]]; then
  echo "Could not resolve training output_dir from config; see ${bootstrap_log}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
exec > "${output_dir}/train_model.out"
exec 2> "${output_dir}/train_model.err"

echo "Project root: ${project_root}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Bootstrap log: ${bootstrap_log}"
echo "Command: ${train_python} -m evo2_go.cli train-model --project-root ${project_root} ${args[*]}"

"${train_python}" - <<'PY'
import torch
print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
PY

"${train_python}" -m evo2_go.cli train-model \
  --project-root "${project_root}" \
  "${args[@]}"

echo "Finished: $(date --iso-8601=seconds)"
