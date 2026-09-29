#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -p pdebug
#SBATCH -J evo2-go-split

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/prepare_training_split.slurm.sh [prepare_training_inputs.py args]

Runs only the CPU-side split-manifest preparation step:
  split_manifest

The split job expects the embedding root and label cache configured in
configs/training_inputs.example.toml to already exist.

Common examples:
  sbatch scripts/prepare_training_split.slurm.sh --force
  sbatch scripts/prepare_training_split.slurm.sh --dry-run

Environment overrides:
  PROJECT_ROOT             Remodel project root. Default: submit dir if it is the remodel tree.
  TRAINING_INPUTS_CONFIG   Config path. Default: configs/training_inputs.example.toml
  BUILD_ID                 Used only for default log dir.
  LOG_DIR                  Default: outputs/builds/${BUILD_ID}/logs/training_inputs
  EVO2_PREP_PYTHON         Python executable. Default: envs/ontology/bin/python if present, else python3
  EVO2_PREP_ACTIVATE_VENV  Set 1 to source a venv before running. Default: 0
  EVO2_PREP_VENV           Venv path used when EVO2_PREP_ACTIVATE_VENV=1
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
elif [[ -d "${submit_dir}/src/evo2_go" && -f "${submit_dir}/prepare_training_inputs.py" ]]; then
  PROJECT_ROOT="${submit_dir}"
else
  PROJECT_ROOT="${script_parent}"
fi

BUILD_ID="${BUILD_ID:-all_labels_true_path_42taxa_human_test_min_count_100_v1}"
CONFIG="${TRAINING_INPUTS_CONFIG:-configs/training_inputs.example.toml}"
LOG_DIR="${LOG_DIR:-outputs/builds/${BUILD_ID}/logs/training_inputs}"

cd "${PROJECT_ROOT}"

if [[ -n "${EVO2_PREP_PYTHON:-}" ]]; then
  PYTHON_BIN="${EVO2_PREP_PYTHON}"
elif [[ -x "envs/ontology/bin/python" ]]; then
  PYTHON_BIN="envs/ontology/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ "${EVO2_PREP_ACTIVATE_VENV:-0}" == "1" ]]; then
  if [[ -z "${EVO2_PREP_VENV:-}" ]]; then
    echo "EVO2_PREP_ACTIVATE_VENV=1 requires EVO2_PREP_VENV" >&2
    exit 2
  fi
  source "${EVO2_PREP_VENV}/bin/activate"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing training-input config: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

cmd=(
  "${PYTHON_BIN}" prepare_training_inputs.py
  --config "${CONFIG}"
  --project-root "${PROJECT_ROOT}"
  --steps split_manifest
)
cmd+=("$@")

echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG}"
echo "Log dir: ${LOG_DIR}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}" \
  > "${LOG_DIR}/prepare_training_split.out" \
  2> "${LOG_DIR}/prepare_training_split.err"
