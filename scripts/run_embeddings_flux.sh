#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_embeddings_flux.sh [tax_id ...]
  scripts/run_embeddings_flux.sh --tax-id TAX_ID [--tax-id TAX_ID ...]
  scripts/run_embeddings_flux.sh --tax-list PATH
  scripts/run_embeddings_flux.sh

With one or more tax IDs, only those taxa are submitted to the embedding
runner. With no tax IDs and no --tax-list, the script repeatedly selects the
next incomplete taxon until no runnable taxa remain or the Flux allocation ends.

Common options:
  --tax-id TAX_ID                    Taxon to run. May be repeated or comma-delimited.
  --tax-list PATH                    File containing tax IDs, one per line.
  --build-id BUILD_ID                Build ID. Defaults to the current 42-taxon true-path build.
  --project-root PATH                Remodel project root. Defaults to this script's parent.
  --taxa-per-process N               Queue mode taxon cap per Python process. Default: 1.
  --stop-after-seconds N             Stop launching new queue iterations after N seconds. Default: 0.
  --dry-run                          Plan tasks without loading Evo2.
  -h, --help                         Show this help.

Environment overrides:
  EVO2_ROCM_MODULE                   Default: rocm/6.4
  EVO2_VENV                          Default: /p/vast1/ncpbr/evo2_tuo/venv/evo2_tuo
  EVO2_LOAD_MODULES                  Set 0 to skip module load. Default: 1
  EVO2_ACTIVATE_VENV                 Set 0 to skip venv activation. Default: 1
  EVO2_PYTHON                        Python executable after activation. Default: python3
  EVO2_EMBED_MAX_WORKERS             Default: 4
  EVO2_EMBED_GPU_IDS                 Default: 0,1,2,3
  EVO2_GPU_VISIBILITY_ENV            Default: CUDA_VISIBLE_DEVICES
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_parent="$(cd "${script_dir}/.." && pwd)"
if [[ -d "${PWD}/src/evo2_go" && -f "${PWD}/configs/build.example.toml" ]]; then
  default_project_root="${PWD}"
else
  default_project_root="${script_parent}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${default_project_root}}"
BUILD_ID="${BUILD_ID:-all_labels_true_path_42taxa_human_test_min_count_100_v1}"
INPUT_ROOT=""
OUTPUT_ROOT=""
MODEL_NAME="${EVO2_MODEL_NAME:-evo2_7b_base}"
MODEL_PATH="${EVO2_MODEL_PATH:-}"
EXPECTED_MAX_SEQLEN="${EVO2_EXPECTED_MAX_SEQLEN:-32768}"
MAX_WORKERS="${EVO2_EMBED_MAX_WORKERS:-4}"
GPU_IDS="${EVO2_EMBED_GPU_IDS:-0,1,2,3}"
GPU_VISIBILITY_ENV="${EVO2_GPU_VISIBILITY_ENV:-CUDA_VISIBLE_DEVICES}"
BUCKET_BATCH_SIZES="${EVO2_BUCKET_BATCH_SIZES:-bucket_1:16,bucket_2:8,bucket_3:16,bucket_4:8,bucket_5:8,bucket_6:6}"
NO_PROGRESS_TIMEOUT_SECONDS="${EVO2_NO_PROGRESS_TIMEOUT_SECONDS:-1500}"
MAX_FAILED_BATCHES="${EVO2_MAX_FAILED_BATCHES:-10}"
TAXA_PER_PROCESS="${EVO2_EMBED_TAXA_PER_PROCESS:-1}"
STOP_AFTER_SECONDS="${EVO2_EMBED_STOP_AFTER_SECONDS:-0}"
PYTHON_BIN="${EVO2_PYTHON:-python3}"
DRY_RUN=0

TAX_IDS=()
TAX_LIST=""

add_tax_ids() {
  local raw="$1"
  local old_ifs="${IFS}"
  IFS=","
  read -r -a parts <<< "${raw}"
  IFS="${old_ifs}"
  local part
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "${part}" ]]; then
      TAX_IDS+=("${part}")
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tax-id)
      [[ $# -ge 2 ]] || { echo "Missing value for --tax-id" >&2; exit 2; }
      add_tax_ids "$2"
      shift 2
      ;;
    --tax-list)
      [[ $# -ge 2 ]] || { echo "Missing value for --tax-list" >&2; exit 2; }
      TAX_LIST="$2"
      shift 2
      ;;
    --build-id)
      [[ $# -ge 2 ]] || { echo "Missing value for --build-id" >&2; exit 2; }
      BUILD_ID="$2"
      shift 2
      ;;
    --project-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --project-root" >&2; exit 2; }
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --input-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --input-root" >&2; exit 2; }
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --output-root" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --model-name)
      [[ $# -ge 2 ]] || { echo "Missing value for --model-name" >&2; exit 2; }
      MODEL_NAME="$2"
      shift 2
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "Missing value for --model-path" >&2; exit 2; }
      MODEL_PATH="$2"
      shift 2
      ;;
    --expected-max-seqlen)
      [[ $# -ge 2 ]] || { echo "Missing value for --expected-max-seqlen" >&2; exit 2; }
      EXPECTED_MAX_SEQLEN="$2"
      shift 2
      ;;
    --max-workers)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-workers" >&2; exit 2; }
      MAX_WORKERS="$2"
      shift 2
      ;;
    --gpu-ids)
      [[ $# -ge 2 ]] || { echo "Missing value for --gpu-ids" >&2; exit 2; }
      GPU_IDS="$2"
      shift 2
      ;;
    --gpu-visibility-env)
      [[ $# -ge 2 ]] || { echo "Missing value for --gpu-visibility-env" >&2; exit 2; }
      GPU_VISIBILITY_ENV="$2"
      shift 2
      ;;
    --bucket-batch-sizes)
      [[ $# -ge 2 ]] || { echo "Missing value for --bucket-batch-sizes" >&2; exit 2; }
      BUCKET_BATCH_SIZES="$2"
      shift 2
      ;;
    --no-progress-timeout-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --no-progress-timeout-seconds" >&2; exit 2; }
      NO_PROGRESS_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --max-failed-batches)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-failed-batches" >&2; exit 2; }
      MAX_FAILED_BATCHES="$2"
      shift 2
      ;;
    --taxa-per-process)
      [[ $# -ge 2 ]] || { echo "Missing value for --taxa-per-process" >&2; exit 2; }
      TAXA_PER_PROCESS="$2"
      shift 2
      ;;
    --stop-after-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --stop-after-seconds" >&2; exit 2; }
      STOP_AFTER_SECONDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        add_tax_ids "$1"
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      add_tax_ids "$1"
      shift
      ;;
  esac
done

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/outputs/map_gene_cds/length_filtered_max_8192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/builds/${BUILD_ID}/data/embeddings}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/evo2_model/evo2_7b_base.pt}"

if [[ "${EVO2_LOAD_MODULES:-1}" != "0" ]]; then
  module load "${EVO2_ROCM_MODULE:-rocm/6.4}"
fi

if [[ "${EVO2_ACTIVATE_VENV:-1}" != "0" ]]; then
  venv="${EVO2_VENV:-/p/vast1/ncpbr/evo2_tuo/venv/evo2_tuo}"
  source "${venv}/bin/activate"
fi

if [[ ! -s "${MODEL_PATH}" ]]; then
  echo "Missing Evo2 model checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing length-filtered CDS input root: ${INPUT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

base_cmd=(
  "${PYTHON_BIN}" -m evo2_go.embeddings.run_embeddings
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --model-name "${MODEL_NAME}"
  --model-path "${MODEL_PATH}"
  --expected-max-seqlen "${EXPECTED_MAX_SEQLEN}"
  --incomplete-only
  --max-workers "${MAX_WORKERS}"
  --gpu-ids "${GPU_IDS}"
  --gpu-visibility-env "${GPU_VISIBILITY_ENV}"
  --bucket-batch-sizes "${BUCKET_BATCH_SIZES}"
  --no-progress-timeout-seconds "${NO_PROGRESS_TIMEOUT_SECONDS}"
  --max-failed-batches "${MAX_FAILED_BATCHES}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  base_cmd+=(--dry-run)
fi

run_summary_value() {
  local key="$1"
  local summary="${OUTPUT_ROOT}/run_summary.tsv"
  if [[ -f "${summary}" ]]; then
    awk -F $'\t' -v key="${key}" '$1 == key {print $2; found=1} END {if (!found) exit 1}' "${summary}" 2>/dev/null || true
  fi
}

echo "Project root: ${PROJECT_ROOT}"
echo "Build ID: ${BUILD_ID}"
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "Workers: ${MAX_WORKERS}; GPUs: ${GPU_IDS}; visibility env: ${GPU_VISIBILITY_ENV}"

if [[ ${#TAX_IDS[@]} -gt 0 || -n "${TAX_LIST}" ]]; then
  cmd=("${base_cmd[@]}")
  for tax_id in "${TAX_IDS[@]}"; do
    cmd+=(--tax-id "${tax_id}")
  done
  if [[ -n "${TAX_LIST}" ]]; then
    cmd+=(--tax-list "${TAX_LIST}")
  fi
  echo "Mode: explicit taxa"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
  exit 0
fi

echo "Mode: queue incomplete taxa, ${TAXA_PER_PROCESS} taxon/taxa per Python process"
start_epoch="$(date +%s)"
iteration=1
while true; do
  if [[ "${STOP_AFTER_SECONDS}" -gt 0 ]]; then
    now_epoch="$(date +%s)"
    elapsed=$((now_epoch - start_epoch))
    if [[ "${elapsed}" -ge "${STOP_AFTER_SECONDS}" ]]; then
      echo "Stop-after-seconds reached before iteration ${iteration}: ${elapsed}/${STOP_AFTER_SECONDS}"
      break
    fi
  fi

  cmd=("${base_cmd[@]}" --limit-runnable-taxa "${TAXA_PER_PROCESS}")
  echo "Queue iteration ${iteration}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry-run queue mode stops after one planning pass."
    break
  fi

  runnable_taxa="$(run_summary_value runnable_taxa)"
  status="$(run_summary_value status)"
  abort_reason="$(run_summary_value abort_reason)"
  message="$(run_summary_value message)"

  if [[ "${runnable_taxa:-}" == "0" ]]; then
    echo "No runnable taxa remain. ${message:-}"
    break
  fi

  if [[ "${status:-}" != "complete" ]]; then
    echo "Embedding iteration ended with status=${status:-missing}" >&2
    if [[ -n "${abort_reason:-}" ]]; then
      echo "abort_reason=${abort_reason}" >&2
    fi
    exit 1
  fi

  iteration=$((iteration + 1))
done
