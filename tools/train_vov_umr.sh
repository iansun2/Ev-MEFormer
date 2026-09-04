#!/usr/bin/env bash
# Train the first two VoV UMR learning modules:
#   finalized EUD VoV -> branch reliability learning -> damage-aware routing.

set -euo pipefail

usage() {
    echo "Usage: $0 SOURCE_CHECKPOINT [GPUS]"
    echo
    echo "Fresh EUD -> branch reliability -> damage-aware routing chain:"
    echo "  conda activate meformerv2"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 $0 /path/to/eud_vov_epoch_20.pth 4"
    echo
    echo "Start/retry damage-aware routing from the reliability checkpoint:"
    echo "  START_MODULE=damage_aware_routing CHAIN_ID=<original-id> $0 /path/to/branch_reliability_epoch_6.pth 4"
    echo
    echo "Optional environment variables:"
    echo "  START_MODULE branch_reliability_learning for the full chain (default),"
    echo "               or damage_aware_routing to skip reliability learning"
    echo "  PYTHON_BIN   Python executable from the MEFormerv2 environment"
    echo "  CHAIN_ID     Output grouping ID (default: UTC timestamp)"
    echo "  OUTPUT_ROOT  Parent output directory (default: work_dirs/vov_umr_chain)"
    echo "  SEED         Training seed passed to both modules (default: 0)"
    echo "  PORT         reliability-learning port; routing uses PORT+1 (default: 29500)"
    echo "  RUN_PREFLIGHT=0 disables checks for diagnostic smoke only (default: 1)"
    echo "  ALLOW_NONFORMAL_GPU_COUNT=1 permits a non-4-GPU diagnostic smoke (default: 0)"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

CALLER_DIR=$PWD
SOURCE_CHECKPOINT=$1
if [[ ${SOURCE_CHECKPOINT} != /* ]]; then
    SOURCE_CHECKPOINT=${CALLER_DIR}/${SOURCE_CHECKPOINT}
fi
GPUS=${2:-4}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
# shellcheck source=tools/umr_module_names.sh
source "${REPO_ROOT}/tools/umr_module_names.sh"

START_MODULE=$(umr_resolve_start_module)
CHAIN_ID=${CHAIN_ID:-$(date -u +%Y%m%d-%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-work_dirs/vov_umr_chain}
SEED=${SEED:-0}
PORT=${PORT:-29500}
RUN_PREFLIGHT=${RUN_PREFLIGHT:-1}
ALLOW_NONFORMAL_GPU_COUNT=${ALLOW_NONFORMAL_GPU_COUNT:-0}

BRANCH_RELIABILITY_CONFIG=projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_v8_a4_clean_anchor_domain_p50_full_e6.py
DAMAGE_AWARE_ROUTING_CONFIG=projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_umr.py
BRANCH_RELIABILITY_WORK_ROOT=${OUTPUT_ROOT}/${CHAIN_ID}/branch_reliability_learning
DAMAGE_AWARE_ROUTING_WORK_ROOT=${OUTPUT_ROOT}/${CHAIN_ID}/damage_aware_routing

if [[ ${START_MODULE} != branch_reliability_learning && \
      ${START_MODULE} != damage_aware_routing ]]; then
    echo "ERROR: START_MODULE must be branch_reliability_learning or damage_aware_routing" >&2
    exit 2
fi
if [[ ! ${CHAIN_ID} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: CHAIN_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
fi
if [[ ${OUTPUT_ROOT} == *" "* ]]; then
    echo "ERROR: OUTPUT_ROOT paths containing spaces are not supported" >&2
    exit 2
fi
if [[ ${SOURCE_CHECKPOINT} == *" "* ]]; then
    echo "ERROR: checkpoint paths containing spaces are not supported by MMCV DictAction" >&2
    exit 2
fi
if [[ ! -f ${SOURCE_CHECKPOINT} ]]; then
    echo "ERROR: source checkpoint does not exist: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! ${GPUS} =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPUS must be a positive integer, got: ${GPUS}" >&2
    exit 2
fi
if [[ ${ALLOW_NONFORMAL_GPU_COUNT} != 0 && ${ALLOW_NONFORMAL_GPU_COUNT} != 1 ]]; then
    echo "ERROR: ALLOW_NONFORMAL_GPU_COUNT must be 0 or 1" >&2
    exit 2
fi
if [[ ${GPUS} -ne 4 && ${ALLOW_NONFORMAL_GPU_COUNT} != 1 ]]; then
    echo "ERROR: formal VoV UMR migration requires 4 GPUs for global effective batch 16" >&2
    echo "       Set ALLOW_NONFORMAL_GPU_COUNT=1 only for a diagnostic smoke run." >&2
    exit 2
fi
if [[ ! ${PORT} =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65534 )); then
    echo "ERROR: PORT must be between 1 and 65534, got: ${PORT}" >&2
    exit 2
fi
if [[ ${RUN_PREFLIGHT} != 0 && ${RUN_PREFLIGHT} != 1 ]]; then
    echo "ERROR: RUN_PREFLIGHT must be 0 or 1, got: ${RUN_PREFLIGHT}" >&2
    exit 2
fi

if [[ -z ${PYTHON_BIN:-} ]]; then
    python_candidates=(python)
    if [[ -x /opt/conda/envs/meformerv2/bin/python ]]; then
        python_candidates+=(/opt/conda/envs/meformerv2/bin/python)
    fi
    for candidate in "${python_candidates[@]}"; do
        if command -v "${candidate}" >/dev/null 2>&1 && \
                "${candidate}" -c "import mmcv, torch, mmdet3d" >/dev/null 2>&1; then
            PYTHON_BIN=${candidate}
            break
        fi
    done
fi
if [[ -z ${PYTHON_BIN:-} ]] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: activate the meformerv2 conda environment or set PYTHON_BIN" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c "import mmcv, torch, mmdet3d" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} cannot import mmcv, torch, and mmdet3d" >&2
    echo "       Activate the meformerv2 conda environment or set PYTHON_BIN explicitly." >&2
    exit 2
fi

VISIBLE_GPU_COUNT=$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")
if [[ ! ${VISIBLE_GPU_COUNT} =~ ^[0-9]+$ ]] || (( VISIBLE_GPU_COUNT < GPUS )); then
    echo "ERROR: requested ${GPUS} GPUs but PyTorch sees ${VISIBLE_GPU_COUNT}" >&2
    echo "       Check CUDA_VISIBLE_DEVICES before starting the chain." >&2
    exit 2
fi

if [[ ${NNODES:-1} != 1 || ${NODE_RANK:-0} != 0 ]]; then
    echo "ERROR: train_vov_umr.sh supports single-node orchestration only" >&2
    exit 2
fi

if [[ ${START_MODULE} == branch_reliability_learning && \
      ( -e ${BRANCH_RELIABILITY_WORK_ROOT} || \
        -e ${DAMAGE_AWARE_ROUTING_WORK_ROOT} || \
        -e ${OUTPUT_ROOT}/${CHAIN_ID}/a4 || \
        -e ${OUTPUT_ROOT}/${CHAIN_ID}/umr ) ]]; then
    echo "ERROR: fresh chain output already exists for CHAIN_ID=${CHAIN_ID}" >&2
    echo "       Choose a new CHAIN_ID, or start damage_aware_routing with the reliability checkpoint." >&2
    exit 2
fi
if [[ ${START_MODULE} == damage_aware_routing ]]; then
    for routing_root in \
        "${DAMAGE_AWARE_ROUTING_WORK_ROOT}" \
        "${OUTPUT_ROOT}/${CHAIN_ID}/umr"; do
        if find "${routing_root}" -type f -name epoch_4.pth -print -quit \
                2>/dev/null | grep -q .; then
            echo "ERROR: an epoch_4.pth already exists under ${routing_root}" >&2
            exit 2
        fi
    done
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

find_module_checkpoint() {
    local work_root=$1
    local checkpoint_name=$2
    local -a matches=()
    mapfile -t matches < <(find "${work_root}" -type f -name "${checkpoint_name}" -print | sort)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "ERROR: expected exactly one ${checkpoint_name} under ${work_root}; found ${#matches[@]}" >&2
        if [[ ${#matches[@]} -gt 0 ]]; then
            printf '  %s\n' "${matches[@]}" >&2
        fi
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

run_distributed_train() {
    local config=$1
    local checkpoint=$2
    local work_root=$3
    local launcher_port=$4

    "${PYTHON_BIN}" -m torch.distributed.launch \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="${MASTER_ADDR:-127.0.0.1}" \
        --nproc_per_node="${GPUS}" \
        --master_port="${launcher_port}" \
        tools/train.py \
        "${config}" \
        --work-dir "${work_root}" \
        --launcher pytorch \
        --seed "${SEED}" \
        --cfg-options "load_from=${checkpoint}" "resume_from=None"
}

echo "[VoV UMR chain] id=${CHAIN_ID}"
echo "[VoV UMR chain] start_module=${START_MODULE}"
echo "[VoV UMR chain] module=$(umr_module_label "${START_MODULE}")"
echo "[VoV UMR chain] source checkpoint=${SOURCE_CHECKPOINT}"
echo "[VoV UMR chain] GPUs=${GPUS}, seed=${SEED}"

if [[ ${START_MODULE} == branch_reliability_learning ]]; then
    EUD_CHECKPOINT=${SOURCE_CHECKPOINT}
    if [[ ${RUN_PREFLIGHT} == 1 ]]; then
        echo "[1/6] Preflight EUD VoV -> branch reliability learning"
        "${PYTHON_BIN}" tools/vov_a4_preflight.py \
            "${BRANCH_RELIABILITY_CONFIG}" --checkpoint "${EUD_CHECKPOINT}" \
            --expected-checkpoint-epoch 20
    else
        echo "[1/6] Reliability-learning preflight disabled by RUN_PREFLIGHT=0"
    fi

    echo "[2/6] Train six-epoch VoV branch reliability representation"
    run_distributed_train \
        "${BRANCH_RELIABILITY_CONFIG}" "${EUD_CHECKPOINT}" \
        "${BRANCH_RELIABILITY_WORK_ROOT}" "${PORT}"
    BRANCH_RELIABILITY_CHECKPOINT=$(find_module_checkpoint \
        "${BRANCH_RELIABILITY_WORK_ROOT}" epoch_6.pth)
    echo "[VoV UMR chain] branch reliability checkpoint=${BRANCH_RELIABILITY_CHECKPOINT}"
else
    BRANCH_RELIABILITY_CHECKPOINT=${SOURCE_CHECKPOINT}
    echo "[1-2/6] Skipping reliability learning; using checkpoint=${BRANCH_RELIABILITY_CHECKPOINT}"
fi

if [[ ${RUN_PREFLIGHT} == 1 ]]; then
    echo "[3/6] Preflight reliability representation -> damage-aware routing"
    "${PYTHON_BIN}" tools/v9_rev20_preflight.py \
        "${DAMAGE_AWARE_ROUTING_CONFIG}" \
        --checkpoint "${BRANCH_RELIABILITY_CHECKPOINT}" \
        --require-full-checkpoint-coverage \
        --expected-checkpoint-epoch 6
else
    echo "[3/6] Damage-aware-routing preflight disabled by RUN_PREFLIGHT=0"
fi

echo "[4/6] Train four-epoch VoV routing and damage-state heads"
run_distributed_train \
    "${DAMAGE_AWARE_ROUTING_CONFIG}" "${BRANCH_RELIABILITY_CHECKPOINT}" \
    "${DAMAGE_AWARE_ROUTING_WORK_ROOT}" "$((PORT + 1))"
DAMAGE_AWARE_ROUTING_CHECKPOINT=$(find_module_checkpoint \
    "${DAMAGE_AWARE_ROUTING_WORK_ROOT}" epoch_4.pth)
echo "[VoV UMR chain] damage-aware routing checkpoint=${DAMAGE_AWARE_ROUTING_CHECKPOINT}"

if [[ ${RUN_PREFLIGHT} == 1 ]]; then
    echo "[5/6] Verify damage-aware-routing checkpoint coverage and guards"
    "${PYTHON_BIN}" tools/v9_rev20_preflight.py \
        "${DAMAGE_AWARE_ROUTING_CONFIG}" \
        --checkpoint "${DAMAGE_AWARE_ROUTING_CHECKPOINT}" \
        --require-full-checkpoint-coverage \
        --expected-checkpoint-epoch 4
else
    echo "[5/6] Final routing preflight disabled by RUN_PREFLIGHT=0"
fi

echo "[6/6] Complete"
echo "BRANCH_RELIABILITY_CHECKPOINT=${BRANCH_RELIABILITY_CHECKPOINT}"
echo "DAMAGE_AWARE_ROUTING_CHECKPOINT=${DAMAGE_AWARE_ROUTING_CHECKPOINT}"
