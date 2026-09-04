#!/usr/bin/env bash
set -Eeuo pipefail

# One-command VoV UMR chain, named by thesis-level module function:
#   EUD -> branch reliability learning -> damage-aware routing ->
#   conservative head routing -> bounded candidate supplement.

usage() {
  cat <<'EOF'
Usage: tools/train_vov_umr_complete.sh SOURCE_CHECKPOINT [GPUS]

Fresh complete chain (SOURCE_CHECKPOINT is VoV EUD epoch 20):
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CHAIN_ID=vov-umr-full-run1 \
  tools/train_vov_umr_complete.sh /path/to/vov_eud_epoch_20.pth 4

Resume from the branch-reliability checkpoint:
  START_MODULE=damage_aware_routing CHAIN_ID=<same-id> \
  tools/train_vov_umr_complete.sh /path/to/branch_reliability_epoch_6.pth 4

Start from the damage-aware-routing checkpoint:
  START_MODULE=conservative_head_routing CHAIN_ID=<id> \
  tools/train_vov_umr_complete.sh /path/to/damage_aware_routing_epoch_4.pth 4

Train only conservative head routing (skip candidate-supplement calibration):
  START_MODULE=conservative_head_routing RUN_BOUNDED_SUPPLEMENT=0 CHAIN_ID=<id> \
  tools/train_vov_umr_complete.sh /path/to/damage_aware_routing_epoch_4.pth 4

Run only bounded-candidate calibration from a conservative-routing checkpoint:
  START_MODULE=bounded_candidate_supplement CHAIN_ID=<id> \
  CONSERVATIVE_ROUTING_CONFIG=/path/to/conservative_head_routing.py \
  CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT=/path/to/damage_aware_routing_epoch_4.pth \
  CONSERVATIVE_ROUTING_CALIBRATION=/path/to/routing_quality_calibration.json \
  tools/train_vov_umr_complete.sh /path/to/conservative_head_routing_epoch_6.pth 4

Environment:
  START_MODULE      branch_reliability_learning (default),
                    damage_aware_routing, conservative_head_routing, or
                    bounded_candidate_supplement
  OUTPUT_ROOT       default: work_dirs/vov_umr_complete
  CHAIN_ID          default: UTC timestamp
  PYTHON_BIN        MEFormerv2 Python executable
  PORT              first training-module port (default: 29500)
  SUPPLEMENT_PORT   calibration collection port (default: PORT+3)
  RUN_PREFLIGHT     1 for formal checks (default: 1)
  RUN_BOUNDED_SUPPLEMENT=1 builds the finite supplement (default: 1)
  RUN_COLLECT       collect train-only supplement artifacts (default: 1)
  ALLOW_EXISTING_OUT_ROOT=1 permits an explicit supplement-calibration resume
  ALLOW_NONFORMAL_GPU_COUNT=1 permits diagnostic non-4-GPU runs
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck source=tools/umr_module_names.sh
source "$ROOT_DIR/tools/umr_module_names.sh"

SOURCE_CHECKPOINT="$1"
if [[ "$SOURCE_CHECKPOINT" != /* ]]; then
  SOURCE_CHECKPOINT="$ROOT_DIR/$SOURCE_CHECKPOINT"
fi
GPUS="${2:-4}"
START_MODULE="$(umr_resolve_start_module)"
CHAIN_ID="${CHAIN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/vov_umr_complete}"
CHAIN_ROOT="${OUTPUT_ROOT}/${CHAIN_ID}"
BRANCH_RELIABILITY_WORK_ROOT="${CHAIN_ROOT}/branch_reliability_learning"
DAMAGE_AWARE_ROUTING_WORK_ROOT="${CHAIN_ROOT}/damage_aware_routing"
CONSERVATIVE_ROUTING_WORK_ROOT="${CHAIN_ROOT}/conservative_head_routing"
CONSERVATIVE_ROUTING_CONFIG_DIR="${CHAIN_ROOT}/configs"
CONSERVATIVE_ROUTING_PLACEHOLDER_CONFIG="projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_v10_router_selector_placeholder.py"
CONSERVATIVE_ROUTING_CONFIG="${CONSERVATIVE_ROUTING_CONFIG:-${V10_CONFIG:-${CONSERVATIVE_ROUTING_CONFIG_DIR}/vov_conservative_head_routing.py}}"
CONSERVATIVE_ROUTING_CALIBRATION="${CONSERVATIVE_ROUTING_CALIBRATION:-${V10_CALIBRATION:-${CHAIN_ROOT}/conservative_head_routing_calibration.json}}"
CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT="${CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT:-${V10_SOURCE_CHECKPOINT:-}}"
TRAIN_INFOS="${TRAIN_INFOS:-data/nuscenes/nuscenes_infos_train.pkl}"
ROUTING_CALIBRATION_BATCH_SIZE="${ROUTING_CALIBRATION_BATCH_SIZE:-${V10_CALIBRATION_BATCH_SIZE:-1}}"
BOUNDED_SUPPLEMENT_ROOT="${BOUNDED_SUPPLEMENT_ROOT:-${SGSUP_ROOT:-${CHAIN_ROOT}/bounded_candidate_supplement}}"
PORT="${PORT:-29500}"
SUPPLEMENT_PORT="${SUPPLEMENT_PORT:-${SGSUP_PORT:-}}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_BOUNDED_SUPPLEMENT="${RUN_BOUNDED_SUPPLEMENT:-${RUN_SGSUP:-1}}"
RUN_COLLECT="${RUN_COLLECT:-1}"
ALLOW_EXISTING_OUT_ROOT="${ALLOW_EXISTING_OUT_ROOT:-0}"
ALLOW_NONFORMAL_GPU_COUNT="${ALLOW_NONFORMAL_GPU_COUNT:-0}"
SEED="${SEED:-0}"

for flag in RUN_PREFLIGHT RUN_BOUNDED_SUPPLEMENT RUN_COLLECT ALLOW_EXISTING_OUT_ROOT ALLOW_NONFORMAL_GPU_COUNT; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "[ERROR] $flag must be 0 or 1" >&2
    exit 2
  fi
done
if [[ ! "$CHAIN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "[ERROR] invalid CHAIN_ID: $CHAIN_ID" >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" == *" "* || "$SOURCE_CHECKPOINT" == *" "* || \
      "$CONSERVATIVE_ROUTING_CONFIG" == *" "* || \
      "$CONSERVATIVE_ROUTING_CALIBRATION" == *" "* || \
      "$TRAIN_INFOS" == *" "* ]]; then
  echo "[ERROR] paths containing spaces are not supported" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[ERROR] source checkpoint does not exist: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_INFOS" ]]; then
  echo "[ERROR] train infos does not exist: $TRAIN_INFOS" >&2
  exit 2
fi
if [[ ! "$ROUTING_CALIBRATION_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] ROUTING_CALIBRATION_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] GPUS must be a positive integer" >&2
  exit 2
fi
if [[ "$GPUS" -ne 4 && "$ALLOW_NONFORMAL_GPU_COUNT" != "1" ]]; then
  echo "[ERROR] formal VoV UMR requires 4 GPUs" >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ || "$PORT" -lt 1 || "$PORT" -gt 65532 ]]; then
  echo "[ERROR] invalid PORT" >&2
  exit 2
fi
if [[ -z "$SUPPLEMENT_PORT" ]]; then
  SUPPLEMENT_PORT="$((PORT + 3))"
fi
if [[ ! "$SUPPLEMENT_PORT" =~ ^[0-9]+$ || \
      "$SUPPLEMENT_PORT" -lt 1 || "$SUPPLEMENT_PORT" -gt 65535 ]]; then
  echo "[ERROR] invalid SUPPLEMENT_PORT" >&2
  exit 2
fi

if [[ -z ${PYTHON_BIN:-} ]]; then
  if [[ -x /opt/conda/envs/meformerv2/bin/python ]]; then
    PYTHON_BIN=/opt/conda/envs/meformerv2/bin/python
  else
    PYTHON_BIN=python
  fi
fi
if ! "$PYTHON_BIN" -c "import mmcv, torch, mmdet3d" >/dev/null 2>&1; then
  echo "[ERROR] $PYTHON_BIN cannot import mmcv, torch, and mmdet3d" >&2
  exit 2
fi
VISIBLE_GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if [[ ! "$VISIBLE_GPU_COUNT" =~ ^[0-9]+$ || "$VISIBLE_GPU_COUNT" -lt "$GPUS" ]]; then
  echo "[ERROR] requested $GPUS GPUs but PyTorch sees $VISIBLE_GPU_COUNT" >&2
  exit 2
fi

checkpoint_epoch() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import torch
try:
    checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(int((checkpoint.get("meta") or {}).get("epoch", -1)))
PY
}

require_epoch() {
  local checkpoint="$1" expected="$2" actual
  actual="$(checkpoint_epoch "$checkpoint")"
  if [[ "$actual" != "$expected" ]]; then
    echo "[ERROR] checkpoint epoch mismatch: expected=$expected got=$actual path=$checkpoint" >&2
    exit 3
  fi
}

find_unique_checkpoint() {
  local root="$1" name="$2"
  local -a matches=()
  mapfile -t matches < <(find "$root" -type f -name "$name" -print | sort)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "[ERROR] expected exactly one $name under $root; found ${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 4
  fi
  printf '%s\n' "${matches[0]}"
}

run_distributed_train() {
  local config="$1" checkpoint="$2" work_root="$3" launcher_port="$4"
  "$PYTHON_BIN" -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --nproc_per_node="$GPUS" \
    --master_port="$launcher_port" \
    tools/train.py \
    "$config" \
    --work-dir "$work_root" \
    --launcher pytorch \
    --seed "$SEED" \
    --cfg-options "load_from=$checkpoint" "resume_from=None"
}

case "$START_MODULE" in
  branch_reliability_learning) require_epoch "$SOURCE_CHECKPOINT" 20 ;;
  damage_aware_routing) require_epoch "$SOURCE_CHECKPOINT" 6 ;;
  conservative_head_routing) require_epoch "$SOURCE_CHECKPOINT" 4 ;;
  bounded_candidate_supplement) require_epoch "$SOURCE_CHECKPOINT" 6 ;;
esac

if [[ "$START_MODULE" != "bounded_candidate_supplement" && \
      ( -e "$CONSERVATIVE_ROUTING_WORK_ROOT" || -e "${CHAIN_ROOT}/v10" ) ]]; then
  echo "[ERROR] conservative-head-routing output already exists under: $CHAIN_ROOT" >&2
  exit 4
fi

echo "[VoV complete] chain_id=$CHAIN_ID start_module=$START_MODULE"
echo "[VoV complete] module=$(umr_module_label "$START_MODULE")"
echo "[VoV complete] source=$SOURCE_CHECKPOINT GPUs=$GPUS"

BRANCH_RELIABILITY_CHECKPOINT=""
DAMAGE_AWARE_ROUTING_CHECKPOINT=""
if [[ "$START_MODULE" == "branch_reliability_learning" || \
      "$START_MODULE" == "damage_aware_routing" ]]; then
  TRAIN_START_MODULE="$START_MODULE"
  START_MODULE="$TRAIN_START_MODULE" \
  START_STAGE= \
  CHAIN_ID="$CHAIN_ID" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  SEED="$SEED" \
  PORT="$PORT" \
  RUN_PREFLIGHT="$RUN_PREFLIGHT" \
  ALLOW_NONFORMAL_GPU_COUNT="$ALLOW_NONFORMAL_GPU_COUNT" \
  bash tools/train_vov_umr.sh "$SOURCE_CHECKPOINT" "$GPUS"
  DAMAGE_AWARE_ROUTING_CHECKPOINT="$(find_unique_checkpoint "$DAMAGE_AWARE_ROUTING_WORK_ROOT" epoch_4.pth)"
  if [[ -d "$BRANCH_RELIABILITY_WORK_ROOT" ]]; then
    BRANCH_RELIABILITY_CHECKPOINT="$(find_unique_checkpoint "$BRANCH_RELIABILITY_WORK_ROOT" epoch_6.pth)"
  elif [[ "$TRAIN_START_MODULE" == "damage_aware_routing" ]]; then
    BRANCH_RELIABILITY_CHECKPOINT="$SOURCE_CHECKPOINT"
  fi
  require_epoch "$BRANCH_RELIABILITY_CHECKPOINT" 6
elif [[ "$START_MODULE" == "conservative_head_routing" ]]; then
  DAMAGE_AWARE_ROUTING_CHECKPOINT="$SOURCE_CHECKPOINT"
fi

if [[ "$START_MODULE" != "bounded_candidate_supplement" ]]; then
  require_epoch "$DAMAGE_AWARE_ROUTING_CHECKPOINT" 4
  mkdir -p "$CONSERVATIVE_ROUTING_CONFIG_DIR"
  echo "[VoV complete] Calibrate checkpoint-specific conservative-routing quality thresholds"
  "$PYTHON_BIN" tools/v10_quality_calibration.py \
    --config "$CONSERVATIVE_ROUTING_PLACEHOLDER_CONFIG" \
    --checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
    --train-infos "$TRAIN_INFOS" \
    --subset 0.02 \
    --epochs 1 \
    --batch-size "$ROUTING_CALIBRATION_BATCH_SIZE" \
    --workers-per-gpu 0 \
    --device cuda:0 \
    --seed 20260522 \
    --output "$CONSERVATIVE_ROUTING_CALIBRATION"

  "$PYTHON_BIN" tools/build_vov_v10_selector_config.py \
    --base-config "$CONSERVATIVE_ROUTING_PLACEHOLDER_CONFIG" \
    --calibration "$CONSERVATIVE_ROUTING_CALIBRATION" \
    --source-checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
    --output "$CONSERVATIVE_ROUTING_CONFIG"
  "$PYTHON_BIN" tools/v10_preflight.py \
    "$CONSERVATIVE_ROUTING_CONFIG" --profile vov_v10 --config-only \
    --checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
    --source-checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
    --calibration "$CONSERVATIVE_ROUTING_CALIBRATION"
  if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    echo "[VoV complete] Preflight damage-aware routing -> conservative head routing"
    "$PYTHON_BIN" tools/v10_preflight.py \
      "$CONSERVATIVE_ROUTING_CONFIG" --profile vov_v10 \
      --checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
      --source-checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
      --calibration "$CONSERVATIVE_ROUTING_CALIBRATION" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch 4 \
      --dump-path "${CHAIN_ROOT}/conservative_routing_input_preflight.json"
  fi
  echo "[VoV complete] Train VoV conservative head routing (6 epochs)"
  run_distributed_train \
    "$CONSERVATIVE_ROUTING_CONFIG" "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
    "$CONSERVATIVE_ROUTING_WORK_ROOT" "$((PORT + 2))"
  CONSERVATIVE_ROUTING_CHECKPOINT="$(find_unique_checkpoint "$CONSERVATIVE_ROUTING_WORK_ROOT" epoch_6.pth)"
  require_epoch "$CONSERVATIVE_ROUTING_CHECKPOINT" 6
  if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    "$PYTHON_BIN" tools/v10_preflight.py \
      "$CONSERVATIVE_ROUTING_CONFIG" --profile vov_v10 \
      --checkpoint "$CONSERVATIVE_ROUTING_CHECKPOINT" \
      --source-checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
      --calibration "$CONSERVATIVE_ROUTING_CALIBRATION" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch 6 \
      --dump-path "${CHAIN_ROOT}/conservative_routing_final_preflight.json"
  fi
else
  CONSERVATIVE_ROUTING_CHECKPOINT="$SOURCE_CHECKPOINT"
  if [[ ! -f "$CONSERVATIVE_ROUTING_CONFIG" || \
        ! -f "$CONSERVATIVE_ROUTING_CALIBRATION" ]]; then
    echo "[ERROR] bounded_candidate_supplement requires CONSERVATIVE_ROUTING_CONFIG and CONSERVATIVE_ROUTING_CALIBRATION" >&2
    exit 4
  fi
  if [[ -z "$CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT" && \
        -d "$DAMAGE_AWARE_ROUTING_WORK_ROOT" ]]; then
    CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT="$(find_unique_checkpoint "$DAMAGE_AWARE_ROUTING_WORK_ROOT" epoch_4.pth)"
  elif [[ -z "$CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT" && -d "${CHAIN_ROOT}/umr" ]]; then
    CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT="$(find_unique_checkpoint "${CHAIN_ROOT}/umr" epoch_4.pth)"
  fi
  if [[ -z "$CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT" || \
        ! -f "$CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT" ]]; then
    echo "[ERROR] bounded_candidate_supplement requires CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT epoch 4" >&2
    exit 4
  fi
  DAMAGE_AWARE_ROUTING_CHECKPOINT="$CONSERVATIVE_ROUTING_SOURCE_CHECKPOINT"
  require_epoch "$DAMAGE_AWARE_ROUTING_CHECKPOINT" 4
fi

DAMAGE_AWARE_ROUTING_CKPT_SHA256="$(sha256sum "$DAMAGE_AWARE_ROUTING_CHECKPOINT" | awk '{print $1}')"
CONSERVATIVE_ROUTING_CKPT_SHA256="$(sha256sum "$CONSERVATIVE_ROUTING_CHECKPOINT" | awk '{print $1}')"

FINAL_CONFIG=""
if [[ "$RUN_BOUNDED_SUPPLEMENT" == "1" ]]; then
  echo "[VoV complete] Build checkpoint-specific bounded candidate supplement"
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PROFILE=vov_v10 \
  BASE_CONFIG="$CONSERVATIVE_ROUTING_CONFIG" \
  BASE_CKPT="$CONSERVATIVE_ROUTING_CHECKPOINT" \
  EXPECTED_CKPT_SHA256="$CONSERVATIVE_ROUTING_CKPT_SHA256" \
  V10_SOURCE_CHECKPOINT="$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
  V10_CALIBRATION="$CONSERVATIVE_ROUTING_CALIBRATION" \
  OUT_ROOT="$BOUNDED_SUPPLEMENT_ROOT" \
  GPUS="$GPUS" \
  PORT="$SUPPLEMENT_PORT" \
  RUN_PREFLIGHT="$RUN_PREFLIGHT" \
  RUN_COLLECT="$RUN_COLLECT" \
  RUN_CALIBRATE=1 \
  RUN_PACK=1 \
  ALLOW_EXISTING_OUT_ROOT="$ALLOW_EXISTING_OUT_ROOT" \
  bash tools/prepare_vov_umr_sgsup_b.sh
  FINAL_CONFIG="${BOUNDED_SUPPLEMENT_ROOT}/runtime_pack/configs/RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py"
fi

mkdir -p "$CHAIN_ROOT"
"$PYTHON_BIN" - \
  "$CHAIN_ROOT/chain_manifest.json" "$CHAIN_ID" "$START_MODULE" \
  "$SOURCE_CHECKPOINT" "$BRANCH_RELIABILITY_CHECKPOINT" \
  "$DAMAGE_AWARE_ROUTING_CHECKPOINT" "$DAMAGE_AWARE_ROUTING_CKPT_SHA256" \
  "$CONSERVATIVE_ROUTING_CHECKPOINT" "$CONSERVATIVE_ROUTING_CKPT_SHA256" \
  "$CONSERVATIVE_ROUTING_CONFIG" "$CONSERVATIVE_ROUTING_CALIBRATION" \
  "$BOUNDED_SUPPLEMENT_ROOT" "$FINAL_CONFIG" <<'PY'
import json
import os
import sys

(output, chain_id, start_module, source, reliability, routing, routing_sha,
 conservative, conservative_sha, conservative_config, calibration,
 supplement, final_config) = sys.argv[1:]
payload = {
    "format": "vov_umr_functional_chain_v3",
    "chain_id": chain_id,
    "start_module": start_module,
    "module_sequence": [
        "branch_reliability_learning",
        "damage_aware_routing",
        "conservative_head_routing",
        "bounded_candidate_supplement",
    ],
    "source_checkpoint": os.path.realpath(source),
    "branch_reliability_checkpoint": (
        os.path.realpath(reliability) if reliability else ""
    ),
    "damage_aware_routing_checkpoint": os.path.realpath(routing),
    "damage_aware_routing_checkpoint_sha256": routing_sha,
    "conservative_head_routing_checkpoint": os.path.realpath(conservative),
    "conservative_head_routing_checkpoint_sha256": conservative_sha,
    "conservative_head_routing_config": os.path.realpath(conservative_config),
    "conservative_head_routing_calibration": os.path.realpath(calibration),
    "bounded_candidate_supplement_root": os.path.realpath(supplement),
    "formal_final_config": os.path.realpath(final_config) if final_config else "",
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo
echo "VoV complete UMR chain finished."
echo "DAMAGE_AWARE_ROUTING_CHECKPOINT=$DAMAGE_AWARE_ROUTING_CHECKPOINT"
echo "DAMAGE_AWARE_ROUTING_CKPT_SHA256=$DAMAGE_AWARE_ROUTING_CKPT_SHA256"
echo "CONSERVATIVE_ROUTING_CHECKPOINT=$CONSERVATIVE_ROUTING_CHECKPOINT"
echo "CONSERVATIVE_ROUTING_CKPT_SHA256=$CONSERVATIVE_ROUTING_CKPT_SHA256"
echo "CONSERVATIVE_ROUTING_CONFIG=$CONSERVATIVE_ROUTING_CONFIG"
echo "CONSERVATIVE_ROUTING_CALIBRATION=$CONSERVATIVE_ROUTING_CALIBRATION"
echo "FINAL_CONFIG=$FINAL_CONFIG"
echo "CHAIN_MANIFEST=${CHAIN_ROOT}/chain_manifest.json"
