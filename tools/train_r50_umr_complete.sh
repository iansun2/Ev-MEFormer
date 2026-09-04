#!/usr/bin/env bash
set -Eeuo pipefail

# One-command R50 UMR chain, named by thesis-level module function:
#   EUD -> branch reliability learning -> damage-aware routing ->
#   conservative head routing -> bounded candidate supplement.

usage() {
  cat <<'EOF'
Usage: tools/train_r50_umr_complete.sh SOURCE_CHECKPOINT [GPUS]

Fresh complete chain (SOURCE_CHECKPOINT is R50 EUD epoch 20):
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CHAIN_ID=r50-umr-full-run1 \
  tools/train_r50_umr_complete.sh /path/to/r50_eud_epoch_20.pth 4

Resume examples:
  START_MODULE=damage_aware_routing \
    tools/train_r50_umr_complete.sh /path/to/branch_reliability_epoch_6.pth 4
  START_MODULE=conservative_head_routing \
    tools/train_r50_umr_complete.sh /path/to/damage_aware_routing_epoch_4.pth 4
  START_MODULE=bounded_candidate_supplement \
    tools/train_r50_umr_complete.sh /path/to/conservative_head_routing_epoch_6.pth 4

Environment:
  START_MODULE      branch_reliability_learning (default),
                    damage_aware_routing, conservative_head_routing, or
                    bounded_candidate_supplement
  OUTPUT_ROOT       default: work_dirs/r50_umr_complete
  CHAIN_ID          default: UTC timestamp
  PYTHON_BIN        MEFormerv2 Python executable
  PORT              first module port; later learned modules use PORT+1/+2
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
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/r50_umr_complete}"
CHAIN_ROOT="${OUTPUT_ROOT}/${CHAIN_ID}"
BRANCH_RELIABILITY_WORK_ROOT="${CHAIN_ROOT}/branch_reliability_learning"
DAMAGE_AWARE_ROUTING_WORK_ROOT="${CHAIN_ROOT}/damage_aware_routing"
CONSERVATIVE_ROUTING_WORK_ROOT="${CHAIN_ROOT}/conservative_head_routing"
BOUNDED_SUPPLEMENT_ROOT="${BOUNDED_SUPPLEMENT_ROOT:-${SGSUP_ROOT:-${CHAIN_ROOT}/bounded_candidate_supplement}}"
PORT="${PORT:-29500}"
SUPPLEMENT_PORT="${SUPPLEMENT_PORT:-${SGSUP_PORT:-}}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_BOUNDED_SUPPLEMENT="${RUN_BOUNDED_SUPPLEMENT:-${RUN_SGSUP:-1}}"
RUN_COLLECT="${RUN_COLLECT:-1}"
ALLOW_EXISTING_OUT_ROOT="${ALLOW_EXISTING_OUT_ROOT:-0}"
ALLOW_NONFORMAL_GPU_COUNT="${ALLOW_NONFORMAL_GPU_COUNT:-0}"
SEED="${SEED:-0}"

BRANCH_RELIABILITY_CONFIG=projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v8_a4_clean_anchor_domain_p50_full_e6.py
DAMAGE_AWARE_ROUTING_CONFIG=projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v9_rev20_a4_routing_only.py
CONSERVATIVE_ROUTING_CONFIG=projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v10_router_selector_frozen.py

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
if [[ "$OUTPUT_ROOT" == *" "* || "$SOURCE_CHECKPOINT" == *" "* ]]; then
  echo "[ERROR] paths containing spaces are not supported" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[ERROR] source checkpoint does not exist: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] GPUS must be a positive integer" >&2
  exit 2
fi
if [[ "$GPUS" -ne 4 && "$ALLOW_NONFORMAL_GPU_COUNT" != "1" ]]; then
  echo "[ERROR] formal R50 UMR requires 4 GPUs (global batch 16)" >&2
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

if [[ "$START_MODULE" == "branch_reliability_learning" && \
      ( -e "$BRANCH_RELIABILITY_WORK_ROOT" || \
        -e "$DAMAGE_AWARE_ROUTING_WORK_ROOT" || \
        -e "$CONSERVATIVE_ROUTING_WORK_ROOT" || \
        -e "${CHAIN_ROOT}/a4" || -e "${CHAIN_ROOT}/umr" || \
        -e "${CHAIN_ROOT}/v10" ) ]]; then
  echo "[ERROR] fresh-chain output already exists: $CHAIN_ROOT" >&2
  exit 4
fi
if [[ "$START_MODULE" == "damage_aware_routing" && \
      ( -e "$DAMAGE_AWARE_ROUTING_WORK_ROOT" || \
        -e "$CONSERVATIVE_ROUTING_WORK_ROOT" || \
        -e "${CHAIN_ROOT}/umr" || -e "${CHAIN_ROOT}/v10" ) ]]; then
  echo "[ERROR] routing output already exists: $CHAIN_ROOT" >&2
  exit 4
fi
if [[ "$START_MODULE" == "conservative_head_routing" && \
      ( -e "$CONSERVATIVE_ROUTING_WORK_ROOT" || -e "${CHAIN_ROOT}/v10" ) ]]; then
  echo "[ERROR] conservative-routing output already exists: $CONSERVATIVE_ROUTING_WORK_ROOT" >&2
  exit 4
fi

echo "[R50 complete] chain_id=$CHAIN_ID start_module=$START_MODULE"
echo "[R50 complete] module=$(umr_module_label "$START_MODULE")"
echo "[R50 complete] source=$SOURCE_CHECKPOINT GPUs=$GPUS"

BRANCH_RELIABILITY_CHECKPOINT=""
DAMAGE_AWARE_ROUTING_CHECKPOINT=""
if [[ "$START_MODULE" == "branch_reliability_learning" ]]; then
  if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    echo "[1/8] Preflight R50 EUD -> branch reliability learning"
    "$PYTHON_BIN" tools/r50_a4_preflight.py \
      "$BRANCH_RELIABILITY_CONFIG" --checkpoint "$SOURCE_CHECKPOINT" \
      --expected-checkpoint-epoch 20
  fi
  echo "[2/8] Train R50 branch reliability representation (6 epochs)"
  run_distributed_train "$BRANCH_RELIABILITY_CONFIG" "$SOURCE_CHECKPOINT" \
    "$BRANCH_RELIABILITY_WORK_ROOT" "$PORT"
  BRANCH_RELIABILITY_CHECKPOINT="$(find_unique_checkpoint "$BRANCH_RELIABILITY_WORK_ROOT" epoch_6.pth)"
  require_epoch "$BRANCH_RELIABILITY_CHECKPOINT" 6
elif [[ "$START_MODULE" == "damage_aware_routing" ]]; then
  BRANCH_RELIABILITY_CHECKPOINT="$SOURCE_CHECKPOINT"
fi

if [[ "$START_MODULE" == "branch_reliability_learning" || \
      "$START_MODULE" == "damage_aware_routing" ]]; then
  if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    echo "[3/8] Preflight branch reliability -> damage-aware routing"
    "$PYTHON_BIN" tools/v9_rev20_preflight.py \
      "$DAMAGE_AWARE_ROUTING_CONFIG" \
      --checkpoint "$BRANCH_RELIABILITY_CHECKPOINT" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch 6
  fi
  echo "[4/8] Train R50 routing and damage-state heads (4 epochs)"
  run_distributed_train "$DAMAGE_AWARE_ROUTING_CONFIG" \
    "$BRANCH_RELIABILITY_CHECKPOINT" "$DAMAGE_AWARE_ROUTING_WORK_ROOT" \
    "$((PORT + 1))"
  DAMAGE_AWARE_ROUTING_CHECKPOINT="$(find_unique_checkpoint "$DAMAGE_AWARE_ROUTING_WORK_ROOT" epoch_4.pth)"
elif [[ "$START_MODULE" == "conservative_head_routing" ]]; then
  DAMAGE_AWARE_ROUTING_CHECKPOINT="$SOURCE_CHECKPOINT"
fi

if [[ "$START_MODULE" != "bounded_candidate_supplement" ]]; then
  require_epoch "$DAMAGE_AWARE_ROUTING_CHECKPOINT" 4
  if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    echo "[5/8] Preflight damage-aware routing -> conservative head routing"
    "$PYTHON_BIN" tools/v9_rev20_preflight.py \
      "$DAMAGE_AWARE_ROUTING_CONFIG" \
      --checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch 4
    "$PYTHON_BIN" tools/v10_preflight.py \
      "$CONSERVATIVE_ROUTING_CONFIG" \
      --checkpoint "$DAMAGE_AWARE_ROUTING_CHECKPOINT" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch 4 \
      --dump-path "${CHAIN_ROOT}/conservative_routing_input_preflight.json"
  fi
  echo "[6/8] Train R50 conservative head routing (6 epochs)"
  run_distributed_train "$CONSERVATIVE_ROUTING_CONFIG" \
    "$DAMAGE_AWARE_ROUTING_CHECKPOINT" "$CONSERVATIVE_ROUTING_WORK_ROOT" \
    "$((PORT + 2))"
  CONSERVATIVE_ROUTING_CHECKPOINT="$(find_unique_checkpoint "$CONSERVATIVE_ROUTING_WORK_ROOT" epoch_6.pth)"
else
  CONSERVATIVE_ROUTING_CHECKPOINT="$SOURCE_CHECKPOINT"
fi
require_epoch "$CONSERVATIVE_ROUTING_CHECKPOINT" 6

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "[7/8] Verify final R50 conservative-routing checkpoint"
  "$PYTHON_BIN" tools/v10_preflight.py \
    "$CONSERVATIVE_ROUTING_CONFIG" \
    --checkpoint "$CONSERVATIVE_ROUTING_CHECKPOINT" \
    --require-full-checkpoint-coverage \
    --expected-checkpoint-epoch 6 \
    --dump-path "${CHAIN_ROOT}/conservative_routing_final_preflight.json"
fi

CONSERVATIVE_ROUTING_CKPT_SHA256="$(sha256sum "$CONSERVATIVE_ROUTING_CHECKPOINT" | awk '{print $1}')"
FINAL_CONFIG=""
if [[ "$RUN_BOUNDED_SUPPLEMENT" == "1" ]]; then
  echo "[8/8] Build checkpoint-specific R50 bounded candidate supplement"
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_CKPT="$CONSERVATIVE_ROUTING_CHECKPOINT" \
  EXPECTED_CKPT_SHA256="$CONSERVATIVE_ROUTING_CKPT_SHA256" \
  OUT_ROOT="$BOUNDED_SUPPLEMENT_ROOT" \
  GPUS="$GPUS" \
  PORT="$SUPPLEMENT_PORT" \
  RUN_PREFLIGHT="$RUN_PREFLIGHT" \
  RUN_COLLECT="$RUN_COLLECT" \
  RUN_CALIBRATE=1 \
  RUN_PACK=1 \
  ALLOW_EXISTING_OUT_ROOT="$ALLOW_EXISTING_OUT_ROOT" \
  bash tools/prepare_r50_umr_sgsup_b.sh
  FINAL_CONFIG="${BOUNDED_SUPPLEMENT_ROOT}/runtime_pack/configs/RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py"
fi

mkdir -p "$CHAIN_ROOT"
"$PYTHON_BIN" - \
  "$CHAIN_ROOT/chain_manifest.json" "$CHAIN_ID" "$START_MODULE" \
  "$SOURCE_CHECKPOINT" "$BRANCH_RELIABILITY_CHECKPOINT" \
  "$DAMAGE_AWARE_ROUTING_CHECKPOINT" "$CONSERVATIVE_ROUTING_CHECKPOINT" \
  "$CONSERVATIVE_ROUTING_CKPT_SHA256" "$BOUNDED_SUPPLEMENT_ROOT" \
  "$FINAL_CONFIG" <<'PY'
import json
import os
import sys

(output, chain_id, start_module, source, reliability, routing, conservative,
 conservative_sha, supplement, final_config) = sys.argv[1:]
payload = {
    "format": "r50_umr_functional_chain_v2",
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
    "damage_aware_routing_checkpoint": (
        os.path.realpath(routing) if routing else ""
    ),
    "conservative_head_routing_checkpoint": os.path.realpath(conservative),
    "conservative_head_routing_checkpoint_sha256": conservative_sha,
    "bounded_candidate_supplement_root": os.path.realpath(supplement),
    "formal_final_config": os.path.realpath(final_config) if final_config else "",
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo
echo "R50 complete UMR chain finished."
echo "CONSERVATIVE_ROUTING_CHECKPOINT=$CONSERVATIVE_ROUTING_CHECKPOINT"
echo "CONSERVATIVE_ROUTING_CKPT_SHA256=$CONSERVATIVE_ROUTING_CKPT_SHA256"
echo "FINAL_CONFIG=$FINAL_CONFIG"
echo "CHAIN_MANIFEST=${CHAIN_ROOT}/chain_manifest.json"
