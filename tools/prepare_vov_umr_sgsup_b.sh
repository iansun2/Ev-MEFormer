#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end, train-only calibration and runtime-pack build for the VoV UMR
# migration of the finalized R50 SGSUP-B context/height rule.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
MODEL_PROFILE="${MODEL_PROFILE:-vov_umr}"
case "$MODEL_PROFILE" in
  vov_umr)
    DEFAULT_BASE_CONFIG="projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_umr.py"
    DEFAULT_BASE_CKPT="work_dirs/vov_umr_chain/vov-umr-run1/umr/20260809-161531/epoch_4.pth"
    DEFAULT_CKPT_SHA256="b4c05cf233190ee23c73eb1f5e94388e3dae3a0e58fba0d222e1c65e849b7750"
    EXPECTED_CKPT_EPOCH=4
    EV_MODEL_LABEL="VoV_UMR_epoch4_SGSUP_calibration"
    ;;
  vov_v10)
    DEFAULT_BASE_CONFIG=""
    DEFAULT_BASE_CKPT=""
    DEFAULT_CKPT_SHA256=""
    EXPECTED_CKPT_EPOCH=6
    EV_MODEL_LABEL="VoV_V10_epoch6_SGSUP_calibration"
    ;;
  *) echo "[ERROR] MODEL_PROFILE must be vov_umr or vov_v10" >&2; exit 2 ;;
esac
BASE_CONFIG="${BASE_CONFIG:-$DEFAULT_BASE_CONFIG}"
BASE_CKPT="${BASE_CKPT:-$DEFAULT_BASE_CKPT}"
EXPECTED_CKPT_SHA256="${EXPECTED_CKPT_SHA256:-$DEFAULT_CKPT_SHA256}"
V10_SOURCE_CHECKPOINT="${V10_SOURCE_CHECKPOINT:-}"
V10_CALIBRATION="${V10_CALIBRATION:-}"
TRAIN_INFOS="${TRAIN_INFOS:-data/nuscenes/nuscenes_infos_train.pkl}"
EXPECTED_TRAIN_INFOS_SHA256="${EXPECTED_TRAIN_INFOS_SHA256:-6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854}"
OUT_ROOT="${OUT_ROOT:-work_dirs/vov_umr_sgsup_b_$(date +%Y%m%d_%H%M%S)}"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_CALIBRATE="${RUN_CALIBRATE:-1}"
RUN_PACK="${RUN_PACK:-1}"
ALLOW_EXISTING_OUT_ROOT="${ALLOW_EXISTING_OUT_ROOT:-0}"

if [[ "$RUN_CALIBRATE" != "1" ]]; then
  echo "[ERROR] Formal VoV SGSUP-B forbids RUN_CALIBRATE=0; artifacts must be" >&2
  echo "        rebuilt from the currently audited train-only sidecars." >&2
  exit 2
fi

GPUS="${GPUS:-4}"
PORT="${PORT:-29500}"
TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-8}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
EV_LOG_COMPRESSION="${EV_LOG_COMPRESSION:-zstd}"
EV_LOG_FLUSH_INTERVAL="${EV_LOG_FLUSH_INTERVAL:-200}"
MAX_LABEL_RECORDS="${MAX_LABEL_RECORDS:-0}"
MAX_MAPPER_RECORDS="${MAX_MAPPER_RECORDS:-800000}"
MAPPER_EPOCHS="${MAPPER_EPOCHS:-80}"

if [[ "$SAMPLES_PER_GPU" != "1" ]]; then
  echo "[ERROR] Formal VoV SGSUP-B calibration requires SAMPLES_PER_GPU=1." >&2
  exit 2
fi

LOG_ROOT="${OUT_ROOT}/ev_logs"
CALIB_DIR="${OUT_ROOT}/calibration"
BRANCH_LABEL_ROOT="${CALIB_DIR}/branch_labels"
BRANCH_LABELS="${BRANCH_LABEL_ROOT}/train_oc/branch_candidate_labels.jsonl.gz"
BRANCH_LABEL_SUMMARY="${BRANCH_LABEL_ROOT}/train_oc/branch_candidate_labels_summary.json"
ACTION_MAPPER_DIR="${CALIB_DIR}/action_utility"
SCENE_DIR="${CALIB_DIR}/scene_gate"
CLEAN_REFERENCE_STATS="${CALIB_DIR}/clean_reference_stats.json"
SCENE_LABELS="${SCENE_DIR}/scene_gate_labels.jsonl"
SCENE_LABEL_SUMMARY="${SCENE_DIR}/scene_gate_labels_summary.json"
SCENE_MAPPER_PATH="${SCENE_DIR}/mapper_scene_bev_damage_applicable.json"
UTILITY_MAPPER_PATH="${ACTION_MAPPER_DIR}/mapper_candidate_useful.json"
MANIFEST="${CALIB_DIR}/${MODEL_PROFILE}_sgsup_b_manifest.json"
FINGERPRINT_FILE="${CALIB_DIR}/model_fingerprint.txt"
RUNTIME_ROOT="${OUT_ROOT}/runtime_pack"
RAW_CANDIDATE="RULE_SGSUP_B_runtime"
FINAL_CANDIDATE="RULE_SGSUP_B_contextheight055_v9bev_dyn10_30"
FINAL_CONFIG="${RUNTIME_ROOT}/configs/${FINAL_CANDIDATE}.py"
UMR_BASE_CANDIDATE="UMR_MATCHED_BASE"
UMR_BASE_CONFIG="${RUNTIME_ROOT}/configs/${UMR_BASE_CANDIDATE}.py"

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
export PATH="$(dirname "$(command -v "$PYTHON_BIN")"):${PATH}"

for path in "$BASE_CONFIG" "$BASE_CKPT" "$TRAIN_INFOS"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing required input: $path" >&2
    exit 2
  fi
done
if [[ "$MODEL_PROFILE" == "vov_v10" ]]; then
  for path in "$V10_SOURCE_CHECKPOINT" "$V10_CALIBRATION"; do
    if [[ ! -f "$path" ]]; then
      echo "[ERROR] vov_v10 requires calibration lineage input: $path" >&2
      exit 2
    fi
  done
fi

reject_eval_path() {
  local value="$1"
  local lower tokens
  lower="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  tokens="$(printf '%s' "$lower" | tr '/_.' '   ' | tr '-' ' ')"
  if [[ "$lower" == *"multicorrupt_results"* || \
        "$lower" == *"targeted_probe"* || \
        "$lower" == *"nuscenes_infos_val"* || \
        "$lower" == *"corrupted-val"* || \
        "$lower" == *"corrupted_val"* || \
        "$lower" == *"validation"* || \
        " $tokens " == *" val "* ]]; then
    echo "[ERROR] refusing validation/evaluation-derived path: $value" >&2
    exit 3
  fi
}

reject_eval_path "$BASE_CONFIG"
reject_eval_path "$TRAIN_INFOS"
reject_eval_path "$OUT_ROOT"

if [[ "$RUN_COLLECT" == "1" && -d "$OUT_ROOT" && \
      -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && \
      "$ALLOW_EXISTING_OUT_ROOT" != "1" ]]; then
  echo "[ERROR] OUT_ROOT is not empty: $OUT_ROOT" >&2
  echo "        Use a new OUT_ROOT, or resume with RUN_COLLECT=0." >&2
  exit 4
fi
mkdir -p "$CALIB_DIR" "$SCENE_DIR" "$ACTION_MAPPER_DIR"

if [[ -z "$EXPECTED_CKPT_SHA256" ]]; then
  EXPECTED_CKPT_SHA256="$(sha256sum "$BASE_CKPT" | awk '{print $1}')"
fi

echo "[VoV SGSUP-B] output=$OUT_ROOT"
echo "[VoV SGSUP-B] profile=$MODEL_PROFILE"
echo "[VoV SGSUP-B] base=$BASE_CONFIG"
echo "[VoV SGSUP-B] checkpoint=$BASE_CKPT"
echo "[VoV SGSUP-B] GPUs=$GPUS"

if [[ "$MODEL_PROFILE" == "vov_v10" ]]; then
  "$PYTHON_BIN" tools/v10_preflight.py \
    "$BASE_CONFIG" --profile vov_v10 --config-only \
    --checkpoint "$BASE_CKPT" \
    --source-checkpoint "$V10_SOURCE_CHECKPOINT" \
    --calibration "$V10_CALIBRATION"
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "===== [1/7] Verify VoV final checkpoint ====="
  if [[ "$MODEL_PROFILE" == "vov_umr" ]]; then
    "$PYTHON_BIN" tools/v9_rev20_preflight.py \
      "$BASE_CONFIG" --checkpoint "$BASE_CKPT" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch "$EXPECTED_CKPT_EPOCH"
  else
    "$PYTHON_BIN" tools/v10_preflight.py \
      "$BASE_CONFIG" --profile vov_v10 \
      --checkpoint "$BASE_CKPT" \
      --source-checkpoint "$V10_SOURCE_CHECKPOINT" \
      --calibration "$V10_CALIBRATION" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch "$EXPECTED_CKPT_EPOCH" \
      --dump-path "${CALIB_DIR}/v10_sgsup_input_preflight.json"
  fi
fi

if [[ "$RUN_COLLECT" == "1" ]]; then
  echo "===== [2/7] Collect VoV train-clean/train-O.C sidecars ====="
  BASE_CONFIG="$BASE_CONFIG" \
  BASE_CKPT="$BASE_CKPT" \
  TRAIN_INFOS="$TRAIN_INFOS" \
  OUT_ROOT="$OUT_ROOT" \
  GPUS="$GPUS" \
  PORT="$PORT" \
  TEST_WORKERS_PER_GPU="$TEST_WORKERS_PER_GPU" \
  SAMPLES_PER_GPU="$SAMPLES_PER_GPU" \
  EV_LOG_COMPRESSION="$EV_LOG_COMPRESSION" \
  EV_LOG_FLUSH_INTERVAL="$EV_LOG_FLUSH_INTERVAL" \
  EV_LOG_BRANCH_CANDIDATES=1 \
  EV_BRANCH_CANDIDATE_TOPK=300 \
  EV_BRANCH_CANDIDATE_SCORE_THR=0.0 \
  EV_BRANCH_CANDIDATE_BRANCHES=fused,bev,img \
  EV_MODEL_LABEL="$EV_MODEL_LABEL" \
    bash tools/umr_utility_collect_phase1_full.sh
elif [[ ! -d "$LOG_ROOT/train_clean" || ! -d "$LOG_ROOT/train_oc" ]]; then
  echo "[ERROR] RUN_COLLECT=0 requires train_clean and train_oc logs under $LOG_ROOT" >&2
  exit 5
fi

if [[ "$RUN_CALIBRATE" == "1" ]]; then
  echo "===== [3/7] Build bounded branch-candidate utility mapper ====="
  BR_ROOT="$OUT_ROOT" \
  INFOS="$TRAIN_INFOS" \
  OUT_DIR="$BRANCH_LABEL_ROOT" \
  SPLITS=train_oc \
  WRITE_LABELS=1 \
  LABEL_COMPRESSION=gz \
  MAX_RECORDS="$MAX_LABEL_RECORDS" \
  PYTHON="$PYTHON_BIN" \
    bash tools/umr_branch_candidate_label_phase1.sh

  LABEL_ROOT="$BRANCH_LABEL_ROOT" \
  SPLIT=train_oc \
  OUT_DIR="$ACTION_MAPPER_DIR" \
  MAX_MAPPER_RECORDS="$MAX_MAPPER_RECORDS" \
  EPOCHS="$MAPPER_EPOCHS" \
  PYTHON="$PYTHON_BIN" \
    bash tools/train_umr_action_utility_from_branch_labels.sh

  echo "===== [4/7] Build VoV clean reference and scene mapper ====="
  "$PYTHON_BIN" tools/export_umr_clean_reference_stats.py \
    --input "$LOG_ROOT/train_clean" \
    --output "$CLEAN_REFERENCE_STATS"

  "$PYTHON_BIN" tools/build_umr_scene_gate_labels.py \
    --scene-log "$LOG_ROOT" \
    --output "$SCENE_LABELS" \
    --summary "$SCENE_LABEL_SUMMARY" \
    --beams-s2-target 0.35 \
    --beams-s2-weight 0.45 \
    --beams-s3-weight 1.0 \
    --clean-weight 1.5 \
    --hard-negative-weight 2.0 \
    --other-negative-weight 1.5 \
    --other-corruptions negative

  "$PYTHON_BIN" tools/train_umr_scene_damage_mapper.py \
    --labels "$SCENE_LABELS" \
    --label-summary "$SCENE_LABEL_SUMMARY" \
    --output "$SCENE_MAPPER_PATH"
fi

for path in \
    "$UTILITY_MAPPER_PATH" "$SCENE_MAPPER_PATH" "$CLEAN_REFERENCE_STATS" \
    "$BRANCH_LABELS" "$BRANCH_LABEL_SUMMARY" \
    "$SCENE_LABELS" "$SCENE_LABEL_SUMMARY"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing calibrated VoV artifact: $path" >&2
    exit 6
  fi
done

echo "===== [5/7] Stamp model provenance ====="
"$PYTHON_BIN" tools/stamp_umr_model_provenance.py \
  --profile "$MODEL_PROFILE" \
  --base-config "$BASE_CONFIG" \
  --checkpoint "$BASE_CKPT" \
  --checkpoint-epoch "$EXPECTED_CKPT_EPOCH" \
  --expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA256" \
  --train-infos "$TRAIN_INFOS" \
  --expected-train-infos-sha256 "$EXPECTED_TRAIN_INFOS_SHA256" \
  --source-log-root "$LOG_ROOT" \
  --branch-labels "$BRANCH_LABELS" \
  --branch-label-summary "$BRANCH_LABEL_SUMMARY" \
  --scene-labels "$SCENE_LABELS" \
  --scene-label-summary "$SCENE_LABEL_SUMMARY" \
  --artifact "$SCENE_MAPPER_PATH" \
  --artifact "$UTILITY_MAPPER_PATH" \
  --artifact "$CLEAN_REFERENCE_STATS" \
  --manifest "$MANIFEST" \
  --fingerprint-output "$FINGERPRINT_FILE"
MODEL_FINGERPRINT="$(tr -d '[:space:]' < "$FINGERPRINT_FILE")"

if [[ "$RUN_PACK" == "1" ]]; then
  echo "===== [6/7] Build raw parity control and finalized guarded runtime config ====="
  OUT_ROOT="$RUNTIME_ROOT" \
  BASE_CONFIG="$BASE_CONFIG" \
  BASE_CKPT="$BASE_CKPT" \
  CLEAN_REFERENCE_STATS="$CLEAN_REFERENCE_STATS" \
  SCENE_MAPPER_PATH="$SCENE_MAPPER_PATH" \
  UTILITY_MAPPER_PATH="$UTILITY_MAPPER_PATH" \
  BRANCH_LABEL_ROOT="$BRANCH_LABEL_ROOT" \
  EXPECTED_MODEL_FINGERPRINT="$MODEL_FINGERPRINT" \
  SUPP_REQUIRE_REFERENCE_STATS=1 \
  RUN_STATIC_AUDIT=1 \
  REQUIRE_RUNTIME_PREFLIGHT=1 \
  CANDIDATES="$RAW_CANDIDATE $FINAL_CANDIDATE" \
    bash tools/umr_rule_runtime_pack.sh

  BASE_CONFIG_REL="$($PYTHON_BIN - "$BASE_CONFIG" "$(dirname "$UMR_BASE_CONFIG")" <<'PY'
import os
import sys
print(os.path.relpath(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])))
PY
  )"
  {
    printf "_base_ = '%s'\n" "${BASE_CONFIG_REL}"
  } > "$UMR_BASE_CONFIG"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$UMR_BASE_CANDIDATE" "matched_${MODEL_PROFILE}_baseline" "supplement_off" \
    "base_runtime" "action_off" "$UMR_BASE_CONFIG" "$BASE_CKPT" \
    >> "${RUNTIME_ROOT}/summaries/candidates.tsv"

  cat > "${RUNTIME_ROOT}/preflight.sh" <<SH
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "\$ROOT_DIR"
PYTHON_BIN="\${PYTHON_BIN:-$PYTHON_BIN}"
BASE_CKPT="\${BASE_CKPT:-$BASE_CKPT}"
EVALUATION_CONFIG="\${1:-$FINAL_CONFIG}"
"\$PYTHON_BIN" tools/preflight_vov_umr_sgsup_b.py \\
  --profile "$MODEL_PROFILE" \\
  --config "$FINAL_CONFIG" \\
  --evaluation-config "\$EVALUATION_CONFIG" \\
  --base-config "$BASE_CONFIG" \\
  --checkpoint "\$BASE_CKPT" \\
  --manifest "$MANIFEST" \\
  --repo-root "\$ROOT_DIR" \\
  --expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA256"
SH
  chmod +x "${RUNTIME_ROOT}/preflight.sh"

  "$PYTHON_BIN" tools/register_vov_umr_sgsup_runtime_configs.py \
    --profile "$MODEL_PROFILE" \
    --manifest "$MANIFEST" \
    --base-config "$BASE_CONFIG" \
    --matched-base "$UMR_BASE_CONFIG" \
    --raw-control "${RUNTIME_ROOT}/configs/${RAW_CANDIDATE}.py" \
    --formal-final "$FINAL_CONFIG"
fi

if [[ ! -f "$FINAL_CONFIG" ]]; then
  echo "[ERROR] final runtime config was not generated: $FINAL_CONFIG" >&2
  exit 7
fi

echo "===== [7/7] Fully merged config and checkpoint audit ====="
for evaluation_config in \
    "$UMR_BASE_CONFIG" \
    "${RUNTIME_ROOT}/configs/${RAW_CANDIDATE}.py" \
    "$FINAL_CONFIG"; do
  "$PYTHON_BIN" tools/preflight_vov_umr_sgsup_b.py \
    --profile "$MODEL_PROFILE" \
    --config "$FINAL_CONFIG" \
    --evaluation-config "$evaluation_config" \
    --base-config "$BASE_CONFIG" \
    --checkpoint "$BASE_CKPT" \
    --manifest "$MANIFEST" \
    --repo-root "$ROOT_DIR" \
    --require-source-files \
    --expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA256"
done

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  if [[ "$MODEL_PROFILE" == "vov_umr" ]]; then
    "$PYTHON_BIN" tools/v9_rev20_preflight.py \
      "$FINAL_CONFIG" --checkpoint "$BASE_CKPT" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch "$EXPECTED_CKPT_EPOCH"
  else
    "$PYTHON_BIN" tools/v10_preflight.py \
      "$FINAL_CONFIG" --profile vov_v10 \
      --checkpoint "$BASE_CKPT" \
      --source-checkpoint "$V10_SOURCE_CHECKPOINT" \
      --calibration "$V10_CALIBRATION" \
      --require-full-checkpoint-coverage \
      --expected-checkpoint-epoch "$EXPECTED_CKPT_EPOCH" \
      --dump-path "${CALIB_DIR}/v10_sgsup_final_preflight.json"
  fi
fi

echo
echo "VoV SGSUP-B runtime pack is ready."
echo "FINAL_CONFIG=$FINAL_CONFIG"
echo "BASE_CKPT=$BASE_CKPT"
echo "MANIFEST=$MANIFEST"
echo "MODEL_FINGERPRINT=$MODEL_FINGERPRINT"
echo "RAW_PARITY_CONTROL=${RUNTIME_ROOT}/configs/${RAW_CANDIDATE}.py"
echo "UMR_MATCHED_BASE=$UMR_BASE_CONFIG"
echo "TARGETED_HELPER=${RUNTIME_ROOT}/validate_targeted_all.sh"
echo "FULL_HELPER=${RUNTIME_ROOT}/validate_full_candidate.sh"
