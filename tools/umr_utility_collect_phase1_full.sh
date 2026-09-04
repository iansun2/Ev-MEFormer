#!/usr/bin/env bash
set -Eeuo pipefail

# Collect train-clean and train-online-corruption EV sidecar logs for UMR
# utility calibration. This script is server-side and train-only by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"

BASE_CONFIG="${BASE_CONFIG:-projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v10_router_selector_frozen.py}"
BASE_CKPT="${BASE_CKPT:-work_dirs/v10_router_selector/20260522-143929/latest.pth}"
TRAIN_INFOS="${TRAIN_INFOS:-data/nuscenes/nuscenes_infos_train.pkl}"
OUT_ROOT="${OUT_ROOT:-work_dirs/umr_utility_phase1_full_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="${OUT_ROOT}/configs"
LOG_ROOT="${OUT_ROOT}/ev_logs"

TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-8}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
EV_LOG_COMPRESSION="${EV_LOG_COMPRESSION:-zstd}"
EV_LOG_FLUSH_INTERVAL="${EV_LOG_FLUSH_INTERVAL:-200}"
EV_LOG_BRANCH_CANDIDATES="${EV_LOG_BRANCH_CANDIDATES:-0}"
EV_BRANCH_CANDIDATE_TOPK="${EV_BRANCH_CANDIDATE_TOPK:-300}"
EV_BRANCH_CANDIDATE_SCORE_THR="${EV_BRANCH_CANDIDATE_SCORE_THR:-0.0}"
EV_BRANCH_CANDIDATE_BRANCHES="${EV_BRANCH_CANDIDATE_BRANCHES:-fused,bev,img}"
OC_CORRUPT_PROB="${OC_CORRUPT_PROB:-0.50}"
OC_SEVERITIES="${OC_SEVERITIES:-1,2,3}"
OC_WEIGHTS="${OC_WEIGHTS:-beamsreducing=0.24,pointsreducing=0.16,spatialmisalignment=0.13,temporalmisalignment=0.09,fog=0.14,snow=0.08,brightness=0.04,dark=0.05,motionblur=0.05,missingcamera=0.02}"
CONDITIONS="${CONDITIONS:-train_clean train_oc}"
SEED="${SEED:-0}"

if [[ -n "${GPUS:-}" ]]; then
  PHASE1_GPUS="$GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _phase1_visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  PHASE1_GPUS="${#_phase1_visible_gpus[@]}"
else
  PHASE1_GPUS=1
fi
PORT="${PORT:-${DIST_PORT:-29500}}"
export PORT

if [[ "$SAMPLES_PER_GPU" != "1" ]]; then
  echo "[ERROR] Formal UMR collection requires SAMPLES_PER_GPU=1; got $SAMPLES_PER_GPU." >&2
  echo "        Scene quantiles and one-record-per-scene provenance are defined for batch 1." >&2
  exit 2
fi

fail_val_like() {
  local value="$1"
  local lower tokens
  lower="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  tokens="$(printf '%s' "$lower" | tr '/_.' '   ' | tr '-' ' ')"
  if [[ "$lower" == *"nuscenes_infos_val"* || \
        "$lower" == *"multicorrupt_results"* || \
        "$lower" == *"targeted_probe"* || \
        "$lower" == *"corrupted_val"* || \
        "$lower" == *"corrupted-val"* || \
        "$lower" == *"validation"* || \
        " $tokens " == *" val "* ]]; then
    echo "[ERROR] Refusing val/corrupted-val-like input: $value" >&2
    exit 3
  fi
}

fail_val_like "$BASE_CONFIG"
fail_val_like "$TRAIN_INFOS"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[ERROR] Missing BASE_CONFIG: $BASE_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[ERROR] Missing BASE_CKPT: $BASE_CKPT" >&2
  echo "        Set BASE_CKPT to the final UMR checkpoint on this machine." >&2
  exit 1
fi
if [[ ! -f "$TRAIN_INFOS" ]]; then
  echo "[ERROR] Missing TRAIN_INFOS: $TRAIN_INFOS" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR" "$LOG_ROOT"

echo "===== Generate train-only Phase 1 logging configs ====="
python tools/make_umr_phase1_log_configs.py \
  --base-config "$BASE_CONFIG" \
  --train-infos "$TRAIN_INFOS" \
  --out-dir "$CONFIG_DIR" \
  --workers-per-gpu "$TEST_WORKERS_PER_GPU" \
  --samples-per-gpu "$SAMPLES_PER_GPU" \
  --oc-corrupt-prob "$OC_CORRUPT_PROB" \
  --oc-severities "$OC_SEVERITIES" \
  --oc-weights "$OC_WEIGHTS"

run_condition() {
  local condition="$1"
  local cfg split corruption severity log_dir
  case "$condition" in
    train_clean)
      cfg="${CONFIG_DIR}/umr_phase1_train_clean_log.py"
      split="train_clean"
      corruption="clean"
      severity="0"
      ;;
    train_oc)
      cfg="${CONFIG_DIR}/umr_phase1_train_oc_log.py"
      split="train_oc"
      corruption=""
      severity=""
      ;;
    *)
      echo "[ERROR] Unknown condition: $condition" >&2
      exit 2
      ;;
  esac
  log_dir="${LOG_ROOT}/${condition}"
  mkdir -p "$log_dir"
  echo
  echo "===== Collect EV sidecar: ${condition} ====="
  echo "config=$cfg"
  echo "ckpt=$BASE_CKPT"
  echo "log_dir=$log_dir"
  echo "workers=$TEST_WORKERS_PER_GPU samples_per_gpu=$SAMPLES_PER_GPU compression=$EV_LOG_COMPRESSION"
  echo "gpus=$PHASE1_GPUS port=$PORT"
  common_args=(
      --save-ev-log
      --ev-log-only
      --ev-log-dir "$log_dir"
      --ev-log-mode sidecar
      --ev-log-compression "$EV_LOG_COMPRESSION"
      --ev-log-flush-interval "$EV_LOG_FLUSH_INTERVAL"
      --seed "$SEED"
      --cfg-options
        data.workers_per_gpu="$TEST_WORKERS_PER_GPU"
        data.test_dataloader.workers_per_gpu="$TEST_WORKERS_PER_GPU"
        data.test_dataloader.samples_per_gpu="$SAMPLES_PER_GPU"
  )
  if [[ "$EV_LOG_BRANCH_CANDIDATES" == "1" ]]; then
    common_args+=(
      --ev-log-branch-candidates
      --ev-branch-candidate-topk "$EV_BRANCH_CANDIDATE_TOPK"
      --ev-branch-candidate-score-thr "$EV_BRANCH_CANDIDATE_SCORE_THR"
      --ev-branch-candidate-branches "$EV_BRANCH_CANDIDATE_BRANCHES"
    )
  fi
  if [[ "$PHASE1_GPUS" -gt 1 ]]; then
    EV_LOG_RECORD_SOURCE_HASHES=1 \
      EV_SPLIT="$split" EV_MODEL_LABEL="${EV_MODEL_LABEL:-EvMEFormer_UMR_phase1}" \
      EV_CORRUPTION="$corruption" EV_SEVERITY="$severity" \
      bash tools/dist_test.sh "$cfg" "$BASE_CKPT" "$PHASE1_GPUS" "${common_args[@]}"
  else
    EV_LOG_RECORD_SOURCE_HASHES=1 \
      EV_SPLIT="$split" EV_MODEL_LABEL="${EV_MODEL_LABEL:-EvMEFormer_UMR_phase1}" \
      EV_CORRUPTION="$corruption" EV_SEVERITY="$severity" \
      python tools/test.py "$cfg" "$BASE_CKPT" \
      "${common_args[@]}"
  fi
}

for condition in $CONDITIONS; do
  run_condition "$condition"
done

cat > "${OUT_ROOT}/README.txt" <<EOF
UMR utility calibration Phase 1-full EV log pack

Base config: $BASE_CONFIG
Checkpoint:  $BASE_CKPT
Train infos: $TRAIN_INFOS
Conditions:  $CONDITIONS
Logs:        $LOG_ROOT
Branch candidates: $EV_LOG_BRANCH_CANDIDATES
Branch top-k:      $EV_BRANCH_CANDIDATE_TOPK
Branch score thr:  $EV_BRANCH_CANDIDATE_SCORE_THR
Branch names:      $EV_BRANCH_CANDIDATE_BRANCHES

This pack is train-only by construction. Do not replace TRAIN_INFOS with val.
The train-O.C config inserts OnlineSyntheticMultiCorrupt into the test pipeline
so GT boxes in nuscenes_infos_train.pkl remain in the original coordinate frame.
EOF

echo
echo "Done."
echo "OUT_ROOT=$OUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
