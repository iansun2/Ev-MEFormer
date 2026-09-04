#!/usr/bin/env bash
set -Eeuo pipefail

LABEL_ROOT="${LABEL_ROOT:-work_dirs/umr_branch_candidate_phase1_20260627_154805/label_analysis}"
SPLIT="${SPLIT:-train_oc}"
OUT_DIR="${OUT_DIR:-work_dirs/umr_action_utility_$(date +%Y%m%d_%H%M%S)}"
MAX_MAPPER_RECORDS="${MAX_MAPPER_RECORDS:-800000}"
EPOCHS="${EPOCHS:-80}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR"

find_labels() {
  local split_dir="$1"
  local candidates=(
    "${split_dir}/branch_candidate_labels.jsonl"
    "${split_dir}/branch_candidate_labels.jsonl.gz"
    "${split_dir}/branch_candidate_labels.jsonl.zst"
    "${split_dir}/branch_candidate_labels.jsonl.zstd"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

case "$SPLIT" in
  train|train_clean|train_oc) ;;
  *)
    echo "[ERROR] unsupported formal split: $SPLIT" >&2
    echo "        Use train/train_clean/train_oc labels only." >&2
    exit 2
    ;;
esac

lower="$(printf '%s' "$LABEL_ROOT" | tr '[:upper:]' '[:lower:]')"
if [[ "$lower" == *"multicorrupt_results"* || "$lower" == *"targeted_probe"* || \
      "$lower" == *"nuscenes_infos_val"* || "$lower" == *"corrupted-val"* || \
      "$lower" == *"corrupted_val"* || "$lower" == *"corrupted/val"* || \
      "$lower" == *"val_corrupt"* || "$lower" == *"val_clean"* || \
      "$lower" == *"validation"* || "$lower" == "val/"* || \
      "$lower" == *"/val/"* || "$lower" == *"/val" ]]; then
  echo "[ERROR] refusing evaluation-derived label root: $LABEL_ROOT" >&2
  exit 3
fi

LABELS="$(find_labels "${LABEL_ROOT}/${SPLIT}")" || {
  echo "[ERROR] branch candidate row labels not found under: ${LABEL_ROOT}/${SPLIT}" >&2
  echo "        Rebuild labels with: WRITE_LABELS=1 bash tools/umr_branch_candidate_label_phase1.sh" >&2
  exit 4
}

echo "===== Train UMR action utility mappers ====="
echo "labels=${LABELS}"
echo "out_dir=${OUT_DIR}"
echo "max_records=${MAX_MAPPER_RECORDS}"

"$PYTHON" tools/train_umr_utility_mapper.py \
  --labels "$LABELS" \
  --output "${OUT_DIR}/mapper_candidate_useful.json" \
  --target candidate_useful \
  --feature-preset action_branch \
  --max-records "$MAX_MAPPER_RECORDS" \
  --epochs "$EPOCHS"

"$PYTHON" tools/train_umr_utility_mapper.py \
  --labels "$LABELS" \
  --output "${OUT_DIR}/mapper_img_supp_opportunity.json" \
  --target img_supp_opportunity \
  --feature-preset action_branch \
  --branch-filter img \
  --max-records "$MAX_MAPPER_RECORDS" \
  --epochs "$EPOCHS"

"$PYTHON" tools/train_umr_utility_mapper.py \
  --labels "$LABELS" \
  --output "${OUT_DIR}/mapper_bev_supp_opportunity.json" \
  --target bev_supp_opportunity \
  --feature-preset action_branch \
  --branch-filter bev \
  --max-records "$MAX_MAPPER_RECORDS" \
  --epochs "$EPOCHS"

"$PYTHON" tools/train_umr_utility_mapper.py \
  --labels "$LABELS" \
  --output "${OUT_DIR}/mapper_fused_slot_candidate_useful.json" \
  --target candidate_useful \
  --feature-preset action_branch \
  --branch-filter fused \
  --max-records "$MAX_MAPPER_RECORDS" \
  --epochs "$EPOCHS"

cat > "${OUT_DIR}/README.txt" <<EOF
UMR action utility mappers

label_root: ${LABEL_ROOT}
split:      ${SPLIT}
labels:     ${LABELS}
max_rows:   ${MAX_MAPPER_RECORDS}

Outputs to copy to local runtime validation:
  ${OUT_DIR}/mapper_candidate_useful.json
  ${OUT_DIR}/mapper_candidate_useful.md
  ${OUT_DIR}/mapper_img_supp_opportunity.json
  ${OUT_DIR}/mapper_img_supp_opportunity.md
  ${OUT_DIR}/mapper_bev_supp_opportunity.json
  ${OUT_DIR}/mapper_bev_supp_opportunity.md
  ${OUT_DIR}/mapper_fused_slot_candidate_useful.json
  ${OUT_DIR}/mapper_fused_slot_candidate_useful.md

Formal notes:
  - These mappers are fit only from train/train-O.C branch-candidate labels.
  - Runtime configs must not use corruption/severity/GT/evaluation fields.
  - The first local smoke validation should compare against RULE_SGSUP_B_runtime
    on clean, beamsreducing-s3, and fog-s3 before any broader run.
EOF

echo "[UMR action utility] output root: ${OUT_DIR}"
