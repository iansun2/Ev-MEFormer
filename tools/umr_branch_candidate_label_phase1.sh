#!/usr/bin/env bash
set -Eeuo pipefail

BR_ROOT="${BR_ROOT:-work_dirs/umr_branch_candidate_phase1_20260627_154805}"
INFOS="${INFOS:-data/nuscenes/nuscenes_infos_train.pkl}"
OUT_DIR="${OUT_DIR:-${BR_ROOT}/label_analysis}"
SPLITS="${SPLITS:-train_clean train_oc}"
MAX_RECORDS="${MAX_RECORDS:-0}"
WRITE_LABELS="${WRITE_LABELS:-0}"
LABEL_COMPRESSION="${LABEL_COMPRESSION:-gz}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1000000}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR"
processed=0

for split in $SPLITS; do
  case "$split" in
    train_clean|train_oc|train) ;;
    *)
      echo "[ERROR] unsupported formal split: $split" >&2
      echo "        Use train_clean/train_oc/train only, or call the Python tool with --allow-debug-val for throwaway diagnostics." >&2
      exit 1
      ;;
  esac
  log_root="${BR_ROOT}/ev_logs/${split}"
  split_out="${OUT_DIR}/${split}"
  mkdir -p "$split_out"
  if [[ ! -d "$log_root" ]]; then
    echo "[WARN] missing split log root: $log_root" >&2
    continue
  fi
  label_args=()
  if [[ "$WRITE_LABELS" == "1" ]]; then
    ext="jsonl"
    if [[ "$LABEL_COMPRESSION" == "gz" || "$LABEL_COMPRESSION" == "gzip" ]]; then
      ext="jsonl.gz"
    fi
    label_args+=(--label-output "branch_candidate_labels.${ext}")
  fi
  echo "===== Branch-candidate labels: ${split} ====="
  "$PYTHON" tools/build_umr_branch_candidate_labels.py \
    --branch-log-root "$log_root" \
    --infos "$INFOS" \
    --out-dir "$split_out" \
    --max-records "$MAX_RECORDS" \
    --progress-interval "$PROGRESS_INTERVAL" \
    --expected-split "$split" \
    "${label_args[@]}"
  processed=$((processed + 1))
done

if [[ "$processed" -eq 0 ]]; then
  echo "[ERROR] no requested split log roots were found under: ${BR_ROOT}/ev_logs" >&2
  exit 1
fi

{
cat <<EOF
UMR branch-candidate Phase-1 label analysis

branch root: ${BR_ROOT}
infos:       ${INFOS}
splits:      ${SPLITS}
max_records: ${MAX_RECORDS}
write_labels:${WRITE_LABELS}
python:      ${PYTHON}

Summary files:
EOF
for split in $SPLITS; do
  if [[ -d "${OUT_DIR}/${split}" ]]; then
    echo "  ${OUT_DIR}/${split}/branch_candidate_labels_summary.md"
  fi
done
cat <<EOF

Formal notes:
  - Uses train/train-O.C branch-candidate logs and train infos only.
  - Corruption/severity are diagnostic grouping metadata only.
  - Row labels are center-distance proxy labels, not official nuScenes AP matching.
EOF
} > "${OUT_DIR}/README.txt"

echo "[BranchLabel] output root: ${OUT_DIR}"
