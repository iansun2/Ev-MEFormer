#!/usr/bin/env bash
set -Eeuo pipefail

# Ev-MEFormer R50 three-line validation:
#   line1_eud            final EUD checkpoint
#   line2_umr_base       matched UMR checkpoint, bounded supplement disabled
#   line3_umr_supplement same UMR checkpoint, guarded bounded supplement enabled
#
# A formal line2/line3 run must use the checkpoint-specific runtime pack created
# by tools/prepare_r50_umr_sgsup_b.sh. Historical configs are accepted only when
# ALLOW_NONFORMAL_RUNTIME=1 is set explicitly.

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0 MULTICORRUPT_ROOT=/path/to/MultiCorrupt \
  CHAIN_MANIFEST=/path/to/r50/chain_manifest.json \
    bash run_r50_three_lines.sh

  CUDA_VISIBLE_DEVICES=0 MULTICORRUPT_ROOT=/path/to/MultiCorrupt \
  RUNTIME_PACK=/path/to/bounded_candidate_supplement/runtime_pack \
  UMR_CKPT=/path/to/conservative_head_routing/epoch_6.pth \
    bash run_r50_three_lines.sh

Common examples:
  # Canary: matched baseline and guarded supplement, clean + beams S3.
  EVAL_MODE=targeted \
  LINES="line2_umr_base line3_umr_supplement" \
  CORRUPTIONS=beamsreducing SEVERITY_LEVELS=3 \
  MULTICORRUPT_ROOT=/path/to/MultiCorrupt \
  CHAIN_MANIFEST=/path/to/chain_manifest.json \
    bash run_r50_three_lines.sh

  # Full-10 for line3 only.
  EVAL_MODE=full LINES=line3_umr_supplement \
  MULTICORRUPT_ROOT=/path/to/MultiCorrupt \
  CHAIN_MANIFEST=/path/to/chain_manifest.json \
    bash run_r50_three_lines.sh

Main environment variables:
  CHAIN_MANIFEST       Preferred. Resolves UMR_CKPT and RUNTIME_PACK.
  RUNTIME_PACK         Formal runtime_pack directory (if no manifest).
  EUD_CKPT             R50 EUD epoch_20.pth.
  UMR_CKPT             R50 conservative-head-routing epoch_6.pth.
  EVAL_MODE            full (default) or targeted.
  LINES                Space-separated line1_eud, line2_umr_base,
                       line3_umr_supplement (default: all three).
  RESULT_ROOT          Fresh result root; default includes a UTC timestamp.
  NUSCENES_DATA_DIR    Must be this checkout's real data/nuscenes directory.
  NUSCENES_BACKUP_DIR  Must be an absent sibling of data/nuscenes.
  TEST_WORKERS_PER_GPU Default: 8.
  RUN_VIZ              Default: 0.
  EV_LOG_MODE          Default: off for full, sidecar for targeted.

Non-formal diagnostic override:
  Set ALLOW_NONFORMAL_RUNTIME=1 and provide both UMR_BASE_CONFIG and
  SUPP_CONFIG explicitly. Such a run is not a formal handoff result.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 2
}

say() {
  printf '\n\033[1m%s\033[0m\n' "$*"
}

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

canonical_path() {
  realpath -m -- "$1"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_EXE="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
[[ -n "$PYTHON_EXE" ]] || die "Python executable not found: $PYTHON_BIN"
# run_targeted_corruption_probe.sh also invokes python/python3 directly. Put the
# selected environment first so every nested command uses the same conda env.
export PATH="$(dirname "$PYTHON_EXE"):$PATH"

CHAIN_MANIFEST="${CHAIN_MANIFEST:-}"
RUNTIME_PACK="${RUNTIME_PACK:-}"
UMR_CKPT="${UMR_CKPT:-}"
RUNTIME_MANIFEST=""

if [[ -n "$CHAIN_MANIFEST" ]]; then
  [[ -f "$CHAIN_MANIFEST" ]] || die "chain manifest not found: $CHAIN_MANIFEST"
  mapfile -t manifest_values < <(
    "$PYTHON_BIN" - "$CHAIN_MANIFEST" "$ROOT_DIR" <<'PY'
import json
import os
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if not str(payload.get("format", "")).startswith("r50_umr_functional_chain_"):
    raise SystemExit("not an R50 functional-chain manifest")

repo_root = os.path.realpath(sys.argv[2])

def portable_path(path):
    if not path:
        return ""
    resolved = os.path.realpath(path)
    if os.path.exists(resolved):
        return resolved
    normalized = path.replace("\\", "/")
    marker = "/work_dirs/"
    if marker in normalized:
        candidate = os.path.join(
            repo_root, "work_dirs", normalized.split(marker, 1)[1])
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    return resolved

checkpoint = payload.get("conservative_head_routing_checkpoint", "")
supplement_root = payload.get("bounded_candidate_supplement_root", "")
print(portable_path(checkpoint))
print(portable_path(os.path.join(supplement_root, "runtime_pack")) if supplement_root else "")
PY
  )
  [[ ${#manifest_values[@]} -eq 2 ]] || die "invalid chain manifest: $CHAIN_MANIFEST"
  if [[ -n "$UMR_CKPT" && "$(canonical_path "$UMR_CKPT")" != "${manifest_values[0]}" ]]; then
    die "UMR_CKPT conflicts with chain manifest"
  fi
  if [[ -n "$RUNTIME_PACK" && "$(canonical_path "$RUNTIME_PACK")" != "${manifest_values[1]}" ]]; then
    die "RUNTIME_PACK conflicts with chain manifest"
  fi
  UMR_CKPT="${manifest_values[0]}"
  RUNTIME_PACK="${manifest_values[1]}"
fi

EUD_CONFIG="${EUD_CONFIG:-${EUD_CFG:-projects/configs/edl_pme_bbox_r3c_cls_voxel0100_r50_800x320_cbgs.py}}"
EUD_CKPT="${EUD_CKPT:-work_dirs/edl_pme_bbox_r3c_cls_voxel0100_r50_800x320_cbgs/edl-moad/epoch_20.pth}"
UMR_BASE_CONFIG="${UMR_BASE_CONFIG:-${UMR_BASE_CFG:-}}"
SUPP_CONFIG="${SUPP_CONFIG:-${SUPP_CFG:-}}"

if [[ -n "$RUNTIME_PACK" ]]; then
  RUNTIME_PACK="$(canonical_path "$RUNTIME_PACK")"
  [[ -d "$RUNTIME_PACK" ]] || die "runtime pack not found: $RUNTIME_PACK"
  [[ -z "$UMR_BASE_CONFIG" ]] || die "do not combine RUNTIME_PACK with UMR_BASE_CONFIG"
  [[ -z "$SUPP_CONFIG" ]] || die "do not combine RUNTIME_PACK with SUPP_CONFIG"
  UMR_BASE_CONFIG="$RUNTIME_PACK/configs/UMR_MATCHED_BASE.py"
  SUPP_CONFIG="$RUNTIME_PACK/configs/RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py"
  RUNTIME_MANIFEST="$(dirname "$RUNTIME_PACK")/calibration/r50_umr_sgsup_b_manifest.json"
fi

EVAL_MODE="${EVAL_MODE:-full}"
LINES="${LINES:-line1_eud line2_umr_base line3_umr_supplement}"
CORRUPTIONS="${CORRUPTIONS:-beamsreducing}"
SEVERITY_LEVELS="${SEVERITY_LEVELS:-3}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/MultiCorrupt_results/r50_three_lines_${RUN_ID}}"
MULTICORRUPT_ROOT="${MULTICORRUPT_ROOT:-}"
NUSCENES_DATA_DIR="${NUSCENES_DATA_DIR:-$ROOT_DIR/data/nuscenes}"
NUSCENES_BACKUP_DIR="${NUSCENES_BACKUP_DIR:-$ROOT_DIR/data/nuscenes.r50-three-lines-backup}"
TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-8}"
RUN_VIZ="${RUN_VIZ:-0}"
EV_LOG_SCENE_ONLY="${EV_LOG_SCENE_ONLY:-0}"
EV_LOG_COMPRESSION="${EV_LOG_COMPRESSION:-none}"
ALLOW_NONFORMAL_RUNTIME="${ALLOW_NONFORMAL_RUNTIME:-0}"
ALLOW_EXISTING_RESULTS="${ALLOW_EXISTING_RESULTS:-0}"
CHECKPOINT_META="${CHECKPOINT_META:-1}"
ADMISSION_MIN_RECORDS="${ADMISSION_MIN_RECORDS:-6019}"
REQUIRE_POSITIVE_SUPPLEMENT="${REQUIRE_POSITIVE_SUPPLEMENT:-0}"

if [[ -z ${EV_LOG_MODE+x} ]]; then
  if [[ "$EVAL_MODE" == "targeted" ]]; then
    EV_LOG_MODE=sidecar
  else
    EV_LOG_MODE=off
  fi
fi

case "$EVAL_MODE" in
  full|targeted) ;;
  *) die "EVAL_MODE must be full or targeted" ;;
esac
[[ -n "$MULTICORRUPT_ROOT" ]] || die "set MULTICORRUPT_ROOT to the extracted dataset"
[[ "$TEST_WORKERS_PER_GPU" =~ ^[0-9]+$ ]] || die "TEST_WORKERS_PER_GPU must be a non-negative integer"
[[ "$ADMISSION_MIN_RECORDS" =~ ^[1-9][0-9]*$ ]] || die "ADMISSION_MIN_RECORDS must be positive"

read -r -a LINE_LIST <<< "$LINES"
read -r -a CORRUPTION_LIST <<< "$CORRUPTIONS"
read -r -a SEVERITY_LIST <<< "$SEVERITY_LEVELS"
[[ ${#LINE_LIST[@]} -gt 0 ]] || die "LINES is empty"
[[ ${#CORRUPTION_LIST[@]} -gt 0 ]] || die "CORRUPTIONS is empty"
[[ ${#SEVERITY_LIST[@]} -gt 0 ]] || die "SEVERITY_LEVELS is empty"

need_eud=0
need_umr=0
for line in "${LINE_LIST[@]}"; do
  case "$line" in
    line1_eud) need_eud=1 ;;
    line2_umr_base|line3_umr_supplement) need_umr=1 ;;
    *) die "unknown line: $line" ;;
  esac
done

if [[ "$need_umr" == "1" ]]; then
  [[ -n "$UMR_CKPT" ]] || die "set CHAIN_MANIFEST or UMR_CKPT for UMR validation"
  if [[ -z "$RUNTIME_PACK" ]]; then
    is_enabled "$ALLOW_NONFORMAL_RUNTIME" || \
      die "formal UMR validation requires CHAIN_MANIFEST or RUNTIME_PACK"
    [[ -n "$UMR_BASE_CONFIG" && -n "$SUPP_CONFIG" ]] || \
      die "non-formal mode requires UMR_BASE_CONFIG and SUPP_CONFIG"
    printf '[WARN] ALLOW_NONFORMAL_RUNTIME=1: results are diagnostic, not formal handoff results.\n' >&2
  fi
fi

say "[preflight] R50 three-line validation"
[[ -f "$ROOT_DIR/multicorrupt_evaluation.sh" ]] || die "run from an Ev-MEFormer checkout"
[[ -f "$ROOT_DIR/run_targeted_corruption_probe.sh" ]] || die "targeted probe script is missing"
[[ -d "$MULTICORRUPT_ROOT" ]] || die "MultiCorrupt root not found: $MULTICORRUPT_ROOT"
[[ -d "$NUSCENES_DATA_DIR" ]] || die "clean nuScenes directory not found: $NUSCENES_DATA_DIR"
[[ ! -L "$NUSCENES_DATA_DIR" ]] || \
  die "data/nuscenes must be a real directory; use symlinks inside it, not on the directory itself"
[[ "$(canonical_path "$NUSCENES_DATA_DIR")" == "$(canonical_path "$ROOT_DIR/data/nuscenes")" ]] || \
  die "NUSCENES_DATA_DIR must be this checkout's data/nuscenes"
[[ "$(dirname "$(canonical_path "$NUSCENES_BACKUP_DIR")")" == \
   "$(dirname "$(canonical_path "$NUSCENES_DATA_DIR")")" ]] || \
  die "NUSCENES_BACKUP_DIR must be beside data/nuscenes"
[[ ! -e "$NUSCENES_BACKUP_DIR" ]] || \
  die "backup already exists; inspect and restore it first: $NUSCENES_BACKUP_DIR"
[[ -f "$NUSCENES_DATA_DIR/nuscenes_infos_val.pkl" ]] || \
  die "missing data/nuscenes/nuscenes_infos_val.pkl"
"$PYTHON_BIN" -c 'import mmcv, torch, mmdet3d' >/dev/null 2>&1 || \
  die "$PYTHON_BIN cannot import mmcv, torch, and mmdet3d"
"$PYTHON_BIN" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' || \
  die "CUDA is unavailable in the selected Python environment"

if [[ "$EVAL_MODE" == "full" ]]; then
  FULL_CORRUPTIONS=(beamsreducing brightness dark fog missingcamera motionblur pointsreducing snow spatialmisalignment temporalmisalignment)
  FULL_SEVERITIES=(1 2 3)
  for corruption in "${FULL_CORRUPTIONS[@]}"; do
    for severity in "${FULL_SEVERITIES[@]}"; do
      [[ -d "$MULTICORRUPT_ROOT/$corruption/$severity" ]] || \
        die "incomplete MultiCorrupt tree: $MULTICORRUPT_ROOT/$corruption/$severity"
    done
  done
else
  for corruption in "${CORRUPTION_LIST[@]}"; do
    for severity in "${SEVERITY_LIST[@]}"; do
      [[ -d "$MULTICORRUPT_ROOT/$corruption/$severity" ]] || \
        die "targeted condition not found: $MULTICORRUPT_ROOT/$corruption/$severity"
    done
  done
fi

if [[ "$need_eud" == "1" ]]; then
  [[ -f "$EUD_CONFIG" ]] || die "EUD config not found: $EUD_CONFIG"
  [[ -f "$EUD_CKPT" ]] || die "EUD checkpoint not found: $EUD_CKPT"
fi
if [[ "$need_umr" == "1" ]]; then
  [[ -f "$UMR_CKPT" ]] || die "UMR checkpoint not found: $UMR_CKPT"
  [[ -f "$UMR_BASE_CONFIG" ]] || die "matched UMR config not found: $UMR_BASE_CONFIG"
  [[ -f "$SUPP_CONFIG" ]] || die "supplement config not found: $SUPP_CONFIG"
  if [[ -n "$RUNTIME_PACK" ]]; then
    [[ -f "$RUNTIME_MANIFEST" ]] || die "formal runtime manifest is missing: $RUNTIME_MANIFEST"
  fi
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

if is_enabled "$CHECKPOINT_META"; then
  if [[ "$need_eud" == "1" ]]; then
    [[ "$(checkpoint_epoch "$EUD_CKPT")" == "20" ]] || die "EUD checkpoint is not epoch 20"
  fi
  if [[ "$need_umr" == "1" ]]; then
    [[ "$(checkpoint_epoch "$UMR_CKPT")" == "6" ]] || die "UMR checkpoint is not epoch 6"
  fi
fi

if [[ -e "$RESULT_ROOT" ]] && ! is_enabled "$ALLOW_EXISTING_RESULTS"; then
  die "RESULT_ROOT already exists; choose a fresh path or set ALLOW_EXISTING_RESULTS=1"
fi
mkdir -p "$RESULT_ROOT"
DRIVER_LOG="$RESULT_ROOT/driver.log"
: > "$DRIVER_LOG"

warn_unrestored_dataset() {
  if [[ -e "$NUSCENES_BACKUP_DIR" ]]; then
    printf '[WARNING] nuScenes backup still exists: %s\n' "$NUSCENES_BACKUP_DIR" >&2
    printf '          Inspect it before another validation run.\n' >&2
  fi
}
trap warn_unrestored_dataset EXIT

runtime_preflight() {
  local config="$1"
  local expected_checkpoint_sha
  if [[ -n "$RUNTIME_PACK" ]]; then
    expected_checkpoint_sha="$(
      "$PYTHON_BIN" - "$RUNTIME_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
print(str((manifest.get("model_provenance") or {}).get("checkpoint_sha256") or ""))
PY
    )"
    [[ "$expected_checkpoint_sha" =~ ^[0-9a-f]{64}$ ]] || \
      die "runtime manifest has no valid checkpoint SHA256"
    "$PYTHON_BIN" tools/preflight_vov_umr_sgsup_b.py \
      --profile r50_v10 \
      --config "$SUPP_CONFIG" \
      --evaluation-config "$config" \
      --base-config projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v10_router_selector_frozen.py \
      --checkpoint "$UMR_CKPT" \
      --manifest "$RUNTIME_MANIFEST" \
      --repo-root "$ROOT_DIR" \
      --expected-checkpoint-sha256 "$expected_checkpoint_sha"
  fi
}

audit_condition() {
  local output_root="$1"
  local condition_dir="$2"
  local model_label="$3"
  local mode="$4"
  local -a summaries=()
  local -a checker_args=(
    tools/check_sgsup_admission_contract.py
    --min-records "$ADMISSION_MIN_RECORDS"
    --expected-model-label "$model_label"
  )
  mapfile -t summaries < <(compgen -G "$output_root/edl_logs/$condition_dir/*scene_summary*" | sort)
  [[ ${#summaries[@]} -gt 0 ]] || die "no scene-summary sidecar: $condition_dir"
  if [[ "$mode" == "off" ]]; then
    checker_args+=(--expect-supplement-off)
  elif is_enabled "$REQUIRE_POSITIVE_SUPPLEMENT" && [[ "$condition_dir" != "clean" ]]; then
    checker_args+=(--require-positive-action)
  fi
  "$PYTHON_BIN" "${checker_args[@]}" "${summaries[@]}"
}

audit_targeted_line() {
  local name="$1"
  local output_root="$2"
  local model_label="$3"
  case "${EV_LOG_MODE,,}" in
    off|0|false|none|no)
      printf '[WARN] admission audit skipped because EV_LOG_MODE=%s\n' "$EV_LOG_MODE" >&2
      return 0
      ;;
  esac
  if [[ "$name" == "line2_umr_base" ]]; then
    audit_condition "$output_root" clean "$model_label" off
    for corruption in "${CORRUPTION_LIST[@]}"; do
      for severity in "${SEVERITY_LIST[@]}"; do
        audit_condition "$output_root" "$corruption/severity_$severity" "$model_label" off
      done
    done
  elif [[ "$name" == "line3_umr_supplement" ]]; then
    audit_condition "$output_root" clean "$model_label" on
    for corruption in "${CORRUPTION_LIST[@]}"; do
      for severity in "${SEVERITY_LIST[@]}"; do
        audit_condition "$output_root" "$corruption/severity_$severity" "$model_label" on
      done
    done
  fi
}

run_line() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local model_label="$4"
  local output_root="$RESULT_ROOT/$name"

  say "===== [$name] start ====="
  printf '  config: %s\n' "$config"
  printf '  ckpt:   %s\n' "$checkpoint"
  printf '  output: %s\n' "$output_root"
  if [[ "$name" != "line1_eud" ]]; then
    runtime_preflight "$config"
  fi

  if [[ "$EVAL_MODE" == "full" ]]; then
    CONFIG="$config" CKPT="$checkpoint" MODEL_LABEL="$model_label" \
    OUT_DIR="$output_root" MULTICORRUPT_ROOT="$MULTICORRUPT_ROOT" \
    NUSCENES_DATA_DIR="$NUSCENES_DATA_DIR" NUSCENES_BACKUP_DIR="$NUSCENES_BACKUP_DIR" \
    TEST_WORKERS_PER_GPU="$TEST_WORKERS_PER_GPU" RUN_VIZ="$RUN_VIZ" \
    EV_LOG_MODE="$EV_LOG_MODE" EV_LOG_SCENE_ONLY="$EV_LOG_SCENE_ONLY" \
    EV_LOG_COMPRESSION="$EV_LOG_COMPRESSION" PYTHON_BIN="$PYTHON_BIN" \
      bash multicorrupt_evaluation.sh 2>&1 | tee -a "$DRIVER_LOG"
  else
    CONFIG="$config" CKPT="$checkpoint" MODEL_LABEL="$model_label" \
    OUT_DIR="$output_root" MULTICORRUPT_ROOT="$MULTICORRUPT_ROOT" \
    NUSCENES_DATA_DIR="$NUSCENES_DATA_DIR" NUSCENES_BACKUP_DIR="$NUSCENES_BACKUP_DIR" \
    CORRUPTIONS="$CORRUPTIONS" SEVERITY_LEVELS="$SEVERITY_LEVELS" \
    TEST_WORKERS_PER_GPU="$TEST_WORKERS_PER_GPU" RUN_VIZ="$RUN_VIZ" \
    EV_LOG_MODE="$EV_LOG_MODE" EV_LOG_SCENE_ONLY="$EV_LOG_SCENE_ONLY" \
    EV_LOG_COMPRESSION="$EV_LOG_COMPRESSION" PYTHON_BIN="$PYTHON_BIN" \
      bash run_targeted_corruption_probe.sh 2>&1 | tee -a "$DRIVER_LOG"
    audit_targeted_line "$name" "$output_root" "$model_label" 2>&1 | tee -a "$DRIVER_LOG"
  fi

  [[ ! -e "$NUSCENES_BACKUP_DIR" ]] || \
    die "[$name] backup remains after evaluation; dataset restoration failed"
  [[ -f "$output_root/metrics_summary.txt" ]] || \
    die "[$name] metrics_summary.txt was not generated"
  say "===== [$name] complete ====="
}

printf '  mode:         %s\n' "$EVAL_MODE"
printf '  lines:        %s\n' "$LINES"
printf '  result root:  %s\n' "$RESULT_ROOT"
if [[ "$need_umr" == "0" ]]; then
  runtime_pack_label='<not needed>'
else
  runtime_pack_label="${RUNTIME_PACK:-<non-formal override>}"
fi
printf '  runtime pack: %s\n' "$runtime_pack_label"
printf '  EV logging:   %s (%s)\n' "$EV_LOG_MODE" "$EV_LOG_COMPRESSION"

for line in "${LINE_LIST[@]}"; do
  case "$line" in
    line1_eud)
      run_line "$line" "$EUD_CONFIG" "$EUD_CKPT" R50_EUD_FINAL
      ;;
    line2_umr_base)
      run_line "$line" "$UMR_BASE_CONFIG" "$UMR_CKPT" R50_UMR_MATCHED_BASE
      ;;
    line3_umr_supplement)
      run_line "$line" "$SUPP_CONFIG" "$UMR_CKPT" R50_UMR_BOUNDED_SUPPLEMENT
      ;;
  esac
done

say "===== R50 three-line summary ====="
for line in "${LINE_LIST[@]}"; do
  metrics="$RESULT_ROOT/$line/metrics_summary.txt"
  printf '%s\n' "--- $line"
  grep -E 'Your model:|mRA \(mean RA\)|mRA_targeted|^mRRA|mRRA_targeted' "$metrics" || \
    printf '  See: %s\n' "$metrics"
done | tee -a "$DRIVER_LOG"

printf '\nResults:    %s\n' "$RESULT_ROOT"
printf 'Driver log: %s\n' "$DRIVER_LOG"
