#!/usr/bin/env bash
set -Eeuo pipefail

SWEEP_ROOT="${SWEEP_ROOT:-work_dirs/umr_rule_runtime_latest}"
RESULTS_DIR="${RESULTS_DIR:-MultiCorrupt_results}"
CANDIDATES="${CANDIDATES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/tools/test.py" && -f "$dir/run_targeted_corruption_probe.sh" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  echo "[ERROR] Could not locate MEFormer repo root from: $SCRIPT_DIR" >&2
  return 2
}
ROOT_DIR="$(find_repo_root)"
cd "$ROOT_DIR"

if [[ ! -d "$SWEEP_ROOT" ]]; then
  echo "[ERROR] missing SWEEP_ROOT: $SWEEP_ROOT" >&2
  exit 2
fi

if [[ -z "$CANDIDATES" && -f "$SWEEP_ROOT/summaries/candidates.tsv" ]]; then
  CANDIDATES="$(awk 'NR>1 {print $1}' "$SWEEP_ROOT/summaries/candidates.tsv")"
fi

audit_args=(--sweep-root "$SWEEP_ROOT" --repo-root "$ROOT_DIR")
if [[ -n "$CANDIDATES" ]]; then
  audit_args+=(--candidates "$CANDIDATES")
fi
python tools/check_umr_rule_runtime_pack.py "${audit_args[@]}"

if [[ -n "$CANDIDATES" ]]; then
  found_any=0
  for name in $CANDIDATES; do
    if [[ -d "$RESULTS_DIR/UMR_${name}_targeted_probe" ]]; then
      found_any=1
      break
    fi
  done
  if [[ "$found_any" == "1" ]]; then
    CANDIDATES="$CANDIDATES" RESULTS_DIR="$RESULTS_DIR" \
      bash tools/diagnose_ugsup_probe_pack.sh
  else
    echo "[Rule UMR diag] no targeted probe result dirs found under $RESULTS_DIR; static audit only."
  fi
fi
