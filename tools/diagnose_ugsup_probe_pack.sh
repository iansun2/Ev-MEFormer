#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"

RESULTS_DIR="${RESULTS_DIR:-MultiCorrupt_results}"
CANDIDATES="${CANDIDATES:-UGSUP_A_util050_k30 UGSUP_B_util035_k50 UGSUP_C_util065_k20}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_DIR}/ugsup_edl_diagnostic.json}"
OUTPUT_MD="${OUTPUT_MD:-${RESULTS_DIR}/ugsup_edl_diagnostic.md}"

python tools/diagnose_ugsup_edl_logs.py \
  --results-dir "$RESULTS_DIR" \
  --candidates "$CANDIDATES" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"

