#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/umr_module_names.sh
source "$ROOT_DIR/tools/umr_module_names.sh"

assert_equal() {
  local expected="$1" actual="$2"
  if [[ "$expected" != "$actual" ]]; then
    echo "[FAIL] expected=$expected actual=$actual" >&2
    exit 1
  fi
}

assert_equal branch_reliability_learning "$(umr_normalize_module a4)"
assert_equal damage_aware_routing "$(umr_normalize_module umr)"
assert_equal conservative_head_routing "$(umr_normalize_module v10)"
assert_equal bounded_candidate_supplement "$(umr_normalize_module sgsup)"

for module in \
  branch_reliability_learning \
  damage_aware_routing \
  conservative_head_routing \
  bounded_candidate_supplement; do
  assert_equal "$module" "$(umr_normalize_module "$module")"
done

START_MODULE=damage_aware_routing
START_STAGE=
assert_equal damage_aware_routing "$(umr_resolve_start_module)"
START_MODULE=
START_STAGE=v10
assert_equal conservative_head_routing "$(umr_resolve_start_module)"
START_MODULE=branch_reliability_learning
START_STAGE=a4
assert_equal branch_reliability_learning "$(umr_resolve_start_module)"
START_MODULE=damage_aware_routing
START_STAGE=a4
if umr_resolve_start_module >/dev/null 2>&1; then
  echo "[FAIL] conflicting functional and legacy inputs must be rejected" >&2
  exit 1
fi
unset START_MODULE START_STAGE
assert_equal branch_reliability_learning "$(umr_resolve_start_module)"

for script in train_vov_umr_complete.sh train_r50_umr_complete.sh; do
  help_text="$(bash "$ROOT_DIR/tools/$script" --help)"
  for module in \
    branch_reliability_learning \
    damage_aware_routing \
    conservative_head_routing \
    bounded_candidate_supplement; do
    if ! grep -Fq "$module" <<<"$help_text"; then
      echo "[FAIL] $script help is missing $module" >&2
      exit 1
    fi
  done
  if grep -Eq 'START_STAGE|V8-A4|Rev20-A|V10 selector|SGSUP' <<<"$help_text"; then
    echo "[FAIL] $script exposes experiment aliases in its public help" >&2
    exit 1
  fi
done

worker_help="$(bash "$ROOT_DIR/tools/train_vov_umr.sh" --help)"
for module in branch_reliability_learning damage_aware_routing; do
  if ! grep -Fq "$module" <<<"$worker_help"; then
    echo "[FAIL] train_vov_umr.sh help is missing $module" >&2
    exit 1
  fi
done
if grep -Eq 'START_STAGE|V8-A4|Rev20-A|V10 selector|SGSUP' <<<"$worker_help"; then
  echo "[FAIL] train_vov_umr.sh exposes experiment aliases in its public help" >&2
  exit 1
fi

echo "UMR functional naming tests passed."
