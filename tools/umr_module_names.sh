#!/usr/bin/env bash

# Canonical UMR training-module names.  These names follow the functional
# terminology used in master-thesis/thesis/chapters/03_method.tex.  Historical
# experiment aliases are accepted only by umr_normalize_module() so existing
# checkpoint-resume commands remain usable.

readonly UMR_MODULE_BRANCH_RELIABILITY="branch_reliability_learning"
readonly UMR_MODULE_DAMAGE_AWARE_ROUTING="damage_aware_routing"
readonly UMR_MODULE_CONSERVATIVE_HEAD_ROUTING="conservative_head_routing"
readonly UMR_MODULE_BOUNDED_CANDIDATE_SUPPLEMENT="bounded_candidate_supplement"

umr_normalize_module() {
  case "${1:-}" in
    branch_reliability_learning|a4)
      printf '%s\n' "$UMR_MODULE_BRANCH_RELIABILITY"
      ;;
    damage_aware_routing|umr)
      printf '%s\n' "$UMR_MODULE_DAMAGE_AWARE_ROUTING"
      ;;
    conservative_head_routing|v10)
      printf '%s\n' "$UMR_MODULE_CONSERVATIVE_HEAD_ROUTING"
      ;;
    bounded_candidate_supplement|sgsup)
      printf '%s\n' "$UMR_MODULE_BOUNDED_CANDIDATE_SUPPLEMENT"
      ;;
    *)
      return 1
      ;;
  esac
}

# Human-readable module labels used in progress logs.
umr_module_label() {
  case "${1:-}" in
    branch_reliability_learning)
      printf '%s\n' 'branch candidate and reliability signal learning'
      ;;
    damage_aware_routing)
      printf '%s\n' 'routing training with damage-state supervision'
      ;;
    conservative_head_routing)
      printf '%s\n' 'per-head conservative routing'
      ;;
    bounded_candidate_supplement)
      printf '%s\n' 'bounded candidate supplement'
      ;;
    *)
      return 1
      ;;
  esac
}

umr_resolve_start_module() {
  local requested_module="${START_MODULE:-}"
  local legacy_stage="${START_STAGE:-}"
  local normalized_module=""
  local normalized_legacy=""

  if [[ -n "$requested_module" ]]; then
    normalized_module="$(umr_normalize_module "$requested_module")" || {
      echo "[ERROR] unknown START_MODULE: $requested_module" >&2
      return 2
    }
  fi
  if [[ -n "$legacy_stage" ]]; then
    normalized_legacy="$(umr_normalize_module "$legacy_stage")" || {
      echo "[ERROR] unsupported legacy START_STAGE value: $legacy_stage" >&2
      return 2
    }
  fi
  if [[ -n "$normalized_module" && -n "$normalized_legacy" && \
        "$normalized_module" != "$normalized_legacy" ]]; then
    echo "[ERROR] START_MODULE and legacy START_STAGE select different modules" >&2
    return 2
  fi

  printf '%s\n' "${normalized_module:-${normalized_legacy:-$UMR_MODULE_BRANCH_RELIABILITY}}"
}
