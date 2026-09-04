#!/usr/bin/env bash
set -Eeuo pipefail

OUT_ROOT="${OUT_ROOT:-work_dirs/umr_rule_runtime_$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${BASE_CONFIG:-projects/configs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v10_router_selector_frozen.py}"
BASE_CKPT="${BASE_CKPT:-work_dirs/v10_router_selector/20260522-143929/latest.pth}"
CLEAN_REFERENCE_STATS="${CLEAN_REFERENCE_STATS:-work_dirs/umr_crg_param_sweep_20260626_091245/clean_reference/clean_reference_stats.json}"
SCENE_MAPPER_PATH="${SCENE_MAPPER_PATH:-work_dirs/umr_scene_gate_phase1_offline_20260627_113259_formal/calibration/mapper_scene_bev_damage_applicable.json}"
DEFAULT_UTILITY_MAPPER_PATH="work_dirs/umr_action_utility_actm_20260628_170942/mapper_candidate_useful.json"
if [[ ! -e "$DEFAULT_UTILITY_MAPPER_PATH" ]]; then
  DEFAULT_UTILITY_MAPPER_PATH="work_dirs/umr_utility_phase1_full_20260626_202030/calibration/mapper_candidate_useful.json"
fi
UTILITY_MAPPER_PATH="${UTILITY_MAPPER_PATH:-$DEFAULT_UTILITY_MAPPER_PATH}"
IMG_ACTION_MAPPER_PATH="${IMG_ACTION_MAPPER_PATH:-}"
BEV_ACTION_MAPPER_PATH="${BEV_ACTION_MAPPER_PATH:-}"
FUSED_SLOT_MAPPER_PATH="${FUSED_SLOT_MAPPER_PATH:-}"
DEFAULT_FUSED_SLOT_MAPPER_PATH="work_dirs/umr_action_utility_actm_20260628_170942/mapper_fused_slot_candidate_useful.json"
if [[ -z "$FUSED_SLOT_MAPPER_PATH" && -e "$DEFAULT_FUSED_SLOT_MAPPER_PATH" ]]; then
  FUSED_SLOT_MAPPER_PATH="$DEFAULT_FUSED_SLOT_MAPPER_PATH"
fi
IMG_ACTION_UTILITY_THRESHOLD_A="${IMG_ACTION_UTILITY_THRESHOLD_A:-0.25}"
IMG_ACTION_UTILITY_THRESHOLD_B="${IMG_ACTION_UTILITY_THRESHOLD_B:-0.40}"
DUAL_ACTION_UTILITY_THRESHOLD="${DUAL_ACTION_UTILITY_THRESHOLD:-0.35}"
ACT_M_MARGIN_A="${ACT_M_MARGIN_A:-0.15}"
ACT_M_DUAL_MARGIN_A="${ACT_M_DUAL_MARGIN_A:-0.40}"
ACT_M_MARGIN_B="${ACT_M_MARGIN_B:-0.30}"
ACT_M_DUAL_MARGIN_B="${ACT_M_DUAL_MARGIN_B:-0.60}"
BRANCH_LABEL_ROOT="${BRANCH_LABEL_ROOT:-work_dirs/umr_branch_candidate_phase1_20260627_154805/label_analysis}"
SCENE_THRESHOLD_KEY="${SCENE_THRESHOLD_KEY:-clean_fpr_005}"
UTILITY_THRESHOLD="${UTILITY_THRESHOLD:-0.45}"
EXPECTED_MODEL_FINGERPRINT="${EXPECTED_MODEL_FINGERPRINT:-}"
SUPP_REQUIRE_REFERENCE_STATS="${SUPP_REQUIRE_REFERENCE_STATS:-0}"
DEFAULT_CANDIDATES="RULE_SGSUP_B_runtime RULE_SGSUP_B_bevonly_abstain_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_nms030_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_nms050_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_nms080_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_nms250_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_nms500_dyn10_30 RULE_SGSUP_B_contextheight055_v9bev_nms1000_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_imgsens_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev145_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_contextheight055_v9bev_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_dual010_img015_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsem070_m085_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsem065_m085_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsemtop100_d070_m085_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsemtop300_d070_m085_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_qsemtop300_d065_marginoff_dyn10_30 RULE_SGSUP_B_relbev_fusedguard_purebev_comp155u175_dyn10_30 RULE_UMR2_A_imgguard_dyn10_30 RULE_UMR2_B_dualbranch_dyn10_30 RULE_UMR2_C_dualmapper_dyn10_30 RULE_UMR2_D_dualmapper_bevboost_dyn10_30"
if [[ -n "$IMG_ACTION_MAPPER_PATH" ]]; then
  DEFAULT_CANDIDATES="${DEFAULT_CANDIDATES} RULE_ACT_A_imgmargin025_dyn10_30 RULE_ACT_B_imgmargin040_dyn10_30"
fi
if [[ -n "$IMG_ACTION_MAPPER_PATH" && -n "$BEV_ACTION_MAPPER_PATH" ]]; then
  DEFAULT_CANDIDATES="${DEFAULT_CANDIDATES} RULE_ACT_C_dualbranch_margin035_dyn10_30"
fi
if [[ -n "$IMG_ACTION_MAPPER_PATH" && -n "$FUSED_SLOT_MAPPER_PATH" ]]; then
  DEFAULT_CANDIDATES="${DEFAULT_CANDIDATES} RULE_ACT_M_A_slotmargin015_dyn10_30 RULE_ACT_M_B_slotmargin030_dyn10_30"
fi
CANDIDATES="${CANDIDATES:-$DEFAULT_CANDIDATES}"
RUN_STATIC_AUDIT="${RUN_STATIC_AUDIT:-1}"
REQUIRE_RUNTIME_PREFLIGHT="${REQUIRE_RUNTIME_PREFLIGHT:-0}"

if [[ "$REQUIRE_RUNTIME_PREFLIGHT" != "0" && "$REQUIRE_RUNTIME_PREFLIGHT" != "1" ]]; then
  echo "[ERROR] REQUIRE_RUNTIME_PREFLIGHT must be 0 or 1" >&2
  exit 2
fi

CONFIG_DIR="${OUT_ROOT}/configs"
SUMMARY_DIR="${OUT_ROOT}/summaries"
mkdir -p "$CONFIG_DIR" "$SUMMARY_DIR"

portable_path() {
  python - "$1" <<'PY'
import os
import sys

p = sys.argv[1]
root = os.getcwd()
if os.path.isabs(p) and (p == root or p.startswith(root + os.sep)):
    print(os.path.relpath(p, root))
else:
    print(p)
PY
}

for p in "$BASE_CONFIG" "$BASE_CKPT" "$CLEAN_REFERENCE_STATS" "$SCENE_MAPPER_PATH" "$UTILITY_MAPPER_PATH"; do
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] missing required input: $p" >&2
    exit 2
  fi
done

action_img_required=0
action_bev_required=0
fused_slot_required=0
for name in $CANDIDATES; do
  case "$name" in
    RULE_ACT_A_*|RULE_ACT_B_*|RULE_ACT_C_*|RULE_ACT_M_*) action_img_required=1 ;;
  esac
  case "$name" in
    RULE_ACT_C_*) action_bev_required=1 ;;
  esac
  case "$name" in
    RULE_ACT_M_*|RULE_SGSUP_B_relbev_fusedguard_purebev_comp*_margin*) fused_slot_required=1 ;;
  esac
done
if [[ "$action_img_required" == "1" && ( -z "$IMG_ACTION_MAPPER_PATH" || ! -e "$IMG_ACTION_MAPPER_PATH" ) ]]; then
  echo "[ERROR] selected action-utility candidate requires IMG_ACTION_MAPPER_PATH" >&2
  exit 2
fi
if [[ "$action_bev_required" == "1" && ( -z "$BEV_ACTION_MAPPER_PATH" || ! -e "$BEV_ACTION_MAPPER_PATH" ) ]]; then
  echo "[ERROR] selected dual action-utility candidate requires BEV_ACTION_MAPPER_PATH" >&2
  exit 2
fi
if [[ "$fused_slot_required" == "1" && ( -z "$FUSED_SLOT_MAPPER_PATH" || ! -e "$FUSED_SLOT_MAPPER_PATH" ) ]]; then
  echo "[ERROR] selected ACT_M candidate requires FUSED_SLOT_MAPPER_PATH" >&2
  exit 2
fi

for p in "$CLEAN_REFERENCE_STATS" "$SCENE_MAPPER_PATH" "$UTILITY_MAPPER_PATH" "$IMG_ACTION_MAPPER_PATH" "$BEV_ACTION_MAPPER_PATH" "$FUSED_SLOT_MAPPER_PATH"; do
  [[ -z "$p" ]] && continue
  lower="$(printf '%s' "$p" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower" == *"multicorrupt_results"* || "$lower" == *"targeted_probe"* || \
        "$lower" == *"nuscenes_infos_val"* || "$lower" == *"corrupted-val"* || \
        "$lower" == *"corrupted_val"* || "$lower" == *"corrupted/val"* || \
        "$lower" == *"val_corrupt"* || "$lower" == *"val_clean"* || \
        "$lower" == *"validation"* || "$lower" == "val/"* || \
        "$lower" == *"/val/"* || "$lower" == *"/val" ]]; then
    echo "[ERROR] forbidden validation/evaluation-derived runtime input: $p" >&2
    exit 3
  fi
done

if [[ -d "$BRANCH_LABEL_ROOT" ]]; then
  lower="$(printf '%s' "$BRANCH_LABEL_ROOT" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower" == *"multicorrupt_results"* || "$lower" == *"targeted_probe"* || \
        "$lower" == *"nuscenes_infos_val"* || "$lower" == *"corrupted-val"* || \
        "$lower" == *"corrupted_val"* || "$lower" == *"corrupted/val"* || \
        "$lower" == *"val_corrupt"* || "$lower" == *"val_clean"* || \
        "$lower" == *"validation"* || "$lower" == "val/"* || \
        "$lower" == *"/val/"* || "$lower" == *"/val" ]]; then
    echo "[ERROR] forbidden validation/evaluation-derived branch label root: $BRANCH_LABEL_ROOT" >&2
    exit 3
  fi
fi

BASE_CKPT_PORTABLE="$(portable_path "$BASE_CKPT")"
CLEAN_REFERENCE_STATS_PORTABLE="$(portable_path "$CLEAN_REFERENCE_STATS")"
SCENE_MAPPER_PORTABLE="$(portable_path "$SCENE_MAPPER_PATH")"
UTILITY_MAPPER_PORTABLE="$(portable_path "$UTILITY_MAPPER_PATH")"
IMG_ACTION_MAPPER_PORTABLE=""
BEV_ACTION_MAPPER_PORTABLE=""
FUSED_SLOT_MAPPER_PORTABLE=""
if [[ -n "$IMG_ACTION_MAPPER_PATH" ]]; then
  IMG_ACTION_MAPPER_PORTABLE="$(portable_path "$IMG_ACTION_MAPPER_PATH")"
fi
if [[ -n "$BEV_ACTION_MAPPER_PATH" ]]; then
  BEV_ACTION_MAPPER_PORTABLE="$(portable_path "$BEV_ACTION_MAPPER_PATH")"
fi
if [[ -n "$FUSED_SLOT_MAPPER_PATH" ]]; then
  FUSED_SLOT_MAPPER_PORTABLE="$(portable_path "$FUSED_SLOT_MAPPER_PATH")"
fi
BASE_CONFIG_REL="$(
  python - "$BASE_CONFIG" "$CONFIG_DIR" <<'PY'
import os
import sys

base = os.path.abspath(sys.argv[1])
cfg_dir = os.path.abspath(sys.argv[2])
print(os.path.relpath(base, cfg_dir))
PY
)"

SCENE_THRESHOLD="$(
  python - "$SCENE_MAPPER_PATH" "$SCENE_THRESHOLD_KEY" <<'PY'
import json
import sys

mapper_path, key = sys.argv[1:3]
with open(mapper_path, "r") as f:
    mapper = json.load(f)
thresholds = mapper.get("thresholds") or {}
if key not in thresholds:
    raise SystemExit(f"threshold key not found in scene mapper: {key}")
print(thresholds[key])
PY
)"

write_config() {
  local name="$1"
  local supp_mode="$2"
  local max_add="$3"
  local budget_min="$4"
  local budget_max="$5"
  local rank_mode="$6"
  local guard_enable="$7"
  local guard_mode="$8"
  local dual_budget="$9"
  local dual_utility="${10}"
  local dual_gate_mode="${11:-strict_img}"
  local dual_scene_threshold="${12:-0.90}"
  local dual_bev_scale="${13:-1.00}"
  local dual_bev_boost="${14:-1.00}"
  local utility_mapper_path="${15-$UTILITY_MAPPER_PORTABLE}"
  local utility_threshold="${16-$UTILITY_THRESHOLD}"
  local img_utility_mapper_path="${17-}"
  local bev_utility_mapper_path="${18-}"
  local img_utility_threshold="${19--1.0}"
  local bev_utility_threshold="${20--1.0}"
  local fused_slot_mapper_path="${21-}"
  local marginal_enable="${22-False}"
  local marginal_margin="${23-0.0}"
  local dual_marginal_margin="${24--1.0}"
  local fused_damage_scale="${25-2.0}"
  local bev_fused_dominance="${26-1.0}"
  local img_damage_scale="${27-1.0}"
  local damage_signal_space="${28-center}"
  local bev_damage_signal="bev_center_unc"
  local img_damage_signal="img_center_unc"
  local fused_damage_signal="fused_center_unc"
  local bev_damage_metric="bev_top10_unc"
  local img_damage_metric="img_top10_unc"
  local fused_damage_metric="fused_top10_unc"
  local img_guard_signal="${29-}"
  local img_guard_metric="${30-}"
  local img_guard_threshold_key="${31-p99}"
  local bev_upper_scale="${32-0.0}"
  local dual_abstain_signal="${33-}"
  local dual_abstain_threshold="${34--1.0}"
  local img_abstain_signal="${35-}"
  local img_abstain_threshold="${36--1.0}"
  local qsem_gate_enable="${37-False}"
  local qsem_disagree_threshold="${38--1.0}"
  local qsem_margin_threshold="${39--1.0}"
  local qsem_disagree_signal="${40-qsem_top300_img_bev_class_disagree_ratio}"
  local qsem_margin_signal="${41-qsem_top300_img_margin_mean}"
  local contextheight_gate_enable="${42-False}"
  local contextheight_context_threshold="${43--1.0}"
  local contextheight_height_threshold="${44--1.0}"
  local contextheight_v9_bev_threshold="${45--1.0}"
  local contextheight_context_signal="${46-v8_context_img_damage_mean}"
  local contextheight_height_signal="${47-distance_bin_abnormality_stats.height.bev.mid}"
  local contextheight_v9_bev_signal="${48-v9_a_bev_above_tau_mean}"
  local supp_nms_center_dist="${49-0.005}"
  if [[ "$damage_signal_space" == "raw" ]]; then
    bev_damage_signal="bev_raw_modality_unc"
    img_damage_signal="img_raw_modality_unc"
    fused_damage_signal="fused_raw_modality_unc"
    bev_damage_metric="v8_gscrg_raw_unc_bev_top10"
    img_damage_metric="v8_gscrg_raw_unc_img_top10"
    fused_damage_metric="v8_gscrg_raw_unc_fused_top10"
  fi
  local cfg="${CONFIG_DIR}/${name}.py"

  cat > "$cfg" <<PY
_base_ = '${BASE_CONFIG_REL}'

model = dict(
    pts_bbox_head=dict(
        ensemble=dict(
            ensemble_mode='damage_gated_soft_router',
            supp_enable=True,
            supp_mode='${supp_mode}',
            supp_max_add_queries=${max_add},
            supp_img_score_threshold=0.02,
            supp_bev_score_threshold=0.02,
            supp_fused_replace_score_ceiling=0.35,
            supp_nms_center_dist=${supp_nms_center_dist},
            supp_per_class_cap=0,
            supp_expected_model_fingerprint='${EXPECTED_MODEL_FINGERPRINT}',
            supp_require_reference_stats=$( [[ "$SUPP_REQUIRE_REFERENCE_STATS" == "1" ]] && printf 'True' || printf 'False' ),
            supp_utility_mapper_path='${utility_mapper_path}',
            supp_utility_threshold=${utility_threshold},
            supp_img_utility_mapper_path='${img_utility_mapper_path}',
            supp_bev_utility_mapper_path='${bev_utility_mapper_path}',
            supp_img_utility_threshold=${img_utility_threshold},
            supp_bev_utility_threshold=${bev_utility_threshold},
            supp_fused_slot_utility_mapper_path='${fused_slot_mapper_path}',
            supp_marginal_utility_enable=${marginal_enable},
            supp_marginal_utility_margin=${marginal_margin},
            supp_dual_damage_marginal_margin=${dual_marginal_margin},
            supp_scene_mapper_path='${SCENE_MAPPER_PORTABLE}',
            supp_scene_mapper_threshold=${SCENE_THRESHOLD},
            supp_scene_mapper_gate_mode='override',
            supp_scene_budget_min_ratio=${budget_min},
            supp_scene_budget_max_ratio=${budget_max},
            supp_scene_budget_full_threshold=0.990,
            supp_rank_mode='${rank_mode}',
            supp_utility_rank_weight=1.0,
            supp_img_damage_guard_enable=${guard_enable},
            supp_img_damage_guard_mode='${guard_mode}',
            supp_img_damage_signal='${img_guard_signal}',
            supp_img_damage_metric_name='${img_guard_metric}',
            supp_img_damage_threshold_key='${img_guard_threshold_key}',
            supp_img_damage_threshold_scale=${img_damage_scale},
            supp_fused_damage_signal='${fused_damage_signal}',
            supp_fused_damage_metric_name='${fused_damage_metric}',
            supp_fused_damage_threshold_key='p99',
            supp_fused_damage_threshold_scale=${fused_damage_scale},
            supp_relative_damage_reference_key='median',
            supp_bev_fused_dominance_ratio=${bev_fused_dominance},
            supp_bev_damage_upper_threshold_scale=${bev_upper_scale},
            supp_dual_abstain_signal='${dual_abstain_signal}',
            supp_dual_abstain_threshold=${dual_abstain_threshold},
            supp_img_abstain_signal='${img_abstain_signal}',
            supp_img_abstain_threshold=${img_abstain_threshold},
            supp_qsem_gate_enable=${qsem_gate_enable},
            supp_qsem_disagree_signal='${qsem_disagree_signal}',
            supp_qsem_disagree_threshold=${qsem_disagree_threshold},
            supp_qsem_margin_signal='${qsem_margin_signal}',
            supp_qsem_margin_threshold=${qsem_margin_threshold},
            supp_contextheight_gate_enable=${contextheight_gate_enable},
            supp_contextheight_context_signal='${contextheight_context_signal}',
            supp_contextheight_context_threshold=${contextheight_context_threshold},
            supp_contextheight_height_signal='${contextheight_height_signal}',
            supp_contextheight_height_threshold=${contextheight_height_threshold},
            supp_contextheight_v9_bev_signal='${contextheight_v9_bev_signal}',
            supp_contextheight_v9_bev_threshold=${contextheight_v9_bev_threshold},
            supp_dual_damage_budget_ratio=${dual_budget},
            supp_dual_damage_utility_threshold=${dual_utility},
            supp_dual_damage_gate_mode='${dual_gate_mode}',
            supp_dual_scene_mapper_threshold=${dual_scene_threshold},
            supp_dual_bev_threshold_scale=${dual_bev_scale},
            supp_dual_bev_rank_boost=${dual_bev_boost},
            scrg_enable=False,
            scrg_mode='flag',
            scrg_switch_heads=('center',),
            scrg_scene_metric='top10',
            scrg_top_fraction=0.10,
            scrg_bev_threshold_key='p99',
            scrg_img_ceiling_key='p99',
            scrg_margin_threshold_key='p99',
            scrg_bev_threshold_scale=1.00,
            scrg_img_ceiling_scale=1.00,
            scrg_margin_threshold_scale=1.00,
            scrg_bev_metric_name='${bev_damage_metric}',
            scrg_img_metric_name='${img_damage_metric}',
            scrg_aux_metric_name='none',
            scrg_bev_signal='${bev_damage_signal}',
            scrg_img_signal='${img_damage_signal}',
            scrg_aux_signal='none',
            scrg_max_switch_ratio=0.10,
            v7_reference_stats_path='${CLEAN_REFERENCE_STATS_PORTABLE}',
            v7_allow_missing_stats_fallback=False,
        ),
    ),
)
PY
}

SUMMARY="${SUMMARY_DIR}/candidates.tsv"
printf 'name\tpolicy\tbudget\trank\tguard\tconfig\tcheckpoint\n' > "$SUMMARY"

for name in $CANDIDATES; do
  case "$name" in
    RULE_SGSUP_B_runtime)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_rule" "dyn10-30/k90" "img_score" "off" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_bevonly_abstain_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "bev_only_abstain" "0.0" "0.0"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_bevonly_abstain" "dyn10-30/k90" "img_score" "pure_bev_only" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev" "dyn10-30/k90" "img_score" "context_lt-0.6488_or_height055/v9bev060" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_nms030_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean" "0.030"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev_nms_sweep" "dyn10-30/k90/nms030" "img_score" "context_lt-0.6488_or_height055/v9bev060/nms030" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_nms050_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean" "0.050"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev_nms_sweep" "dyn10-30/k90/nms050" "img_score" "context_lt-0.6488_or_height055/v9bev060/nms050" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_nms080_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean" "0.080"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev_nms_sweep" "dyn10-30/k90/nms080" "img_score" "context_lt-0.6488_or_height055/v9bev060/nms080" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_nms250_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean" "0.250"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev_nms_sweep" "dyn10-30/k90/nms250" "img_score" "context_lt-0.6488_or_height055/v9bev060/nms250" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_nms500_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean" "0.500"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev_nms_sweep" "dyn10-30/k90/nms500" "img_score" "context_lt-0.6488_or_height055/v9bev060/nms500" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_contextheight055_v9bev_nms1000_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.00" "1.00" "center" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean" "1.000"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_contextheight_v9bev_nms_sweep" "dyn10-30/k90/nms1000" "img_score" "context_lt-0.6488_or_height055/v9bev060/nms1000" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "2.00" "1.05" "1.00" "raw"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_relative_bev_fused" "dyn10-30/k90" "img_score" "fused_p99x2/bev_dom105" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_imgsens_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.15" "0.85" "raw" "v8_a4_cls_unc_z_img" "v8_a4_cls_unc_z_img_top10" "p99"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_relative_bev_fused_strict" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom115/img085" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev145_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_dominance" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/no_img_signal" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_contextheight055_v9bev_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "False" "-1.0" "-1.0" "" "" "True" "-0.64882296" "0.55" "0.60" "v8_context_img_damage_mean" "distance_bin_abnormality_stats.height.bev.mid" "v9_a_bev_above_tau_mean"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_contextheight_v9bev" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/context_or_height055/v9bev060" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_dual010_img015_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_dual_abstain" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "v9_state_confirmed_dual_mean" "0.10" "v9_a_img_above_tau_mean" "0.15"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_dual_abstain" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/dual010/img015" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_dual_abstain" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "v9_state_confirmed_dual_mean" "0.12" "v9_a_img_above_tau_mean" "0.18"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_dual_abstain" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/dual012/img018" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsem070_m085_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_dual_abstain" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "v9_state_confirmed_dual_mean" "0.12" "v9_a_img_above_tau_mean" "0.18" "True" "0.70" "0.085" "qsem_img_bev_class_disagree_flag_ratio" "qsem_img_margin_mean"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_qsem_gate" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/dual012/img018/qsemD070M085" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsem065_m085_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_dual_abstain" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "v9_state_confirmed_dual_mean" "0.12" "v9_a_img_above_tau_mean" "0.18" "True" "0.65" "0.085" "qsem_img_bev_class_disagree_flag_ratio" "qsem_img_margin_mean"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_qsem_gate" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/dual012/img018/qsemD065M085" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsemtop100_d070_m085_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_dual_abstain" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "v9_state_confirmed_dual_mean" "0.12" "v9_a_img_above_tau_mean" "0.18" "True" "0.70" "0.085" "qsem_top100_img_bev_class_disagree_ratio" "qsem_top100_img_margin_mean"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_qsem_topk_gate" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/dual012/img018/qsemTop100D070M085" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_dual012_img018_qsemtop300_d070_m085_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_dual_abstain" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "v9_state_confirmed_dual_mean" "0.12" "v9_a_img_above_tau_mean" "0.18" "True" "0.70" "0.085" "qsem_top300_img_bev_class_disagree_ratio" "qsem_top300_img_margin_mean"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_qsem_topk_gate" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/dual012/img018/qsemTop300D070M085" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_qsemtop300_d065_marginoff_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.45" "1.00" "raw" "" "" "p99" "0.0" "" "-1.0" "" "-1.0" "True" "0.65" "-1.0" "qsem_top300_img_bev_class_disagree_ratio" ""
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_qsem_topk_gate" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom145/qsemTop300D065MarginOff" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_comp155u175_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_composite" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "" "False" "0.0" "-1.0" "1.50" "1.55" "1.00" "raw" "" "" "p99" "1.75"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_composite" "dyn10-30/k90" "img_score" "fused_p99x1.5/bev_dom155/bev_upper175" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_SGSUP_B_relbev_fusedguard_purebev_comp160u160_margin015_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "relative_bev_fused_composite" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "${UTILITY_MAPPER_PORTABLE}" "${UTILITY_THRESHOLD}" "" "" "-1.0" "-1.0" "${FUSED_SLOT_MAPPER_PORTABLE}" "True" "0.15" "0.40" "1.50" "1.60" "1.00" "raw" "" "" "p99" "1.60"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "sgsup_b_pure_bev_composite_margin" "dyn10-30/k90/slot015" "img_score" "fused_p99x1.5/bev_dom160/bev_upper160/slot015" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_UMR2_A_imgguard_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "img_score" "True" "suppress" "0.0" "0.0"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "rule_umr_v2" "dyn10-30/k90" "img_score" "img_damage_suppress" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_UMR2_B_dualbranch_dyn10_30|RULE_UMR2_B_dualsmall_dyn10_30)
      write_config "$name" "dual_branch_highscore" 90 "0.10" "0.30" "utility_product" "True" "dual_small" "0.10" "0.65"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "rule_umr_v2" "dyn10-30/k90/dual10" "utility_product" "dual_branch_util65" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_UMR2_C_dualmapper_dyn10_30)
      write_config "$name" "dual_branch_highscore" 90 "0.10" "0.30" "utility_product" "True" "dual_small" "0.10" "0.65" "scene_mapper_or_img" "0.90" "1.00" "1.00"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "rule_umr_v2" "dyn10-30/k90/dual10/mapper90" "utility_product" "dual_mapper_util65" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_UMR2_D_dualmapper_bevboost_dyn10_30)
      write_config "$name" "dual_branch_highscore" 90 "0.10" "0.30" "utility_product" "True" "dual_small" "0.10" "0.65" "scene_mapper_or_img" "0.90" "1.00" "1.35"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "rule_umr_v2" "dyn10-30/k90/dual10/mapper90" "utility_product" "dual_mapper_bevboost135" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_ACT_A_imgmargin025_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "utility_product" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "" "0.0" "${IMG_ACTION_MAPPER_PORTABLE}" "" "${IMG_ACTION_UTILITY_THRESHOLD_A}" "-1.0"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "action_img_supp" "dyn10-30/k90/img_margin025" "utility_product" "off" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_ACT_B_imgmargin040_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "utility_product" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "" "0.0" "${IMG_ACTION_MAPPER_PORTABLE}" "" "${IMG_ACTION_UTILITY_THRESHOLD_B}" "-1.0"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "action_img_supp" "dyn10-30/k90/img_margin040" "utility_product" "off" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_ACT_M_A_slotmargin015_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "utility_product" "False" "off" "0.0" "0.0" "strict_img" "0.90" "1.00" "1.00" "" "0.0" "${IMG_ACTION_MAPPER_PORTABLE}" "" "${IMG_ACTION_UTILITY_THRESHOLD_A}" "-1.0" "${FUSED_SLOT_MAPPER_PORTABLE}" "True" "${ACT_M_MARGIN_A}" "${ACT_M_DUAL_MARGIN_A}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "action_marginal_img_supp" "dyn10-30/k90/img025/slot015/dual040" "utility_product" "slot_margin" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_ACT_M_B_slotmargin030_dyn10_30)
      write_config "$name" "img_highscore" 90 "0.10" "0.30" "utility_product" "False" "off" "0.0" "0.0" "scene_mapper_or_img" "0.90" "1.00" "1.00" "" "0.0" "${IMG_ACTION_MAPPER_PORTABLE}" "" "${IMG_ACTION_UTILITY_THRESHOLD_B}" "-1.0" "${FUSED_SLOT_MAPPER_PORTABLE}" "True" "${ACT_M_MARGIN_B}" "${ACT_M_DUAL_MARGIN_B}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "action_marginal_img_supp" "dyn10-30/k90/img040/slot030/dual060" "utility_product" "slot_margin_strict" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    RULE_ACT_C_dualbranch_margin035_dyn10_30)
      write_config "$name" "dual_branch_highscore" 90 "0.10" "0.30" "utility_product" "True" "dual_small" "0.10" "${DUAL_ACTION_UTILITY_THRESHOLD}" "scene_mapper_or_img" "0.90" "1.00" "1.10" "" "0.0" "${IMG_ACTION_MAPPER_PORTABLE}" "${BEV_ACTION_MAPPER_PORTABLE}" "${DUAL_ACTION_UTILITY_THRESHOLD}" "${DUAL_ACTION_UTILITY_THRESHOLD}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "action_dual_branch" "dyn10-30/k90/dual10/margin035" "utility_product" "dual_branch_action" \
        "${CONFIG_DIR}/${name}.py" "$BASE_CKPT_PORTABLE" >> "$SUMMARY"
      ;;
    *)
      echo "[ERROR] unknown candidate: $name" >&2
      exit 4
      ;;
  esac
done

cat > "${OUT_ROOT}/validate_targeted_all.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
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
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"
SWEEP_ROOT="$SCRIPT_DIR"
BASE_CKPT="${BASE_CKPT:-__BASE_CKPT__}"
CORRUPTIONS="${CORRUPTIONS:-beamsreducing}"
SEVERITY_LEVELS="${SEVERITY_LEVELS:-3}"
RUN_VIZ="${RUN_VIZ:-0}"
EV_LOG_MODE="${EV_LOG_MODE:-off}"
EV_LOG_COMPRESSION="${EV_LOG_COMPRESSION:-none}"
EV_LOG_SCENE_ONLY="${EV_LOG_SCENE_ONLY:-0}"
TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-8}"
CANDIDATES="${CANDIDATES:-}"
REQUIRE_RUNTIME_PREFLIGHT=__REQUIRE_RUNTIME_PREFLIGHT__

if [[ -z "$CANDIDATES" ]]; then
  CANDIDATES="$(awk 'NR>1 {print $1}' "$SWEEP_ROOT/summaries/candidates.tsv")"
fi

for name in $CANDIDATES; do
  cfg="$SWEEP_ROOT/configs/${name}.py"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] missing config: $cfg" >&2
    exit 1
  fi
  if [[ "$REQUIRE_RUNTIME_PREFLIGHT" == "1" ]]; then
    if [[ ! -x "$SWEEP_ROOT/preflight.sh" ]]; then
      echo "[ERROR] formal preflight is required but unavailable" >&2
      exit 1
    fi
    "$SWEEP_ROOT/preflight.sh" "$cfg"
  elif [[ -x "$SWEEP_ROOT/preflight.sh" ]]; then
    "$SWEEP_ROOT/preflight.sh" "$cfg"
  fi
  echo "===== targeted probe: $name ====="
  CONFIG="$cfg" CKPT="$BASE_CKPT" MODEL_LABEL="UMR_${name}" \
  CORRUPTIONS="$CORRUPTIONS" SEVERITY_LEVELS="$SEVERITY_LEVELS" \
  RUN_VIZ="$RUN_VIZ" EV_LOG_MODE="$EV_LOG_MODE" EV_LOG_COMPRESSION="$EV_LOG_COMPRESSION" \
  EV_LOG_SCENE_ONLY="$EV_LOG_SCENE_ONLY" \
  TEST_WORKERS_PER_GPU="$TEST_WORKERS_PER_GPU" \
    bash run_targeted_corruption_probe.sh
done
SH
sed -i "s#__BASE_CKPT__#${BASE_CKPT_PORTABLE}#g" "${OUT_ROOT}/validate_targeted_all.sh"
sed -i "s#__REQUIRE_RUNTIME_PREFLIGHT__#${REQUIRE_RUNTIME_PREFLIGHT}#g" "${OUT_ROOT}/validate_targeted_all.sh"
chmod +x "${OUT_ROOT}/validate_targeted_all.sh"

cat > "${OUT_ROOT}/validate_full_candidate.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <candidate_name>"
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/tools/test.py" && -f "$dir/multicorrupt_evaluation.sh" ]]; then
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
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:${PYTHONPATH}}"
name="$1"
cfg="$SCRIPT_DIR/configs/${name}.py"
BASE_CKPT="${BASE_CKPT:-__BASE_CKPT__}"
REQUIRE_RUNTIME_PREFLIGHT=__REQUIRE_RUNTIME_PREFLIGHT__
if [[ ! -f "$cfg" ]]; then
  echo "[ERROR] missing config: $cfg"
  exit 1
fi
if [[ "$REQUIRE_RUNTIME_PREFLIGHT" == "1" ]]; then
  if [[ ! -x "$SCRIPT_DIR/preflight.sh" ]]; then
    echo "[ERROR] formal preflight is required but unavailable" >&2
    exit 1
  fi
  "$SCRIPT_DIR/preflight.sh" "$cfg"
elif [[ -x "$SCRIPT_DIR/preflight.sh" ]]; then
  "$SCRIPT_DIR/preflight.sh" "$cfg"
fi
CONFIG="$cfg" CKPT="$BASE_CKPT" MODEL_LABEL="UMR_${name}" \
RUN_VIZ="${RUN_VIZ:-0}" EV_LOG_MODE="${EV_LOG_MODE:-off}" \
EV_LOG_COMPRESSION="${EV_LOG_COMPRESSION:-none}" \
EV_LOG_SCENE_ONLY="${EV_LOG_SCENE_ONLY:-0}" \
TEST_WORKERS_PER_GPU="${TEST_WORKERS_PER_GPU:-8}" \
  bash multicorrupt_evaluation.sh
SH
sed -i "s#__BASE_CKPT__#${BASE_CKPT_PORTABLE}#g" "${OUT_ROOT}/validate_full_candidate.sh"
sed -i "s#__REQUIRE_RUNTIME_PREFLIGHT__#${REQUIRE_RUNTIME_PREFLIGHT}#g" "${OUT_ROOT}/validate_full_candidate.sh"
chmod +x "${OUT_ROOT}/validate_full_candidate.sh"

PACK_CANDIDATES="$(awk 'NR>1 {printf "%s%s", sep, $1; sep=" "} END {print ""}' "$SUMMARY")"
FIRST_CANDIDATE="$(awk 'NR==2 {print $1}' "$SUMMARY")"

cat > "${OUT_ROOT}/README.txt" <<EOF
Rule-based UMR runtime pack

Base config:            ${BASE_CONFIG}
Base checkpoint:        ${BASE_CKPT_PORTABLE}
Clean reference stats:  ${CLEAN_REFERENCE_STATS_PORTABLE}
Scene mapper:           ${SCENE_MAPPER_PORTABLE}
Scene threshold key:    ${SCENE_THRESHOLD_KEY}
Scene threshold:        ${SCENE_THRESHOLD}
Utility mapper:         ${UTILITY_MAPPER_PORTABLE}
Utility threshold:      ${UTILITY_THRESHOLD}
Model fingerprint:      ${EXPECTED_MODEL_FINGERPRINT:-<legacy/unpinned>}
Require reference:      ${SUPP_REQUIRE_REFERENCE_STATS}
IMG action mapper:      ${IMG_ACTION_MAPPER_PORTABLE:-<not configured>}
BEV action mapper:      ${BEV_ACTION_MAPPER_PORTABLE:-<not configured>}
Fused-slot mapper:      ${FUSED_SLOT_MAPPER_PORTABLE:-<not configured>}
Branch-label evidence:  ${BRANCH_LABEL_ROOT} (design/audit only; not read by runtime configs)

Candidates:
$(awk 'NR>1 {print "  - " $1 ": policy=" $2 ", budget=" $3 ", guard=" $5}' "$SUMMARY")

Targeted smoke / faster metric run:
  CANDIDATES="${PACK_CANDIDATES}" CORRUPTIONS="beamsreducing fog" SEVERITY_LEVELS="3" \\
  RUN_VIZ=0 EV_LOG_MODE=sidecar EV_LOG_SCENE_ONLY=1 EV_LOG_COMPRESSION=none TEST_WORKERS_PER_GPU=8 bash ${OUT_ROOT}/validate_targeted_all.sh
  Note: scene-only mode is for fast metric/scene-gate smoke checks. It may not
  provide prediction-light rows, so use EV_LOG_SCENE_ONLY=0 for supplement
  admission diagnostics.

Targeted probe with prediction telemetry for the first packed candidate:
  CANDIDATES="${FIRST_CANDIDATE}" \\
  CORRUPTIONS="beamsreducing fog" SEVERITY_LEVELS="3" \\
  RUN_VIZ=0 EV_LOG_MODE=sidecar EV_LOG_SCENE_ONLY=0 EV_LOG_COMPRESSION=none TEST_WORKERS_PER_GPU=8 bash ${OUT_ROOT}/validate_targeted_all.sh

Remaining SGSUP_B/Rule runtime corruptions for packed candidates:
  CANDIDATES="${PACK_CANDIDATES}" CORRUPTIONS="beamsreducing fog motionblur pointsreducing spatialmisalignment temporalmisalignment" SEVERITY_LEVELS="1 2 3" \\
  RUN_VIZ=0 EV_LOG_MODE=sidecar EV_LOG_SCENE_ONLY=1 EV_LOG_COMPRESSION=none TEST_WORKERS_PER_GPU=8 bash ${OUT_ROOT}/validate_targeted_all.sh

Post-run diagnostics:
  SWEEP_ROOT=${OUT_ROOT} CANDIDATES="$CANDIDATES" bash tools/diagnose_rule_umr_runtime_pack.sh
  For full pred_added/candidate/utility diagnostics, rerun the specific
  candidate/corruption with EV_LOG_SCENE_ONLY=0.

Thesis safety notes:
  - Runtime does not read corruption type, severity, GT, or eval outputs.
  - Mappers/reference stats must be train/train-O.C derived only.
  - qsemtop candidates use runtime top-k high-score query aggregates so the
    gate reads the same high-confidence query subset concept as the offline
    query-semantic diagnostic, instead of all-query scene means.
  - Dual/img abstain thresholds in pure-BEV dual-abstain candidates are
    predeclared diagnostic constants selected from offline signal screening;
    they are not learned mappers and are not read from validation metrics.
  - Action utility mappers should be trained on the server with:
      LABEL_ROOT=<branch_candidate_label_analysis_root> OUT_DIR=<mapper_out> bash tools/train_umr_action_utility_from_branch_labels.sh
    Then copy mapper_candidate_useful.json/md, mapper_img_supp_opportunity.json/md,
    mapper_bev_supp_opportunity.json/md, and
    mapper_fused_slot_candidate_useful.json/md to local. ACT_M candidates require
    FUSED_SLOT_MAPPER_PATH=<mapper_out>/mapper_fused_slot_candidate_useful.json.
  - Rule-based UMR v2 guards are diagnostic candidates until branch-candidate logs confirm utility under img/dual damage.
EOF

if [[ "$RUN_STATIC_AUDIT" != "0" ]]; then
  python tools/check_umr_rule_runtime_pack.py \
    --sweep-root "$OUT_ROOT" \
    --repo-root "$(pwd)" \
    --candidates "$CANDIDATES"
fi

echo "Rule runtime root: ${OUT_ROOT}"
echo "Candidate summary: ${SUMMARY}"
echo "Targeted helper: ${OUT_ROOT}/validate_targeted_all.sh"
echo "Full helper: ${OUT_ROOT}/validate_full_candidate.sh"
