#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import OrderedDict


FORBIDDEN_PATH_MARKERS = (
    "multicorrupt_results",
    "targeted_probe",
    "nuscenes_infos_val",
    "corrupted-val",
    "corrupted_val",
    "corrupted/val",
    "val_corrupt",
    "val_clean",
    "validation",
)
FORBIDDEN_FEATURE_MARKERS = (
    "corruption",
    "severity",
    "condition",
    "gt",
    "ground_truth",
    "eval",
    "metric_summary",
)
SUPPORTED_UTILITY_FEATURES = {
    "score",
    "distance_to_ego",
    "center_unc",
    "height_unc",
    "cls_unc",
    "combined_unc",
    "branch_idx",
    "branch_rank",
    "cls_weight",
    "center_weight",
    "height_weight",
    "non_fused_prob",
    "fused_keep",
    "damage_gate",
    "bev_damage",
    "img_damage",
    "dual_damage",
    "branch_is_fused",
    "branch_is_bev",
    "branch_is_img",
    "v8_a4_route_score_bev",
    "v8_a4_route_score_img",
    "v8_bev_reliability",
    "v8_img_reliability",
    "v8_reliability_margin",
    "v9_a_bev",
    "v9_a_img",
    "v9_tail_prob_bev",
    "v9_tail_prob_img",
    "v9_route_score_bev",
    "v9_route_score_img",
}
VALID_GUARD_MODES = {
    "suppress",
    "drop",
    "block",
    "dual_small",
    "dual_tiny",
    "cap_dual",
    "bev_only_abstain",
    "pure_bev",
    "bev_only",
    "relative_bev_fused",
    "bev_dominant_fused_safe",
    "fused_safe_bev_dominant",
    "relative_bev_fused_composite",
    "pure_bev_composite",
    "bev_dominant_fused_safe_composite",
    "relative_bev_fused_dual_abstain",
    "pure_bev_dual_abstain",
    "bev_dominant_dual_abstain",
}


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_sgsup_b_qsem(row, fields, candidate):
    if fields.get("supp_qsem_gate_enable") is not True:
        row["errors"].append("sgsup_b_qsem_gate_not_enabled")
    if "qsemtop100" in candidate:
        expected_disagree_signal = (
            "qsem_top100_img_bev_class_disagree_ratio")
        expected_margin_signal = "qsem_top100_img_margin_mean"
    elif "qsemtop300" in candidate:
        expected_disagree_signal = (
            "qsem_top300_img_bev_class_disagree_ratio")
        expected_margin_signal = "qsem_top300_img_margin_mean"
    else:
        expected_disagree_signal = (
            "qsem_img_bev_class_disagree_flag_ratio")
        expected_margin_signal = "qsem_img_margin_mean"
    if fields.get("supp_qsem_disagree_signal") != expected_disagree_signal:
        row["errors"].append("sgsup_b_qsem_disagree_signal_invalid")
    if "marginoff" in candidate:
        if fields.get("supp_qsem_margin_signal") not in ("", None):
            row["errors"].append("sgsup_b_qsem_margin_signal_should_be_empty")
    elif fields.get("supp_qsem_margin_signal") != expected_margin_signal:
        row["errors"].append("sgsup_b_qsem_margin_signal_invalid")
    disagree_thr = float(fields.get("supp_qsem_disagree_threshold") or -1.0)
    margin_thr = float(fields.get("supp_qsem_margin_threshold") or -1.0)
    if (
            ("qsem070" in candidate or "d070" in candidate) and
            abs(disagree_thr - 0.70) > 1e-9):
        row["errors"].append("sgsup_b_qsem_disagree_suffix_mismatch")
    if (
            ("qsem065" in candidate or "d065" in candidate) and
            abs(disagree_thr - 0.65) > 1e-9):
        row["errors"].append("sgsup_b_qsem_disagree_suffix_mismatch")
    if "m085" in candidate and abs(margin_thr - 0.085) > 1e-9:
        row["errors"].append("sgsup_b_qsem_margin_suffix_mismatch")
    if "marginoff" in candidate and margin_thr >= 0.0:
        row["errors"].append("sgsup_b_qsem_margin_should_be_disabled")
    if not (0.60 <= disagree_thr <= 0.80):
        row["errors"].append("sgsup_b_qsem_disagree_threshold_invalid")
    if "marginoff" not in candidate and not (0.07 <= margin_thr <= 0.11):
        row["errors"].append("sgsup_b_qsem_margin_threshold_invalid")


def check_sgsup_b_contextheight(row, fields, candidate):
    if fields.get("supp_contextheight_gate_enable") is not True:
        row["errors"].append("sgsup_b_contextheight_gate_not_enabled")
    if fields.get("supp_qsem_gate_enable") not in (False, None):
        row["errors"].append("sgsup_b_contextheight_unexpected_qsem_gate")
    if fields.get("supp_qsem_disagree_signal"):
        row["errors"].append("sgsup_b_contextheight_unexpected_qsem_disagree_signal")
    if fields.get("supp_qsem_margin_signal"):
        row["errors"].append("sgsup_b_contextheight_unexpected_qsem_margin_signal")
    if fields.get("supp_dual_abstain_signal"):
        row["errors"].append("sgsup_b_contextheight_unexpected_dual_abstain_signal")
    if fields.get("supp_img_abstain_signal"):
        row["errors"].append("sgsup_b_contextheight_unexpected_img_abstain_signal")
    if fields.get("supp_marginal_utility_enable") is True:
        row["errors"].append("sgsup_b_contextheight_unexpected_marginal_gate")
    if fields.get("supp_fused_slot_utility_mapper_path"):
        row["errors"].append("sgsup_b_contextheight_unexpected_fused_slot_mapper")
    expected = {
        "supp_contextheight_context_signal": "v8_context_img_damage_mean",
        "supp_contextheight_height_signal": "distance_bin_abnormality_stats.height.bev.mid",
        "supp_contextheight_v9_bev_signal": "v9_a_bev_above_tau_mean",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            row["errors"].append(f"sgsup_b_{key}_invalid")
    context_thr = float(fields.get("supp_contextheight_context_threshold") or 0.0)
    height_thr = float(fields.get("supp_contextheight_height_threshold") or -1.0)
    v9_bev_thr = float(fields.get("supp_contextheight_v9_bev_threshold") or -1.0)
    if abs(context_thr - (-0.64882296)) > 1e-8:
        row["errors"].append("sgsup_b_contextheight_context_threshold_invalid")
    if "contextheight055" in candidate and abs(height_thr - 0.55) > 1e-9:
        row["errors"].append("sgsup_b_contextheight_height_suffix_mismatch")
    if "v9bev" in candidate and abs(v9_bev_thr - 0.60) > 1e-9:
        row["errors"].append("sgsup_b_contextheight_v9bev_threshold_invalid")


def rel_or_abs(root, path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def extract_value(text, key):
    patterns = [
        rf"{re.escape(key)}\s*=\s*'([^']*)'",
        rf'{re.escape(key)}\s*=\s*"([^"]*)"',
        rf"{re.escape(key)}\s*=\s*([A-Za-z0-9_.+-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            raw = m.group(1)
            if raw == "True":
                return True
            if raw == "False":
                return False
            try:
                return float(raw)
            except ValueError:
                return raw
    return None


def forbidden_path(path):
    lower = str(path or "").lower()
    norm = lower.replace("\\", "/")
    markers = [marker for marker in FORBIDDEN_PATH_MARKERS if marker in norm]
    parts = [part for part in norm.split("/") if part]
    if "val" in parts:
        markers.append("val")
    return markers


def check_mapper(repo_root, mapper_path, expected_target=None,
                 expected_branch_filter=None, runtime_features=False,
                 expected_model_fingerprint=""):
    result = {
        "path": mapper_path,
        "exists": False,
        "formal_calibration_safe": None,
        "debug_only": None,
        "target": None,
        "branch_filter": None,
        "unsafe_features": None,
        "features": None,
        "feature_count": None,
        "label_provenance": None,
        "forbidden_path_markers": forbidden_path(mapper_path),
        "errors": [],
    }
    full_path = rel_or_abs(repo_root, mapper_path)
    if not mapper_path or not os.path.exists(full_path):
        result["errors"].append("missing_mapper")
        return result
    if result["forbidden_path_markers"]:
        result["errors"].append(
            "forbidden_mapper_path:" + ",".join(result["forbidden_path_markers"]))
    result["exists"] = True
    try:
        data = load_json(full_path)
    except Exception as exc:
        result["errors"].append(f"load_failed:{exc}")
        return result
    result["formal_calibration_safe"] = data.get("formal_calibration_safe")
    result["debug_only"] = data.get("debug_only")
    result["target"] = data.get("target")
    result["branch_filter"] = data.get("branch_filter")
    result["unsafe_features"] = data.get("unsafe_features")
    result["label_provenance"] = data.get("label_provenance")
    result["model_provenance"] = data.get("model_provenance")
    features = data.get("features") or []
    result["features"] = features
    result["feature_count"] = len(features) if isinstance(features, list) else None
    if data.get("formal_calibration_safe") is not True:
        result["errors"].append("formal_calibration_safe_not_true")
    if data.get("debug_only") is True:
        result["errors"].append("debug_only_true")
    if data.get("unsafe_features"):
        result["errors"].append("unsafe_features_nonempty")
    if data.get("bad_missing_features"):
        result["errors"].append("bad_missing_features_nonempty")
    if data.get("label_summary_missing") is True:
        result["errors"].append("label_summary_missing")
    if expected_model_fingerprint:
        actual_fingerprint = str(
            (data.get("model_provenance") or {}).get("fingerprint") or "")
        if actual_fingerprint != expected_model_fingerprint:
            result["errors"].append("model_fingerprint_mismatch")
    if data.get("forbidden_input_markers"):
        result["errors"].append(
            f"forbidden_input_markers:{','.join(map(str, data.get('forbidden_input_markers')))}")
    if expected_target and data.get("target") != expected_target:
        result["errors"].append(f"unexpected_target:{data.get('target')}")
    if expected_branch_filter is not None:
        branch_filter = str(data.get("branch_filter") or "")
        provenance = data.get("label_provenance") or {}
        provenance_branch_filter = str(provenance.get("branch_filter") or "")
        if branch_filter != expected_branch_filter:
            result["errors"].append(
                f"unexpected_branch_filter:{branch_filter or 'all'}")
        if provenance_branch_filter != expected_branch_filter:
            result["errors"].append(
                "unexpected_provenance_branch_filter:"
                f"{provenance_branch_filter or 'all'}")
    if not isinstance(features, list) or not features:
        result["errors"].append("features_missing")
    else:
        for feature in features:
            lower = str(feature).lower()
            markers = [m for m in FORBIDDEN_FEATURE_MARKERS if m in lower]
            if markers:
                result["errors"].append(
                    f"forbidden_feature:{feature}:{','.join(markers)}")
            if runtime_features and feature not in SUPPORTED_UTILITY_FEATURES:
                result["errors"].append(f"unsupported_runtime_feature:{feature}")
    model = data.get("model") or {}
    weights = model.get("weights")
    if isinstance(features, list) and isinstance(weights, list) and len(weights) != len(features):
        result["errors"].append("model_weight_feature_len_mismatch")
    normalization = data.get("normalization") or {}
    for norm_key in ("mean", "std"):
        values = normalization.get(norm_key)
        if isinstance(features, list) and isinstance(values, list) and len(values) != len(features):
            result["errors"].append(f"normalization_{norm_key}_feature_len_mismatch")
    for key in ("label_summary_path", "prediction_log_root", "infos"):
        if key in data:
            markers = forbidden_path(data.get(key))
            if markers:
                result["errors"].append(f"forbidden_{key}:{','.join(markers)}")
    if runtime_features:
        provenance = data.get("label_provenance") or {}
        if not isinstance(provenance, dict) or not provenance:
            result["errors"].append("label_provenance_missing")
        else:
            if provenance.get("formal_source_kind") != "train_or_train_oc":
                result["errors"].append(
                    f"label_provenance_not_train:{provenance.get('formal_source_kind')}")
            if not provenance.get("labels_content_sha256"):
                result["errors"].append("label_provenance_missing_sha256")
            if provenance.get("forbidden_input_markers"):
                result["errors"].append(
                    "label_provenance_forbidden_markers:"
                    + ",".join(map(str, provenance.get("forbidden_input_markers"))))
            for key in ("labels_path", "labels_abspath", "labels_realpath",
                        "label_root", "infos", "prediction_log_root"):
                if provenance.get(key):
                    markers = forbidden_path(provenance.get(key))
                    if markers:
                        result["errors"].append(
                            f"label_provenance_forbidden_{key}:{','.join(markers)}")
    return result


def check_reference_stats(repo_root, stats_path, required_scene_metrics=None,
                          expected_model_fingerprint=""):
    result = {
        "path": stats_path,
        "exists": False,
        "forbidden_path_markers": forbidden_path(stats_path),
        "errors": [],
    }
    full_path = rel_or_abs(repo_root, stats_path)
    if not stats_path or not os.path.exists(full_path):
        result["errors"].append("missing_reference_stats")
        return result
    result["exists"] = True
    try:
        data = load_json(full_path)
    except Exception as exc:
        result["errors"].append(f"load_failed:{exc}")
        return result
    if data.get("format") != "umr_clean_reference_stats_v1":
        result["errors"].append(
            f"unexpected_reference_stats_format:{data.get('format')}")
    if expected_model_fingerprint:
        actual_fingerprint = str(
            (data.get("model_provenance") or {}).get("fingerprint") or "")
        if actual_fingerprint != expected_model_fingerprint:
            result["errors"].append("reference_stats_model_fingerprint_mismatch")
    source_files = data.get("source_files")
    source_meta_samples = 0
    source_meta_seen = 0
    source_meta_predictions = 0
    if not isinstance(source_files, list) or not source_files:
        result["errors"].append("reference_stats_missing_source_files")
    else:
        for source in source_files:
            markers = forbidden_path(source)
            if markers:
                result["errors"].append(
                    "reference_stats_forbidden_source:"
                    + ",".join(markers))
            source_full_path = rel_or_abs(repo_root, source)
            source_dir = os.path.dirname(source_full_path)
            source_name = os.path.basename(source_full_path)
            if ".jsonl" in source_name:
                source_stem = source_name.split(".jsonl", 1)[0]
            else:
                source_stem = os.path.splitext(source_name)[0]
            suffix = ""
            for prefix in ("ev_prediction_light", "ev_scene_summary"):
                if source_stem.startswith(prefix):
                    suffix = source_stem[len(prefix):]
                    break
            meta_path = os.path.join(source_dir, f"ev_meta{suffix}.json")
            if not os.path.exists(meta_path):
                result["errors"].append("reference_stats_missing_source_meta")
                continue
            try:
                meta = load_json(meta_path)
            except Exception as exc:
                result["errors"].append(f"reference_stats_meta_load_failed:{exc}")
                continue
            source_meta_seen += 1
            if not (meta.get("extra_meta") or {}).get("completed"):
                result["errors"].append("reference_stats_source_not_completed")
            source_meta_samples += int(meta.get("num_samples") or 0)
            source_meta_predictions += int(meta.get("num_predictions") or 0)
    # nuScenes train has 28k samples; val has 6k. This is a coarse but useful
    # static guard against accidentally using clean val reference stats.
    if source_meta_seen and source_meta_samples < 20000:
        result["errors"].append(
            f"reference_stats_source_samples_too_small:{source_meta_samples}")
    if source_meta_seen and int(data.get("records") or 0) != source_meta_predictions:
        result["errors"].append(
            "reference_stats_record_meta_mismatch:"
            f"{data.get('records')}!={source_meta_predictions}")
    scene_global = data.get("scene_global") or {}
    required_scene_metrics = required_scene_metrics or {
        "bev_top10_unc": ("p99",),
        "img_top10_unc": ("p99",),
    }
    for metric, stat_keys in required_scene_metrics.items():
        metric_stats = scene_global.get(metric) or {}
        for stat_key in stat_keys:
            if stat_key not in metric_stats:
                result["errors"].append(
                    f"missing_scene_global_{metric}_{stat_key}")
    return result


def check_config(repo_root, candidate, config_path):
    full_config = rel_or_abs(repo_root, config_path)
    row = OrderedDict()
    row["candidate"] = candidate
    row["config"] = config_path
    row["exists"] = os.path.exists(full_config)
    row["errors"] = []
    row["warnings"] = []
    row["abstain_threshold_policy"] = ""
    if not row["exists"]:
        row["errors"].append("missing_config")
        return row
    text = read_text(full_config)
    fields = {
        key: extract_value(text, key)
        for key in (
            "ensemble_mode",
            "supp_enable",
            "supp_mode",
            "supp_max_add_queries",
            "supp_img_score_threshold",
            "supp_bev_score_threshold",
            "supp_nms_center_dist",
            "supp_expected_model_fingerprint",
            "supp_require_reference_stats",
            "supp_utility_mapper_path",
            "supp_utility_threshold",
            "supp_img_utility_mapper_path",
            "supp_bev_utility_mapper_path",
            "supp_img_utility_threshold",
            "supp_bev_utility_threshold",
            "supp_fused_slot_utility_mapper_path",
            "supp_marginal_utility_enable",
            "supp_marginal_utility_margin",
            "supp_dual_damage_marginal_margin",
            "supp_scene_mapper_path",
            "supp_scene_mapper_threshold",
            "supp_scene_mapper_gate_mode",
            "supp_scene_budget_min_ratio",
            "supp_scene_budget_max_ratio",
            "supp_rank_mode",
            "supp_img_damage_guard_enable",
            "supp_img_damage_guard_mode",
            "supp_img_damage_signal",
            "supp_img_damage_metric_name",
            "supp_img_damage_threshold_key",
            "supp_img_damage_threshold_scale",
            "supp_fused_damage_signal",
            "supp_fused_damage_metric_name",
            "supp_fused_damage_threshold_key",
            "supp_fused_damage_threshold_scale",
            "supp_relative_damage_reference_key",
            "supp_bev_fused_dominance_ratio",
            "supp_bev_damage_upper_threshold_scale",
            "supp_dual_abstain_signal",
            "supp_dual_abstain_threshold",
            "supp_img_abstain_signal",
            "supp_img_abstain_threshold",
            "supp_qsem_gate_enable",
            "supp_qsem_disagree_signal",
            "supp_qsem_disagree_threshold",
            "supp_qsem_margin_signal",
            "supp_qsem_margin_threshold",
            "supp_contextheight_gate_enable",
            "supp_contextheight_context_signal",
            "supp_contextheight_context_threshold",
            "supp_contextheight_height_signal",
            "supp_contextheight_height_threshold",
            "supp_contextheight_v9_bev_signal",
            "supp_contextheight_v9_bev_threshold",
            "scrg_bev_metric_name",
            "scrg_img_metric_name",
            "scrg_bev_signal",
            "scrg_img_signal",
            "supp_dual_damage_budget_ratio",
            "supp_dual_damage_utility_threshold",
            "supp_dual_damage_gate_mode",
            "supp_dual_scene_mapper_threshold",
            "supp_dual_bev_threshold_scale",
            "supp_dual_bev_rank_boost",
            "v7_reference_stats_path",
            "v7_allow_missing_stats_fallback",
        )
    }
    row.update(fields)
    if fields["ensemble_mode"] != "damage_gated_soft_router":
        row["errors"].append("not_damage_gated_soft_router")
    if fields["supp_enable"] is not True:
        row["errors"].append("supp_not_enabled")
    if fields["v7_allow_missing_stats_fallback"] is not False:
        row["errors"].append("v7_missing_stats_not_fail_closed")
    expected_model_fingerprint = str(
        fields.get("supp_expected_model_fingerprint") or "")
    if expected_model_fingerprint and fields.get(
            "supp_require_reference_stats") is not True:
        row["errors"].append("pinned_artifacts_without_required_reference")
    required_paths = ("supp_scene_mapper_path", "v7_reference_stats_path")
    for key in required_paths:
        markers = forbidden_path(fields.get(key))
        if markers:
            row["errors"].append(f"forbidden_{key}:{','.join(markers)}")
        if not fields.get(key):
            row["errors"].append(f"missing_{key}")
        elif not os.path.exists(rel_or_abs(repo_root, str(fields.get(key)))):
            row["errors"].append(f"path_not_found:{key}")
    optional_mapper_paths = (
        "supp_utility_mapper_path",
        "supp_img_utility_mapper_path",
        "supp_bev_utility_mapper_path",
        "supp_fused_slot_utility_mapper_path",
    )
    for key in optional_mapper_paths:
        markers = forbidden_path(fields.get(key))
        if markers:
            row["errors"].append(f"forbidden_{key}:{','.join(markers)}")
        if fields.get(key) and not os.path.exists(rel_or_abs(repo_root, str(fields.get(key)))):
            row["errors"].append(f"path_not_found:{key}")
    is_action_candidate = candidate.startswith("RULE_ACT")
    if not is_action_candidate and not fields.get("supp_utility_mapper_path"):
        row["errors"].append("missing_supp_utility_mapper_path")
    if candidate.startswith("RULE_SGSUP_B"):
        is_bevonly_abstain = candidate.startswith("RULE_SGSUP_B_bevonly_abstain")
        is_relative_fused = candidate.startswith("RULE_SGSUP_B_relbev_fusedguard")
        is_imgsens = candidate.startswith("RULE_SGSUP_B_relbev_fusedguard_imgsens")
        is_purebev = candidate.startswith("RULE_SGSUP_B_relbev_fusedguard_purebev")
        is_dual_abstain = candidate.startswith(
            "RULE_SGSUP_B_relbev_fusedguard_purebev_dual")
        is_qsem_gate = (
            candidate.startswith("RULE_SGSUP_B_relbev_fusedguard_purebev") and
            "_qsem" in candidate)
        is_contextheight_gate = "contextheight" in candidate
        if is_bevonly_abstain:
            if fields["supp_img_damage_guard_enable"] is not True:
                row["errors"].append("sgsup_b_bevonly_guard_not_enabled")
            if str(fields.get("supp_img_damage_guard_mode") or "").lower() not in (
                    "bev_only_abstain", "pure_bev", "bev_only"):
                row["errors"].append("sgsup_b_bevonly_guard_mode_invalid")
        elif is_relative_fused:
            if fields["supp_img_damage_guard_enable"] is not True:
                row["errors"].append("sgsup_b_relative_guard_not_enabled")
            if str(fields.get("supp_img_damage_guard_mode") or "").lower() not in (
                    "relative_bev_fused", "bev_dominant_fused_safe",
                    "fused_safe_bev_dominant",
                    "relative_bev_fused_composite", "pure_bev_composite",
                    "bev_dominant_fused_safe_composite",
                    "relative_bev_fused_dual_abstain",
                    "pure_bev_dual_abstain",
                    "bev_dominant_dual_abstain"):
                row["errors"].append("sgsup_b_relative_guard_mode_invalid")
            if fields.get("supp_fused_damage_signal") != "fused_raw_modality_unc":
                row["errors"].append("sgsup_b_relative_fused_signal_invalid")
            if fields.get("supp_fused_damage_metric_name") != "v8_gscrg_raw_unc_fused_top10":
                row["errors"].append("sgsup_b_relative_fused_metric_invalid")
            if fields.get("supp_fused_damage_threshold_key") != "p99":
                row["errors"].append("sgsup_b_relative_fused_threshold_key_invalid")
            if float(fields.get("supp_fused_damage_threshold_scale") or 0.0) <= 0.0:
                row["errors"].append("sgsup_b_relative_fused_scale_missing")
            if fields.get("supp_relative_damage_reference_key") != "median":
                row["errors"].append("sgsup_b_relative_reference_key_invalid")
            if float(fields.get("supp_bev_fused_dominance_ratio") or 0.0) <= 0.0:
                row["errors"].append("sgsup_b_relative_dominance_missing")
            if fields.get("scrg_bev_signal") != "bev_raw_modality_unc":
                row["errors"].append("sgsup_b_relative_bev_signal_invalid")
            if fields.get("scrg_img_signal") != "img_raw_modality_unc":
                row["errors"].append("sgsup_b_relative_img_signal_invalid")
            if fields.get("scrg_bev_metric_name") != "v8_gscrg_raw_unc_bev_top10":
                row["errors"].append("sgsup_b_relative_bev_metric_invalid")
            if fields.get("scrg_img_metric_name") != "v8_gscrg_raw_unc_img_top10":
                row["errors"].append("sgsup_b_relative_img_metric_invalid")
            if is_imgsens:
                if fields.get("supp_img_damage_signal") != "v8_a4_cls_unc_z_img":
                    row["errors"].append("sgsup_b_imgsens_signal_invalid")
                if fields.get("supp_img_damage_metric_name") != "v8_a4_cls_unc_z_img_top10":
                    row["errors"].append("sgsup_b_imgsens_metric_invalid")
                if fields.get("supp_img_damage_threshold_key") != "p99":
                    row["errors"].append("sgsup_b_imgsens_threshold_key_invalid")
                if float(fields.get("supp_img_damage_threshold_scale") or 0.0) >= 1.0:
                    row["errors"].append("sgsup_b_imgsens_threshold_not_sensitive")
            if is_purebev:
                if fields.get("supp_img_damage_signal"):
                    row["errors"].append("sgsup_b_purebev_should_not_use_img_signal")
                if fields.get("supp_img_damage_metric_name"):
                    row["errors"].append("sgsup_b_purebev_should_not_use_img_metric")
                if float(fields.get("supp_fused_damage_threshold_scale") or 0.0) > 1.5:
                    row["errors"].append("sgsup_b_purebev_fused_guard_too_loose")
                if float(fields.get("supp_bev_fused_dominance_ratio") or 0.0) < 1.40:
                    row["errors"].append("sgsup_b_purebev_dominance_too_low")
                if is_dual_abstain:
                    row["abstain_threshold_policy"] = (
                        "predeclared_diagnostic_constants")
                    guard_mode = str(fields.get("supp_img_damage_guard_mode") or "").lower()
                    if guard_mode not in (
                            "relative_bev_fused_dual_abstain",
                            "pure_bev_dual_abstain",
                            "bev_dominant_dual_abstain"):
                        row["errors"].append("sgsup_b_dual_abstain_guard_mode_invalid")
                    if fields.get("supp_dual_abstain_signal") != "v9_state_confirmed_dual_mean":
                        row["errors"].append("sgsup_b_dual_abstain_signal_invalid")
                    if fields.get("supp_img_abstain_signal") != "v9_a_img_above_tau_mean":
                        row["errors"].append("sgsup_b_img_abstain_signal_invalid")
                    dual_thr = float(fields.get("supp_dual_abstain_threshold") or -1.0)
                    img_thr = float(fields.get("supp_img_abstain_threshold") or -1.0)
                    if not (0.05 <= dual_thr <= 0.20):
                        row["errors"].append("sgsup_b_dual_abstain_threshold_invalid")
                    if not (0.08 <= img_thr <= 0.25):
                        row["errors"].append("sgsup_b_img_abstain_threshold_invalid")
                    expected_thresholds = {
                        "dual010_img015": (0.10, 0.15),
                        "dual012_img018": (0.12, 0.18),
                    }
                    for suffix, (expected_dual, expected_img) in expected_thresholds.items():
                        if suffix in candidate:
                            if abs(dual_thr - expected_dual) > 1e-9:
                                row["errors"].append(
                                    "sgsup_b_dual_abstain_threshold_suffix_mismatch")
                            if abs(img_thr - expected_img) > 1e-9:
                                row["errors"].append(
                                    "sgsup_b_img_abstain_threshold_suffix_mismatch")
                if is_qsem_gate:
                    check_sgsup_b_qsem(row, fields, candidate)
                elif fields.get("supp_qsem_gate_enable") not in (False, None):
                    row["errors"].append("sgsup_b_unexpected_qsem_gate")
                if candidate.startswith("RULE_SGSUP_B_relbev_fusedguard_purebev_comp"):
                    guard_mode = str(fields.get("supp_img_damage_guard_mode") or "").lower()
                    if guard_mode not in (
                            "relative_bev_fused_composite",
                            "pure_bev_composite",
                            "bev_dominant_fused_safe_composite"):
                        row["errors"].append("sgsup_b_composite_guard_mode_invalid")
                    if float(fields.get("supp_bev_fused_dominance_ratio") or 0.0) < 1.50:
                        row["errors"].append("sgsup_b_composite_dominance_too_low")
                    upper_scale = float(fields.get("supp_bev_damage_upper_threshold_scale") or 0.0)
                    if upper_scale <= 1.0:
                        row["errors"].append("sgsup_b_composite_upper_missing")
                    if upper_scale > 2.0:
                        row["errors"].append("sgsup_b_composite_upper_too_loose")
                    if "margin" in candidate:
                        if fields.get("supp_marginal_utility_enable") is not True:
                            row["errors"].append("sgsup_b_composite_margin_not_enabled")
                        if not fields.get("supp_fused_slot_utility_mapper_path"):
                            row["errors"].append("sgsup_b_composite_missing_fused_slot_mapper")
                        if float(fields.get("supp_marginal_utility_margin") or 0.0) <= 0.0:
                            row["errors"].append("sgsup_b_composite_margin_missing")
        elif fields["supp_img_damage_guard_enable"] not in (False, None):
            row["errors"].append("sgsup_b_guard_should_be_off")
        if is_contextheight_gate:
            check_sgsup_b_contextheight(row, fields, candidate)
        elif fields.get("supp_contextheight_gate_enable") not in (False, None):
            row["errors"].append("sgsup_b_unexpected_contextheight_gate")
        if fields.get("supp_mode") != "img_highscore":
            row["errors"].append("sgsup_b_unexpected_supp_mode")
        if fields.get("supp_img_utility_mapper_path") or fields.get("supp_bev_utility_mapper_path"):
            row["errors"].append("sgsup_b_should_not_use_action_mapper")
        nms_match = re.search(r"_nms([0-9]{3,4})_", candidate)
        if nms_match:
            expected_nms = float(int(nms_match.group(1))) / 1000.0
            actual_nms = float(fields.get("supp_nms_center_dist") or 0.0)
            if abs(actual_nms - expected_nms) > 1e-9:
                row["errors"].append("sgsup_b_nms_center_dist_suffix_mismatch")
    if is_action_candidate:
        if not fields.get("supp_img_utility_mapper_path"):
            row["errors"].append("action_candidate_missing_img_mapper")
        if candidate.startswith("RULE_ACT_C") and not fields.get("supp_bev_utility_mapper_path"):
            row["errors"].append("dual_action_candidate_missing_bev_mapper")
        if candidate.startswith("RULE_ACT_M"):
            if not fields.get("supp_fused_slot_utility_mapper_path"):
                row["errors"].append("act_m_missing_fused_slot_mapper")
            if fields.get("supp_marginal_utility_enable") is not True:
                row["errors"].append("act_m_marginal_gate_not_enabled")
            if float(fields.get("supp_marginal_utility_margin") or 0.0) <= 0.0:
                row["errors"].append("act_m_margin_missing")
            if float(fields.get("supp_dual_damage_marginal_margin") or 0.0) <= 0.0:
                row["errors"].append("act_m_dual_margin_missing")
        if candidate.startswith(("RULE_ACT_A", "RULE_ACT_B")):
            if fields.get("supp_mode") != "img_highscore":
                row["errors"].append("img_action_candidate_unexpected_supp_mode")
            if fields.get("supp_img_damage_guard_enable") not in (False, None):
                row["errors"].append("img_action_candidate_guard_should_be_off")
        if candidate.startswith("RULE_ACT_M"):
            if fields.get("supp_mode") != "img_highscore":
                row["errors"].append("act_m_unexpected_supp_mode")
            if fields.get("supp_img_damage_guard_enable") not in (False, None):
                row["errors"].append("act_m_guard_should_be_off")
        if candidate.startswith("RULE_ACT_C"):
            if fields.get("supp_mode") != "dual_branch_highscore":
                row["errors"].append("dual_action_candidate_mode_not_enabled")
            if fields.get("supp_rank_mode") != "utility_product":
                row["errors"].append("dual_action_rank_not_utility_product")
    if candidate.startswith("RULE_UMR2"):
        if fields["supp_img_damage_guard_enable"] is not True:
            row["errors"].append("umr2_guard_not_enabled")
        guard_mode = str(fields["supp_img_damage_guard_mode"]).lower()
        if guard_mode in ("", "none", "off"):
            row["errors"].append("umr2_guard_mode_off")
        elif guard_mode not in VALID_GUARD_MODES:
            row["errors"].append(f"invalid_umr2_guard_mode:{guard_mode}")
        is_rule_b = candidate.startswith("RULE_UMR2_B")
        is_relaxed_dual = (
            candidate.startswith("RULE_UMR2_C") or
            candidate.startswith("RULE_UMR2_D")
        )
        is_dual_branch_candidate = (
            is_rule_b or is_relaxed_dual or
            "dualbranch" in candidate.lower() or
            "dualsmall" in candidate.lower() or
            fields.get("supp_mode") == "dual_branch_highscore"
        )
        if is_dual_branch_candidate:
            if fields.get("supp_mode") != "dual_branch_highscore":
                row["errors"].append("umr2_dualbranch_mode_not_enabled")
            if fields.get("supp_rank_mode") != "utility_product":
                row["errors"].append("umr2_dualbranch_rank_not_utility_product")
            if float(fields.get("supp_dual_damage_budget_ratio") or 0.0) <= 0.0:
                row["errors"].append("umr2_dualbranch_budget_missing")
            if float(fields.get("supp_dual_damage_utility_threshold") or 0.0) <= 0.0:
                row["errors"].append("umr2_dualbranch_utility_threshold_missing")
            if float(fields.get("supp_bev_score_threshold") or 0.0) <= 0.0:
                row["errors"].append("umr2_dualbranch_bev_threshold_missing")
            dual_gate_mode = str(fields.get("supp_dual_damage_gate_mode") or "strict_img").lower()
            if dual_gate_mode not in (
                    "strict", "strict_img", "img", "img_only",
                    "scene_mapper", "mapper", "scene_mapper_or_img", "mapper_or_img",
                    "bev_score", "bev", "bev_or_img",
                    "mapper_or_bev", "scene_mapper_or_bev", "relaxed", "relaxed_or_img"):
                row["errors"].append(f"invalid_dual_damage_gate_mode:{dual_gate_mode}")
            if is_relaxed_dual:
                if dual_gate_mode not in ("scene_mapper_or_img", "mapper_or_img",
                                          "mapper_or_bev", "scene_mapper_or_bev",
                                          "relaxed", "relaxed_or_img"):
                    row["errors"].append("relaxed_dual_candidate_missing_mapper_gate")
                if float(fields.get("supp_dual_scene_mapper_threshold") or 0.0) <= 0.0:
                    row["errors"].append("relaxed_dual_scene_threshold_missing")
            if candidate.startswith("RULE_UMR2_D"):
                if float(fields.get("supp_dual_bev_rank_boost") or 0.0) <= 1.0:
                    row["errors"].append("dual_bevboost_candidate_boost_missing")
    row["scene_mapper"] = check_mapper(
        repo_root,
        fields.get("supp_scene_mapper_path"),
        expected_target="scene_bev_damage_applicable",
        expected_model_fingerprint=expected_model_fingerprint)
    if fields.get("supp_utility_mapper_path"):
        row["utility_mapper"] = check_mapper(
            repo_root,
            fields.get("supp_utility_mapper_path"),
            expected_target="candidate_useful",
            runtime_features=True,
            expected_model_fingerprint=expected_model_fingerprint)
    else:
        row["utility_mapper"] = {"errors": [], "target": None, "path": ""}
    if fields.get("supp_img_utility_mapper_path"):
        row["img_action_mapper"] = check_mapper(
            repo_root,
            fields.get("supp_img_utility_mapper_path"),
            expected_target="img_supp_opportunity",
            runtime_features=True,
            expected_model_fingerprint=expected_model_fingerprint)
    else:
        row["img_action_mapper"] = {"errors": [], "target": None, "path": ""}
    if fields.get("supp_bev_utility_mapper_path"):
        row["bev_action_mapper"] = check_mapper(
            repo_root,
            fields.get("supp_bev_utility_mapper_path"),
            expected_target="bev_supp_opportunity",
            runtime_features=True,
            expected_model_fingerprint=expected_model_fingerprint)
    else:
        row["bev_action_mapper"] = {"errors": [], "target": None, "path": ""}
    if fields.get("supp_fused_slot_utility_mapper_path"):
        row["fused_slot_mapper"] = check_mapper(
            repo_root,
            fields.get("supp_fused_slot_utility_mapper_path"),
            expected_target="candidate_useful",
            expected_branch_filter="fused",
            runtime_features=True,
            expected_model_fingerprint=expected_model_fingerprint)
    else:
        row["fused_slot_mapper"] = {"errors": [], "target": None, "path": ""}
    reference_requirements = None
    if candidate.startswith("RULE_SGSUP_B_relbev_fusedguard"):
        reference_requirements = {
            "v8_gscrg_raw_unc_bev_top10": ("p99", "median"),
            "v8_gscrg_raw_unc_img_top10": ("p99",),
            "v8_gscrg_raw_unc_fused_top10": ("p99", "median"),
        }
        if candidate.startswith("RULE_SGSUP_B_relbev_fusedguard_imgsens"):
            reference_requirements["v8_a4_cls_unc_z_img_top10"] = ("p99",)
    row["reference_stats"] = check_reference_stats(
        repo_root,
        fields.get("v7_reference_stats_path"),
        required_scene_metrics=reference_requirements,
        expected_model_fingerprint=expected_model_fingerprint)
    for mapper_key in ("scene_mapper", "utility_mapper", "img_action_mapper",
                       "bev_action_mapper", "fused_slot_mapper"):
        if row[mapper_key]["errors"]:
            row["errors"].append(f"{mapper_key}_invalid")
    if row["reference_stats"]["errors"]:
        row["errors"].append("reference_stats_invalid")
    return row


def read_candidates(sweep_root, explicit):
    if explicit:
        return [(name, os.path.join(sweep_root, "configs", f"{name}.py")) for name in explicit.split()]
    summary = os.path.join(sweep_root, "summaries", "candidates.tsv")
    out = []
    with open(summary, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append((row["name"], row.get("config") or os.path.join(sweep_root, "configs", f"{row['name']}.py")))
    return out


def write_markdown(results, path):
    lines = [
        "# Rule-based UMR Runtime Pack Audit",
        "",
        "| candidate | policy_ok | guard | scene_mapper | utility_mapper | img_action | bev_action | fused_slot | reference | errors |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        scene_ok = not row.get("scene_mapper", {}).get("errors")
        util_ok = not row.get("utility_mapper", {}).get("errors")
        img_action_ok = not row.get("img_action_mapper", {}).get("errors")
        bev_action_ok = not row.get("bev_action_mapper", {}).get("errors")
        fused_slot_ok = not row.get("fused_slot_mapper", {}).get("errors")
        ref_ok = not row.get("reference_stats", {}).get("errors")
        errors = ", ".join(row.get("errors") or []) or "none"
        guard = (
            f"mode={row.get('supp_mode')}, "
            f"guard={row.get('supp_img_damage_guard_enable')}:{row.get('supp_img_damage_guard_mode')}, "
            f"fused={row.get('supp_fused_damage_threshold_key')}x"
            f"{row.get('supp_fused_damage_threshold_scale')}, "
            f"bev/fused={row.get('supp_bev_fused_dominance_ratio')}, "
            f"bev_upper={row.get('supp_bev_damage_upper_threshold_scale')}, "
            f"dual_abstain={row.get('supp_dual_abstain_signal')}@"
            f"{row.get('supp_dual_abstain_threshold')}, "
            f"img_abstain={row.get('supp_img_abstain_signal')}@"
            f"{row.get('supp_img_abstain_threshold')}, "
            f"qsem={row.get('supp_qsem_gate_enable')}:"
            f"{row.get('supp_qsem_disagree_signal')}@"
            f"{row.get('supp_qsem_disagree_threshold')}/"
            f"{row.get('supp_qsem_margin_signal')}@"
            f"{row.get('supp_qsem_margin_threshold')}, "
            f"contextheight={row.get('supp_contextheight_gate_enable')}:"
            f"{row.get('supp_contextheight_context_signal')}@"
            f"{row.get('supp_contextheight_context_threshold')}/"
            f"{row.get('supp_contextheight_height_signal')}@"
            f"{row.get('supp_contextheight_height_threshold')}/"
            f"{row.get('supp_contextheight_v9_bev_signal')}@"
            f"{row.get('supp_contextheight_v9_bev_threshold')}, "
            f"nms_center={row.get('supp_nms_center_dist')}, "
            f"abstain_policy={row.get('abstain_threshold_policy') or 'none'}, "
            f"dual={row.get('supp_dual_damage_gate_mode')}@{row.get('supp_dual_scene_mapper_threshold')}, "
            f"bev_boost={row.get('supp_dual_bev_rank_boost')}"
        )
        lines.append(
            f"| {row['candidate']} | {str(not row.get('errors')).lower()} | "
            f"{guard} | {str(scene_ok).lower()} | {str(util_ok).lower()} | "
            f"{str(img_action_ok).lower()} | {str(bev_action_ok).lower()} | "
            f"{str(fused_slot_ok).lower()} | {str(ref_ok).lower()} | {errors} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- This is a static provenance/config audit. It does not prove AP/NDS behavior.",
        "- Formal inputs must be train/train-O.C derived. Paths containing MultiCorrupt results, targeted probes, or nuScenes val info are rejected.",
        "- `RULE_SGSUP_B_runtime` intentionally has image/dual guard disabled to preserve the measured SGSUP_B behavior.",
        "- `RULE_SGSUP_B_bevonly_abstain_*` keeps SGSUP-B action/budget but admits only BEV-damaged, IMG-clean scenes; all other modality states abstain to fused.",
        "- `RULE_SGSUP_B_relbev_fusedguard_*` replaces the absolute BEV-p99 requirement with train-reference normalized BEV-over-fused dominance and abstains on severe fused or IMG damage.",
        "- `RULE_SGSUP_B_relbev_fusedguard_purebev*` is a conservative pure-BEV admissibility probe: it avoids image-side semantic thresholds and requires stronger BEV-over-fused dominance before applying SGSUP-B.",
        "- `RULE_SGSUP_B_relbev_fusedguard_purebev_dual*` keeps the pure-BEV dominance contract and additionally abstains on dual-state and IMG-anomaly scene summaries.",
        "- `RULE_SGSUP_B_relbev_fusedguard_purebev_dual*` abstain thresholds are predeclared diagnostic constants from offline signal-screening, not learned mappers or eval/GT-derived runtime inputs.",
        "- `RULE_SGSUP_B_relbev_fusedguard_purebev*_qsem*` additionally gates on runtime query-semantic image/BEV class disagreement; `marginoff` variants intentionally disable the image confidence-margin gate to isolate disagreement-only behavior.",
        "- `RULE_SGSUP_B_contextheight*` applies the replay-matched pure-BEV subset rule `(context < -0.64882296 OR height.bev.mid < 0.55) AND v9_a_bev_above_tau_mean < 0.60` before SGSUP-B admission.",
        "- `RULE_SGSUP_B_contextheight*_nms*` keeps the same admission rule and sweeps the supplement same-class center-distance duplicate suppression threshold.",
        "- `RULE_SGSUP_B_relbev_fusedguard_purebev_contextheight*` keeps the older relative BEV/fused guard as an ablation around the same context-height subset rule.",
        "- `RULE_SGSUP_B_relbev_fusedguard_purebev_comp*` additionally requires BEV damage, rejects excessively broad BEV damage, and can optionally apply fused-slot marginal utility.",
        "- `RULE_UMR2_*` candidates must have the image-damage guard enabled and fail closed if reference evidence is unavailable.",
        "- `RULE_ACT_*` candidates use train/train-O.C branch-candidate action mappers and are independent from the SGSUP_B baseline path.",
        "- `RULE_ACT_M_*` candidates additionally require a train/train-O.C fused-slot mapper and a positive candidate-minus-slot utility margin.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--candidates", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    sweep_root = os.path.abspath(args.sweep_root)
    results = [
        check_config(repo_root, name, cfg)
        for name, cfg in read_candidates(sweep_root, args.candidates)
    ]
    out_json = args.out_json or os.path.join(sweep_root, "summaries", "rule_runtime_static_audit.json")
    out_md = args.out_md or os.path.join(sweep_root, "summaries", "rule_runtime_static_audit.md")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    write_markdown(results, out_md)
    print(f"[Rule UMR audit] wrote JSON: {out_json}")
    print(f"[Rule UMR audit] wrote MD:   {out_md}")
    failed = [row["candidate"] for row in results if row.get("errors")]
    if failed:
        print(f"[Rule UMR audit][FAIL] {', '.join(failed)}")
        raise SystemExit(1)
    print("[Rule UMR audit] all candidates passed static checks")


if __name__ == "__main__":
    main()
