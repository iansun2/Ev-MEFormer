#!/usr/bin/env python3
"""Audit a fully merged VoV or R50 UMR SGSUP-B runtime configuration."""

import argparse
import ast
import hashlib
import json
import math
import os

from mmcv import Config

try:
    from umr_collector_contract import current_collector_contract
except ImportError:  # Support importing this file as a tools.* module.
    from tools.umr_collector_contract import current_collector_contract


EXPECTED_SCENE_FEATURES = [
    "scene_fused_weight", "scene_selector_entropy",
    "scene_v8_a4_bev_damage_z_mean", "scene_v8_a4_route_score_bev_mean",
    "scene_v8_bev_damage_mean", "scene_v8_context_bev_damage_mean",
    "scene_v8_high_damage_query_ratio", "scene_v8_bev_reliability_mean",
    "scene_v8_reliability_margin_mean", "scene_v9_a_bev_mean",
    "scene_v9_a_bev_above_tau_mean", "scene_v9_tail_prob_bev_mean",
    "scene_v9_tail_excess_bev_mean", "scene_v9_q95_shift_bev_mean",
    "scene_v9_route_score_bev_mean", "scene_v9_state_confirmed_bev_mean",
    "scene_v9_center_l1_p95", "scene_v9_fused_keep_unit_q95",
]


def source_log_collector_errors(record, expected_collector):
    """Return portable-preflight errors for one formal source-log shard."""
    errors = []
    if int(record.get("samples_per_gpu") or 0) != 1:
        errors.append(
            "source-log shard was not collected with samples_per_gpu=1: {}".format(
                record.get("path")))
    if record.get("collector_code") != expected_collector:
        errors.append(
            "source-log collector code differs from current runtime: {}".format(
                record.get("path")))
    return errors


EXPECTED_UTILITY_FEATURES = [
    "score", "distance_to_ego", "branch_idx", "branch_rank", "cls_weight",
    "center_weight", "height_weight", "non_fused_prob", "fused_keep",
    "damage_gate", "bev_damage", "img_damage", "dual_damage",
    "branch_is_fused", "branch_is_bev", "branch_is_img",
]
EXPECTED_SCENE_POLICY = {
    "beams_s1": {"target": 0.0, "weight": 1.0},
    "beams_s2": {"target": 0.35, "weight": 0.45},
    "beams_s3": {"target": 1.0, "weight": 1.0},
    "clean": {"target": 0.0, "weight": 1.5},
    "fog_motionblur": {"target": 0.0, "weight": 2.0},
    "other_corruptions": "negative",
}

EXPECTED_FIELDS = {
    "ensemble_mode": "damage_gated_soft_router",
    "supp_enable": True,
    "supp_mode": "img_highscore",
    "supp_max_add_queries": 90,
    "supp_img_score_threshold": 0.02,
    "supp_bev_score_threshold": 0.02,
    "supp_fused_replace_score_ceiling": 0.35,
    "supp_nms_center_dist": 0.005,
    "supp_per_class_cap": 0,
    "supp_utility_threshold": 0.45,
    "supp_img_utility_mapper_path": "",
    "supp_bev_utility_mapper_path": "",
    "supp_img_utility_threshold": -1.0,
    "supp_bev_utility_threshold": -1.0,
    "supp_fused_slot_utility_mapper_path": "",
    "supp_marginal_utility_enable": False,
    "supp_marginal_utility_margin": 0.0,
    "supp_dual_damage_marginal_margin": -1.0,
    "supp_scene_mapper_gate_mode": "override",
    "supp_scene_budget_min_ratio": 0.10,
    "supp_scene_budget_max_ratio": 0.30,
    "supp_scene_budget_full_threshold": 0.990,
    "supp_rank_mode": "img_score",
    "supp_utility_rank_weight": 1.0,
    "supp_img_damage_guard_enable": False,
    "supp_img_damage_guard_mode": "off",
    "supp_img_damage_signal": "",
    "supp_img_damage_metric_name": "",
    "supp_img_damage_threshold_key": "p99",
    "supp_img_damage_threshold_scale": 1.0,
    "supp_fused_damage_signal": "fused_center_unc",
    "supp_fused_damage_metric_name": "fused_top10_unc",
    "supp_fused_damage_threshold_key": "p99",
    "supp_fused_damage_threshold_scale": 2.0,
    "supp_relative_damage_reference_key": "median",
    "supp_bev_fused_dominance_ratio": 1.0,
    "supp_bev_damage_upper_threshold_scale": 0.0,
    "supp_dual_abstain_signal": "",
    "supp_dual_abstain_threshold": -1.0,
    "supp_img_abstain_signal": "",
    "supp_img_abstain_threshold": -1.0,
    "supp_qsem_gate_enable": False,
    "supp_qsem_disagree_signal": "",
    "supp_qsem_disagree_threshold": -1.0,
    "supp_qsem_margin_signal": "",
    "supp_qsem_margin_threshold": -1.0,
    "supp_contextheight_gate_enable": True,
    "supp_contextheight_context_signal": "v8_context_img_damage_mean",
    "supp_contextheight_context_threshold": -0.64882296,
    "supp_contextheight_height_signal": (
        "distance_bin_abnormality_stats.height.bev.mid"),
    "supp_contextheight_height_threshold": 0.55,
    "supp_contextheight_v9_bev_signal": "v9_a_bev_above_tau_mean",
    "supp_contextheight_v9_bev_threshold": 0.60,
    "supp_dual_damage_budget_ratio": 0.0,
    "supp_dual_damage_utility_threshold": 0.0,
    "supp_dual_damage_gate_mode": "strict_img",
    "supp_dual_scene_mapper_threshold": 0.90,
    "supp_dual_bev_threshold_scale": 1.0,
    "supp_dual_bev_rank_boost": 1.0,
    "scrg_enable": False,
    "scrg_mode": "flag",
    "scrg_switch_heads": ("center",),
    "scrg_scene_metric": "top10",
    "scrg_top_fraction": 0.10,
    "scrg_bev_threshold_key": "p99",
    "scrg_img_ceiling_key": "p99",
    "scrg_margin_threshold_key": "p99",
    "scrg_bev_threshold_scale": 1.0,
    "scrg_img_ceiling_scale": 1.0,
    "scrg_margin_threshold_scale": 1.0,
    "scrg_bev_metric_name": "bev_top10_unc",
    "scrg_img_metric_name": "img_top10_unc",
    "scrg_aux_metric_name": "none",
    "scrg_bev_signal": "bev_center_unc",
    "scrg_img_signal": "img_center_unc",
    "scrg_aux_signal": "none",
    "scrg_max_switch_ratio": 0.10,
    "v7_allow_missing_stats_fallback": False,
    "supp_require_reference_stats": True,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def equal(left, right):
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return left == right


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            result.update(flatten(item, child))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: list(value)}
    return {prefix: value}


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def resolved_config_sha256(config_path):
    cfg = Config.fromfile(config_path)
    payload = json.dumps(
        plain(dict(cfg)), sort_keys=True, separators=(",", ":"),
        default=repr).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def direct_base_path(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=config_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_base_"
                   for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return ""
            value = value[0]
        if not isinstance(value, str):
            return ""
        return os.path.realpath(
            os.path.join(os.path.dirname(os.path.abspath(config_path)), value))
    return ""


def resolve_repo_path(repo_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


def check_artifact(repo_root, path, expected_fingerprint, expected_sha, errors):
    full_path = resolve_repo_path(repo_root, path)
    if not os.path.isfile(full_path):
        errors.append("missing artifact: {}".format(path))
        return None
    actual_sha = sha256_file(full_path)
    if actual_sha != expected_sha:
        errors.append("artifact SHA256 mismatch: {}".format(path))
    with open(full_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fingerprint = str(
        (payload.get("model_provenance") or {}).get("fingerprint") or "")
    if fingerprint != expected_fingerprint:
        errors.append("artifact fingerprint mismatch: {}".format(path))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--profile", choices=("vov_umr", "vov_v10", "r50_v10"),
        default="vov_umr")
    parser.add_argument(
        "--evaluation-config", default="",
        help="Actual pack config about to run; defaults to --config.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument(
        "--require-source-files", action="store_true",
        help="Build-time audit: re-hash large train-only sidecars/intermediates.")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    expected_collector = current_collector_contract(repo_root)
    cfg = Config.fromfile(args.config)
    base = Config.fromfile(args.base_config)
    evaluation_config = args.evaluation_config or args.config
    evaluation_cfg = Config.fromfile(evaluation_config)
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    errors = []

    provenance = manifest.get("model_provenance") or {}
    if manifest.get("profile") != args.profile:
        errors.append("manifest top-level profile mismatch")
    if provenance.get("profile") != args.profile:
        errors.append("manifest model profile mismatch")
    fingerprint = str(provenance.get("fingerprint") or "")
    if not fingerprint:
        errors.append("manifest model fingerprint is missing")
    fingerprint_payload = {
        "profile": provenance.get("profile"),
        "config_sha256": provenance.get("base_config_sha256"),
        "resolved_config_sha256": provenance.get("resolved_config_sha256"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "checkpoint_epoch": provenance.get("checkpoint_epoch"),
        "identity": provenance.get("identity"),
        "critical_code": provenance.get("critical_code"),
    }
    computed_fingerprint = hashlib.sha256(json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if fingerprint != computed_fingerprint:
        errors.append("manifest model fingerprint is internally inconsistent")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != provenance.get("checkpoint_sha256"):
        errors.append("checkpoint does not match manifest")
    if (args.expected_checkpoint_sha256 and
            checkpoint_sha != args.expected_checkpoint_sha256):
        errors.append("checkpoint does not match pre-registered SHA256")
    expected_checkpoint_epoch = 4 if args.profile == "vov_umr" else 6
    if int(provenance.get("checkpoint_epoch") or -1) != expected_checkpoint_epoch:
        errors.append(
            "manifest checkpoint epoch is not {}".format(expected_checkpoint_epoch))
    critical_code = provenance.get("critical_code") or []
    if not critical_code:
        errors.append("manifest critical code provenance is missing")
    for record in critical_code:
        source_path = resolve_repo_path(repo_root, str(record.get("path") or ""))
        if not os.path.isfile(source_path):
            errors.append("critical source is missing: {}".format(record.get("path")))
        elif sha256_file(source_path) != record.get("sha256"):
            errors.append("critical source hash mismatch: {}".format(record.get("path")))
    if sha256_file(args.base_config) != provenance.get("base_config_sha256"):
        errors.append("base config does not match manifest")
    if (resolved_config_sha256(args.base_config) !=
            provenance.get("resolved_config_sha256")):
        errors.append("fully resolved base config does not match manifest")
    train_infos_path = resolve_repo_path(
        repo_root, str(provenance.get("train_infos") or ""))
    if not os.path.isfile(train_infos_path):
        errors.append("manifest train infos is missing")
    elif sha256_file(train_infos_path) != provenance.get("train_infos_sha256"):
        errors.append("train infos does not match manifest")
    if provenance.get("train_infos_sha256") != (
            "6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854"):
        errors.append("train infos is not the registered nuScenes train split")
    if direct_base_path(args.config) != os.path.realpath(args.base_config):
        errors.append("runtime config does not directly inherit the formal UMR base")
    if manifest.get("parity_source") != (
            "RULE_SGSUP_B_contextheight055_v9bev_dyn10_30"):
        errors.append("unexpected R50 parity source")
    if manifest.get("parity_source_record") != {
            "path": (
                "work_dirs/umr_rule_runtime_dup_sweep_large_20260703_192741/"
                "configs/RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py"),
            "sha256": (
                "c250ad2d40284a5d0594a93637c7b426e0b751303755ca6c0955bb0266b6fc52"),
            "generator_commit": "40b0efa929089b8ba92a8a9720564d3b7fbb4d8a",
            "tracked_contract": (
                "projects/configs/umr_sgsup_b_r50_parity_contract.json"),
            "tracked_contract_sha256": (
                "3ab13f9097696f78af0457306562e3fbad3e763d9f9c9cc90388dd14481fde31"),
    }:
        errors.append("R50 parity source path/hash/commit is not canonical")
    if manifest.get("scene_threshold_key") != "clean_fpr_005":
        errors.append("scene threshold policy is not clean_fpr_005")
    if plain(manifest.get("scene_label_policy") or {}) != EXPECTED_SCENE_POLICY:
        errors.append("manifest scene label policy differs from R50 formal policy")
    runtime_configs = manifest.get("runtime_configs") or {}
    expected_runtime_names = {
        "UMR_MATCHED_BASE", "RULE_SGSUP_B_runtime",
        "RULE_SGSUP_B_contextheight055_v9bev_dyn10_30",
    }
    if set(runtime_configs) != expected_runtime_names:
        errors.append("manifest runtime-config set is incomplete or unexpected")
    evaluation_real = os.path.realpath(evaluation_config)
    evaluation_matches = [
        (name, record) for name, record in runtime_configs.items()
        if os.path.realpath(resolve_repo_path(
            repo_root, str(record.get("path") or ""))) == evaluation_real
    ]
    if len(evaluation_matches) != 1:
        errors.append("actual evaluation config is not uniquely registered")
    else:
        evaluation_name, evaluation_record = evaluation_matches[0]
        if (sha256_file(evaluation_config) != evaluation_record.get("sha256") or
                resolved_config_sha256(evaluation_config) !=
                evaluation_record.get("resolved_sha256")):
            errors.append("actual evaluation config hash mismatch")
        expected_roles = {
            "UMR_MATCHED_BASE": {
                "vov_umr": "matched_vov_umr_action_off",
                "vov_v10": "matched_vov_v10_action_off",
                "r50_v10": "matched_r50_v10_action_off",
            }[args.profile],
            "RULE_SGSUP_B_runtime": "unguarded_action_parity_control",
            "RULE_SGSUP_B_contextheight055_v9bev_dyn10_30":
                "formal_guarded_final",
        }
        if evaluation_record.get("role") != expected_roles[evaluation_name]:
            errors.append("actual evaluation config role mismatch")
        evaluation_ensemble = evaluation_cfg.model.pts_bbox_head.ensemble
        if direct_base_path(evaluation_config) != os.path.realpath(args.base_config):
            errors.append("actual evaluation config does not directly inherit UMR base")
        if evaluation_name == "UMR_MATCHED_BASE":
            if (plain(dict(evaluation_cfg)) != plain(dict(base)) or
                    bool(evaluation_ensemble.get("supp_enable", False))):
                errors.append("matched UMR baseline is not action-off base parity")
        else:
            expected_guard = evaluation_name.endswith(
                "contextheight055_v9bev_dyn10_30")
            if (evaluation_ensemble.get("supp_enable") is not True or
                    evaluation_ensemble.get("supp_mode") != "img_highscore" or
                    int(evaluation_ensemble.get("supp_max_add_queries") or 0) != 90 or
                    bool(evaluation_ensemble.get(
                        "supp_contextheight_gate_enable")) != expected_guard or
                    bool(evaluation_ensemble.get("v10_enable", False)) !=
                    (args.profile in ("vov_v10", "r50_v10"))):
                errors.append("actual evaluation config action/guard contract mismatch")

    model = cfg.model
    backbone = model.img_backbone
    ensemble = model.pts_bbox_head.ensemble
    if args.profile in ("vov_umr", "vov_v10"):
        identity_checks = {
            "backbone.type": (backbone.type, "VoVNet"),
            "backbone.spec_name": (backbone.get("spec_name", ""), "V-99-eSE"),
            "voxel_size": (list(model.pts_voxel_layer.voxel_size),
                           [0.075, 0.075, 0.2]),
            "bbox_voxel_size": (list(model.pts_bbox_head.bbox_coder.voxel_size),
                                [0.075, 0.075, 0.2]),
            "image_final_dim": (list(cfg.ida_aug_conf.final_dim), [640, 1600]),
            "num_query": (int(model.pts_bbox_head.get("num_query", 900)), 900),
            "v10_enable": (
                bool(ensemble.get("v10_enable", False)),
                args.profile == "vov_v10"),
            "v10_matching_mode": (
                str(model.pts_bbox_head.get("v10_matching_mode", "union")),
                "single_best" if args.profile == "vov_v10" else "union"),
        }
    else:
        identity_checks = {
            "backbone.type": (backbone.type, "ResNet"),
            "voxel_size": (list(model.pts_voxel_layer.voxel_size),
                           [0.1, 0.1, 0.2]),
            "bbox_voxel_size": (list(model.pts_bbox_head.bbox_coder.voxel_size),
                                [0.1, 0.1, 0.2]),
            "image_final_dim": (list(cfg.ida_aug_conf.final_dim), [320, 800]),
            "num_query": (int(model.pts_bbox_head.get("num_query", 900)), 900),
            "v10_enable": (bool(ensemble.get("v10_enable", False)), True),
            "v10_matching_mode": (
                str(model.pts_bbox_head.get("v10_matching_mode", "union")),
                "single_best"),
        }
    for name, (actual, expected) in identity_checks.items():
        if actual != expected:
            errors.append(
                "{}: expected={!r}, got={!r}".format(name, expected, actual))

    for key, expected in EXPECTED_FIELDS.items():
        actual = ensemble.get(key)
        if not equal(actual, expected):
            errors.append(
                "{}: expected={!r}, got={!r}".format(key, expected, actual))
    if str(ensemble.get("supp_expected_model_fingerprint") or "") != fingerprint:
        errors.append("config model fingerprint does not match manifest")

    base_flat = flatten(dict(base))
    cfg_flat = flatten(dict(cfg))
    all_keys = sorted(set(base_flat) | set(cfg_flat))
    changed = [
        key for key in all_keys
        if not equal(base_flat.get(key), cfg_flat.get(key))
    ]
    dynamic_fields = {
        "supp_expected_model_fingerprint", "supp_utility_mapper_path",
        "supp_scene_mapper_path", "supp_scene_mapper_threshold",
        "v7_reference_stats_path",
    }
    allowed_fields = set(EXPECTED_FIELDS) | dynamic_fields
    allowed_keys = {
        "model.pts_bbox_head.ensemble.{}".format(key)
        for key in allowed_fields
    }
    unexpected = [
        key for key in changed
        if key not in allowed_keys
    ]
    if unexpected:
        errors.append(
            "non-runtime model fields changed: {}".format(", ".join(unexpected)))

    artifacts = manifest.get("artifacts") or []
    artifact_by_path = {item.get("path"): item for item in artifacts}
    configured_paths = (
        str(ensemble.supp_scene_mapper_path),
        str(ensemble.supp_utility_mapper_path),
        str(ensemble.v7_reference_stats_path),
    )
    payloads = {}
    for path in configured_paths:
        record = artifact_by_path.get(path)
        if not record:
            errors.append("configured artifact is absent from manifest: {}".format(path))
            continue
        payloads[path] = check_artifact(
            repo_root, path, fingerprint, record.get("sha256"), errors)
    scene_payload = payloads.get(str(ensemble.supp_scene_mapper_path)) or {}
    if list(scene_payload.get("features") or ()) != EXPECTED_SCENE_FEATURES:
        errors.append("scene mapper feature schema differs from R50")
    scene_threshold = (scene_payload.get("thresholds") or {}).get("clean_fpr_005")
    if scene_threshold is None or not equal(
            ensemble.supp_scene_mapper_threshold, scene_threshold):
        errors.append("scene threshold is not bound to model-specific clean_fpr_005")
    utility_payload = payloads.get(str(ensemble.supp_utility_mapper_path)) or {}
    if utility_payload.get("feature_preset") != "action_branch":
        errors.append("utility mapper is not bounded branch-candidate action_branch")
    if list(utility_payload.get("features") or ()) != EXPECTED_UTILITY_FEATURES:
        errors.append("utility mapper feature schema differs from R50")

    calibration_sources = manifest.get("calibration_sources") or {}
    if set(calibration_sources) != {
            "branch_labels", "branch_label_summary",
            "scene_labels", "scene_label_summary"}:
        errors.append("intermediate calibration-source set is incomplete")
    for payload in payloads.values():
        if payload and payload.get("calibration_sources") != calibration_sources:
            errors.append("artifact calibration-source binding differs from manifest")

    source_log_meta = manifest.get("source_log_meta") or []
    split_records = {"train_clean": [], "train_oc": []}
    for record in source_log_meta:
        split = str(record.get("formal_split") or "")
        if split not in split_records:
            errors.append("unexpected formal source split: {}".format(split))
        else:
            split_records[split].append(record)
        errors.extend(source_log_collector_errors(
            record, expected_collector))
        source_files = record.get("source_files") or {}
        if set(source_files) != {
                "scene_summary_path", "prediction_light_path",
                "branch_candidates_path"}:
            errors.append("source-log shard file set is incomplete")
        expected_source_records = {
            "scene_summary_path": int(record.get("num_samples") or 0),
            "prediction_light_path": int(record.get("num_predictions") or 0),
            "branch_candidates_path": int(
                record.get("num_branch_candidates") or 0),
        }
        for source_name, source in source_files.items():
            source_path = resolve_repo_path(repo_root, str(source.get("path") or ""))
            if (not str(source.get("sha256") or "") or
                    int(source.get("records") or 0) !=
                    expected_source_records.get(source_name, -1)):
                errors.append("logged calibration source record is incomplete: {}".format(
                    source.get("path")))
            if args.require_source_files:
                if not os.path.isfile(source_path):
                    errors.append("logged calibration source is missing: {}".format(
                        source.get("path")))
                elif sha256_file(source_path) != source.get("sha256"):
                    errors.append("logged calibration source hash mismatch: {}".format(
                        source.get("path")))
        generated = record.get("generated_config") or {}
        generated_path = resolve_repo_path(
            repo_root, str(generated.get("path") or ""))
        if not generated.get("sha256") or not generated.get("resolved_sha256"):
            errors.append("generated calibration config record is incomplete: {}".format(
                generated.get("path")))
        if args.require_source_files:
            if not os.path.isfile(generated_path):
                errors.append("generated calibration config is missing: {}".format(
                    generated.get("path")))
            elif (sha256_file(generated_path) != generated.get("sha256") or
                  resolved_config_sha256(generated_path) != generated.get("resolved_sha256")):
                errors.append("generated calibration config hash mismatch: {}".format(
                    generated.get("path")))
    for split, records in split_records.items():
        if not records:
            errors.append("missing source-log metadata for {}".format(split))
            continue
        world_sizes = {int(record.get("world_size") or 0) for record in records}
        ranks = {int(record.get("rank", -1)) for record in records}
        samples = sum(int(record.get("num_samples") or 0) for record in records)
        if (len(world_sizes) != 1 or next(iter(world_sizes)) <= 0 or
                ranks != set(range(next(iter(world_sizes)))) or samples != 28130):
            errors.append("incomplete/inconsistent source-log shards for {}".format(split))
    for name, source in calibration_sources.items():
        source_path = resolve_repo_path(repo_root, str(source.get("path") or ""))
        if not source.get("sha256"):
            errors.append("intermediate calibration source record is incomplete: {}".format(
                source.get("path")))
        if args.require_source_files and not os.path.isfile(source_path):
            errors.append("intermediate calibration source is missing: {}".format(
                source.get("path")))
        elif args.require_source_files and sha256_file(source_path) != source.get("sha256"):
            errors.append("intermediate calibration source hash mismatch: {}".format(
                source.get("path")))
        elif args.require_source_files and name == "scene_label_summary":
            with open(source_path, "r", encoding="utf-8") as handle:
                scene_summary = json.load(handle)
            if plain(scene_summary.get("label_policy") or {}) != EXPECTED_SCENE_POLICY:
                errors.append("scene label policy differs from R50 formal policy")

    if errors:
        print("[{} SGSUP-B preflight] FAIL".format(args.profile))
        for error in errors:
            print("  - {}".format(error))
        raise SystemExit(1)
    print("[{} SGSUP-B preflight] PASS".format(args.profile))
    print("  fingerprint={}".format(fingerprint))
    print("  changed_model_fields={}".format(len(changed)))
    print("  checkpoint_sha256={}".format(checkpoint_sha))


if __name__ == "__main__":
    main()
