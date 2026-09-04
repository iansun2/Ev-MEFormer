#!/usr/bin/env python3
"""Bind train-only UMR runtime artifacts to one model/config checkpoint.

The bounded supplement mappers are distribution dependent.  This tool stamps
their JSON payloads with a portable fingerprint derived from the fully
specified model configuration and checkpoint content, then writes an auditable
manifest.  It intentionally refuses validation/MultiCorrupt-like inputs.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile

from mmcv import Config

try:
    from umr_collector_contract import current_collector_contract
except ImportError:  # Support importing this file as a tools.* module.
    from tools.umr_collector_contract import current_collector_contract


FORBIDDEN_MARKERS = (
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
FORMAL_SPLITS = ("train_clean", "train_oc")
EXPECTED_TRAIN_INFOS_SHA256 = (
    "6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854")
EXPECTED_DATASET_SIZE = 28130
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
EXPECTED_UTILITY_FEATURES = [
    "score", "distance_to_ego", "branch_idx", "branch_rank", "cls_weight",
    "center_weight", "height_weight", "non_fused_prob", "fused_keep",
    "damage_gate", "bev_damage", "img_damage", "dual_damage",
    "branch_is_fused", "branch_is_bev", "branch_is_img",
]
EXPECTED_OC_WEIGHTS = {
    "beamsreducing": 0.24, "pointsreducing": 0.16,
    "spatialmisalignment": 0.13, "temporalmisalignment": 0.09,
    "fog": 0.14, "snow": 0.08, "brightness": 0.04, "dark": 0.05,
    "motionblur": 0.05, "missingcamera": 0.02,
}
CRITICAL_CODE_PATHS = (
    "projects/mmdet3d_plugin/models/utils/edl_pme_transformer.py",
    "projects/mmdet3d_plugin/models/utils/v10_quality.py",
    "projects/mmdet3d_plugin/models/utils/edl_uncertainty_logger.py",
    "projects/mmdet3d_plugin/models/dense_heads/edl_head.py",
    "projects/mmdet3d_plugin/datasets/pipelines/synthetic_corruptions.py",
    "tools/dist_test.sh",
    "tools/test.py",
    "tools/make_umr_phase1_log_configs.py",
    "tools/umr_utility_collect_phase1_full.sh",
    "tools/umr_collector_contract.py",
    "tools/build_umr_branch_candidate_labels.py",
    "tools/train_umr_utility_mapper.py",
    "tools/build_umr_scene_gate_labels.py",
    "tools/train_umr_scene_damage_mapper.py",
    "tools/export_umr_clean_reference_stats.py",
    "tools/register_vov_umr_sgsup_runtime_configs.py",
    "tools/stamp_umr_model_provenance.py",
    "tools/preflight_vov_umr_sgsup_b.py",
    "tools/umr_rule_runtime_pack.sh",
    "projects/configs/umr_sgsup_b_r50_parity_contract.json",
)
PROFILE_CRITICAL_CODE_PATHS = {
    "vov_umr": ("tools/prepare_vov_umr_sgsup_b.sh",),
    "vov_v10": (
        "tools/prepare_vov_umr_sgsup_b.sh",
        "tools/v10_preflight.py",
        "tools/v10_quality_calibration.py",
        "tools/build_vov_v10_selector_config.py",
    ),
    "r50_v10": (
        "tools/prepare_r50_umr_sgsup_b.sh",
        "tools/v10_preflight.py",
    ),
}
R50_PARITY_CONTRACT = "projects/configs/umr_sgsup_b_r50_parity_contract.json"
R50_PARITY_SOURCE = (
    "work_dirs/umr_rule_runtime_dup_sweep_large_20260703_192741/configs/"
    "RULE_SGSUP_B_contextheight055_v9bev_dyn10_30.py")
R50_PARITY_SOURCE_SHA256 = (
    "c250ad2d40284a5d0594a93637c7b426e0b751303755ca6c0955bb0266b6fc52")
R50_PARITY_SOURCE_COMMIT = "40b0efa929089b8ba92a8a9720564d3b7fbb4d8a"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_records(path):
    if path.endswith((".gz", ".gzip")):
        import gzip
        handle = gzip.open(path, "rb")
    elif path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise SystemExit(
                "[UMR provenance][ERROR] zstandard is required to audit logs") from exc
        raw = open(path, "rb")
        handle = zstd.ZstdDecompressor().stream_reader(raw, closefd=True)
    else:
        handle = open(path, "rb")
    count = 0
    tail = b""
    try:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
            tail = chunk[-1:]
    finally:
        handle.close()
    return count + (1 if tail and tail != b"\n" else 0)


def jsonl_content_sha256(path):
    raw = None
    reader = None
    if path.endswith((".gz", ".gzip")):
        handle = gzip.open(path, "rt", encoding="utf-8")
    elif path.endswith((".zst", ".zstd")):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise SystemExit(
                "[UMR provenance][ERROR] zstandard is required to audit labels") from exc
        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        handle = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        handle = open(path, "rt", encoding="utf-8")
    digest = hashlib.sha256()
    try:
        for line in handle:
            if line.strip():
                digest.update(line.encode("utf-8"))
    finally:
        handle.close()
        if reader is not None:
            reader.close()
        if raw is not None:
            raw.close()
    return digest.hexdigest()


def path_tokens(path):
    normalized = str(path or "").lower().replace("\\", "/")
    tokens = []
    for part in normalized.split("/"):
        tokens.extend(part.replace("-", "_").replace(".", "_").split("_"))
    return normalized, {token for token in tokens if token}


def assert_train_only_path(path, label):
    normalized, tokens = path_tokens(path)
    hits = [marker for marker in FORBIDDEN_MARKERS if marker in normalized]
    if "val" in tokens:
        hits.append("path_token=val")
    if hits:
        raise SystemExit(
            "[UMR provenance][ERROR] refusing evaluation-derived {}: {}; "
            "markers={}".format(label, path, sorted(set(hits))))


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


def real(path):
    return os.path.realpath(os.path.abspath(path))


def assert_within(path, root, label):
    path_real = real(path)
    root_real = real(root)
    try:
        inside = os.path.commonpath((path_real, root_real)) == root_real
    except ValueError:
        inside = False
    if not inside:
        raise SystemExit(
            "[UMR provenance][ERROR] {} is outside current calibration root: "
            "{} not under {}".format(label, path, root))
    return path_real


def load_json(path, label):
    if not os.path.isfile(path):
        raise SystemExit(
            "[UMR provenance][ERROR] missing {}: {}".format(label, path))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def declared_path(path):
    if not path:
        return ""
    return real(path)


def checkpoint_epoch(path):
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SystemExit(
            "[UMR provenance][ERROR] cannot read checkpoint metadata: {}".format(
                exc))
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
    epoch = (meta or {}).get("epoch") if isinstance(meta, dict) else None
    del checkpoint
    try:
        return int(epoch)
    except (TypeError, ValueError):
        raise SystemExit(
            "[UMR provenance][ERROR] checkpoint meta.epoch is missing or invalid")


def walk_pipeline(pipeline):
    for step in pipeline or ():
        if not isinstance(step, dict):
            continue
        yield step
        for child in walk_pipeline(step.get("transforms") or ()):
            yield child


def validate_log_config(path, split, base_config, train_infos, profile):
    cfg = Config.fromfile(path)
    base = Config.fromfile(base_config)
    if plain(cfg.model) != plain(base.model):
        raise SystemExit(
            "[UMR provenance][ERROR] generated log config changes the base model")
    if real(cfg.data.test.ann_file) != real(train_infos):
        raise SystemExit(
            "[UMR provenance][ERROR] log config train infos mismatch: {}".format(
                path))
    if (int(cfg.data.get("samples_per_gpu", 0)) != 1 or
            int(cfg.data.get("test_dataloader", {}).get(
                "samples_per_gpu", 0)) != 1):
        raise SystemExit(
            "[UMR provenance][ERROR] formal log config must use "
            "samples_per_gpu=1: {}".format(path))
    steps = list(walk_pipeline(cfg.data.test.pipeline))
    corrupt = [step for step in steps
               if step.get("type") == "OnlineSyntheticMultiCorrupt"]
    if split == "train_clean" and corrupt:
        raise SystemExit(
            "[UMR provenance][ERROR] clean log config contains online corruption")
    if split == "train_oc":
        if len(corrupt) != 1:
            raise SystemExit(
                "[UMR provenance][ERROR] train-O.C config must contain exactly one "
                "OnlineSyntheticMultiCorrupt")
        policy = corrupt[0]
        if (float(policy.get("corrupt_prob", -1.0)) != 0.5 or
                list(policy.get("severity_choices") or ()) != [1, 2, 3] or
                plain(policy.get("corruption_weights") or {}) != EXPECTED_OC_WEIGHTS or
                policy.get("record_meta") is not True):
            raise SystemExit(
                "[UMR provenance][ERROR] train-O.C policy differs from R50 formal "
                "calibration policy")
    if config_identity(path, profile) != config_identity(base_config, profile):
        raise SystemExit(
            "[UMR provenance][ERROR] generated log config model identity mismatch")


def validate_collector_meta(extra, expected_collector, path):
    """Fail closed when formal collection metadata is stale or incomplete."""
    if int(extra.get("samples_per_gpu") or 0) != 1:
        raise SystemExit(
            "[UMR provenance][ERROR] formal EV log was not collected "
            "with samples_per_gpu=1: {}".format(path))
    if (extra.get("collector_code_format") !=
            expected_collector["format"] or
            extra.get("collector_code_fingerprint") !=
            expected_collector["fingerprint"] or
            list(extra.get("collector_code_records") or ()) !=
            expected_collector["records"]):
        raise SystemExit(
            "[UMR provenance][ERROR] collection-time code fingerprint "
            "differs from the current formal collector: {}".format(path))


def config_identity(config_path, profile="vov_umr"):
    cfg = Config.fromfile(config_path)
    model = cfg.model
    backbone = model.img_backbone
    head = model.pts_bbox_head
    voxel_size = plain(model.pts_voxel_layer.voxel_size)
    bbox_voxel_size = plain(head.bbox_coder.voxel_size)
    ensemble = head.ensemble
    identity = {
        "backbone_type": str(backbone.type),
        "backbone_spec": str(backbone.get("spec_name", "")),
        "backbone_out_features": plain(backbone.get("out_features", ())),
        "image_final_dim": plain(cfg.ida_aug_conf.final_dim),
        "voxel_size": voxel_size,
        "bbox_voxel_size": bbox_voxel_size,
        # EDLHead's constructor default is 900 and the formal configs leave it
        # implicit.  An explicit override, when present, must still be 900.
        "num_query": int(head.get("num_query", 900)),
        "ensemble_mode": str(ensemble.ensemble_mode),
        "v8_router_variant": str(ensemble.v8_router_variant),
        "v10_enable": bool(ensemble.get("v10_enable", False)),
        "v10_matching_mode": str(head.get("v10_matching_mode", "union")),
        "total_epochs": int(cfg.total_epochs),
    }
    expected_profiles = {
        "vov_umr": {
            "backbone_type": "VoVNet",
            "backbone_spec": "V-99-eSE",
            "image_final_dim": [640, 1600],
            "voxel_size": [0.075, 0.075, 0.2],
            "bbox_voxel_size": [0.075, 0.075, 0.2],
            "num_query": 900,
            "ensemble_mode": "damage_gated_soft_router",
            "v8_router_variant": "v9_lite_domain_anchored",
            "v10_enable": False,
            "v10_matching_mode": "union",
            "total_epochs": 4,
        },
        "vov_v10": {
            "backbone_type": "VoVNet",
            "backbone_spec": "V-99-eSE",
            "image_final_dim": [640, 1600],
            "voxel_size": [0.075, 0.075, 0.2],
            "bbox_voxel_size": [0.075, 0.075, 0.2],
            "num_query": 900,
            "ensemble_mode": "damage_gated_soft_router",
            "v8_router_variant": "v9_lite_domain_anchored",
            "v10_enable": True,
            "v10_matching_mode": "single_best",
            "total_epochs": 6,
        },
        "r50_v10": {
            "backbone_type": "ResNet",
            "backbone_spec": "",
            "image_final_dim": [320, 800],
            "voxel_size": [0.1, 0.1, 0.2],
            "bbox_voxel_size": [0.1, 0.1, 0.2],
            "num_query": 900,
            "ensemble_mode": "damage_gated_soft_router",
            "v8_router_variant": "v9_lite_domain_anchored",
            "v10_enable": True,
            "v10_matching_mode": "single_best",
            "total_epochs": 6,
        },
    }
    if profile not in expected_profiles:
        raise SystemExit(
            "[UMR provenance][ERROR] unsupported model profile: {}".format(profile))
    expected = expected_profiles[profile]
    errors = []
    for key, expected_value in expected.items():
        if identity.get(key) != expected_value:
            errors.append(
                "{}: expected={!r}, got={!r}".format(
                    key, expected_value, identity.get(key)))
    if errors:
        raise SystemExit(
            "[UMR provenance][ERROR] base config is not the formal {} "
            "contract:\n  - {}".format(profile, "\n  - ".join(errors)))
    return identity


def atomic_write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".umr-provenance-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def validate_artifact(path, payload):
    basename = os.path.basename(path)
    if basename == "mapper_scene_bev_damage_applicable.json":
        if payload.get("target") != "scene_bev_damage_applicable":
            raise SystemExit("[UMR provenance][ERROR] invalid scene mapper target")
        if "clean_fpr_005" not in (payload.get("thresholds") or {}):
            raise SystemExit("[UMR provenance][ERROR] scene mapper lacks clean_fpr_005")
        if list(payload.get("features") or ()) != EXPECTED_SCENE_FEATURES:
            raise SystemExit(
                "[UMR provenance][ERROR] scene mapper feature schema differs from R50")
        threshold = (payload.get("thresholds") or {}).get("clean_fpr_005")
        if (not isinstance(threshold, (int, float)) or
                not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0):
            raise SystemExit(
                "[UMR provenance][ERROR] invalid scene clean_fpr_005 threshold")
    elif basename == "mapper_candidate_useful.json":
        if payload.get("target") != "candidate_useful":
            raise SystemExit("[UMR provenance][ERROR] invalid utility mapper target")
        if payload.get("feature_preset") != "action_branch":
            raise SystemExit(
                "[UMR provenance][ERROR] utility mapper is not action_branch")
        if list(payload.get("features") or ()) != EXPECTED_UTILITY_FEATURES:
            raise SystemExit(
                "[UMR provenance][ERROR] utility feature schema differs from R50")
        label_provenance = payload.get("label_provenance") or {}
        if label_provenance.get("formal_source_kind") != "train_or_train_oc":
            raise SystemExit(
                "[UMR provenance][ERROR] utility labels are not train/train-O.C")
        if not label_provenance.get("labels_content_sha256"):
            raise SystemExit(
                "[UMR provenance][ERROR] utility labels lack content SHA256")
    elif basename == "clean_reference_stats.json":
        if payload.get("format") != "umr_clean_reference_stats_v1":
            raise SystemExit("[UMR provenance][ERROR] invalid clean reference format")
        if int(payload.get("records") or 0) <= 0:
            raise SystemExit("[UMR provenance][ERROR] empty clean reference")
        scene_global = payload.get("scene_global") or {}
        for key in ("bev_top10_unc", "img_top10_unc"):
            stat = scene_global.get(key) or {}
            p99 = stat.get("p99")
            if (not isinstance(p99, (int, float)) or
                    not math.isfinite(float(p99)) or int(stat.get("count") or 0) <= 0):
                raise SystemExit(
                    "[UMR provenance][ERROR] clean reference lacks finite {} p99".format(
                        key))
    else:
        raise SystemExit(
            "[UMR provenance][ERROR] unsupported formal artifact: {}".format(path))
    if basename.startswith("mapper_"):
        if payload.get("formal_calibration_safe") is not True:
            raise SystemExit(
                "[UMR provenance][ERROR] mapper is not formal_calibration_safe")
        if payload.get("debug_only") is True:
            raise SystemExit("[UMR provenance][ERROR] debug mapper is forbidden")
        if payload.get("unsafe_features"):
            raise SystemExit("[UMR provenance][ERROR] unsafe mapper features")
        features = list(payload.get("features") or ())
        weights = list((payload.get("model") or {}).get("weights") or ())
        means = list((payload.get("normalization") or {}).get("mean") or ())
        stds = list((payload.get("normalization") or {}).get("std") or ())
        bias = (payload.get("model") or {}).get("bias")
        if not (len(features) == len(weights) == len(means) == len(stds)):
            raise SystemExit(
                "[UMR provenance][ERROR] mapper numeric vector length mismatch")
        numeric = weights + means + stds + [bias]
        if any(not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) for value in numeric):
            raise SystemExit(
                "[UMR provenance][ERROR] mapper contains non-finite parameters")
        if any(float(value) <= 0.0 for value in stds):
            raise SystemExit(
                "[UMR provenance][ERROR] mapper normalization std must be positive")


def collect_log_meta(log_root, checkpoint, base_config, train_infos, profile):
    records = []
    split_samples = {split: 0 for split in FORMAL_SPLITS}
    split_ranks = {split: set() for split in FORMAL_SPLITS}
    split_world_sizes = {split: set() for split in FORMAL_SPLITS}
    split_config_hashes = {split: set() for split in FORMAL_SPLITS}
    expected_checkpoint = os.path.realpath(checkpoint)
    expected_checkpoint_sha256 = sha256_file(checkpoint)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_collector = current_collector_contract(repo_root)
    for root, _dirs, files in os.walk(log_root):
        for filename in sorted(files):
            if not filename.startswith("ev_meta") or not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            context = payload.get("sample_context") or {}
            extra = payload.get("extra_meta") or {}
            split = str(context.get("ev_split") or "")
            if split not in FORMAL_SPLITS:
                raise SystemExit(
                    "[UMR provenance][ERROR] non-formal EV split in {}: {}".format(
                        path, split or "missing"))
            assert_within(path, os.path.join(log_root, split), "EV metadata")
            if extra.get("completed") is not True:
                raise SystemExit(
                    "[UMR provenance][ERROR] incomplete EV log: {}".format(path))
            logged_checkpoint = str(extra.get("checkpoint") or "")
            if (not logged_checkpoint or
                    os.path.realpath(logged_checkpoint) != expected_checkpoint):
                raise SystemExit(
                    "[UMR provenance][ERROR] EV log checkpoint mismatch: {}".format(
                        path))
            if str(extra.get("checkpoint_sha256") or "") != expected_checkpoint_sha256:
                raise SystemExit(
                    "[UMR provenance][ERROR] collection-time checkpoint SHA256 "
                    "mismatch: {}".format(path))
            rank = int(payload.get("rank", -1))
            world_size = int(payload.get("world_size") or 0)
            if rank < 0 or world_size <= 0 or rank >= world_size:
                raise SystemExit(
                    "[UMR provenance][ERROR] invalid rank/world_size: {}".format(path))
            if rank in split_ranks[split]:
                raise SystemExit(
                    "[UMR provenance][ERROR] duplicate {} rank {}".format(
                        split, rank))
            split_ranks[split].add(rank)
            split_world_sizes[split].add(world_size)
            if int(extra.get("dataset_size") or 0) != EXPECTED_DATASET_SIZE:
                raise SystemExit(
                    "[UMR provenance][ERROR] unexpected dataset_size: {}".format(path))
            validate_collector_meta(extra, expected_collector, path)
            generated_config = str(extra.get("config") or "")
            if not generated_config or not os.path.isfile(generated_config):
                raise SystemExit(
                    "[UMR provenance][ERROR] missing generated log config: {}".format(
                        path))
            if str(extra.get("config_sha256") or "") != sha256_file(generated_config):
                raise SystemExit(
                    "[UMR provenance][ERROR] collection-time log config SHA256 "
                    "mismatch: {}".format(path))
            validate_log_config(
                generated_config, split, base_config=base_config,
                train_infos=train_infos, profile=profile)
            split_config_hashes[split].add(sha256_file(generated_config))
            topk = int(payload.get("branch_candidate_topk_per_branch") or 0)
            score_threshold = float(payload.get("branch_candidate_score_thr") or 0.0)
            branches = list(payload.get("branch_candidate_branches") or ())
            if topk != 300 or abs(score_threshold) > 1e-12:
                raise SystemExit(
                    "[UMR provenance][ERROR] branch-candidate budget mismatch: {}".format(
                        path))
            if branches != ["fused", "bev", "img"]:
                raise SystemExit(
                    "[UMR provenance][ERROR] branch order mismatch: {}: {}".format(
                        path, branches))
            num_samples = int(payload.get("num_samples") or 0)
            num_predictions = int(payload.get("num_predictions") or 0)
            num_branch_candidates = int(payload.get("num_branch_candidates") or 0)
            if (num_samples <= 0 or num_predictions != num_samples * 300 or
                    num_branch_candidates != num_samples * 900):
                raise SystemExit(
                    "[UMR provenance][ERROR] incomplete EV row coverage: {}".format(
                        path))
            source_files = {}
            expected_rows = {
                "scene_summary_path": num_samples,
                "prediction_light_path": num_predictions,
                "branch_candidates_path": num_branch_candidates,
            }
            for source_key in (
                    "scene_summary_path", "prediction_light_path",
                    "branch_candidates_path"):
                declared = payload.get(source_key)
                if not declared:
                    raise SystemExit(
                        "[UMR provenance][ERROR] EV meta lacks {}: {}".format(
                            source_key, path))
                source_path = (declared if os.path.isabs(str(declared)) else
                               os.path.join(root, str(declared)))
                assert_within(source_path, os.path.join(log_root, split), source_key)
                if not os.path.isfile(source_path):
                    raise SystemExit(
                        "[UMR provenance][ERROR] EV source file missing: {}".format(
                            source_path))
                record_count = count_jsonl_records(source_path)
                if record_count != expected_rows[source_key]:
                    raise SystemExit(
                        "[UMR provenance][ERROR] {} row count mismatch: "
                        "expected={}, got={}: {}".format(
                            source_key, expected_rows[source_key], record_count,
                            source_path))
                source_files[source_key] = {
                    "path": os.path.relpath(real(source_path), os.getcwd()),
                    "sha256": sha256_file(source_path),
                    "size_bytes": int(os.path.getsize(source_path)),
                    "records": record_count,
                }
            split_samples[split] += num_samples
            records.append({
                "path": os.path.relpath(path, os.getcwd()),
                "sha256": sha256_file(path),
                "num_samples": num_samples,
                "num_predictions": num_predictions,
                "num_branch_candidates": num_branch_candidates,
                "branch_candidates": bool(payload.get("branch_candidates", False)),
                "world_size": int(payload.get("world_size") or 0),
                "rank": rank,
                "formal_split": split,
                "generated_config": {
                    "path": os.path.relpath(real(generated_config), os.getcwd()),
                    "sha256": sha256_file(generated_config),
                    "resolved_sha256": resolved_config_sha256(generated_config),
                },
                "samples_per_gpu": 1,
                "collector_code": expected_collector,
                "source_files": source_files,
            })
    if not records:
        raise SystemExit("[UMR provenance][ERROR] no EV metadata under log root")
    if not all(item["branch_candidates"] for item in records):
        raise SystemExit(
            "[UMR provenance][ERROR] branch-candidate logging was not enabled")
    for split, samples in split_samples.items():
        if samples != EXPECTED_DATASET_SIZE:
            raise SystemExit(
                "[UMR provenance][ERROR] incomplete/duplicate {} coverage: {} samples".format(
                    split, samples))
        if len(split_world_sizes[split]) != 1:
            raise SystemExit(
                "[UMR provenance][ERROR] inconsistent {} world sizes".format(split))
        if len(split_config_hashes[split]) != 1:
            raise SystemExit(
                "[UMR provenance][ERROR] inconsistent {} log configs".format(split))
        world_size = next(iter(split_world_sizes[split]))
        if split_ranks[split] != set(range(world_size)):
            raise SystemExit(
                "[UMR provenance][ERROR] incomplete {} rank set: {}".format(
                    split, sorted(split_ranks[split])))
    if split_world_sizes["train_clean"] != split_world_sizes["train_oc"]:
        raise SystemExit(
            "[UMR provenance][ERROR] clean and train-O.C world sizes differ")
    return records


def source_paths(log_meta, split, source_key):
    return {
        real(item["source_files"][source_key]["path"])
        for item in log_meta if item["formal_split"] == split
    }


def source_hashes(log_meta, splits, source_key):
    if isinstance(splits, str):
        splits = (splits,)
    return {
        real(item["source_files"][source_key]["path"]):
            item["source_files"][source_key]["sha256"]
        for item in log_meta if item["formal_split"] in splits
    }


def declared_hashes(payload):
    return {declared_path(path): str(value) for path, value in payload.items()}


def validate_calibration_chain(args, payloads, log_meta):
    artifact_by_name = {
        os.path.basename(path): (path, payload)
        for path, payload in payloads
    }
    utility = artifact_by_name["mapper_candidate_useful.json"][1]
    scene = artifact_by_name["mapper_scene_bev_damage_applicable.json"][1]
    reference = artifact_by_name["clean_reference_stats.json"][1]

    branch_labels = assert_within(
        args.branch_labels, os.path.dirname(args.branch_label_summary),
        "branch labels")
    branch_summary = load_json(args.branch_label_summary, "branch-label summary")
    if (branch_summary.get("formal_calibration_safe") is not True or
            branch_summary.get("debug_only") is True or
            branch_summary.get("expected_split") != "train_oc"):
        raise SystemExit(
            "[UMR provenance][ERROR] branch-label summary is not formal train-O.C")
    if declared_path(branch_summary.get("infos")) != real(args.train_infos):
        raise SystemExit(
            "[UMR provenance][ERROR] branch-label train infos mismatch")
    if declared_path(branch_summary.get("branch_log_root")) != real(
            os.path.join(args.source_log_root, "train_oc")):
        raise SystemExit(
            "[UMR provenance][ERROR] branch-label log root mismatch")
    expected_branch_logs = source_paths(
        log_meta, "train_oc", "branch_candidates_path")
    actual_branch_logs = {
        declared_path(path) for path in branch_summary.get("input_logs") or ()}
    if actual_branch_logs != expected_branch_logs:
        raise SystemExit(
            "[UMR provenance][ERROR] branch-label input logs do not match current VoV logs")
    if declared_hashes(branch_summary.get("input_log_sha256") or {}) != source_hashes(
            log_meta, "train_oc", "branch_candidates_path"):
        raise SystemExit(
            "[UMR provenance][ERROR] branch labels are stale relative to current logs")
    if (declared_path(branch_summary.get("output_labels")) != branch_labels or
            branch_summary.get("output_labels_sha256") != sha256_file(branch_labels) or
            branch_summary.get("output_labels_content_sha256") !=
            jsonl_content_sha256(branch_labels) or
            int(branch_summary.get("output_label_records") or 0) <= 0 or
            count_jsonl_records(branch_labels) !=
            int(branch_summary.get("output_label_records") or 0) or
            int((branch_summary.get("counts") or {}).get("labels_written") or 0) !=
            int(branch_summary.get("output_label_records") or 0)):
        raise SystemExit(
            "[UMR provenance][ERROR] branch-label output provenance mismatch")
    utility_labels = (utility.get("label_provenance") or {}).get(
        "labels_realpath") or (utility.get("label_provenance") or {}).get("labels_path")
    if not utility_labels or declared_path(utility_labels) != branch_labels:
        raise SystemExit(
            "[UMR provenance][ERROR] utility mapper was not fit from current labels")
    if jsonl_content_sha256(branch_labels) != (utility.get("label_provenance") or {}).get(
            "labels_content_sha256"):
        raise SystemExit(
            "[UMR provenance][ERROR] utility label content hash mismatch")

    scene_labels = assert_within(
        args.scene_labels, os.path.dirname(args.scene_label_summary),
        "scene labels")
    scene_summary = load_json(args.scene_label_summary, "scene-label summary")
    if (scene_summary.get("formal_calibration_safe") is not True or
            scene_summary.get("debug_only") is True or
            declared_path(scene_summary.get("scene_log")) != real(args.source_log_root)):
        raise SystemExit(
            "[UMR provenance][ERROR] scene-label summary is not from current logs")
    expected_scene_logs = (
        source_paths(log_meta, "train_clean", "scene_summary_path") |
        source_paths(log_meta, "train_oc", "scene_summary_path"))
    actual_scene_logs = {
        declared_path(path) for path in scene_summary.get("log_paths") or ()}
    if actual_scene_logs != expected_scene_logs:
        raise SystemExit(
            "[UMR provenance][ERROR] scene-label input logs do not match current VoV logs")
    if declared_hashes(scene_summary.get("log_path_sha256") or {}) != source_hashes(
            log_meta, FORMAL_SPLITS, "scene_summary_path"):
        raise SystemExit(
            "[UMR provenance][ERROR] scene labels are stale relative to current logs")
    if (declared_path(scene_summary.get("output_labels")) != scene_labels or
            scene_summary.get("output_labels_sha256") != sha256_file(scene_labels) or
            int(scene_summary.get("written") or 0) <= 0 or
            count_jsonl_records(scene_labels) != int(scene_summary.get("written") or 0)):
        raise SystemExit(
            "[UMR provenance][ERROR] scene-label output provenance mismatch")
    expected_policy = {
        "beams_s1": {"target": 0.0, "weight": 1.0},
        "beams_s2": {"target": 0.35, "weight": 0.45},
        "beams_s3": {"target": 1.0, "weight": 1.0},
        "clean": {"target": 0.0, "weight": 1.5},
        "fog_motionblur": {"target": 0.0, "weight": 2.0},
        "other_corruptions": "negative",
    }
    if plain(scene_summary.get("label_policy") or {}) != expected_policy:
        raise SystemExit(
            "[UMR provenance][ERROR] scene-label policy differs from R50 formal policy")
    if declared_path(scene.get("label_summary_path")) != real(args.scene_label_summary):
        raise SystemExit(
            "[UMR provenance][ERROR] scene mapper label summary mismatch")
    if (declared_path(scene.get("labels_path")) != scene_labels or
            scene.get("labels_sha256") != sha256_file(scene_labels) or
            scene.get("label_summary_sha256") != sha256_file(args.scene_label_summary)):
        raise SystemExit(
            "[UMR provenance][ERROR] scene mapper is stale relative to current labels")

    expected_reference_logs = source_paths(
        log_meta, "train_clean", "prediction_light_path")
    actual_reference_logs = {
        declared_path(path) for path in reference.get("source_files") or ()}
    if actual_reference_logs != expected_reference_logs:
        raise SystemExit(
            "[UMR provenance][ERROR] clean reference does not match current VoV logs")
    if declared_hashes(reference.get("source_file_sha256") or {}) != source_hashes(
            log_meta, "train_clean", "prediction_light_path"):
        raise SystemExit(
            "[UMR provenance][ERROR] clean reference is stale relative to current logs")

    return {
        "branch_labels": {"path": os.path.relpath(branch_labels, os.getcwd()),
                          "sha256": sha256_file(branch_labels),
                          "content_sha256": jsonl_content_sha256(branch_labels)},
        "branch_label_summary": {
            "path": os.path.relpath(real(args.branch_label_summary), os.getcwd()),
            "sha256": sha256_file(args.branch_label_summary)},
        "scene_labels": {"path": os.path.relpath(scene_labels, os.getcwd()),
                         "sha256": sha256_file(scene_labels)},
        "scene_label_summary": {
            "path": os.path.relpath(real(args.scene_label_summary), os.getcwd()),
            "sha256": sha256_file(args.scene_label_summary)},
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def critical_code_records(profile):
    records = []
    paths = CRITICAL_CODE_PATHS + PROFILE_CRITICAL_CODE_PATHS.get(profile, ())
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(
                "[UMR provenance][ERROR] missing critical source: {}".format(path))
        records.append({"path": path, "sha256": sha256_file(path)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument(
        "--profile", choices=("vov_umr", "vov_v10", "r50_v10"),
        default="vov_umr")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--train-infos", required=True)
    parser.add_argument(
        "--expected-train-infos-sha256", default=EXPECTED_TRAIN_INFOS_SHA256)
    parser.add_argument("--source-log-root", required=True)
    parser.add_argument("--branch-labels", required=True)
    parser.add_argument("--branch-label-summary", required=True)
    parser.add_argument("--scene-labels", required=True)
    parser.add_argument("--scene-label-summary", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fingerprint-output", required=True)
    args = parser.parse_args()

    for path, label in (
            (args.base_config, "base config"),
            (args.train_infos, "train infos"),
            (args.source_log_root, "source log root")):
        assert_train_only_path(path, label)
    for path in args.artifact:
        assert_train_only_path(path, "runtime artifact")
    for path, label in (
            (args.branch_labels, "branch labels"),
            (args.branch_label_summary, "branch-label summary"),
            (args.scene_labels, "scene labels"),
            (args.scene_label_summary, "scene-label summary")):
        assert_train_only_path(path, label)
    for path in (args.base_config, args.checkpoint, args.train_infos):
        if not os.path.isfile(path):
            raise SystemExit("[UMR provenance][ERROR] missing file: {}".format(path))
    identity = config_identity(args.base_config, args.profile)
    config_sha = sha256_file(args.base_config)
    resolved_config_sha = resolved_config_sha256(args.base_config)
    checkpoint_sha = sha256_file(args.checkpoint)
    if (args.expected_checkpoint_sha256 and
            checkpoint_sha != args.expected_checkpoint_sha256):
        raise SystemExit(
            "[UMR provenance][ERROR] checkpoint SHA256 mismatch: expected={}, "
            "got={}".format(args.expected_checkpoint_sha256, checkpoint_sha))
    train_infos_sha = sha256_file(args.train_infos)
    if (not args.expected_train_infos_sha256 or
            train_infos_sha != args.expected_train_infos_sha256):
        raise SystemExit(
            "[UMR provenance][ERROR] train infos SHA256 mismatch: expected={}, "
            "got={}".format(args.expected_train_infos_sha256, train_infos_sha))
    actual_checkpoint_epoch = checkpoint_epoch(args.checkpoint)
    expected_profile_epoch = 4 if args.profile == "vov_umr" else 6
    if (actual_checkpoint_epoch != int(args.checkpoint_epoch) or
            actual_checkpoint_epoch != expected_profile_epoch):
        raise SystemExit(
            "[UMR provenance][ERROR] checkpoint epoch mismatch: CLI={}, "
            "checkpoint={}".format(args.checkpoint_epoch, actual_checkpoint_epoch))
    critical_code = critical_code_records(args.profile)
    fingerprint_payload = {
        "profile": args.profile,
        "config_sha256": config_sha,
        "resolved_config_sha256": resolved_config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": actual_checkpoint_epoch,
        "identity": identity,
        "critical_code": critical_code,
    }
    canonical = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    provenance = {
        "format": "umr_model_provenance_v1",
        "profile": args.profile,
        "fingerprint": fingerprint,
        "base_config": os.path.relpath(os.path.abspath(args.base_config), os.getcwd()),
        "base_config_sha256": config_sha,
        "resolved_config_sha256": resolved_config_sha,
        "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), os.getcwd()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": actual_checkpoint_epoch,
        "train_infos": os.path.relpath(os.path.abspath(args.train_infos), os.getcwd()),
        "train_infos_sha256": train_infos_sha,
        "source_log_root": os.path.relpath(
            os.path.abspath(args.source_log_root), os.getcwd()),
        "formal_splits": list(FORMAL_SPLITS),
        "identity": identity,
        "critical_code": critical_code,
        "generator_commit": git_commit(),
    }

    log_meta = collect_log_meta(
        args.source_log_root, args.checkpoint, args.base_config, args.train_infos,
        args.profile)
    artifact_payloads = []
    for path in args.artifact:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_artifact(path, payload)
        existing_fingerprint = str(
            (payload.get("model_provenance") or {}).get("fingerprint") or "")
        if existing_fingerprint and existing_fingerprint != fingerprint:
            raise SystemExit(
                "[UMR provenance][ERROR] refusing to overwrite artifact with a "
                "different model fingerprint: {}".format(path))
        artifact_payloads.append((path, payload))
    if ({os.path.basename(path) for path, _payload in artifact_payloads} != {
            "mapper_scene_bev_damage_applicable.json",
            "mapper_candidate_useful.json", "clean_reference_stats.json"} or
            len(artifact_payloads) != 3):
        raise SystemExit(
            "[UMR provenance][ERROR] formal artifact set is incomplete or duplicated")
    calibration_sources = validate_calibration_chain(
        args, artifact_payloads, log_meta)

    artifact_records = []
    for path, payload in artifact_payloads:
        payload["model_provenance"] = provenance
        payload["calibration_sources"] = calibration_sources
        atomic_write_json(path, payload)
        artifact_records.append({
            "path": os.path.relpath(os.path.abspath(path), os.getcwd()),
            "sha256": sha256_file(path),
            "target": payload.get("target"),
            "format": payload.get("format"),
        })

    manifest = {
        "format": "{}_sgsup_b_manifest_v1".format(args.profile),
        "profile": args.profile,
        "method": "SGSUP-B context/height pure-BEV subset guard",
        "parity_source": "RULE_SGSUP_B_contextheight055_v9bev_dyn10_30",
        "parity_source_record": {
            "path": R50_PARITY_SOURCE,
            "sha256": R50_PARITY_SOURCE_SHA256,
            "generator_commit": R50_PARITY_SOURCE_COMMIT,
            "tracked_contract": R50_PARITY_CONTRACT,
            "tracked_contract_sha256": sha256_file(R50_PARITY_CONTRACT),
        },
        "model_provenance": provenance,
        "scene_threshold_key": "clean_fpr_005",
        "scene_label_policy": {
            "beams_s1": {"target": 0.0, "weight": 1.0},
            "beams_s2": {"target": 0.35, "weight": 0.45},
            "beams_s3": {"target": 1.0, "weight": 1.0},
            "clean": {"target": 0.0, "weight": 1.5},
            "fog_motionblur": {"target": 0.0, "weight": 2.0},
            "other_corruptions": "negative",
        },
        "formal_calibration_safe": True,
        "debug_only": False,
        "source_log_meta": log_meta,
        "calibration_sources": calibration_sources,
        "artifacts": artifact_records,
    }
    atomic_write_json(args.manifest, manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.fingerprint_output)), exist_ok=True)
    with open(args.fingerprint_output, "w", encoding="utf-8") as handle:
        handle.write(fingerprint + "\n")
    print("[UMR provenance] fingerprint={}".format(fingerprint))
    print("[UMR provenance] manifest={}".format(args.manifest))


if __name__ == "__main__":
    main()
