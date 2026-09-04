#!/usr/bin/env python3
"""Generate an immutable VoV V10 selector config from train-only calibration."""

import argparse
import hashlib
import json
import math
import os
import tempfile

import torch
from mmcv import Config


EXPECTED_TRAIN_INFOS_SHA256 = (
    "6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854")
THRESHOLD_KEYS = ("tau_margin", "margin", "tau_dmg", "T_oracle")
EXPECTED_CRITICAL_CODE_PATHS = (
    "projects/mmdet3d_plugin/models/utils/edl_pme_transformer.py",
    "projects/mmdet3d_plugin/models/utils/v10_quality.py",
    "projects/mmdet3d_plugin/models/dense_heads/edl_head.py",
    "projects/mmdet3d_plugin/datasets/pipelines/synthetic_corruptions.py",
    "tools/v10_quality_calibration.py",
    "tools/build_vov_v10_selector_config.py",
)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return int((payload.get("meta") or {}).get("epoch", -1))


def atomic_write(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".vov-v10-config-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_critical_code(records):
    if not isinstance(records, list):
        raise SystemExit("[VoV V10 config][ERROR] missing critical-code records")
    by_path = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("path"):
            raise SystemExit("[VoV V10 config][ERROR] invalid critical-code record")
        path = str(record["path"])
        if path in by_path:
            raise SystemExit("[VoV V10 config][ERROR] duplicate critical-code path")
        by_path[path] = str(record.get("sha256") or "")
    if tuple(sorted(by_path)) != tuple(sorted(EXPECTED_CRITICAL_CODE_PATHS)):
        raise SystemExit("[VoV V10 config][ERROR] critical-code path set mismatch")
    for path in EXPECTED_CRITICAL_CODE_PATHS:
        absolute = os.path.join(REPO_ROOT, path)
        if not os.path.isfile(absolute) or sha256_file(absolute) != by_path[path]:
            raise SystemExit(
                "[VoV V10 config][ERROR] critical-code SHA mismatch: {}".format(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    for path in (args.base_config, args.calibration, args.source_checkpoint):
        if not os.path.isfile(path):
            raise SystemExit("[VoV V10 config][ERROR] missing input: {}".format(path))
    with open(args.calibration, "r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    provenance = calibration.get("model_provenance") or {}
    diagnostics = calibration.get("_diagnostics") or {}
    if calibration.get("format") != "v10_quality_calibration_v2":
        raise SystemExit("[VoV V10 config][ERROR] unsupported calibration format")
    if calibration.get("formal_train_only") is not True or diagnostics.get("dry_run"):
        raise SystemExit("[VoV V10 config][ERROR] calibration is not formal train-only")
    if int(diagnostics.get("total_pairs") or 0) <= 0:
        raise SystemExit("[VoV V10 config][ERROR] calibration has no matched pairs")
    if provenance.get("checkpoint_sha256") != sha256_file(args.source_checkpoint):
        raise SystemExit("[VoV V10 config][ERROR] source checkpoint SHA mismatch")
    if int(provenance.get("checkpoint_epoch") or -1) != 4:
        raise SystemExit("[VoV V10 config][ERROR] source checkpoint is not epoch 4")
    if checkpoint_epoch(args.source_checkpoint) != 4:
        raise SystemExit("[VoV V10 config][ERROR] checkpoint meta.epoch is not 4")
    if provenance.get("base_config_sha256") != sha256_file(args.base_config):
        raise SystemExit("[VoV V10 config][ERROR] calibration config SHA mismatch")
    if provenance.get("train_infos_sha256") != EXPECTED_TRAIN_INFOS_SHA256:
        raise SystemExit("[VoV V10 config][ERROR] calibration is not nuScenes train")
    validate_critical_code(provenance.get("critical_code"))

    thresholds = {}
    for key in THRESHOLD_KEYS:
        value = float(calibration.get(key))
        if not math.isfinite(value):
            raise SystemExit("[VoV V10 config][ERROR] non-finite {}".format(key))
        thresholds[key] = value
    if (thresholds["tau_margin"] < 0.0 or thresholds["tau_dmg"] < 0.0 or
            thresholds["margin"] < 0.0 or thresholds["T_oracle"] <= 0.0):
        raise SystemExit("[VoV V10 config][ERROR] invalid threshold range")

    base_relative = os.path.relpath(
        os.path.abspath(args.base_config),
        os.path.dirname(os.path.abspath(args.output)))
    checkpoint_relative = os.path.relpath(
        os.path.abspath(args.source_checkpoint),
        os.path.dirname(os.path.abspath(args.output)))
    calibration_sha = sha256_file(args.calibration)
    checkpoint_sha = sha256_file(args.source_checkpoint)
    text = "# Generated from checkpoint-specific VoV train/O.C calibration.\n"
    text += "_base_ = {!r}\n\n".format(base_relative)
    text += "load_from = {!r}\n".format(checkpoint_relative)
    text += "resume_from = None\n\n"
    text += "v10_calibration_provenance = dict(\n"
    text += "    format='v10_quality_calibration_v2',\n"
    text += "    formal_train_only=True,\n"
    text += "    calibration_sha256={!r},\n".format(calibration_sha)
    text += "    calibration_base_config_sha256={!r},\n".format(
        provenance["base_config_sha256"])
    text += "    source_checkpoint_sha256={!r},\n".format(checkpoint_sha)
    text += "    source_checkpoint_epoch=4,\n"
    text += "    train_infos_sha256={!r},\n".format(
        EXPECTED_TRAIN_INFOS_SHA256)
    text += ")\n\n"
    text += "_v10_thresholds = dict(\n"
    for key in THRESHOLD_KEYS:
        text += "    {}={!r},\n".format(key, thresholds[key])
    text += ")\n\n"
    text += "model = dict(pts_bbox_head=dict(\n"
    text += "    v10_oracle_thresholds=_v10_thresholds,\n"
    # Match the finalized R50 selector semantics. The calibration recommendation
    # remains in the JSON for audit, but never silently changes this contract.
    text += "    v10_matching_mode='single_best',\n"
    text += "    ensemble=dict(v10_oracle_thresholds=_v10_thresholds),\n"
    text += "))\n\n"
    text += "work_dir = './work_dirs/vov_v10_router_selector_calibrated'\n"
    atomic_write(args.output, text)

    resolved = Config.fromfile(args.output)
    if (resolved.model.pts_bbox_head.v10_matching_mode != "single_best" or
            not bool(resolved.model.pts_bbox_head.ensemble.v10_enable) or
            int(resolved.total_epochs) != 6 or
            os.path.realpath(os.path.join(
                os.path.dirname(os.path.abspath(args.output)),
                str(resolved.load_from))) != os.path.realpath(args.source_checkpoint)):
        raise SystemExit("[VoV V10 config][ERROR] generated config contract failed")
    print("[VoV V10 config] wrote {}".format(args.output))
    print("  calibration_sha256={}".format(calibration_sha))
    print("  source_checkpoint_sha256={}".format(checkpoint_sha))


if __name__ == "__main__":
    main()
