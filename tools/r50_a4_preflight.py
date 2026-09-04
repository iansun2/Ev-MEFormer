#!/usr/bin/env python3
"""Fail-closed Stage-1 R50 EUD to V8-A4 transition checks."""

import argparse
import os
import sys

from mmcv import Config
from mmdet3d.models import build_model


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import projects.mmdet3d_plugin  # noqa: F401,E402
from tools.vov_a4_preflight import (  # noqa: E402
    ENSEMBLE_PREFIX,
    _check_checkpoint,
    _check_trainable,
    _find_hook,
)


def walk_pipeline(pipeline):
    for step in pipeline or ():
        if not isinstance(step, dict):
            continue
        yield step
        for child in walk_pipeline(step.get("transforms") or ()):
            yield child


def check_config(cfg):
    errors = []
    model = cfg.model
    head = model.pts_bbox_head
    ensemble = head.ensemble
    checks = {
        "img_backbone.type": (model.img_backbone.type, "ResNet"),
        "image_final_dim": (list(cfg.ida_aug_conf.final_dim), [320, 800]),
        "voxel_size": (list(model.pts_voxel_layer.voxel_size), [0.1, 0.1, 0.2]),
        "bbox_voxel_size": (
            list(head.bbox_coder.voxel_size), [0.1, 0.1, 0.2]),
        "use_ensemble": (bool(head.get("use_ensemble", False)), True),
        "ensemble.type": (ensemble.type, "EDLPMETransformer"),
        "v8_router_variant": (
            ensemble.v8_router_variant,
            "a4_clean_anchor_domain_reliability"),
        "test_modalities": (list(head.modalities.test), ["fused", "bev", "img"]),
        "samples_per_gpu": (int(cfg.data.samples_per_gpu), 4),
        "total_epochs": (int(cfg.total_epochs), 6),
        "checkpoint_interval": (int(cfg.checkpoint_config.interval), 1),
    }
    for name, (actual, expected) in checks.items():
        print("  {}={}".format(name, actual))
        if actual != expected:
            errors.append(
                "{}: expected={!r}, got={!r}".format(name, expected, actual))

    steps = list(walk_pipeline(cfg.data.train.dataset.pipeline))
    resize = [step for step in steps if step.get("type") == "ResizeCropFlipImage"]
    corrupt = [
        step for step in steps
        if step.get("type") == "OnlineSyntheticMultiCorrupt"]
    if len(resize) != 1:
        errors.append("R50 A4 must contain exactly one ResizeCropFlipImage")
    elif list(resize[0].get("data_aug_conf", {}).get("final_dim", ())) != [320, 800]:
        errors.append("R50 A4 resize final_dim must be [320, 800]")
    if len(corrupt) != 1:
        errors.append("R50 A4 must contain exactly one OnlineSyntheticMultiCorrupt")
    elif abs(float(corrupt[0].get("corrupt_prob", -1.0)) - 0.50) > 1e-12:
        errors.append("R50 A4 corrupt_prob must be 0.50")

    freeze_hook = _find_hook(cfg, "FreezeWeight")
    if freeze_hook is None:
        errors.append("FreezeWeight hook is missing")
    elif list(freeze_hook.get("finetune_weight", ())) != [ENSEMBLE_PREFIX]:
        errors.append("R50 A4 must train only pts_bbox_head.ensemble")
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="R50 V8-A4 config path")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Stage-1 R50 EUD checkpoint; defaults to cfg.load_from.")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print("[R50 A4] Config contract")
    errors = check_config(cfg)
    if args.no_checkpoint and args.expected_checkpoint_epoch is not None:
        errors.append(
            "--expected-checkpoint-epoch cannot be used with --no-checkpoint")

    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    errors.extend(_check_trainable(cfg, model))
    if not args.no_checkpoint:
        checkpoint = args.checkpoint or cfg.get("load_from")
        if not checkpoint or not os.path.isfile(checkpoint):
            errors.append("Stage-1 R50 checkpoint is missing: {}".format(checkpoint))
        else:
            errors.extend(_check_checkpoint(
                model, checkpoint,
                expected_epoch=args.expected_checkpoint_epoch))

    if errors:
        print("\n[R50 A4 Preflight] FAIL")
        for error in errors:
            print("  - {}".format(error))
        raise SystemExit(1)
    print("\n[R50 A4 Preflight] PASS")


if __name__ == "__main__":
    main()
