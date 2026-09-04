#!/usr/bin/env python
"""Preflight checks for V10 Stage-2 Router/Selector configs.

Dry-run mode uses synthetic ensemble inputs so the tool can run before the
nuScenes data pipeline is available. It still exercises the V10 dispatch path,
FreezeWeight hook, trainable whitelist, buffer guards, oracle-quality numerics,
and gradient leakage guard.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.runner.hooks import HOOKS
from mmcv.utils import build_from_cfg
from mmdet3d.models import build_model


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import projects.mmdet3d_plugin  # noqa: F401,E402  Registers custom modules.
from tools.v9_rev20_preflight import (  # noqa: E402
    _check_checkpoint_coverage,
    _check_checkpoint_epoch,
)


V10_TRAINABLE_PREFIXES = (
    'pts_bbox_head.ensemble.v10_dd_head',
    'pts_bbox_head.ensemble.v10_router_head_cls',
    'pts_bbox_head.ensemble.v10_router_head_height',
    'pts_bbox_head.ensemble.v10_router_head_center',
    'pts_bbox_head.ensemble.v10_center_safe_gate',
)

V10_EDLCUE_TRAINABLE_PREFIX = 'pts_bbox_head.ensemble.v10_edlcue_head'
V10_EDLCUE_TRAIN_SCOPES = ('full_v10', 'head_only')
V10_REV20_BASE_CHECKPOINT_MARKER = 'v9_rev20_a4_routing_only'
V10_EDLCUE_BASE_CHECKPOINT_MARKER = 'v10_router_selector'
EXPECTED_TRAIN_INFOS_SHA256 = (
    '6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854')

DETECTION_BLACKLIST_PREFIXES = (
    'img_backbone',
    'img_neck',
    'pts_voxel_encoder',
    'pts_middle_encoder',
    'pts_backbone',
    'pts_neck',
    'pts_bbox_head.transformer',
    'pts_bbox_head.task_heads',
    'pts_bbox_head.ensemble.decoder',
    'pts_bbox_head.ensemble.task_heads',
    'pts_bbox_head.ensemble.modality_proj',
    'pts_bbox_head.ensemble.box_pos_embedding',
    'pts_bbox_head.ensemble.distance_prior_head',
    'pts_bbox_head.ensemble.residual_router_head',
    'pts_bbox_head.ensemble.alpha_res_head',
    'pts_bbox_head.ensemble.selector_context_proj',
    'pts_bbox_head.ensemble.v9_bev_anomaly_head',
    'pts_bbox_head.ensemble.v9_img_anomaly_head',
    'pts_bbox_head.ensemble.v9_fused_gate_head',
    'pts_bbox_head.ensemble.v8_a3_damage_context_head',
    'pts_bbox_head.ensemble.v8_damage_feature_weights_raw',
    'pts_bbox_head.ensemble.v8_a4_damage_feature_weights_raw',
    'pts_bbox_head.ensemble.v8_damage_feature_bias',
    'pts_bbox_head.ensemble.v8_a4_damage_feature_bias',
)

BUFFER_NAMES = (
    'v8_a4_anchor_mean',
    'v8_a4_anchor_var',
    'v8_a4_anchor_count',
    'v9_tau_bev',
    'v9_tau_img',
    'v9_tau_update_count',
    'v9_clean_distance_threshold_bev',
    'v9_clean_distance_threshold_img',
    'v9_clean_distance_update_count',
    'v9_clean_q95_z_bev',
    'v9_clean_q95_z_img',
)

V10_HEALTH_KEYS = (
    'v10_p_dmg_bev',
    'v10_p_dmg_img',
    'v10_p_dmg_dual',
    'v10_w_fused_cls',
    'v10_w_bev_cls',
    'v10_w_img_cls',
    'v10_w_fused_center',
    'v10_w_bev_center',
    'v10_w_img_center',
    'v10_w_fused_height',
    'v10_w_bev_height',
    'v10_w_img_height',
    'v10_center_floor',
)

PARITY_KEYS = ('cls_logits', 'center', 'center_nig', 'height', 'height_nig')
VOV_V10_CRITICAL_CODE_PATHS = (
    'projects/mmdet3d_plugin/models/utils/edl_pme_transformer.py',
    'projects/mmdet3d_plugin/models/utils/v10_quality.py',
    'projects/mmdet3d_plugin/models/dense_heads/edl_head.py',
    'projects/mmdet3d_plugin/datasets/pipelines/synthetic_corruptions.py',
    'tools/v10_quality_calibration.py',
    'tools/build_vov_v10_selector_config.py',
)


def _starts_with_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _find_hook(cfg: Config, hook_type: str):
    for hook in cfg.get('custom_hooks', []):
        if hook.get('type') == hook_type:
            return hook
    return None


def _v10_train_scope(cfg: Config, edlcue_enabled: bool) -> str:
    scope = str(cfg.get('v10_edlcue_train_scope', 'full_v10'))
    if not edlcue_enabled:
        return 'full_v10'
    return scope


def _expected_trainable_prefixes(cfg: Config) -> Tuple[str, ...]:
    ensemble_cfg = cfg.model['pts_bbox_head']['ensemble']
    edlcue_enabled = bool(ensemble_cfg.get('v10_edlcue_enable', False))
    scope = _v10_train_scope(cfg, edlcue_enabled)
    if edlcue_enabled and scope == 'head_only':
        return (V10_EDLCUE_TRAINABLE_PREFIX,)
    expected = tuple(V10_TRAINABLE_PREFIXES)
    if edlcue_enabled:
        expected = expected + (V10_EDLCUE_TRAINABLE_PREFIX,)
    return expected


def _full_path(path: str) -> str:
    if not path:
        return ''
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    detached = tensor.detach()
    return {
        'shape': list(detached.shape),
        'min': float(detached.min().cpu()) if detached.numel() else 0.0,
        'max': float(detached.max().cpu()) if detached.numel() else 0.0,
        'mean': float(detached.mean().cpu()) if detached.numel() else 0.0,
        'std': float(detached.std(unbiased=False).cpu()) if detached.numel() else 0.0,
    }


class _PrintLogger:
    def info(self, _msg):
        pass


class _SizedLoader:
    def __len__(self):
        return 2


class _DummyRunner:
    def __init__(self, model):
        self.model = model
        self.logger = _PrintLogger()
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.rank = 0
        self.data_loader = _SizedLoader()


@contextmanager
def _temporary_attr(obj, name, value):
    old_value = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old_value)


def _build_freeze_hook(cfg: Config):
    hook_cfg = _find_hook(cfg, 'FreezeWeight')
    if hook_cfg is None:
        return None
    return build_from_cfg(dict(hook_cfg), HOOKS)


def _apply_freeze_hook(cfg: Config, model) -> List[str]:
    hook = _build_freeze_hook(cfg)
    if hook is None:
        return ['FreezeWeight hook is missing']
    runner = _DummyRunner(model)
    hook.before_run(runner)
    return []


def _reapply_bn_freeze(cfg: Config, model) -> List[str]:
    hook = _build_freeze_hook(cfg)
    if hook is None:
        return ['FreezeWeight hook is missing']
    hook.before_train_epoch(_DummyRunner(model))
    return []


def _get_head_and_ensemble(model):
    head = model.pts_bbox_head
    return head, head.ensemble


def _check_config(
        cfg: Config, checkpoint: str, dry_run: bool, profile: str,
        explicit_checkpoint: bool) -> List[str]:
    errors = []
    head_cfg = cfg.model['pts_bbox_head']
    ensemble_cfg = head_cfg['ensemble']
    print('[HG0] Config sanity')

    cfg_load = cfg.get('load_from')
    ckpt_path = _full_path(checkpoint or cfg_load)
    print(f'  load_from={cfg_load}')
    print(f'  checkpoint={checkpoint}')
    edlcue_enabled = bool(ensemble_cfg.get('v10_edlcue_enable', False))
    expected_checkpoint_marker = (
        V10_EDLCUE_BASE_CHECKPOINT_MARKER
        if edlcue_enabled else V10_REV20_BASE_CHECKPOINT_MARKER)
    if (not explicit_checkpoint and profile == 'r50_v10' and
            not (cfg_load and expected_checkpoint_marker in str(cfg_load))):
        errors.append(
            f'cfg.load_from must point to {expected_checkpoint_marker} checkpoint')
    if not os.path.exists(ckpt_path):
        msg = f'checkpoint path does not exist: {ckpt_path}'
        if dry_run:
            print(f'  [dry-run warning] {msg}')
        else:
            errors.append(msg)

    scope = _v10_train_scope(cfg, edlcue_enabled)
    expected = _expected_trainable_prefixes(cfg)
    print(f'  v10_edlcue_train_scope={scope}')
    if scope not in V10_EDLCUE_TRAIN_SCOPES:
        errors.append(
            f'v10_edlcue_train_scope must be one of {V10_EDLCUE_TRAIN_SCOPES}')
    freeze_hook = _find_hook(cfg, 'FreezeWeight')
    if freeze_hook is None:
        errors.append('FreezeWeight hook is missing')
    else:
        prefixes = tuple(freeze_hook.get('finetune_weight', []))
        print(f'  FreezeWeight prefixes={prefixes}')
        if prefixes != expected:
            errors.append(f'FreezeWeight prefixes must exactly match {expected}')
        if not freeze_hook.get('freeze_bn', False):
            errors.append('FreezeWeight.freeze_bn must be True')

    edl_hook = _find_hook(cfg, 'EDLEpochHook')
    if edl_hook is None:
        errors.append('EDLEpochHook is missing')
    elif not edl_hook.get('v10_cls_fused_keep_min_schedule'):
        errors.append('EDLEpochHook must register v10_cls_fused_keep_min_schedule')

    checks = {
        'v8_router_variant': ensemble_cfg.get('v8_router_variant') == 'v9_lite_domain_anchored',
        'update_tau_online': ensemble_cfg.get('update_tau_online') is False,
        'v9_freeze_clean_anchor_epochs': int(ensemble_cfg.get('v9_freeze_clean_anchor_epochs', 0)) >= 999,
        'v10_enable': ensemble_cfg.get('v10_enable') is True,
        'v10_input_detach': ensemble_cfg.get('v10_input_detach') is True,
        'v10_use_conservative_schedule': (
            ensemble_cfg.get('v10_use_conservative_schedule') is False),
        'v10_use_per_head_cls_dispatch': (
            ensemble_cfg.get('v10_use_per_head_cls_dispatch') is False),
        'v10_loss_enabled': head_cfg.get('v10_loss_enabled') is True,
        'lambda_v10_route_kl': abs(float(
            head_cfg.get('lambda_v10_route_kl', -1)) - 0.05) < 1e-9,
        'lambda_v10_route_margin': abs(float(
            head_cfg.get('lambda_v10_route_margin', -1)) - 0.10) < 1e-9,
        'lambda_v10_dd_anomaly': abs(float(
            head_cfg.get('lambda_v10_dd_anomaly', -1)) - 1.00) < 1e-9,
        'lambda_v10_dd_consistency': abs(float(
            head_cfg.get('lambda_v10_dd_consistency', -1)) - 0.05) < 1e-9,
        'lambda_v10_csg': abs(float(
            head_cfg.get('lambda_v10_csg', -1)) - 0.10) < 1e-9,
        'lambda_v9_route_consistency': float(head_cfg.get('lambda_v9_route_consistency', -1)) == 0.0,
        'lambda_v9_gate_domain': float(head_cfg.get('lambda_v9_gate_domain', -1)) == 0.0,
        'lambda_v9_anomaly_supervision': abs(float(head_cfg.get('lambda_v9_anomaly_supervision', -1)) - 0.01) < 1e-9,
        'lambda_consistency_cls': float(head_cfg.get('lambda_consistency_cls', -1)) == 0.0,
        'lambda_consistency_box': float(head_cfg.get('lambda_consistency_box', -1)) == 0.0,
    }
    if edlcue_enabled:
        checks.update({
            'total_epochs_positive': int(cfg.total_epochs) >= 1,
            'runner_matches_total_epochs': (
                int(cfg.runner.max_epochs) == int(cfg.total_epochs)),
        })
    else:
        checks.update({
            'total_epochs': int(cfg.total_epochs) == 6,
            'runner.max_epochs': int(cfg.runner.max_epochs) == 6,
        })
    if profile == 'vov_v10':
        checks.update({
            'v10_matching_mode': str(
                head_cfg.get('v10_matching_mode', '')) == 'single_best',
            'backbone': cfg.model.img_backbone.type == 'VoVNet',
            'backbone_spec': cfg.model.img_backbone.get('spec_name') == 'V-99-eSE',
            'image_final_dim': list(cfg.ida_aug_conf.final_dim) == [640, 1600],
            'voxel_size': list(cfg.model.pts_voxel_layer.voxel_size) == [0.075, 0.075, 0.2],
            'samples_per_gpu': int(cfg.data.samples_per_gpu) == 2,
            'cumulative_iters': int(cfg.optimizer_config.cumulative_iters) == 2,
            'formal_calibration_provenance': bool(
                (cfg.get('v10_calibration_provenance') or {}).get(
                    'formal_train_only', False)),
        })
    else:
        checks.update({
            'v10_matching_mode': str(
                head_cfg.get('v10_matching_mode', 'union')) in (
                    'union', 'single_best'),
            'backbone': cfg.model.img_backbone.type == 'ResNet',
            'image_final_dim': list(cfg.ida_aug_conf.final_dim) == [320, 800],
            'voxel_size': list(cfg.model.pts_voxel_layer.voxel_size) == [0.1, 0.1, 0.2],
        })
    thresholds = head_cfg.get('v10_oracle_thresholds') or {}
    ensemble_thresholds = ensemble_cfg.get('v10_oracle_thresholds') or {}
    checks['head_ensemble_threshold_parity'] = dict(thresholds) == dict(
        ensemble_thresholds)
    for key in ('tau_margin', 'margin', 'tau_dmg', 'T_oracle'):
        try:
            value = float(thresholds[key])
            checks['threshold_{}'.format(key)] = (
                math.isfinite(value) and
                (value > 0.0 if key == 'T_oracle' else value >= 0.0))
        except (KeyError, TypeError, ValueError):
            checks['threshold_{}'.format(key)] = False
    for name, ok in checks.items():
        print(f'  {name}: {ok}')
        if not ok:
            errors.append(f'HG0 config sanity failed: {name}')
    return errors


def _check_vov_calibration_files(cfg, calibration, source_checkpoint):
    if not calibration or not source_checkpoint:
        return [
            'vov_v10 requires --calibration and --source-checkpoint for provenance']
    errors = []
    if not os.path.isfile(calibration):
        return ['VoV V10 calibration JSON is missing: {}'.format(calibration)]
    if not os.path.isfile(source_checkpoint):
        return ['VoV V10 source checkpoint is missing: {}'.format(source_checkpoint)]
    with open(calibration, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    record = cfg.get('v10_calibration_provenance') or {}
    if payload.get('format') != 'v10_quality_calibration_v2':
        errors.append('unsupported VoV V10 calibration format')
    if payload.get('formal_train_only') is not True:
        errors.append('VoV V10 calibration is not formal train-only')
    if record.get('calibration_sha256') != _sha256_file(calibration):
        errors.append('VoV V10 calibration SHA mismatch')
    source_provenance = payload.get('model_provenance') or {}
    if (record.get('calibration_base_config_sha256') !=
            source_provenance.get('base_config_sha256')):
        errors.append('VoV V10 calibration base-config provenance mismatch')
    if record.get('source_checkpoint_sha256') != _sha256_file(source_checkpoint):
        errors.append('VoV V10 source checkpoint SHA mismatch')
    if int(record.get('source_checkpoint_epoch') or -1) != 4:
        errors.append('VoV V10 source checkpoint provenance is not epoch 4')
    if source_provenance.get('checkpoint_sha256') != _sha256_file(source_checkpoint):
        errors.append('calibration JSON source checkpoint mismatch')
    if int(source_provenance.get('checkpoint_epoch') or -1) != 4:
        errors.append('calibration JSON checkpoint epoch is not 4')
    if source_provenance.get('train_infos_sha256') != EXPECTED_TRAIN_INFOS_SHA256:
        errors.append('calibration JSON is not bound to registered nuScenes train')
    if record.get('train_infos_sha256') != EXPECTED_TRAIN_INFOS_SHA256:
        errors.append('generated config train-infos provenance mismatch')
    try:
        try:
            source_checkpoint_payload = torch.load(
                source_checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            source_checkpoint_payload = torch.load(
                source_checkpoint, map_location='cpu')
        if int((source_checkpoint_payload.get('meta') or {}).get('epoch', -1)) != 4:
            errors.append('actual VoV V10 source checkpoint meta.epoch is not 4')
    except Exception as exc:
        errors.append('cannot audit VoV V10 source checkpoint epoch: {}'.format(exc))
    critical_records = source_provenance.get('critical_code')
    if not isinstance(critical_records, list):
        errors.append('VoV V10 calibration critical-code records are missing')
    else:
        critical_by_path = {}
        for item in critical_records:
            if not isinstance(item, dict) or not item.get('path'):
                errors.append('VoV V10 calibration has invalid critical-code record')
                continue
            path = str(item['path'])
            if path in critical_by_path:
                errors.append('VoV V10 calibration has duplicate critical-code path')
            critical_by_path[path] = str(item.get('sha256') or '')
        if set(critical_by_path) != set(VOV_V10_CRITICAL_CODE_PATHS):
            errors.append('VoV V10 calibration critical-code path set mismatch')
        for path in VOV_V10_CRITICAL_CODE_PATHS:
            absolute = os.path.join(REPO_ROOT, path)
            if (not os.path.isfile(absolute) or
                    critical_by_path.get(path) != _sha256_file(absolute)):
                errors.append(
                    'VoV V10 calibration critical-code SHA mismatch: {}'.format(
                        path))
    if int((payload.get('_diagnostics') or {}).get('total_pairs') or 0) <= 0:
        errors.append('VoV V10 calibration has no matched pairs')
    configured_thresholds = cfg.model.pts_bbox_head.get(
        'v10_oracle_thresholds') or {}
    for key in ('tau_margin', 'margin', 'tau_dmg', 'T_oracle'):
        try:
            if abs(float(configured_thresholds[key]) - float(payload[key])) > 1e-12:
                errors.append('VoV V10 calibrated threshold mismatch: {}'.format(key))
        except (KeyError, TypeError, ValueError):
            errors.append('VoV V10 calibrated threshold is invalid: {}'.format(key))
    return errors


def _check_trainable(cfg: Config, model) -> List[str]:
    errors = _apply_freeze_hook(cfg, model)
    expected_prefixes = _expected_trainable_prefixes(cfg)
    trainable = [
        (name, tuple(param.shape), int(param.numel()))
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    total_trainable = sum(count for _, _, count in trainable)
    total_params = sum(int(param.numel()) for param in model.parameters())
    ratio = total_trainable / max(total_params, 1)

    print('[HG1/HG2/HG6] Trainable parameters')
    for name, shape, count in trainable:
        print(f'  {name}: shape={shape}, params={count:,}')
    print(f'  total_trainable={total_trainable:,} / {total_params:,} ({100.0 * ratio:.4f}%)')

    if not trainable:
        errors.append('No V10 trainable parameters found')
    for name, _shape, _count in trainable:
        if not _starts_with_any(name, expected_prefixes):
            errors.append(f'HG1 unexpected trainable parameter: {name}')
        if _starts_with_any(name, DETECTION_BLACKLIST_PREFIXES):
            errors.append(f'HG2 blacklist parameter is trainable: {name}')
    for prefix in expected_prefixes:
        if not any(name.startswith(prefix) for name, _, _ in trainable):
            errors.append(f'HG1 expected prefix has no trainable params: {prefix}')
    for name, param in model.named_parameters():
        if _starts_with_any(name, DETECTION_BLACKLIST_PREFIXES) and param.requires_grad:
            errors.append(f'HG2 blacklist parameter requires_grad=True: {name}')
    if ratio > 0.02:
        errors.append(f'HG6 trainable ratio exceeds 2.0%: {100.0 * ratio:.4f}%')
    return errors


def _synthetic_inputs(ensemble, bs: int = 4, nq: int = 8, seed: int = 0, device='cpu'):
    torch.manual_seed(seed)
    nb_dec = int(ensemble.num_layers)
    embed_dims = int(ensemble.embed_dims)
    total_q = nq * 3
    modalities = ['fused', 'bev', 'img']
    num_queries = [nq, nq, nq]
    dtype = torch.float32
    x = [torch.randn(bs, total_q, embed_dims, device=device, dtype=dtype)]
    reference = torch.rand(bs, total_q, 3, device=device, dtype=dtype)
    outs = {
        'center': torch.rand(nb_dec, bs, total_q, 2, device=device, dtype=dtype),
        'height': torch.rand(nb_dec, bs, total_q, 1, device=device, dtype=dtype),
        'cls_logits': torch.randn(nb_dec, bs, total_q, 10, device=device, dtype=dtype),
        'cls_alpha': torch.rand(nb_dec, bs, total_q, 10, device=device, dtype=dtype) + 1.0,
        'cls_beta': torch.rand(nb_dec, bs, total_q, 10, device=device, dtype=dtype) + 1.0,
        'center_nig': torch.rand(nb_dec, bs, total_q, 8, device=device, dtype=dtype),
        'height_nig': torch.rand(nb_dec, bs, total_q, 4, device=device, dtype=dtype),
        'dim': torch.rand(nb_dec, bs, total_q, 3, device=device, dtype=dtype),
        'dim_nig': torch.rand(nb_dec, bs, total_q, 12, device=device, dtype=dtype),
        'rot': torch.rand(nb_dec, bs, total_q, 2, device=device, dtype=dtype),
        'vel': torch.rand(nb_dec, bs, total_q, 2, device=device, dtype=dtype),
    }
    nig = {
        modality: {
            'center_nig': torch.rand(bs, nq, 8, device=device, dtype=dtype),
            'height_nig': torch.rand(bs, nq, 4, device=device, dtype=dtype),
        }
        for modality in modalities
    }
    img_metas = [{} for _ in range(bs)]
    return x, reference, outs, nig, modalities, num_queries, img_metas


def _ensemble_forward(model, bs=4, nq=8, seed=0, enable=None, grad=False):
    _head, ensemble = _get_head_and_ensemble(model)
    old_enable = ensemble.v10_enable
    if enable is not None:
        ensemble.v10_enable = bool(enable)
    args = _synthetic_inputs(ensemble, bs=bs, nq=nq, seed=seed)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    try:
        with ctx:
            return ensemble(
                args[0], args[1], args[2], args[3], args[4], args[5],
                0, [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], img_metas=args[6])
    finally:
        ensemble.v10_enable = old_enable


def _snapshot_bn(model) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    snapshot = {}
    for name, module in model.named_modules():
        if _starts_with_any(name, V10_TRAINABLE_PREFIXES):
            continue
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            mean = module.running_mean.detach().clone() if module.running_mean is not None else torch.empty(0)
            var = module.running_var.detach().clone() if module.running_var is not None else torch.empty(0)
            nbt = module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else torch.empty(0, dtype=torch.long)
            snapshot[name] = (mean, var, nbt)
    return snapshot


def _check_bn_freeze(cfg: Config, model) -> List[str]:
    print('[HG3] BN freeze check')
    errors = []
    model.train()
    errors.extend(_reapply_bn_freeze(cfg, model))
    before = _snapshot_bn(model)
    for _ in range(2):
        _ensemble_forward(model, bs=2, nq=4, seed=123, enable=False, grad=False)
    after = _snapshot_bn(model)
    changed = []
    for name, old in before.items():
        new = after.get(name)
        if new is None:
            changed.append(name)
            continue
        for old_tensor, new_tensor in zip(old, new):
            if old_tensor.shape != new_tensor.shape or not torch.allclose(old_tensor.cpu(), new_tensor.cpu(), atol=1e-7, rtol=0.0):
                changed.append(name)
                break
    print(f'  checked_bn={len(before)}, changed={changed[:8]}')
    if changed:
        errors.append('HG3 BN running stats changed: ' + ', '.join(changed[:20]))
    return errors


def _snapshot_buffers(ensemble) -> Dict[str, torch.Tensor]:
    result = {}
    for name in BUFFER_NAMES:
        value = getattr(ensemble, name, None)
        if isinstance(value, torch.Tensor):
            result[name] = value.detach().clone()
    return result


def _check_buffer_freeze(model) -> List[str]:
    print('[HG4] Buffer freeze check')
    errors = []
    _head, ensemble = _get_head_and_ensemble(model)
    before = _snapshot_buffers(ensemble)
    state = {name: p.detach().clone() for name, p in ensemble.named_parameters() if name.startswith('v10_')}
    out = _ensemble_forward(model, bs=2, nq=4, seed=7, enable=True, grad=True)
    loss_terms = [out[key].float().sum() for key in out if key.startswith('v10_') and isinstance(out[key], torch.Tensor)]
    if not loss_terms:
        errors.append('HG4 no V10 outputs available for dummy loss')
        return errors
    optim = torch.optim.SGD([p for p in ensemble.parameters() if p.requires_grad], lr=1e-4)
    optim.zero_grad()
    sum(loss_terms).backward()
    optim.step()
    after = _snapshot_buffers(ensemble)
    changed = []
    for name, old in before.items():
        new = after.get(name)
        if new is None or old.shape != new.shape or not torch.allclose(old.cpu(), new.detach().cpu(), atol=1e-7, rtol=0.0):
            changed.append(name)
    for name, param in ensemble.named_parameters():
        if name in state:
            param.data.copy_(state[name])
    print(f'  buffers={list(before.keys())}, changed={changed}')
    if changed:
        errors.append('HG4 buffers changed: ' + ', '.join(changed))
    return errors


def _check_forward_health(model) -> List[str]:
    print('[HG5a] V10 forward health')
    errors = []
    _head, ensemble = _get_head_and_ensemble(model)
    out = _ensemble_forward(model, bs=4, nq=8, seed=42, enable=True, grad=False)
    for key in V10_HEALTH_KEYS:
        tensor = out.get(key)
        if not isinstance(tensor, torch.Tensor):
            errors.append(f'HG5a missing tensor: {key}')
            continue
        if tensor.dim() < 3:
            errors.append(f'HG5a bad shape for {key}: {tuple(tensor.shape)}')
        if not torch.isfinite(tensor).all():
            errors.append(f'HG5a non-finite tensor: {key} stats={_tensor_stats(tensor)}')
    floor = float(getattr(ensemble, 'v10_cls_fused_keep_min', 0.0))
    keep_max = float(getattr(ensemble, 'v8_fused_keep_max', 1.0))
    for key in ('v10_w_fused_cls', 'v10_w_fused_height'):
        tensor = out[key]
        if tensor.min() < floor - 1e-6 or tensor.max() > keep_max + 1e-6:
            errors.append(f'HG5a {key} outside [{floor}, {keep_max}]: {_tensor_stats(tensor)}')
    if (out['v10_w_fused_center'] + 1e-6 < out['v10_center_floor']).any():
        errors.append('HG5a center fused keep is below CSG floor')
    if bool(getattr(ensemble, 'v10_edlcue_enable', False)):
        for key in ('v10_edlcue_admission_logits', 'v10_edlcue_delta'):
            tensor = out.get(key)
            if not isinstance(tensor, torch.Tensor):
                errors.append(f'HG5a missing tensor: {key}')
                continue
            if not torch.isfinite(tensor).all():
                errors.append(f'HG5a non-finite tensor: {key} stats={_tensor_stats(tensor)}')
        delta = out.get('v10_edlcue_delta')
        if isinstance(delta, torch.Tensor):
            max_delta = float(getattr(ensemble, 'v10_edlcue_max_delta', 0.0))
            if delta.abs().max() > max_delta + 1e-6:
                errors.append(
                    f'HG5a v10_edlcue_delta exceeds bound {max_delta}: {_tensor_stats(delta)}')
    for head in ('cls', 'height', 'center'):
        simplex = out[f'v10_w_fused_{head}'] + out[f'v10_w_bev_{head}'] + out[f'v10_w_img_{head}']
        max_err = float((simplex - 1.0).abs().max().cpu())
        print(f'  simplex_{head}_max_err={max_err:.8f}')
        if max_err > 1e-5:
            errors.append(f'HG5a routing simplex failed for {head}: {max_err}')
    return errors


def _v10_last_layer_stats(ensemble) -> Dict[str, Dict[str, float]]:
    stats = {}
    for name in (
            'v10_dd_head',
            'v10_router_head_cls',
            'v10_router_head_height',
            'v10_router_head_center',
            'v10_center_safe_gate'):
        module = getattr(ensemble, name, None)
        if module is None:
            continue
        linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            continue
        last = linear_layers[-1]
        stats[name] = {
            'weight_absmax': float(last.weight.detach().abs().max().cpu()),
            'weight_mean': float(last.weight.detach().mean().cpu()),
            'weight_std': float(last.weight.detach().std(unbiased=False).cpu()),
            'bias_absmax': float(last.bias.detach().abs().max().cpu()) if last.bias is not None else 0.0,
            'bias_mean': float(last.bias.detach().mean().cpu()) if last.bias is not None else 0.0,
            'bias_std': float(last.bias.detach().std(unbiased=False).cpu()) if last.bias is not None else 0.0,
        }
    return stats


def _dump_parity(path: str, ensemble, diffs: Dict[str, float], out_on, out_off):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'v10_last_layer': _v10_last_layer_stats(ensemble),
        'parity_diffs': diffs,
        'top_diff_keys': sorted(diffs.items(), key=lambda item: item[1], reverse=True)[:10],
        'v10_delta_stats': {
            key: _tensor_stats(value)
            for key, value in out_on.items()
            if key.startswith('v10_') and isinstance(value, torch.Tensor)
        },
        'v9_base_stats': {
            key: _tensor_stats(value)
            for key, value in out_off.items()
            if key in ('v9_fused_keep_unit', 'v8_fused_keep', 'v9_soft_weight_fused_cls') and isinstance(value, torch.Tensor)
        },
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def _check_bypass_parity(model, dump_path: str) -> List[str]:
    print('[HG5b] V10 self-bypass parity (v10_enable=False with init vs random V10 weights)')
    errors = []
    _head, ensemble = _get_head_and_ensemble(model)
    model.eval()

    v10_module_names = (
        'v10_dd_head', 'v10_router_head_cls', 'v10_router_head_height',
        'v10_router_head_center', 'v10_center_safe_gate')

    # Save original V10 weights (init=zero from step 1)
    saved_states = {}
    for name in v10_module_names:
        module = getattr(ensemble, name, None)
        if module is not None:
            saved_states[name] = {k: v.detach().clone() for k, v in module.state_dict().items()}

    # Run 1: V10 weights = init (zero)
    torch.manual_seed(42)
    with torch.no_grad():
        out_init = _ensemble_forward(model, bs=16, nq=4, seed=42, enable=False, grad=False)

    # Run 2: V10 weights = random non-zero
    rng = torch.Generator(device='cpu').manual_seed(123)
    for name in v10_module_names:
        module = getattr(ensemble, name, None)
        if module is None:
            continue
        for param in module.parameters():
            with torch.no_grad():
                noise = torch.randn(param.shape, generator=rng) * 0.5
                param.copy_(noise.to(device=param.device, dtype=param.dtype))

    torch.manual_seed(42)
    with torch.no_grad():
        out_rand = _ensemble_forward(model, bs=16, nq=4, seed=42, enable=False, grad=False)

    # Restore V10 weights to init state
    for name, state in saved_states.items():
        module = getattr(ensemble, name, None)
        if module is not None:
            module.load_state_dict(state)

    # Compare: V10 weights should NOT affect forward when v10_enable=False
    diffs = {}
    for key in PARITY_KEYS:
        if key not in out_init or key not in out_rand:
            errors.append(f'HG5b missing parity key: {key}')
            continue
        diff = float((out_init[key] - out_rand[key]).abs().max().cpu())
        diffs[key] = diff
    print(f'  diffs={diffs}')
    if any(value > 1e-4 for value in diffs.values()):
        _dump_parity(dump_path, ensemble, diffs, out_init, out_rand)
        errors.append(f'HG5b self-bypass parity failed; dump written to {dump_path}')
    return errors


def _size_iou(pred_dim, gt_dim):
    min_dim = torch.minimum(pred_dim, gt_dim)
    max_dim = torch.maximum(pred_dim, gt_dim).clamp(min=1e-6)
    return (min_dim.prod(dim=-1) / max_dim.prod(dim=-1).clamp(min=1e-6)).clamp(0.0, 1.0)


def _check_oracle_numerical() -> List[str]:
    print('[HG8] Oracle quality numerical check')
    errors = []
    torch.manual_seed(3)
    n_pairs = 12
    n_classes = 10
    cls_logits = torch.randn(3, n_pairs, n_classes)
    gt_cls = torch.randint(0, n_classes, (n_pairs,))
    pred_center = torch.randn(3, n_pairs, 2)
    gt_center = torch.randn(n_pairs, 2)
    pred_height = torch.randn(3, n_pairs, 1)
    gt_height = torch.randn(n_pairs, 1)
    pred_heading = torch.randn(3, n_pairs)
    gt_heading = torch.randn(n_pairs)
    pred_dim = torch.rand(3, n_pairs, 3) + 0.5
    gt_dim = torch.rand(n_pairs, 3) + 0.5
    pred_vel = torch.randn(3, n_pairs, 2)
    gt_vel = torch.randn(n_pairs, 2)
    gt_vel[0] = float('nan')

    r_center_by_class = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.5, 0.75])
    r_center = r_center_by_class[gt_cls].clamp(min=1e-6)
    gt_vel_fallback = torch.nan_to_num(gt_vel, nan=0.0)
    cls_logp = F.log_softmax(cls_logits, dim=-1)[..., gt_cls].diagonal(dim1=1, dim2=2)
    cls_logp = cls_logp.clamp(min=-20.0)
    quality = (
        5.0 * cls_logp
        - 2.0 * (pred_center - gt_center.unsqueeze(0)).abs().sum(dim=-1) / r_center.unsqueeze(0)
        - 1.0 * (pred_height - gt_height.unsqueeze(0)).abs().squeeze(-1) / 1.0
        - 1.0 * torch.sin(pred_heading - gt_heading.unsqueeze(0)).abs() / 1.0
        - 1.0 * (1.0 - _size_iou(pred_dim, gt_dim.unsqueeze(0)))
        - 1.0 * (pred_vel - gt_vel_fallback.unsqueeze(0)).norm(dim=-1) / 2.0
    )
    if not torch.isfinite(quality).all():
        errors.append('HG8 quality contains NaN/Inf')
    if quality.numel() < 1:
        errors.append('HG8 N_pairs must be >= 1')
    p_oracle = torch.softmax(quality / 0.5, dim=0)
    entropy = -(p_oracle * p_oracle.clamp(min=1e-8).log()).sum(dim=0)
    if not torch.isfinite(entropy).all() or float(entropy.min()) <= 0.0:
        errors.append('HG8 oracle softmax entropy collapsed or non-finite')
    zero_pair_quality = quality.new_zeros(0, requires_grad=True)
    zero_loss = zero_pair_quality.sum() * 0.0
    if not zero_loss.requires_grad:
        errors.append('HG8 zero-pair connected loss fallback is detached')
    print(f'  n_pairs={quality.shape[1]}, entropy_range=({float(entropy.min()):.6f}, {float(entropy.max()):.6f})')
    return errors


def _check_input_detach(model) -> List[str]:
    print('[HG_input_detach] Gradient leakage check')
    errors = []
    out = _ensemble_forward(model, bs=2, nq=4, seed=9, enable=True, grad=True)
    loss_terms = [value.float().sum() for key, value in out.items() if key.startswith('v10_') and isinstance(value, torch.Tensor)]
    if not loss_terms:
        return ['HG_input_detach no V10 outputs available']
    loss = sum(loss_terms)
    blacklist = [
        (name, param)
        for name, param in model.named_parameters()
        if _starts_with_any(name, DETECTION_BLACKLIST_PREFIXES)
    ]
    old_requires_grad = [param.requires_grad for _name, param in blacklist]
    for _name, param in blacklist:
        param.requires_grad_(True)
    params = [param for _name, param in blacklist]
    try:
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
        leaked = []
        for (name, _param), grad in zip(blacklist, grads):
            if grad is not None and grad.detach().abs().max() > 0:
                leaked.append(name)
    finally:
        for (_name, param), old_flag in zip(blacklist, old_requires_grad):
            param.requires_grad_(old_flag)
    print(f'  checked_blacklist_params={len(blacklist)}, leaked={leaked[:8]}')
    if leaked:
        errors.append('HG_input_detach blacklist params received grad: ' + ', '.join(leaked[:20]))
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='V10 config path')
    parser.add_argument(
        '--profile', choices=('r50_v10', 'vov_v10'), default='r50_v10')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint to load; defaults to cfg.load_from')
    parser.add_argument('--source-checkpoint', default=None)
    parser.add_argument('--calibration', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Use synthetic inputs and tolerate missing checkpoint')
    parser.add_argument(
        '--config-only', action='store_true',
        help='Audit the config/calibration/checkpoint lineage without model build.')
    parser.add_argument('--dump-path', default='work_dirs/v10_preflight_dump.json')
    parser.add_argument('--no-load', action='store_true', help='Skip checkpoint loading')
    parser.add_argument(
        '--require-full-checkpoint-coverage', action='store_true',
        help='Require exact persistent checkpoint key and shape coverage.')
    parser.add_argument(
        '--expected-checkpoint-epoch', type=int, default=None,
        help='Require checkpoint meta.epoch to equal this value.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint or cfg.get('load_from')
    errors: List[str] = []
    errors.extend(_check_config(
        cfg, checkpoint, args.dry_run, args.profile,
        explicit_checkpoint=bool(args.checkpoint)))
    if args.profile == 'vov_v10':
        errors.extend(_check_vov_calibration_files(
            cfg, args.calibration, args.source_checkpoint))
    if args.config_only:
        if args.require_full_checkpoint_coverage:
            errors.append(
                '--require-full-checkpoint-coverage cannot be used with --config-only')
        if args.expected_checkpoint_epoch is not None:
            errors.append('--expected-checkpoint-epoch cannot be used with --config-only')
        if args.no_load:
            errors.append('--no-load is redundant with --config-only')
        if errors:
            print('\n[V10 Config Preflight] FAIL')
            for error in errors:
                print(f'  - {error}')
            raise SystemExit(1)
        print('\n[V10 Config Preflight] PASS')
        return

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    _head, ensemble = _get_head_and_ensemble(model)
    ensemble.init_weights()
    ckpt_path = _full_path(checkpoint)
    loaded_checkpoint = None
    if args.no_load and args.require_full_checkpoint_coverage:
        errors.append('--require-full-checkpoint-coverage cannot be used with --no-load')
    if args.no_load and args.expected_checkpoint_epoch is not None:
        errors.append('--expected-checkpoint-epoch cannot be used with --no-load')
    if checkpoint and os.path.exists(ckpt_path) and not args.no_load:
        print(f'[Load] checkpoint={checkpoint}')
        loaded_checkpoint = load_checkpoint(
            model, ckpt_path, map_location='cpu', strict=False)
        if args.expected_checkpoint_epoch is not None:
            errors.extend(_check_checkpoint_epoch(
                loaded_checkpoint, args.expected_checkpoint_epoch))
        if args.require_full_checkpoint_coverage:
            errors.extend(_check_checkpoint_coverage(model, loaded_checkpoint))
    elif not args.no_load and not args.dry_run:
        errors.append(f'checkpoint is required and missing: {ckpt_path}')
    else:
        print('[Load] skipped checkpoint load for dry-run/no-load')

    errors.extend(_check_trainable(cfg, model))
    errors.extend(_check_bn_freeze(cfg, model))
    errors.extend(_check_forward_health(model))
    errors.extend(_check_bypass_parity(model, args.dump_path))
    errors.extend(_check_input_detach(model))
    errors.extend(_check_buffer_freeze(model))
    errors.extend(_check_oracle_numerical())

    if errors:
        print('\n[V10 Preflight] FAIL')
        for error in errors:
            print(f'  - {error}')
        raise SystemExit(1)
    print('\n[V10 Preflight] PASS')


if __name__ == '__main__':
    main()
