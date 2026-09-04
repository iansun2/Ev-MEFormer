#!/usr/bin/env python
"""Preflight checks for Rev20 A4-frozen routing-only configs.

This script covers the non-evaluation hard gates:

HG0: config sanity for router variant / anchor freeze / tau update.
HG1: trainable params must match the Rev20-A whitelist.
HG2: detection-related params must remain frozen.
HG4: A4 anchor readiness plus tau/anchor update-buffer guards.

For newly trained chains, ``--require-full-checkpoint-coverage`` additionally
requires exact persistent state-dict key and tensor-shape coverage. MMCV 1.6
also serializes nonpersistent buffers, unlike ``torch.nn.Module.state_dict``;
the four known config-derived distance-bin buffers are accepted only after an
exact runtime type/shape/dtype/value match. It is opt-in so that historical A4
checkpoints created before later diagnostic heads were added can still use the
legacy preflight behavior.

HG3 (clean + motionblur-s3 metric parity with A4) still requires the normal
validation/corruption probe pipeline.
"""

import argparse
import os
import sys
from numbers import Integral
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from mmcv import Config
from mmcv.utils import build_from_cfg
from mmcv.runner.hooks import HOOKS
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import projects.mmdet3d_plugin  # noqa: F401,E402  Registers custom modules.


REV20_A_PREFIXES = (
    'pts_bbox_head.ensemble.v9_bev_anomaly_head',
    'pts_bbox_head.ensemble.v9_img_anomaly_head',
    'pts_bbox_head.ensemble.v9_fused_gate_head',
)

OPTIMIZER_ZERO_GUARD_PREFIXES = (
    'img_backbone',
    'img_neck',
    'pts_voxel_encoder',
    'pts_middle_encoder',
    'pts_backbone',
    'pts_neck',
    'pts_bbox_head',
)

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

MMCV_NONPERSISTENT_COMPAT_KEYS = (
    'pts_bbox_head.ensemble.distance_bin_edges_tensor',
    'pts_bbox_head.ensemble.v5_distance_bins_tensor',
    'pts_bbox_head.ensemble.v6_distance_bins_tensor',
    'pts_bbox_head.ensemble.v7_distance_bins_tensor',
)


def _as_dict(obj):
    return obj if isinstance(obj, dict) else dict(obj)


def _find_hook(cfg: Config, hook_type: str):
    for hook in cfg.get('custom_hooks', []):
        if hook.get('type') == hook_type:
            return hook
    return None


def _starts_with_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(
        name == prefix or name.startswith(prefix + '.')
        for prefix in prefixes
    )


def _normalize_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Checkpoint must be a dict, got {type(checkpoint)!r}')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        raise TypeError('Checkpoint state_dict is not a mapping')

    normalized = {}
    for name, tensor in state.items():
        clean_name = name[7:] if name.startswith('module.') else name
        if clean_name in normalized:
            raise ValueError(
                f'Checkpoint key collision after module-prefix removal: {clean_name}')
        normalized[clean_name] = tensor
    return normalized


def _validate_nonpersistent_compat_key(
        model, name: str, checkpoint_value) -> str:
    """Return an error string unless an MMCV-only buffer is an exact match."""
    module_name, buffer_name = name.rsplit('.', 1)
    owner = dict(model.named_modules()).get(module_name)
    if owner is None:
        return f'{name}: runtime owner module does not exist'
    if buffer_name not in owner._buffers:
        return f'{name}: runtime leaf is not a registered buffer'
    if buffer_name not in owner._non_persistent_buffers_set:
        return f'{name}: runtime buffer is not marked nonpersistent'

    runtime_value = owner._buffers[buffer_name]
    if not isinstance(checkpoint_value, torch.Tensor):
        return f'{name}: checkpoint value is not a tensor'
    if not isinstance(runtime_value, torch.Tensor):
        return f'{name}: runtime value is not a tensor'
    if tuple(checkpoint_value.shape) != tuple(runtime_value.shape):
        return (
            f'{name}: shape mismatch checkpoint={tuple(checkpoint_value.shape)} '
            f'runtime={tuple(runtime_value.shape)}')
    if checkpoint_value.dtype != runtime_value.dtype:
        return (
            f'{name}: dtype mismatch checkpoint={checkpoint_value.dtype} '
            f'runtime={runtime_value.dtype}')
    if not torch.equal(
            checkpoint_value.detach().cpu(), runtime_value.detach().cpu()):
        return f'{name}: value mismatch against the config-derived runtime buffer'
    return ''


def _check_checkpoint_coverage(model, checkpoint) -> List[str]:
    errors = []
    try:
        checkpoint_state = _normalize_state_dict(checkpoint)
    except (TypeError, ValueError) as exc:
        return [str(exc)]

    model_state = model.state_dict()
    raw_unexpected = sorted(set(checkpoint_state) - set(model_state))
    missing = sorted(set(model_state) - set(checkpoint_state))
    mismatched = sorted(
        name for name in set(checkpoint_state).intersection(model_state)
        if (not hasattr(checkpoint_state[name], 'shape') or
            tuple(checkpoint_state[name].shape) != tuple(model_state[name].shape))
    )

    compatible_nonpersistent = []
    incompatible_nonpersistent = []
    for name in raw_unexpected:
        if name not in MMCV_NONPERSISTENT_COMPAT_KEYS:
            continue
        error = _validate_nonpersistent_compat_key(
            model, name, checkpoint_state[name])
        if error:
            incompatible_nonpersistent.append(error)
        else:
            compatible_nonpersistent.append(name)
    unexpected = sorted(
        set(raw_unexpected) - set(compatible_nonpersistent)
        - {item.split(':', 1)[0] for item in incompatible_nonpersistent}
    )

    print('[Checkpoint] Full persistent state-dict coverage')
    print(f'  checkpoint_keys={len(checkpoint_state):,}')
    print(f'  model_keys={len(model_state):,}')
    print(f'  missing={len(missing):,}')
    print(f'  unexpected={len(unexpected):,}')
    print(f'  shape_mismatches={len(mismatched):,}')
    print(f'  compatible_nonpersistent={len(compatible_nonpersistent):,}')
    print(f'  incompatible_nonpersistent={len(incompatible_nonpersistent):,}')
    for name in compatible_nonpersistent:
        print(f'    compatible: {name}')

    if missing:
        errors.append('Missing checkpoint keys: ' + ', '.join(missing[:20]))
    if unexpected:
        errors.append('Unexpected checkpoint keys: ' + ', '.join(unexpected[:20]))
    if incompatible_nonpersistent:
        errors.append(
            'Incompatible known nonpersistent checkpoint buffers: ' +
            '; '.join(incompatible_nonpersistent[:20]))
    if mismatched:
        details = []
        for name in mismatched[:20]:
            checkpoint_shape = getattr(checkpoint_state[name], 'shape', None)
            details.append(
                f'{name}: checkpoint={checkpoint_shape}, '
                f'model={tuple(model_state[name].shape)}')
        errors.append('Tensor shape mismatches: ' + '; '.join(details))
    return errors


def _check_checkpoint_epoch(checkpoint, expected_epoch: int) -> List[str]:
    meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    actual_epoch = meta.get('epoch') if isinstance(meta, dict) else None
    print('[Checkpoint] Epoch metadata')
    print(f'  expected={expected_epoch}, actual={actual_epoch}')
    if (isinstance(actual_epoch, bool) or
            not isinstance(actual_epoch, Integral) or
            actual_epoch != expected_epoch):
        return [
            'Checkpoint meta.epoch mismatch: '
            f'expected={expected_epoch}, actual={actual_epoch}'
        ]
    return []


class _PrintLogger:
    def info(self, msg):
        # FreezeWeight logs every frozen parameter, which makes preflight output
        # unreadable. The explicit trainable whitelist report below is the
        # auditable summary.
        pass


class _DummyRunner:
    def __init__(self, model):
        self.model = model
        self.logger = _PrintLogger()


def _apply_actual_freeze_hook(cfg: Config, model) -> List[str]:
    freeze_hook_cfg = _find_hook(cfg, 'FreezeWeight')
    if freeze_hook_cfg is None:
        return ['FreezeWeight hook is missing']
    hook = build_from_cfg(dict(freeze_hook_cfg), HOOKS)
    hook.before_run(_DummyRunner(model))
    return []


def _snapshot_buffers(module) -> Dict[str, torch.Tensor]:
    snapshot = {}
    for name in BUFFER_NAMES:
        buf = getattr(module, name, None)
        if isinstance(buf, torch.Tensor):
            snapshot[name] = buf.detach().clone()
    return snapshot


def _assert_buffers_equal(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> List[str]:
    changed = []
    for name, old in before.items():
        new = after.get(name)
        if new is None or old.shape != new.shape or not torch.equal(old.cpu(), new.detach().cpu()):
            changed.append(name)
    return changed


def _check_config(cfg: Config, expected_prefixes: Sequence[str]) -> List[str]:
    errors = []
    ensemble_cfg = cfg.model['pts_bbox_head']['ensemble']
    router_variant = ensemble_cfg.get('v8_router_variant')
    update_tau_online = ensemble_cfg.get('update_tau_online')
    freeze_anchor_epochs = ensemble_cfg.get('v9_freeze_clean_anchor_epochs')

    print('[HG0] Config values')
    print(f'  v8_router_variant={router_variant}')
    print(f'  update_tau_online={update_tau_online}')
    print(f'  v9_freeze_clean_anchor_epochs={freeze_anchor_epochs}')

    if router_variant != 'v9_lite_domain_anchored':
        errors.append('v8_router_variant must be v9_lite_domain_anchored')
    if update_tau_online is not False:
        errors.append('update_tau_online must be False for Rev20-A')
    if int(freeze_anchor_epochs or 0) < 999:
        errors.append('v9_freeze_clean_anchor_epochs must be >= 999')

    backbone_type = cfg.model.img_backbone.get('type')
    samples_per_gpu = int(cfg.data.get('samples_per_gpu', 0))
    cumulative_iters = int(cfg.optimizer_config.get('cumulative_iters', 1))
    warmup_iters = int(cfg.lr_config.get('warmup_iters', 0))
    checkpoint_interval = int(cfg.checkpoint_config.get('interval', 0))
    print(f'  img_backbone.type={backbone_type}')
    print(f'  total_epochs={cfg.get("total_epochs")}')
    print(f'  samples_per_gpu={samples_per_gpu}')
    print(f'  cumulative_iters={cumulative_iters}')
    print(f'  lr_warmup_iters={warmup_iters}')
    print(f'  checkpoint_interval={checkpoint_interval}')
    if int(cfg.get('total_epochs', 0)) != 4:
        errors.append('Rev20-A total_epochs must be 4')
    expected_batch_contract = {
        'ResNet': (4, 1, 500),
        'VoVNet': (2, 2, 1000),
    }.get(backbone_type)
    actual_batch_contract = (
        samples_per_gpu, cumulative_iters, warmup_iters)
    if expected_batch_contract is None:
        errors.append(
            f'Unsupported Rev20-A image backbone type: {backbone_type}')
    elif actual_batch_contract != expected_batch_contract:
        errors.append(
            f'Rev20-A {backbone_type} batch/warmup contract must be '
            f'{expected_batch_contract}, got {actual_batch_contract}')
    if checkpoint_interval != 1:
        errors.append('Rev20-A checkpoint_config.interval must be 1')

    corrupt_steps = []
    for step in cfg.data.train.dataset.pipeline:
        if step.get('type') == 'OnlineSyntheticMultiCorrupt':
            corrupt_steps.append(step)
    if len(corrupt_steps) != 1:
        errors.append(
            'Rev20-A must contain exactly one OnlineSyntheticMultiCorrupt '
            f'step, got {len(corrupt_steps)}')
    else:
        corrupt = corrupt_steps[0]
        corrupt_prob = float(corrupt.get('corrupt_prob', -1.0))
        print(f'  corrupt_prob={corrupt_prob}')
        if abs(corrupt_prob - 0.60) > 1e-12:
            errors.append('Rev20-A corrupt_prob must be 0.60')

    freeze_hook = _find_hook(cfg, 'FreezeWeight')
    if freeze_hook is None:
        errors.append('FreezeWeight hook is missing')
    else:
        prefixes = tuple(freeze_hook.get('finetune_weight', []))
        print(f'  FreezeWeight prefixes={prefixes}')
        if prefixes != tuple(expected_prefixes):
            errors.append(
                'FreezeWeight finetune_weight must exactly match Rev20-A prefixes: '
                f'{expected_prefixes}')
        if not freeze_hook.get('freeze_bn', False):
            errors.append('FreezeWeight.freeze_bn must be True')

    custom_keys = cfg.optimizer.get('paramwise_cfg', {}).get('custom_keys', {})
    print(f'  optimizer custom_keys={tuple(custom_keys.keys())}')
    for prefix in expected_prefixes:
        opts = custom_keys.get(prefix)
        if opts is None:
            errors.append(f'Optimizer guard missing trainable V9 prefix: {prefix}')
        elif float(opts.get('lr_mult', 1.0)) != 1.0:
            errors.append(f'Optimizer lr_mult for {prefix} must be 1.0')
    for prefix in OPTIMIZER_ZERO_GUARD_PREFIXES:
        opts = custom_keys.get(prefix)
        if opts is None:
            errors.append(f'Optimizer zero-lr guard missing prefix: {prefix}')
        else:
            if float(opts.get('lr_mult', 1.0)) != 0.0:
                errors.append(f'Optimizer lr_mult for {prefix} must be 0.0')
            if float(opts.get('decay_mult', 1.0)) != 0.0:
                errors.append(f'Optimizer decay_mult for {prefix} must be 0.0')

    edl_hook = _find_hook(cfg, 'EDLEpochHook')
    if edl_hook is None:
        errors.append('EDLEpochHook is missing')
    else:
        schedule = edl_hook.get('v9_keep_min_schedule', [])
        print(f'  v9_keep_min_schedule={schedule}')
        min_keep = min(float(item['value']) for item in schedule) if schedule else None
        if min_keep is None or min_keep < 0.85:
            errors.append('Rev20-A keep_min schedule must not go below 0.85')

    return errors


def _check_trainable(cfg: Config, model, expected_prefixes: Sequence[str]) -> List[str]:
    errors = []
    errors.extend(_apply_actual_freeze_hook(cfg, model))
    trainable: List[Tuple[str, Tuple[int, ...], int]] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, tuple(param.shape), int(param.numel())))

    print('[HG1/HG2] Trainable parameters')
    for name, shape, count in trainable:
        print(f'  {name}: shape={shape}, params={count:,}')
    total = sum(count for _, _, count in trainable)
    all_params = sum(int(param.numel()) for param in model.parameters())
    ratio = 100.0 * total / max(all_params, 1)
    print(f'  total_trainable={total:,} / {all_params:,} ({ratio:.4f}%)')

    if not trainable:
        errors.append('No trainable parameters matched Rev20-A prefixes')
    for name, _shape, _count in trainable:
        if not _starts_with_any(name, expected_prefixes):
            errors.append(f'Unexpected trainable parameter outside whitelist: {name}')
        if _starts_with_any(name, DETECTION_BLACKLIST_PREFIXES):
            errors.append(f'Detection-related parameter is trainable: {name}')
    for prefix in expected_prefixes:
        if not any(
                _starts_with_any(name, (prefix,))
                for name, _, _ in trainable):
            errors.append(f'Expected trainable prefix has no parameters: {prefix}')
    if ratio > 2.0:
        errors.append(f'Trainable ratio too high for Rev20-A: {ratio:.4f}%')
    return errors


@torch.no_grad()
def _check_buffer_guards(model) -> List[str]:
    errors = []
    ensemble = model.pts_bbox_head.ensemble
    ensemble.train()
    ensemble._current_epoch = 0

    before = _snapshot_buffers(ensemble)
    missing_buffers = sorted(set(BUFFER_NAMES) - set(before))
    if missing_buffers:
        return ['Missing required Rev20 buffers: ' + ', '.join(missing_buffers)]

    # Exercise tau update guard.
    a_bev = torch.full((2, 4, 1), 0.3)
    a_img = torch.full((2, 4, 1), 0.3)
    d_bev = torch.full((2, 4, 1), 0.3)
    d_img = torch.full((2, 4, 1), 0.3)
    z_dim = int(getattr(ensemble, 'v8_a4_anchor_mean').shape[-1])
    bev_z = torch.zeros(2, 4, z_dim)
    img_z = torch.zeros(2, 4, z_dim)
    clean_mask = torch.ones(2, dtype=torch.bool)
    ensemble._v9_update_clean_thresholds(
        a_bev, a_img, d_bev, d_img, bev_z, img_z, clean_mask)

    # Exercise A4 anchor update guard.
    bev_features = torch.ones(2, 4, z_dim)
    img_features = torch.ones(2, 4, z_dim)
    ensemble._v8_update_a4_clean_anchor(bev_features, img_features, clean_mask)

    after = _snapshot_buffers(ensemble)
    changed = _assert_buffers_equal(before, after)
    print('[HG4] Buffer guard check')
    for name in BUFFER_NAMES:
        if name in before:
            print(f'  {name}: unchanged={name not in changed}')
    if changed:
        errors.append('Buffers changed during guarded update calls: ' + ', '.join(changed))
    return errors


@torch.no_grad()
def _check_a4_anchor_ready(model) -> List[str]:
    errors = []
    ensemble = model.pts_bbox_head.ensemble
    count = getattr(ensemble, 'v8_a4_anchor_count', None)
    mean = getattr(ensemble, 'v8_a4_anchor_mean', None)
    variance = getattr(ensemble, 'v8_a4_anchor_var', None)

    print('[HG4] A4 anchor readiness')
    if not isinstance(count, torch.Tensor):
        return ['v8_a4_anchor_count buffer is missing']
    count_float = count.detach().to(dtype=torch.float64)
    count_min = float(count_float.min().item()) if count_float.numel() else 0.0
    print(f'  v8_a4_anchor_count_min={count_min}')
    if count_float.numel() == 0 or not torch.isfinite(count_float).all():
        errors.append('v8_a4_anchor_count must be non-empty and finite')
    elif count_min <= 0.0:
        errors.append('v8_a4_anchor_count must be > 0 before freezing the anchor')

    for name, tensor in (
            ('v8_a4_anchor_mean', mean), ('v8_a4_anchor_var', variance)):
        finite = isinstance(tensor, torch.Tensor) and bool(
            tensor.numel() > 0 and torch.isfinite(tensor).all())
        print(f'  {name}_finite={finite}')
        if not finite:
            errors.append(f'{name} must be non-empty and finite')
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Rev20 config path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint to load. Defaults to cfg.load_from when available.')
    parser.add_argument(
        '--no-load',
        action='store_true',
        help='Do not load a checkpoint; useful for pure config/model checks.')
    parser.add_argument(
        '--require-full-checkpoint-coverage',
        action='store_true',
        help=(
            'Require exact persistent model/checkpoint keys and tensor shapes; '
            'known MMCV-only nonpersistent buffers must exactly match runtime.'))
    parser.add_argument(
        '--expected-checkpoint-epoch',
        type=int,
        default=None,
        help='Require checkpoint meta.epoch to equal this value.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint or cfg.get('load_from')

    errors = _check_config(cfg, REV20_A_PREFIXES)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    loaded_checkpoint = None
    if args.no_load and args.require_full_checkpoint_coverage:
        errors.append(
            '--require-full-checkpoint-coverage cannot be used with --no-load')
    if args.no_load and args.expected_checkpoint_epoch is not None:
        errors.append(
            '--expected-checkpoint-epoch cannot be used with --no-load')
    if checkpoint and not args.no_load:
        print(f'[Load] checkpoint={checkpoint}')
        loaded_checkpoint = load_checkpoint(
            model, checkpoint, map_location='cpu', strict=False)
        if args.expected_checkpoint_epoch is not None:
            errors.extend(_check_checkpoint_epoch(
                loaded_checkpoint, args.expected_checkpoint_epoch))
        if args.require_full_checkpoint_coverage:
            errors.extend(_check_checkpoint_coverage(model, loaded_checkpoint))
        errors.extend(_check_a4_anchor_ready(model))
    elif not args.no_load:
        errors.append('No checkpoint provided and cfg.load_from is empty')

    errors.extend(_check_trainable(cfg, model, REV20_A_PREFIXES))
    errors.extend(_check_buffer_guards(model))

    if errors:
        print('\n[Rev20 Preflight] FAIL')
        for error in errors:
            print(f'  - {error}')
        raise SystemExit(1)
    print('\n[Rev20 Preflight] PASS')


if __name__ == '__main__':
    main()
