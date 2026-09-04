#!/usr/bin/env python
"""Preflight the EUD-VoV -> A4 precursor transition.

The check is intentionally CPU-only and covers the transition invariants that
are otherwise easy to miss when porting the R50 UMR lineage to VoV:

1. VoV image/voxel geometry and finalized EUD loss settings are preserved.
2. The A4 PME ensemble is enabled with the expected online corruption setup.
3. FreezeWeight leaves only ``pts_bbox_head.ensemble`` trainable.
4. A Stage-1 EUD checkpoint differs from the A4 model only by new ensemble
   keys; unexpected keys and tensor-shape mismatches are hard failures.
"""

import argparse
import os
import sys
from numbers import Integral
from typing import Dict, Iterable, List, Tuple

import torch
from mmcv import Config
from mmcv.runner.hooks import HOOKS
from mmcv.utils import build_from_cfg
from mmdet3d.models import build_model


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import projects.mmdet3d_plugin  # noqa: E402,F401  Registers custom modules.


ENSEMBLE_PREFIX = 'pts_bbox_head.ensemble'
OPTIMIZER_ZERO_GUARD_PREFIXES = (
    'img_backbone',
    'img_neck',
    'pts_voxel_encoder',
    'pts_middle_encoder',
    'pts_backbone',
    'pts_neck',
    'pts_bbox_head',
)


class _QuietLogger:
    def info(self, _message):
        pass


class _DummyRunner:
    def __init__(self, model):
        self.model = model
        self.logger = _QuietLogger()


def _find_hook(cfg: Config, hook_type: str):
    for hook in cfg.get('custom_hooks', []):
        if hook.get('type') == hook_type:
            return hook
    return None


def _find_pipeline_step(cfg: Config, step_type: str):
    matches = [
        step for step in cfg.data.train.dataset.pipeline
        if step.get('type') == step_type
    ]
    return matches[0] if len(matches) == 1 else None


def _count_pipeline_steps(cfg: Config, step_type: str) -> int:
    return sum(
        step.get('type') == step_type
        for step in cfg.data.train.dataset.pipeline)


def _load_checkpoint(path: str):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')

    if not isinstance(checkpoint, dict):
        raise TypeError(f'Checkpoint must contain a dict, got {type(checkpoint)!r}')
    return checkpoint


def _normalize_checkpoint_state(checkpoint) -> Dict[str, torch.Tensor]:
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        raise TypeError('Checkpoint state_dict is not a mapping')

    normalized = {}
    for name, tensor in state.items():
        clean_name = name[7:] if name.startswith('module.') else name
        if clean_name in normalized:
            raise ValueError(
                'Checkpoint key collision after module-prefix removal: '
                f'{clean_name}')
        normalized[clean_name] = tensor
    return normalized


def _is_close_list(actual: Iterable[float], expected: Iterable[float]) -> bool:
    actual = list(actual)
    expected = list(expected)
    return len(actual) == len(expected) and all(
        abs(float(left) - float(right)) < 1e-9
        for left, right in zip(actual, expected)
    )


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


def _check_config(cfg: Config) -> List[str]:
    errors = []
    model = cfg.model
    head = model.pts_bbox_head
    ensemble = head.get('ensemble', {})

    print('[Config] VoV/A4 invariants')
    checks: List[Tuple[str, object]] = [
        ('img_backbone.type', model.img_backbone.get('type')),
        ('img_neck.in_channels', list(model.img_neck.get('in_channels', []))),
        ('voxel_size', list(model.pts_voxel_layer.get('voxel_size', []))),
        ('sparse_shape', list(model.pts_middle_encoder.get('sparse_shape', []))),
        ('use_ensemble', head.get('use_ensemble')),
        ('router_variant', ensemble.get('v8_router_variant')),
        ('samples_per_gpu', cfg.data.get('samples_per_gpu')),
        ('cumulative_iters', cfg.optimizer_config.get('cumulative_iters')),
        ('lr_warmup_iters', cfg.lr_config.get('warmup_iters')),
        ('total_epochs', cfg.get('total_epochs')),
        ('checkpoint_interval', cfg.checkpoint_config.get('interval')),
    ]
    for name, value in checks:
        print(f'  {name}={value}')

    if model.img_backbone.get('type') != 'VoVNet':
        errors.append('img_backbone.type must be VoVNet')
    if list(model.img_neck.get('in_channels', [])) != [768, 1024]:
        errors.append('VoV CPFPN in_channels must be [768, 1024]')
    if not _is_close_list(
            model.pts_voxel_layer.get('voxel_size', []), [0.075, 0.075, 0.2]):
        errors.append('voxel_size must be [0.075, 0.075, 0.2]')
    if list(model.pts_middle_encoder.get('sparse_shape', [])) != [41, 1440, 1440]:
        errors.append('sparse_shape must be [41, 1440, 1440]')
    if head.get('use_ensemble') is not True:
        errors.append('pts_bbox_head.use_ensemble must be True')
    if ensemble.get('type') != 'EDLPMETransformer':
        errors.append('ensemble.type must be EDLPMETransformer')
    if ensemble.get('v8_router_variant') != 'a4_clean_anchor_domain_reliability':
        errors.append('A4 router variant is incorrect')
    if list(head.get('modalities', {}).get('test', [])) != ['fused', 'bev', 'img']:
        errors.append('Stage-2 test modalities must expose fused/bev/img')

    loss_cfg = head.get('loss_edl_weights', {})
    expected_loss_values = {
        'nig_nll_weight': 0.3,
        'nig_reg_weight': 0.0001,
        'cls_edl_weight': 0.5,
        'cls_kl_weight': 0.1,
    }
    for key, expected in expected_loss_values.items():
        actual = loss_cfg.get(key)
        print(f'  loss_edl_weights.{key}={actual}')
        if actual is None or abs(float(actual) - expected) > 1e-12:
            errors.append(f'Finalized EUD value changed: {key}={actual}')
    if head.get('use_l1_quality_cls') is not True:
        errors.append('Finalized EUD use_l1_quality_cls must remain enabled')
    if head.get('use_dynamic_kl') is not True:
        errors.append('Finalized EUD use_dynamic_kl must remain enabled')

    resize = _find_pipeline_step(cfg, 'ResizeCropFlipImage')
    corrupt = _find_pipeline_step(cfg, 'OnlineSyntheticMultiCorrupt')
    resize_count = _count_pipeline_steps(cfg, 'ResizeCropFlipImage')
    corrupt_count = _count_pipeline_steps(cfg, 'OnlineSyntheticMultiCorrupt')
    if resize_count != 1:
        errors.append(
            'A4 must contain exactly one ResizeCropFlipImage step, '
            f'got {resize_count}')
    if resize is None:
        errors.append('ResizeCropFlipImage is missing from train pipeline')
    else:
        final_dim = tuple(resize.get('data_aug_conf', {}).get('final_dim', ()))
        resize_lim = tuple(resize.get('data_aug_conf', {}).get('resize_lim', ()))
        print(f'  final_dim={final_dim}, resize_lim={resize_lim}')
        if final_dim != (640, 1600):
            errors.append('VoV train final_dim must be (640, 1600)')
        if resize_lim != (0.94, 1.25):
            errors.append('VoV train resize_lim must be (0.94, 1.25)')
    if corrupt_count != 1:
        errors.append(
            'A4 must contain exactly one OnlineSyntheticMultiCorrupt step, '
            f'got {corrupt_count}')
    if corrupt is None:
        errors.append('OnlineSyntheticMultiCorrupt is missing or duplicated')
    elif abs(float(corrupt.get('corrupt_prob', -1)) - 0.50) > 1e-12:
        errors.append('A4 corrupt_prob must be 0.50')

    if int(cfg.data.get('samples_per_gpu', 0)) != 2:
        errors.append('VoV samples_per_gpu must be 2')
    if int(cfg.optimizer_config.get('cumulative_iters', 0)) != 2:
        errors.append('VoV cumulative_iters must be 2')
    if int(cfg.lr_config.get('warmup_iters', 0)) != 1000:
        errors.append('VoV A4 LR warmup must be 1000 accumulated iterations')
    if int(cfg.get('total_epochs', 0)) != 6:
        errors.append('VoV A4 total_epochs must be 6')
    if int(cfg.checkpoint_config.get('interval', 0)) != 1:
        errors.append('VoV A4 checkpoint_config.interval must be 1')

    freeze_hook = _find_hook(cfg, 'FreezeWeight')
    if freeze_hook is None:
        errors.append('FreezeWeight hook is missing')
    else:
        prefixes = list(freeze_hook.get('finetune_weight', []))
        print(f'  FreezeWeight prefixes={prefixes}')
        if prefixes != [ENSEMBLE_PREFIX]:
            errors.append('A4 FreezeWeight must train only pts_bbox_head.ensemble')
        if freeze_hook.get('freeze_bn') is not True:
            errors.append('A4 FreezeWeight.freeze_bn must be True')

    custom_keys = cfg.optimizer.get('paramwise_cfg', {}).get('custom_keys', {})
    train_opts = custom_keys.get(ENSEMBLE_PREFIX)
    if train_opts is None or float(train_opts.get('lr_mult', 0.0)) != 1.0:
        errors.append('Optimizer ensemble lr guard is missing or not 1.0')
    for prefix in OPTIMIZER_ZERO_GUARD_PREFIXES:
        opts = custom_keys.get(prefix)
        if opts is None:
            errors.append(f'Optimizer zero-LR guard missing: {prefix}')
        elif (float(opts.get('lr_mult', 1.0)) != 0.0 or
              float(opts.get('decay_mult', 1.0)) != 0.0):
            errors.append(f'Optimizer guard must set LR/decay to zero: {prefix}')
    return errors


def _check_trainable(cfg: Config, model) -> List[str]:
    errors = []
    hook_cfg = _find_hook(cfg, 'FreezeWeight')
    if hook_cfg is None:
        return ['Cannot apply missing FreezeWeight hook']
    hook = build_from_cfg(dict(hook_cfg), HOOKS)
    hook.before_run(_DummyRunner(model))

    trainable = [
        (name, tuple(param.shape), int(param.numel()))
        for name, param in model.named_parameters() if param.requires_grad
    ]
    print('[Freeze] Trainable parameters')
    total = 0
    for name, shape, count in trainable:
        print(f'  {name}: shape={shape}, params={count:,}')
        total += count
        if not name.startswith(ENSEMBLE_PREFIX + '.'):
            errors.append(f'Unexpected trainable parameter: {name}')
    print(f'  total_trainable={total:,}')
    if not trainable:
        errors.append('No A4 ensemble parameters remain trainable')
    return errors


def _check_checkpoint(
        model, checkpoint_path: str, expected_epoch: int = None) -> List[str]:
    errors = []
    try:
        checkpoint = _load_checkpoint(checkpoint_path)
        checkpoint_state = _normalize_checkpoint_state(checkpoint)
    except (TypeError, ValueError) as exc:
        return [str(exc)]

    if expected_epoch is not None:
        errors.extend(_check_checkpoint_epoch(checkpoint, expected_epoch))
    model_state = model.state_dict()

    unexpected = sorted(set(checkpoint_state) - set(model_state))
    missing = sorted(set(model_state) - set(checkpoint_state))
    mismatched = sorted(
        name for name in set(checkpoint_state).intersection(model_state)
        if tuple(checkpoint_state[name].shape) != tuple(model_state[name].shape)
    )
    missing_outside_ensemble = [
        name for name in missing if not name.startswith(ENSEMBLE_PREFIX + '.')
    ]
    checkpoint_ensemble_keys = sorted(
        name for name in checkpoint_state
        if name.startswith(ENSEMBLE_PREFIX + '.'))
    model_ensemble_keys = {
        name for name in model_state
        if name.startswith(ENSEMBLE_PREFIX + '.')}

    print('[Checkpoint] Stage-1 EUD -> A4 coverage')
    print(f'  checkpoint_keys={len(checkpoint_state):,}')
    print(f'  model_keys={len(model_state):,}')
    print(f'  missing={len(missing):,} (expected: new ensemble only)')
    print(f'  unexpected={len(unexpected):,}')
    print(f'  shape_mismatches={len(mismatched):,}')
    print(f'  checkpoint_ensemble_keys={len(checkpoint_ensemble_keys):,}')

    if checkpoint_ensemble_keys:
        errors.append(
            'Stage-1 checkpoint must not contain ensemble keys: ' +
            ', '.join(checkpoint_ensemble_keys[:20]))
    if set(missing) != model_ensemble_keys:
        errors.append(
            'Missing-key set must exactly equal the new A4 ensemble state')
    if missing_outside_ensemble:
        errors.append(
            'Checkpoint misses non-ensemble keys: ' +
            ', '.join(missing_outside_ensemble[:20]))
    if unexpected:
        errors.append('Unexpected checkpoint keys: ' + ', '.join(unexpected[:20]))
    if mismatched:
        details = [
            f'{name}: checkpoint={tuple(checkpoint_state[name].shape)} '
            f'model={tuple(model_state[name].shape)}'
            for name in mismatched[:20]
        ]
        errors.append('Tensor shape mismatches: ' + '; '.join(details))
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='VoV A4 config path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Stage-1 EUD VoV checkpoint; defaults to cfg.load_from.',
    )
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Run config/model/freeze checks without checkpoint coverage.',
    )
    parser.add_argument(
        '--expected-checkpoint-epoch',
        type=int,
        default=None,
        help='Require checkpoint meta.epoch to equal this value.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    errors = _check_config(cfg)
    if args.no_checkpoint and args.expected_checkpoint_epoch is not None:
        errors.append(
            '--expected-checkpoint-epoch cannot be used with --no-checkpoint')

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    errors.extend(_check_trainable(cfg, model))

    if not args.no_checkpoint:
        checkpoint = args.checkpoint or cfg.get('load_from')
        if not checkpoint:
            errors.append('No Stage-1 EUD checkpoint was provided')
        elif not os.path.isfile(checkpoint):
            errors.append(f'Checkpoint does not exist: {checkpoint}')
        else:
            errors.extend(_check_checkpoint(
                model, checkpoint, expected_epoch=args.expected_checkpoint_epoch))

    if errors:
        print('\n[VoV A4 Preflight] FAIL')
        for error in errors:
            print(f'  - {error}')
        raise SystemExit(1)
    print('\n[VoV A4 Preflight] PASS')


if __name__ == '__main__':
    main()
