#!/usr/bin/env python
"""V10 quality-threshold calibration driver.

Dry-run mode validates the numerical schema with synthetic tensors. Real mode
runs a deterministic forward-only subset of nuScenes train with the configured
online corruptions and emits checkpoint/config/data provenance.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from torch.utils.data import Subset


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import projects.mmdet3d_plugin  # noqa: F401,E402  Registers custom modules.
from projects.mmdet3d_plugin.models.utils.v10_quality import (  # noqa: E402
    BRANCH_ORDER,
    compute_quality_per_branch,
    damage_label_from_quality,
    oracle_label_from_quality,
    union_matching,
)


def _full_path(path: str) -> str:
    if not path:
        return ''
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _dry_output_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f'{root}_dry_run{ext or ".json"}'


def _get_head_and_ensemble(model):
    head = model.pts_bbox_head
    return head, head.ensemble


def _synthetic_inputs(ensemble, bs: int = 4, nq: int = 12, seed: int = 0, device='cpu'):
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
    img_metas = [{'corruption_type': 'synthetic'} for _ in range(bs)]
    return x, reference, outs, nig, modalities, num_queries, img_metas


def _build_model(cfg: Config, checkpoint: str, dry_run: bool):
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    _head, ensemble = _get_head_and_ensemble(model)
    ensemble.init_weights()
    ckpt_path = _full_path(checkpoint or cfg.get('load_from'))
    if ckpt_path and os.path.exists(ckpt_path):
        print(f'[Load] checkpoint={ckpt_path}')
        load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    elif dry_run:
        print(f'[Load] dry-run skip missing checkpoint: {ckpt_path}')
    else:
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    model.eval()
    return model


def _run_synthetic_forward(model):
    _head, ensemble = _get_head_and_ensemble(model)
    old_enable = ensemble.v10_enable
    ensemble.v10_enable = True
    args = _synthetic_inputs(ensemble, bs=4, nq=12, seed=7)
    with torch.no_grad():
        out = ensemble(
            args[0], args[1], args[2], args[3], args[4], args[5],
            0, [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], img_metas=args[6])
    ensemble.v10_enable = old_enable
    return out


def _mock_gt_from_output(out: Dict[str, object], n_gt: int = 6) -> Dict[str, torch.Tensor]:
    source_center = out['selector_source_center']
    source_height = out['selector_source_height']
    fused_center = source_center['fused']
    fused_height = source_height['fused']
    bs, nq, _ = fused_center.shape
    torch.manual_seed(19)
    q_idx = torch.randint(0, nq, (bs, n_gt), device=fused_center.device)
    gather_2d = q_idx.unsqueeze(-1).expand(bs, n_gt, 2)
    gather_1d = q_idx.unsqueeze(-1).expand(bs, n_gt, 1)
    center = fused_center.gather(1, gather_2d) + 0.05 * torch.randn(bs, n_gt, 2)
    height = fused_height.gather(1, gather_1d) + 0.02 * torch.randn(bs, n_gt, 1)
    dim = torch.rand(bs, n_gt, 3) + 0.5
    yaw = torch.rand(bs, n_gt, 1) * 6.283185307179586
    vel = torch.randn(bs, n_gt, 2) * 0.2
    gt_cls = torch.randint(0, 10, (bs, n_gt), device=fused_center.device)
    gt_box = torch.cat(
        [center.cpu(), height.cpu(), dim, yaw, vel], dim=-1
    ).to(fused_center.device)
    return {'gt_cls': gt_cls, 'gt_box': gt_box}


def _branch_outputs_from_forward(out: Dict[str, object]) -> Dict[str, Dict[str, torch.Tensor]]:
    torch.manual_seed(23)
    branch_outputs = {}
    for branch in BRANCH_ORDER:
        center = out['selector_source_center'][branch]
        height = out['selector_source_height'][branch]
        bs, nq, _ = center.shape
        angle = torch.rand(bs, nq, device=center.device) * 6.283185307179586
        branch_outputs[branch] = {
            'cls_logits': out['selector_source_cls_logits'][branch],
            'center': center,
            'height': height,
            'dim': torch.rand(bs, nq, 3, device=center.device) + 0.5,
            'rot': torch.stack([angle.sin(), angle.cos()], dim=-1),
            'vel': torch.randn(bs, nq, 2, device=center.device) * 0.2,
        }
    return branch_outputs


def _quantile(values: torch.Tensor, q: float, default: float) -> float:
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float(default)
    return float(torch.quantile(values.cpu(), q).item())


def _matching_diagnostics(matching: Dict[str, torch.Tensor], quality: torch.Tensor) -> Dict[str, object]:
    query_to_gts = defaultdict(set)
    matched_counts = []
    for branch_match in matching.values():
        bs = branch_match.shape[0]
        for b in range(bs):
            valid = branch_match[b] >= 0
            matched_counts.append(int(valid.sum().item()))
            for g in torch.where(valid)[0].tolist():
                q = int(branch_match[b, g].item())
                query_to_gts[(b, q)].add(int(g))

    multi_items = {key: gts for key, gts in query_to_gts.items() if len(gts) >= 2}
    oracle = quality.argmax(dim=0)
    conflicts = 0
    for (b, _q), gts in multi_items.items():
        oracle_set = {int(oracle[b, g].item()) for g in gts}
        if len(oracle_set) >= 2:
            conflicts += 1
    denom_all = max(len(query_to_gts), 1)
    denom_multi = max(len(multi_items), 1)
    counts = torch.tensor(matched_counts, dtype=torch.float32) if matched_counts else torch.zeros(1)
    conflict_ratio = conflicts / denom_multi if multi_items else 0.0
    use_single_best = conflict_ratio > 0.30
    n_min = max(1, int(torch.quantile(counts, 0.25).item()))
    return {
        'multi_gt_query_ratio': len(multi_items) / denom_all,
        'conflicting_oracle_ratio': conflict_ratio,
        'per_corruption_matched_count_distribution': {
            'synthetic': {
                'p25': float(torch.quantile(counts, 0.25).item()),
                'p50': float(torch.quantile(counts, 0.50).item()),
                'p75': float(torch.quantile(counts, 0.75).item()),
                'count': int(counts.numel()),
            }
        },
        'fallback_recommendation': {
            'use_single_best_matching': use_single_best,
            '_reason': 'conflicting_oracle_ratio > 30%' if use_single_best else 'multi-pair conflict below 30%',
        },
        'n_min_suggested': n_min,
        '_n_min_reason': 'p25 across per-sample matched-positive counts',
    }


def _calibrate_from_quality(quality: torch.Tensor, matching: Dict[str, torch.Tensor]) -> Dict[str, object]:
    sorted_q, _ = quality.sort(dim=0, descending=True)
    quality_gap = (sorted_q[0] - sorted_q[1]).flatten()
    max_min_gap = (sorted_q[0] - sorted_q[-1]).flatten()
    runner_up_non_fused = torch.maximum(quality[1], quality[2])
    fused_gap = (quality[0] - runner_up_non_fused).flatten()
    tau_margin = _quantile(quality_gap, 0.70, 0.05)
    tau_dmg = max(0.01, _quantile(fused_gap.clamp(min=0.0), 0.70, 0.15))
    t_oracle = min(2.0, max(0.1, _quantile(max_min_gap, 0.50, 0.5)))
    diagnostics = {
        'quality_gap_p50': _quantile(quality_gap, 0.50, 0.0),
        'quality_gap_p95': _quantile(quality_gap, 0.95, 0.0),
        'per_corruption_gap': {
            'synthetic': {
                'p50': _quantile(quality_gap, 0.50, 0.0),
                'p70': _quantile(quality_gap, 0.70, 0.0),
                'p95': _quantile(quality_gap, 0.95, 0.0),
            }
        },
    }
    diagnostics.update(_matching_diagnostics(matching, quality))
    return {
        'tau_margin': tau_margin,
        'margin': 0.10,
        'tau_dmg': tau_dmg,
        'T_oracle': t_oracle,
        '_diagnostics': diagnostics,
    }


def _write_json(path: str, payload: Dict[str, object]):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.v10-calibration-', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


EXPECTED_TRAIN_INFOS_SHA256 = (
    '6e07dac06f515722fa9dbb685dea40f39a561c2d06739d5bd7cb83a340855854')
CRITICAL_CODE_PATHS = (
    'projects/mmdet3d_plugin/models/utils/edl_pme_transformer.py',
    'projects/mmdet3d_plugin/models/utils/v10_quality.py',
    'projects/mmdet3d_plugin/models/dense_heads/edl_head.py',
    'projects/mmdet3d_plugin/datasets/pipelines/synthetic_corruptions.py',
    'tools/v10_quality_calibration.py',
    'tools/build_vov_v10_selector_config.py',
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _critical_code_records():
    records = []
    for relative_path in CRITICAL_CODE_PATHS:
        absolute_path = os.path.join(REPO_ROOT, relative_path)
        if not os.path.isfile(absolute_path):
            raise SystemExit(
                '[V10 calibration][ERROR] missing critical code: {}'.format(
                    relative_path))
        records.append({
            'path': relative_path,
            'sha256': _sha256_file(absolute_path),
        })
    return records


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _resolved_config_sha256(path: str) -> str:
    payload = json.dumps(
        _plain(dict(Config.fromfile(path))), sort_keys=True,
        separators=(',', ':'), default=repr).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_epoch(path: str) -> int:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    return int((payload.get('meta') or {}).get('epoch', -1))


def _walk_pipeline(pipeline):
    for step in pipeline or ():
        if not isinstance(step, dict):
            continue
        yield step
        for child in _walk_pipeline(step.get('transforms') or ()):
            yield child


def _assert_train_only_contract(cfg: Config, args, checkpoint: str):
    if checkpoint is None or not os.path.isfile(_full_path(checkpoint)):
        raise SystemExit('[V10 calibration][ERROR] source checkpoint is missing')
    if not 0.0 < float(args.subset) <= 1.0:
        raise SystemExit('[V10 calibration][ERROR] subset must be in (0, 1]')
    if int(args.epochs) < 1 or int(args.batch_size) < 1:
        raise SystemExit('[V10 calibration][ERROR] epochs/batch-size must be positive')
    if int(args.workers_per_gpu) < 0:
        raise SystemExit('[V10 calibration][ERROR] workers-per-gpu must be non-negative')
    if not os.path.isfile(args.train_infos):
        raise SystemExit('[V10 calibration][ERROR] train infos is missing')
    train_infos_sha = _sha256_file(args.train_infos)
    if train_infos_sha != EXPECTED_TRAIN_INFOS_SHA256:
        raise SystemExit('[V10 calibration][ERROR] unexpected train infos SHA256')
    lower_paths = ' '.join(
        [args.train_infos, args.output, args.config]).lower()
    if any(token in lower_paths for token in (
            'nuscenes_infos_val', 'multicorrupt_results', 'targeted_probe',
            'corrupted-val', 'corrupted_val', '/validation/')):
        raise SystemExit('[V10 calibration][ERROR] evaluation-derived path rejected')
    train_cfg = cfg.data.train
    dataset_cfg = train_cfg.get('dataset', train_cfg)
    ann_file = os.path.realpath(str(dataset_cfg.get('ann_file') or ''))
    if ann_file != os.path.realpath(args.train_infos):
        raise SystemExit('[V10 calibration][ERROR] config ann_file is not train infos')
    corruptions = [
        step for step in _walk_pipeline(dataset_cfg.get('pipeline') or ())
        if step.get('type') == 'OnlineSyntheticMultiCorrupt'
    ]
    if len(corruptions) != 1:
        raise SystemExit('[V10 calibration][ERROR] expected one train O.C transform')
    corruption = corruptions[0]
    expected_weights = {
        'beamsreducing': 12, 'fog': 10, 'pointsreducing': 8,
        'spatialmisalignment': 6, 'temporalmisalignment': 6,
        'motionblur': 5, 'dark': 4, 'brightness': 4,
        'missingcamera': 3, 'snow': 2,
    }
    if (abs(float(corruption.get('corrupt_prob', -1.0)) - 0.60) > 1e-12 or
            list(corruption.get('severity_choices') or ()) != [1, 2, 3] or
            dict(corruption.get('corruption_weights') or {}) != expected_weights or
            corruption.get('record_meta') is not True):
        raise SystemExit('[V10 calibration][ERROR] train O.C policy mismatch')
    model = cfg.model
    ensemble = model.pts_bbox_head.ensemble
    identity = {
        'backbone': str(model.img_backbone.type),
        'spec': str(model.img_backbone.get('spec_name', '')),
        'final_dim': list(cfg.ida_aug_conf.final_dim),
        'voxel_size': list(model.pts_voxel_layer.voxel_size),
        'v10_enable': bool(ensemble.get('v10_enable', False)),
        'total_epochs': int(cfg.total_epochs),
    }
    expected_identity = {
        'backbone': 'VoVNet', 'spec': 'V-99-eSE',
        'final_dim': [640, 1600], 'voxel_size': [0.075, 0.075, 0.2],
        'v10_enable': True, 'total_epochs': 6,
    }
    if identity != expected_identity:
        raise SystemExit(
            '[V10 calibration][ERROR] VoV V10 identity mismatch: {}'.format(identity))
    if _checkpoint_epoch(checkpoint) != 4:
        raise SystemExit('[V10 calibration][ERROR] source checkpoint is not epoch 4')
    if not args.diagnostic:
        if (abs(float(args.subset) - 0.02) > 1e-12 or int(args.epochs) != 1 or
                args.max_batches is not None or int(args.workers_per_gpu) != 0):
            raise SystemExit(
                '[V10 calibration][ERROR] formal policy requires subset=0.02, '
                'epochs=1, no max-batches, workers-per-gpu=0')
    return {
        'train_infos': os.path.realpath(args.train_infos),
        'train_infos_sha256': train_infos_sha,
        'corrupt_prob': 0.60,
        'severity_choices': [1, 2, 3],
        'corruption_weights': expected_weights,
    }


class _SubsetWithAttrs(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        for attr in ('CLASSES', 'PALETTE', 'box_type_3d', 'box_mode_3d'):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))
        if hasattr(dataset, 'flag'):
            self.flag = np.asarray(dataset.flag)[np.asarray(indices, dtype=np.int64)]


def _unwrap_singleton(value):
    if hasattr(value, 'data'):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        inner = value[0]
        if isinstance(inner, (list, tuple)) and inner:
            return list(inner)
        if hasattr(inner, 'tensor') or torch.is_tensor(inner) or isinstance(inner, dict):
            return inner
        value = inner
    return value


def _as_sample_list(value):
    value = _unwrap_singleton(value)
    if isinstance(value, tuple):
        value = list(value)
    return value if isinstance(value, list) else [value]


def _as_img_metas(value):
    value = _unwrap_singleton(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value if isinstance(value, list) else [value]


def _as_tensor_or_none(value):
    value = _unwrap_singleton(value)
    if isinstance(value, list) and len(value) == 1 and torch.is_tensor(value[0]):
        return value[0]
    return value


def _move_and_unwrap_batch(data, device):
    if device.type == 'cuda':
        device_id = device.index if device.index is not None else 0
        data = scatter(data, [device_id])[0]
    return (
        _as_sample_list(data.get('points')),
        _as_tensor_or_none(data.get('img')),
        _as_img_metas(data.get('img_metas')),
        _as_sample_list(data.get('gt_bboxes_3d')),
        _as_sample_list(data.get('gt_labels_3d')),
    )


def _forward_head_outputs(model, data, device):
    points, img, img_metas, gt_bboxes, gt_labels = _move_and_unwrap_batch(
        data, device)
    img_feats, pts_feats = model.extract_feat(points, img=img, img_metas=img_metas)
    outs = model.pts_bbox_head(
        pts_feats if pts_feats is not None else [None],
        img_feats if img_feats is not None else [None], img_metas)
    return outs, img_metas, gt_bboxes, gt_labels


def _iter_task_preds(outs):
    for task_id, task_preds in enumerate(outs):
        if isinstance(task_preds, list):
            task_preds = task_preds[0]
        yield task_id, task_preds


def _pair_corruptions(pairs, img_metas):
    values = []
    for batch_idx in pairs[:, 0].detach().cpu().tolist():
        meta = img_metas[int(batch_idx)]
        corruption = str(meta.get('corruption_type', 'clean')).lower()
        values.append('clean' if corruption in ('', 'none') else corruption)
    return values


def _percentiles(values, keys):
    array = np.asarray(values if values else [0.0], dtype=np.float64)
    return {
        name: float(np.percentile(array, percentile))
        for name, percentile in keys.items()
    }


def _run_real_calibration(cfg, checkpoint, args):
    checkpoint = _full_path(checkpoint)
    policy = _assert_train_only_contract(cfg, args, checkpoint)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise SystemExit('[V10 calibration][ERROR] formal real mode requires CUDA')
    device_index = device.index if device.index is not None else 0
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise SystemExit('[V10 calibration][ERROR] requested CUDA device is unavailable')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(device)
    started = time.time()
    model = _build_model(cfg, checkpoint, dry_run=False).to(device)
    model.eval()
    head, ensemble = _get_head_and_ensemble(model)
    ensemble.v10_enable = True

    dataset = build_dataset(copy.deepcopy(cfg.data.train))
    subset_len = max(1, int(math.ceil(len(dataset) * float(args.subset))))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_len].tolist()
    subset = _SubsetWithAttrs(dataset, indices)
    dataloader = build_dataloader(
        subset, samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers_per_gpu, num_gpus=1, dist=False,
        shuffle=False, seed=args.seed, persistent_workers=False)
    print('[Calibration] samples={}/{} batches={} device={}'.format(
        subset_len, len(dataset), len(dataloader), device))

    quality_gaps = defaultdict(list)
    damage_gaps = defaultdict(list)
    matched_counts = defaultdict(list)
    max_min_gaps = []
    entropy_values = []
    damage_positive = {0: 0.0, 1: 0.0, 2: 0.0}
    total_pairs = 0
    total_queries = 0
    multi_queries = 0
    conflicting_queries = 0
    batches_processed = 0

    with torch.no_grad():
        for _epoch in range(args.epochs):
            for data in dataloader:
                if args.max_batches is not None and batches_processed >= args.max_batches:
                    break
                outs, img_metas, gt_bboxes, gt_labels = _forward_head_outputs(
                    model, data, device)
                for task_id, task_preds in _iter_task_preds(outs):
                    branches = head._build_branch_outputs_for_v10(task_preds)
                    if branches is None:
                        continue
                    gt = head._v10_local_task_gt(
                        gt_bboxes, gt_labels, task_id,
                        branches['fused']['center'].device)
                    matching = union_matching(
                        branches, gt, assigner=head.assigner,
                        matching_mode='union')
                    pairs = matching['pairs']
                    if pairs.numel() == 0:
                        continue
                    quality_full = compute_quality_per_branch(
                        branches, gt, matching['matching'])
                    quality = quality_full[:, pairs[:, 0], pairs[:, 2]]
                    if not torch.isfinite(quality).all():
                        raise RuntimeError('V10 calibration quality contains NaN/Inf')
                    sorted_quality, _ = quality.sort(dim=0, descending=True)
                    gaps = (sorted_quality[0] - sorted_quality[1]).cpu().tolist()
                    damage = (quality[0] - torch.maximum(
                        quality[1], quality[2])).cpu().tolist()
                    corruptions = _pair_corruptions(pairs, img_metas)
                    for corruption, gap, damage_gap in zip(corruptions, gaps, damage):
                        quality_gaps[corruption].append(float(gap))
                        damage_gaps[corruption].append(float(damage_gap))
                    max_min_gaps.extend(
                        (sorted_quality[0] - sorted_quality[-1]).cpu().tolist())
                    p_oracle = oracle_label_from_quality(quality, T_oracle=0.5)
                    entropy_values.extend((-(p_oracle * p_oracle.clamp(
                        min=1e-8).log()).sum(dim=0)).cpu().tolist())
                    y_damage = damage_label_from_quality(quality, tau_dmg=0.15)
                    for branch_idx in damage_positive:
                        damage_positive[branch_idx] += float(
                            y_damage[branch_idx].sum().item())
                    total_pairs += int(pairs.shape[0])

                    grouped = defaultdict(list)
                    oracle_idx = quality.argmax(dim=0).cpu()
                    for idx, (batch_idx, query_idx, _gt_idx) in enumerate(
                            pairs.cpu().tolist()):
                        grouped[(int(batch_idx), int(query_idx))].append(idx)
                    total_queries += len(grouped)
                    for indices_for_query in grouped.values():
                        if len(indices_for_query) < 2:
                            continue
                        multi_queries += 1
                        if len({int(oracle_idx[idx]) for idx in indices_for_query}) > 1:
                            conflicting_queries += 1
                    for batch_idx in pairs[:, 0].unique().cpu().tolist():
                        count = int((pairs[:, 0] == int(batch_idx)).sum().item())
                        meta = img_metas[int(batch_idx)]
                        corruption = str(meta.get('corruption_type', 'clean')).lower()
                        matched_counts['clean' if corruption in ('', 'none')
                                       else corruption].append(count)
                batches_processed += 1
                if batches_processed == 1 or batches_processed % 50 == 0:
                    print('[Calibration] batch={} pairs={} conflicts={}/{}'.format(
                        batches_processed, total_pairs,
                        conflicting_queries, multi_queries))
            if args.max_batches is not None and batches_processed >= args.max_batches:
                break

    if total_pairs <= 0:
        raise SystemExit('[V10 calibration][ERROR] no matched pairs were collected')
    all_quality_gaps = [v for values in quality_gaps.values() for v in values]
    all_damage_gaps = [v for values in damage_gaps.values() for v in values]
    max_min = np.asarray(max_min_gaps, dtype=np.float64)
    quality_array = np.asarray(all_quality_gaps, dtype=np.float64)
    damage_array = np.asarray(all_damage_gaps, dtype=np.float64)
    conflict_ratio = conflicting_queries / max(multi_queries, 1)
    matched_distribution = {
        key: dict(_percentiles(
            values, {'p25': 25, 'p50': 50, 'p75': 75}),
            count=len(values))
        for key, values in sorted(matched_counts.items())
    }
    n_min_suggested = max(
        1, int(min(
            record['p25'] for record in matched_distribution.values())))
    payload = {
        'format': 'v10_quality_calibration_v2',
        'formal_train_only': not args.diagnostic,
        'tau_margin': float(np.quantile(quality_array, 0.70)),
        'margin': 0.10,
        'tau_dmg': float(np.quantile(np.maximum(damage_array, 0.0), 0.70)),
        'T_oracle': max(0.3, float(np.median(max_min) / 3.0)),
        'model_provenance': {
            'base_config': os.path.realpath(args.config),
            'base_config_sha256': _sha256_file(args.config),
            'resolved_config_sha256': _resolved_config_sha256(args.config),
            'checkpoint': os.path.realpath(checkpoint),
            'checkpoint_sha256': _sha256_file(checkpoint),
            'checkpoint_epoch': _checkpoint_epoch(checkpoint),
            'train_infos': policy['train_infos'],
            'train_infos_sha256': policy['train_infos_sha256'],
            'critical_code': _critical_code_records(),
        },
        'calibration_policy': {
            'subset_ratio': float(args.subset), 'epochs': int(args.epochs),
            'seed': int(args.seed), 'batch_size': int(args.batch_size),
            'workers_per_gpu': int(args.workers_per_gpu),
            'matching_mode_for_diagnostics': 'union',
            'corrupt_prob': policy['corrupt_prob'],
            'severity_choices': policy['severity_choices'],
            'corruption_weights': policy['corruption_weights'],
        },
        '_diagnostics': {
            'dry_run': False,
            'total_batches_processed': batches_processed,
            'total_pairs': total_pairs,
            'total_query_count': total_queries,
            'multi_gt_query_ratio': multi_queries / max(total_queries, 1),
            'conflicting_oracle_ratio': conflict_ratio,
            'fallback_recommendation': {
                'use_single_best_matching': conflict_ratio > 0.30,
                '_reason': 'conflicting_oracle_ratio={:.6f}'.format(
                    conflict_ratio),
            },
            'n_min_suggested': n_min_suggested,
            '_n_min_reason': (
                'p25 across per-corruption matched-positive counts'),
            'quality_gap_p50': float(np.quantile(quality_array, 0.50)),
            'quality_gap_p95': float(np.quantile(quality_array, 0.95)),
            'per_corruption_gap': {
                key: _percentiles(values, {'p50': 50, 'p70': 70, 'p95': 95})
                for key, values in sorted(quality_gaps.items())
            },
            'per_corruption_matched_count_distribution': matched_distribution,
            'damage_positive_ratio': {
                str(key): value / total_pairs
                for key, value in damage_positive.items()
            },
            'oracle_entropy_mean': float(np.mean(entropy_values)),
            'subset_ratio': float(args.subset),
            'elapsed_sec': float(time.time() - started),
        },
    }
    _write_json(args.output, payload)
    print('[Calibration] wrote {}'.format(args.output))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='V10 placeholder config path')
    parser.add_argument('--checkpoint', default=None, help='Rev20-A checkpoint; defaults to cfg.load_from')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--subset', type=float, default=0.02)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers-per-gpu', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260522)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--train-infos', default='data/nuscenes/nuscenes_infos_train.pkl')
    parser.add_argument('--output', default='work_dirs/v10_quality_calibration.json')
    parser.add_argument('--dry-run', action='store_true', help='Use one synthetic batch and write *_dry_run.json')
    parser.add_argument(
        '--diagnostic', action='store_true',
        help='Permit nonformal subset/workers/max-batches for a smoke run.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if not args.dry_run:
        _run_real_calibration(
            cfg, args.checkpoint or cfg.get('load_from'), args)
        return

    model = _build_model(cfg, args.checkpoint, dry_run=True)
    out = _run_synthetic_forward(model)
    branch_outputs = _branch_outputs_from_forward(out)
    gt = _mock_gt_from_output(out)
    matching_result = union_matching(branch_outputs, gt, assigner=None)
    matching = matching_result['matching']
    quality = compute_quality_per_branch(branch_outputs, gt, matching)
    if not torch.isfinite(quality).all():
        raise RuntimeError('quality contains NaN/Inf')
    oracle = oracle_label_from_quality(quality, T_oracle=0.5)
    damage = damage_label_from_quality(quality, tau_dmg=0.15)
    if not torch.isfinite(oracle).all() or not torch.isfinite(damage).all():
        raise RuntimeError('oracle/damage labels contain NaN/Inf')

    payload = _calibrate_from_quality(quality, matching)
    payload['_diagnostics']['dry_run'] = True
    payload['_diagnostics']['quality_shape'] = list(quality.shape)
    payload['_diagnostics']['oracle_entropy_mean'] = float(
        -(oracle * oracle.clamp(min=1e-8).log()).sum(dim=0).mean().item())
    payload['_diagnostics']['damage_positive_mean'] = float(damage.mean().item())

    output_path = _dry_output_path(args.output)
    _write_json(output_path, payload)
    print(f'[DryRun] wrote {output_path}')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
