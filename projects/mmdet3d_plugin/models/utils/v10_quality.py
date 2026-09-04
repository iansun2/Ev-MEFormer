"""Shared V10 oracle-quality helpers.

The functions in this module are intentionally stateless.  They operate on
detached fp32 tensors so the oracle labels used by V10 routing do not leak
gradients back into frozen stage-1 branches.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is present in the training env.
    linear_sum_assignment = None


BRANCH_ORDER = ('fused', 'bev', 'img')
CLASS_NAMES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)
DEFAULT_QUALITY_WEIGHTS = {
    'w_cls': 5.0,
    'w_ctr': 2.0,
    'w_hgt': 1.0,
    # V10.0 does not expose per-branch dim/rot/vel source tensors yet.
    # Step 5 feeds the same routed fallback to all branches for these terms,
    # so keep them disabled until branch-specific tensors are available.
    'w_rot': 0.0,
    'w_dim': 0.0,
    'w_vel': 0.0,
}
DEFAULT_R_CENTER = {
    'car': 2.0,
    'truck': 2.0,
    'bus': 2.0,
    'trailer': 2.0,
    'construction_vehicle': 2.0,
    'motorcycle': 1.0,
    'bicycle': 1.0,
    'pedestrian': 0.5,
    'traffic_cone': 0.5,
    'barrier': 0.5,
}
DEFAULT_R_HEIGHT = 1.0
DEFAULT_R_ROT = 3.141592653589793 / 2.0
DEFAULT_R_VEL = 2.0


def _merge_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(DEFAULT_QUALITY_WEIGHTS)
    if weights:
        merged.update(weights)
    return merged


def _class_r_center_tensor(
        gt_cls: torch.Tensor,
        class_r_center: Optional[Dict] = None) -> torch.Tensor:
    mapping = dict(DEFAULT_R_CENTER)
    if class_r_center:
        mapping.update(class_r_center)
    values = []
    for idx, name in enumerate(CLASS_NAMES):
        values.append(float(mapping.get(name, mapping.get(idx, 1.0))))
    table = gt_cls.new_tensor(values, dtype=torch.float32)
    safe_cls = gt_cls.long().clamp(min=0, max=len(values) - 1)
    return table.to(gt_cls.device)[safe_cls].clamp(min=1e-6)


def _gather_queries(tensor: torch.Tensor, matching_idx: torch.Tensor) -> torch.Tensor:
    idx = matching_idx.long().clamp(min=0, max=tensor.shape[1] - 1)
    expand_shape = list(idx.shape) + [tensor.shape[-1]]
    gather_idx = idx.unsqueeze(-1).expand(*expand_shape)
    return tensor.gather(1, gather_idx)


def _size_iou(pred_dim: torch.Tensor, gt_dim: torch.Tensor) -> torch.Tensor:
    pred_dim = pred_dim.clamp(min=1e-4, max=1e3)
    gt_dim = gt_dim.clamp(min=1e-4, max=1e3)
    inter = torch.minimum(pred_dim, gt_dim).prod(dim=-1)
    pred_vol = pred_dim.prod(dim=-1)
    gt_vol = gt_dim.prod(dim=-1)
    return inter / (pred_vol + gt_vol - inter).clamp(min=1e-6)


def compute_quality(
        cls_logits: torch.Tensor,
        center_pred: torch.Tensor,
        height_pred: torch.Tensor,
        dim_pred: torch.Tensor,
        rot_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_box: torch.Tensor,
        matching_idx: torch.Tensor,
        weights: Optional[Dict[str, float]] = None,
        class_r_center: Optional[Dict] = None) -> torch.Tensor:
    """Compute V10 oracle quality for one modality.

    Args follow the plan.  ``gt_box`` accepts shape ``[..., >=8]`` and uses
    [x, y, z, w, l, h, sin, cos, vx, vy] when velocity is available.
    Invalid pairs where ``matching_idx < 0`` or ``gt_cls < 0`` return 0.
    """
    w = _merge_weights(weights)
    cls_logits = cls_logits.detach().float()
    center_pred = center_pred.detach().float()
    height_pred = height_pred.detach().float()
    dim_pred = dim_pred.detach().float()
    rot_pred = rot_pred.detach().float()
    vel_pred = vel_pred.detach().float()
    gt_cls = gt_cls.detach().long()
    gt_box = gt_box.detach().float()
    matching_idx = matching_idx.detach().long()

    valid = (matching_idx >= 0) & (gt_cls >= 0)
    safe_cls = gt_cls.clamp(min=0, max=cls_logits.shape[-1] - 1)

    matched_cls = _gather_queries(cls_logits, matching_idx)
    matched_center = _gather_queries(center_pred, matching_idx)
    matched_height = _gather_queries(height_pred, matching_idx)
    matched_dim = _gather_queries(dim_pred, matching_idx)
    matched_rot = _gather_queries(rot_pred, matching_idx)
    matched_vel = _gather_queries(vel_pred, matching_idx)

    cls_logp = F.log_softmax(matched_cls, dim=-1)
    cls_logp = cls_logp.gather(-1, safe_cls.unsqueeze(-1)).squeeze(-1).clamp(min=-20.0)

    r_center = _class_r_center_tensor(safe_cls, class_r_center)
    center_l1 = (matched_center - gt_box[..., 0:2]).abs().sum(dim=-1).clamp(max=1e3)
    height_l1 = (matched_height - gt_box[..., 2:3]).abs().squeeze(-1).clamp(max=1e3)

    gt_dim = gt_box[..., 3:6] if gt_box.shape[-1] >= 6 else torch.ones_like(matched_dim)
    dim_iou = _size_iou(matched_dim, gt_dim)

    matched_rot = F.normalize(torch.nan_to_num(matched_rot, nan=0.0), dim=-1)
    if gt_box.shape[-1] >= 7:
        yaw = torch.nan_to_num(gt_box[..., 6], nan=0.0, posinf=0.0, neginf=0.0)
        gt_rot = torch.stack([yaw.sin(), yaw.cos()], dim=-1)
    else:
        gt_rot = torch.zeros_like(matched_rot)
        gt_rot[..., 1] = 1.0
    sin_heading_err = (matched_rot[..., 0] * gt_rot[..., 1] - matched_rot[..., 1] * gt_rot[..., 0]).abs()

    if gt_box.shape[-1] >= 9:
        gt_vel = gt_box[..., 7:9]
    else:
        gt_vel = torch.zeros_like(matched_vel)
    gt_vel = torch.nan_to_num(gt_vel, nan=0.0, posinf=0.0, neginf=0.0)
    vel_err = (matched_vel - gt_vel).norm(dim=-1).clamp(max=1e3)

    quality = (
        float(w['w_cls']) * cls_logp
        - float(w['w_ctr']) * center_l1 / r_center
        - float(w['w_hgt']) * height_l1 / DEFAULT_R_HEIGHT
        - float(w['w_rot']) * sin_heading_err / DEFAULT_R_ROT
        - float(w['w_dim']) * (1.0 - dim_iou)
        - float(w['w_vel']) * vel_err / DEFAULT_R_VEL
    )
    quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(valid, quality, quality.new_zeros(()))


def compute_quality_per_branch(
        branch_outputs: Dict[str, Dict[str, torch.Tensor]],
        gt: Dict[str, torch.Tensor],
        matching: Dict[str, torch.Tensor],
        branches: Sequence[str] = BRANCH_ORDER,
        weights: Optional[Dict[str, float]] = None,
        class_r_center: Optional[Dict] = None) -> torch.Tensor:
    qualities = []
    for branch in branches:
        output = branch_outputs[branch]
        qualities.append(compute_quality(
            output['cls_logits'], output['center'], output['height'],
            output['dim'], output['rot'], output['vel'],
            gt['gt_cls'], gt['gt_box'], matching[branch],
            weights=weights, class_r_center=class_r_center))
    return torch.stack(qualities, dim=0)


def _simple_hungarian(cls_logits, center, height, gt_cls, gt_box):
    nq = cls_logits.shape[0]
    n_gt = gt_cls.shape[0]
    if n_gt == 0 or nq == 0:
        return cls_logits.new_full((n_gt,), -1, dtype=torch.long)
    logp = F.log_softmax(cls_logits.float(), dim=-1)[:, gt_cls.long().clamp(0, cls_logits.shape[-1] - 1)]
    center_cost = torch.cdist(center.float(), gt_box[:, 0:2].float(), p=1) / 2.0
    height_cost = torch.cdist(height.float(), gt_box[:, 2:3].float(), p=1)
    cost = (-logp + center_cost + height_cost).detach().cpu()
    if linear_sum_assignment is None:
        rows = torch.arange(min(nq, n_gt))
        cols = rows.clone()
    else:
        row_np, col_np = linear_sum_assignment(cost)
        rows = torch.as_tensor(row_np, dtype=torch.long)
        cols = torch.as_tensor(col_np, dtype=torch.long)
    match = cls_logits.new_full((n_gt,), -1, dtype=torch.long)
    match[cols.to(match.device)] = rows.to(match.device)
    return match


def _compose_assign_bbox(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    pred_full = torch.cat(
        [output['center'], output['height'], output['dim'], output['rot'], output['vel']],
        dim=-1)
    # Network head order: [cx, cy, cz, w, l, h, sin, cos, vx, vy].
    # HungarianAssigner3D expects normalize_bbox order:
    # [cx, cy, w, l, cz, h, sin, cos, vx, vy].
    return torch.cat([
        pred_full[..., 0:2],
        pred_full[..., 3:5],
        pred_full[..., 2:3],
        pred_full[..., 5:6],
        pred_full[..., 6:10],
    ], dim=-1)


def union_matching(
        branch_outputs: Dict[str, Dict[str, torch.Tensor]],
        gt: Dict[str, torch.Tensor],
        assigner=None,
        branches: Sequence[str] = BRANCH_ORDER,
        matching_mode: str = 'union') -> Dict[str, object]:
    """Run V10 matching and return matched ``(b, q, g)`` pairs.

    ``union`` runs Hungarian matching independently on every branch and uses
    the union of their matched queries. ``single_best`` reproduces the
    finalized R50 V10 fallback: it matches the fused branch once and shares
    that assignment with BEV and IMG.
    """
    matching_mode = str(matching_mode or 'union').lower()
    if matching_mode not in ('union', 'single_best'):
        raise ValueError(
            'Unsupported V10 matching_mode: {}'.format(matching_mode))
    gt_cls = gt['gt_cls']
    gt_box = gt['gt_box']
    bs, n_gt = gt_cls.shape
    matching = {}
    pair_set = set()

    def _match_one_branch(output: Dict[str, torch.Tensor]) -> torch.Tensor:
        branch_match = gt_cls.new_full((bs, n_gt), -1)
        for b in range(bs):
            valid = gt_cls[b] >= 0
            valid_ids = torch.where(valid)[0]
            if valid_ids.numel() == 0:
                continue
            if assigner is not None:
                bbox_pred = _compose_assign_bbox({k: v[b] for k, v in output.items() if k != 'cls_logits'})
                assign_result = assigner.assign(
                    bbox_pred,
                    output['cls_logits'][b],
                    gt_box[b, valid_ids],
                    gt_cls[b, valid_ids])
                assigned = assign_result.gt_inds
                pos_q = torch.where(assigned > 0)[0]
                for q in pos_q:
                    gt_local = int(assigned[q].item()) - 1
                    g = int(valid_ids[gt_local].item())
                    branch_match[b, g] = q
                    pair_set.add((b, int(q.item()), g))
            else:
                local_match = _simple_hungarian(
                    output['cls_logits'][b],
                    output['center'][b],
                    output['height'][b],
                    gt_cls[b, valid_ids],
                    gt_box[b, valid_ids])
                for local_g, q in enumerate(local_match):
                    if q >= 0:
                        g = int(valid_ids[local_g].item())
                        branch_match[b, g] = q
                        pair_set.add((b, int(q.item()), g))
        return branch_match

    if matching_mode == 'single_best':
        fused_match = _match_one_branch(branch_outputs['fused'])
        for branch in branches:
            matching[branch] = fused_match
    else:
        for branch in branches:
            matching[branch] = _match_one_branch(branch_outputs[branch])
    if pair_set:
        pairs = gt_cls.new_tensor(sorted(pair_set), dtype=torch.long)
    else:
        pairs = gt_cls.new_zeros((0, 3), dtype=torch.long)
    return {
        'matching': matching,
        'pairs': pairs,
        'branches': tuple(branches),
        'matching_mode': matching_mode,
    }


def oracle_label_from_quality(quality_per_branch: torch.Tensor, T_oracle: float = 0.5) -> torch.Tensor:
    quality = quality_per_branch.detach().float()
    temperature = max(float(T_oracle), 1e-4)
    probs = torch.softmax(quality / temperature, dim=0)
    return torch.nan_to_num(probs, nan=1.0 / max(quality.shape[0], 1))


def damage_label_from_quality(quality_per_branch: torch.Tensor, tau_dmg: float = 0.15) -> torch.Tensor:
    """Build DD targets in ``(bev_dmg, img_dmg, dual_dmg)`` order."""
    quality = quality_per_branch.detach().float()
    fused = quality[0]
    bev_dmg = quality[1] < (fused - float(tau_dmg))
    img_dmg = quality[2] < (fused - float(tau_dmg))
    dual_dmg = bev_dmg & img_dmg
    return torch.stack([bev_dmg, img_dmg, dual_dmg], dim=0).float()
