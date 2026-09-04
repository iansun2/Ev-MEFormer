# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import collections
import copy

import math
from numbers import Number
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean, build_bbox_coder
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models import builder
import torch.nn.functional as F
import torch.distributed as dist

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.v10_quality import (
    BRANCH_ORDER,
    compute_quality_per_branch,
    damage_label_from_quality,
    oracle_label_from_quality,
    union_matching,
)


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


@HEADS.register_module()
class EDLHead(BaseModule):
    def __init__(self,
                 in_channels,
                 use_ensemble=False,
                 ensemble=None,
                 num_query=900,
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 pc_range=[],
                 modalities=dict(),
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.75,
                 use_l1_quality_cls=False,
                 cls_quality_tau=2.0,
                 cls_neg_gamma=2.0,
                 cls_neg_balance=1.0,
                 use_dynamic_kl=False,
                 lambda_consistency_cls=0.0,
                 lambda_consistency_box=0.0,
                 lambda_fused_prior=0.0,
                 lambda_selector_target=0.0,
                 lambda_selector_target_base=0.1,
                 lambda_selector_target_residual=0.5,
                 lambda_residual_usage=0.05,
                 lambda_selector_second_opinion=0.2,
                 lambda_selector_entropy_hard=0.02,
                 lambda_alpha_res_separation=0.05,
                 lambda_v8_gate_supervision=0.0,
                 lambda_v8_modality_supervision=0.0,
                 lambda_v8_domain_reliability_supervision=0.0,
                 lambda_v8_center_delta=0.0,
                 lambda_v9_anomaly_supervision=0.0,
                 lambda_v9_gate_domain=0.0,
                 lambda_v9_route_consistency=0.0,
                 lambda_v10_route_kl=0.20,
                 lambda_v10_route_margin=0.10,
                 lambda_v10_dd_anomaly=1.00,
                 lambda_v10_dd_consistency=0.05,
                 lambda_v10_csg=0.10,
                 lambda_v10_edlcue_kl=0.0,
                 lambda_v10_edlcue_rank=0.0,
                 lambda_v10_edlcue_delta=0.0,
                 v10_edlcue_warmup_iters=0,
                 v10_loss_enabled=False,
                 v10_oracle_thresholds=None,
                 v10_matching_mode='union',
                 v10_route_margin_value=0.10,
                 v10_edlcue_margin_value=0.10,
                 v10_csg_clean_target=0.90,
                 lambda_rev19_query_distill=0.0,
                 v9_anomaly_head_lambda_warmup=False,
                 v9_severity_weights=None,
                 v8_clean_fused_keep_target=0.85,
                 v8_damage_fused_keep_target=0.35,
                 v8_dual_damage_fused_keep_target=0.60,
                 v8_low_damage_fused_keep_weight=0.25,
                 v8_clean_reliability_target=0.90,
                 v8_damage_reliability_target=0.20,
                 v8_safe_reliability_target=0.90,
                 selector_teacher_temperature=0.5,
                 selector_entropy_target_hard=0.35,
                 residual_usage_target=0.2,
                 alpha_res_separation_margin=0.15,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                     type="L1Loss",
                     reduction="mean",
                     loss_weight=0.25,
                 ),
                 loss_iou=None,
                 separate_head=dict(
                     type='EDLNIGHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 loss_edl_weights=None,
                 **kwargs):
        assert init_cfg is None
        super(EDLHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # [EDL] take code_weights from train_cfg for the IoU proxy computation
        if train_cfg is not None and 'pts' in train_cfg:
            self.code_weights = train_cfg['pts'].get(
                'code_weights', [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            )
        else:
            # Default weights: center matters most, velocity least
            self.code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        
        self.use_ensemble = use_ensemble
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.pc_range = pc_range
        self.modalities = modalities
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        self.use_l1_quality_cls = use_l1_quality_cls
        self.cls_quality_tau = cls_quality_tau
        self.cls_neg_gamma = self._expand_task_class_param(cls_neg_gamma, 'cls_neg_gamma')
        self.cls_neg_balance = self._expand_task_class_param(cls_neg_balance, 'cls_neg_balance')
        self.use_dynamic_kl = use_dynamic_kl
        self.lambda_consistency_cls = lambda_consistency_cls
        self.lambda_consistency_box = lambda_consistency_box
        self.lambda_fused_prior = lambda_fused_prior
        self.lambda_selector_target = lambda_selector_target
        self.lambda_selector_target_base = lambda_selector_target_base
        self.lambda_selector_target_residual = lambda_selector_target_residual
        self.lambda_residual_usage = lambda_residual_usage
        self.lambda_selector_second_opinion = lambda_selector_second_opinion
        self.lambda_selector_entropy_hard = lambda_selector_entropy_hard
        self.lambda_alpha_res_separation = lambda_alpha_res_separation
        self.lambda_v8_gate_supervision = lambda_v8_gate_supervision
        self.lambda_v8_modality_supervision = lambda_v8_modality_supervision
        self.lambda_v8_domain_reliability_supervision = lambda_v8_domain_reliability_supervision
        self.lambda_v8_center_delta = lambda_v8_center_delta
        self.lambda_v9_anomaly_supervision = float(lambda_v9_anomaly_supervision)
        self.lambda_v9_gate_domain = float(lambda_v9_gate_domain)
        self.lambda_v9_route_consistency = float(lambda_v9_route_consistency)
        self.lambda_v10_route_kl = float(lambda_v10_route_kl)
        self.lambda_v10_route_margin = float(lambda_v10_route_margin)
        self.lambda_v10_dd_anomaly = float(lambda_v10_dd_anomaly)
        self.lambda_v10_dd_consistency = float(lambda_v10_dd_consistency)
        self.lambda_v10_csg = float(lambda_v10_csg)
        self.lambda_v10_edlcue_kl = float(lambda_v10_edlcue_kl)
        self.lambda_v10_edlcue_rank = float(lambda_v10_edlcue_rank)
        self.lambda_v10_edlcue_delta = float(lambda_v10_edlcue_delta)
        self.v10_edlcue_warmup_iters = max(int(v10_edlcue_warmup_iters), 0)
        self.v10_edlcue_warmup_factor = 1.0
        self.v10_loss_enabled = bool(v10_loss_enabled)
        self.v10_oracle_thresholds = dict(v10_oracle_thresholds or {})
        self.v10_matching_mode = str(v10_matching_mode or 'union')
        self.v10_route_margin_value = float(v10_route_margin_value)
        self.v10_edlcue_margin_value = float(v10_edlcue_margin_value)
        self.v10_csg_clean_target = float(v10_csg_clean_target)
        self.lambda_rev19_query_distill = float(lambda_rev19_query_distill)
        self.v9_anomaly_head_lambda_warmup = bool(v9_anomaly_head_lambda_warmup)
        v9_severity_weights = v9_severity_weights or {0: 1.0, 1: 0.4, 2: 0.7, 3: 1.0}
        self.v9_severity_weights = {
            int(k): float(v) for k, v in v9_severity_weights.items()
        }
        self.v8_clean_fused_keep_target = float(v8_clean_fused_keep_target)
        self.v8_damage_fused_keep_target = float(v8_damage_fused_keep_target)
        self.v8_dual_damage_fused_keep_target = float(v8_dual_damage_fused_keep_target)
        self.v8_low_damage_fused_keep_weight = float(v8_low_damage_fused_keep_weight)
        self.v8_clean_reliability_target = float(v8_clean_reliability_target)
        self.v8_damage_reliability_target = float(v8_damage_reliability_target)
        self.v8_safe_reliability_target = float(v8_safe_reliability_target)
        self.selector_teacher_temperature = max(selector_teacher_temperature, 1e-6)
        self.selector_entropy_target_hard = selector_entropy_target_hard
        self.residual_usage_target = residual_usage_target
        self.alpha_res_separation_margin = alpha_res_separation_margin
        self.residual_floor = (
            float(ensemble.get('residual_floor', 0.1))
            if isinstance(ensemble, dict) else 0.1
        )
        self.v7_router_label_margin = (
            float(ensemble.get('v7_router_label_margin', 0.10))
            if isinstance(ensemble, dict) else 0.10
        )
        self.v7_img_router_loss_weight = (
            float(ensemble.get('v7_img_router_loss_weight', 0.0))
            if isinstance(ensemble, dict) else 0.0
        )
        self.v7_bev_router_loss_weight = (
            float(ensemble.get('v7_bev_router_loss_weight', 0.0))
            if isinstance(ensemble, dict) else 0.0
        )
        
        # [EDL] iteration-based warmup (replaces the epoch-based one)
        self.current_iter = 0
        self.edl_warmup_factor = 0.0

        self.debug = True
        self.debug_counter = 0
        self.debug_interval = 250      # verbose debug output
        self.nig_debug_interval = 50   # NIG parameter reporting (aligned with the torch epoch info)
        self.cls_debug_interval = 50
        self._edl_debug_stats = []
        self.current_epoch = 0

        # [REMOVED] dim_log_range - dim now uses the raw output, as in MEFormer

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.fp16_enabled = False

        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )

        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

        if self.use_ensemble:
            ensemble["heads"] = common_heads
            ensemble["separate_head"] = separate_head
            ensemble["num_classes"] = self.num_classes
            self.ensemble = build_transformer(ensemble)
        else:
            self.ensemble = None
        
        if loss_edl_weights is None:
            self.l1_weight = 1.0
            self.l1_weight_standard = 1.0  # separate L1 weight for plain regression (rot/vel)
            self.nig_nll_weight = 0.1
            self.nig_reg_weight = 0.0001  # the paper suggests lambda = 1e-4
            self.warmup_iters = 1000      # Iteration-based warmup
            self.detach_gamma = False     # no detach by default (original E-DETR behaviour)
            # [EDL CLS] Classification EDL weights
            self.cls_edl_weight = 1.0     # EDL NLL weight
            self.cls_kl_weight = 0.1      # KL regularization weight
            self.cls_warmup_iters = 2000  # KL warmup iterations
            # [MT-ENet] Lipschitz L1 Loss (Proposition 2: U_l threshold)
            self.use_lipschitz_l1 = False  # off by default
            self.ll1_zero_below_threshold = False  # True = original (loss = 0), False = fall back to L1
        else:
            self.l1_weight = loss_edl_weights.get('l1_weight', 1.0)
            self.l1_weight_standard = loss_edl_weights.get('l1_weight_standard', 1.0)  # plain regression (rot/vel)
            self.nig_nll_weight = loss_edl_weights.get('nig_nll_weight', 0.1)
            self.nig_reg_weight = loss_edl_weights.get('nig_reg_weight', 0.0001)
            self.warmup_iters = loss_edl_weights.get('warmup_iters', 1000)
            # detach_gamma=True means L1 supervises gamma while NIG supervises only v, alpha, beta
            self.detach_gamma = loss_edl_weights.get('detach_gamma', False)
            # [EDL CLS] Classification EDL weights
            self.cls_edl_weight = loss_edl_weights.get('cls_edl_weight', 1.0)
            self.cls_kl_weight = loss_edl_weights.get('cls_kl_weight', 0.1)
            self.cls_warmup_iters = loss_edl_weights.get('cls_warmup_iters', 2000)
            # [MT-ENet] Lipschitz L1 Loss (faithful to Proposition 2: U_l threshold with digamma)
            self.use_lipschitz_l1 = loss_edl_weights.get('use_lipschitz_l1', False)
            # ll1_zero_below_threshold:
            #   True  = faithful to the original MT-ENet: loss = 0 when chi^2 <= U_l, leaving it to the NLL
            #   False = pragmatic compromise: fall back to plain L1 when chi^2 <= U_l so gamma is always supervised
            self.ll1_zero_below_threshold = loss_edl_weights.get('ll1_zero_below_threshold', False)

        # [EDL Modification] Task Head Construction
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            # 1. Widen the regression heads
            heads = copy.deepcopy(common_heads)
            for k, v in heads.items():
                # [FIX] only the EDL tasks (center, height) are multiplied by 4 for NIG
                # rot, dim and vel use plain regression and keep their original width
                if k in ('rot', 'dim', 'vel'):
                    heads[k] = (v[0], v[1])
                else:
                    # EDL tasks (center, height) are multiplied by 4 for NIG: gamma, v, alpha, beta
                    heads[k] = (v[0] * 4, v[1])
            
            # 2. Classification Head: EDL Beta Distribution (α, β evidence)
            # Output width = num_cls * 2 (each class has alpha and beta evidence)
            heads.update(dict(cls_logits=(num_cls * 2, 2)))
            
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, 
                num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        if self.ensemble is not None and hasattr(self.ensemble, 'set_epoch'):
            self.ensemble.set_epoch(epoch)
    
    def _update_warmup_factor(self):
        """
        Update EDL warmup factor based on current iteration.
        Linear warmup from 0 to 1 over warmup_iters.
        Called automatically at the beginning of each loss computation.
        """
        if self.training:
            if self.warmup_iters > 0:
                self.edl_warmup_factor = min(1.0, self.current_iter / self.warmup_iters)
            else:
                self.edl_warmup_factor = 1.0
            if self.v10_edlcue_warmup_iters > 0:
                self.v10_edlcue_warmup_factor = min(
                    1.0, self.current_iter / self.v10_edlcue_warmup_iters)
            else:
                self.v10_edlcue_warmup_factor = 1.0
            
            # Debug output every debug_interval iterations
            if self.debug and self.current_iter % self.debug_interval == 0:
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                if is_main_process:
                    print(
                        f"\n[EDL Warmup] Iter {self.current_iter},"
                        f" Factor: {self.edl_warmup_factor:.4f},"
                        f" v10_edlcue: {self.v10_edlcue_warmup_factor:.4f}")
            
            # [FIX] Increment counter EVERY iteration, not just when printing debug
            self.current_iter += 1

    def _expand_task_class_param(self, value, name):
        total_num_cls = sum(self.num_classes)

        if isinstance(value, Number):
            scalar = float(value)
            return [torch.full((num_cls,), scalar, dtype=torch.float32) for num_cls in self.num_classes]

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, (list, tuple)):
            raise TypeError(f'{name} must be a scalar, list, or tuple, but got {type(value)}')

        if len(value) == total_num_cls and all(isinstance(v, Number) for v in value):
            task_values = []
            offset = 0
            for num_cls in self.num_classes:
                task_values.append(torch.tensor(value[offset:offset + num_cls], dtype=torch.float32))
                offset += num_cls
            return task_values

        if len(value) != len(self.num_classes):
            raise ValueError(
                f'{name} must be a scalar, a flat list of length {total_num_cls}, '
                f'or a per-task list of length {len(self.num_classes)}'
            )

        task_values = []
        for task_value, num_cls in zip(value, self.num_classes):
            if isinstance(task_value, Number):
                task_values.append(torch.full((num_cls,), float(task_value), dtype=torch.float32))
                continue
            if isinstance(task_value, np.ndarray):
                task_value = task_value.tolist()
            if not isinstance(task_value, (list, tuple)) or len(task_value) != num_cls:
                raise ValueError(
                    f'{name} per-task entry must be a scalar or a list of length {num_cls}, '
                    f'but got {task_value}'
                )
            if not all(isinstance(v, Number) for v in task_value):
                raise TypeError(f'{name} contains non-numeric values: {task_value}')
            task_values.append(torch.tensor(task_value, dtype=torch.float32))

        return task_values

    def _get_task_class_param(self, task_params, task_id, ref_tensor):
        return task_params[task_id].to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _build_edl_class_weights(self,
                                 labels,
                                 labels_weights,
                                 y_true,
                                 p_pred,
                                 neg_gamma,
                                 neg_balance,
                                 pred_bboxes_flatten=None,
                                 normalized_bbox_targets=None):
        base_weights = labels_weights[:, None].to(dtype=p_pred.dtype).expand_as(p_pred)
        neg_weights = neg_balance[None, :] * p_pred.clamp(min=1e-6).pow(neg_gamma[None, :])
        query_quality = labels_weights.to(dtype=p_pred.dtype).new_ones(labels.shape[0])

        if self.use_l1_quality_cls and pred_bboxes_flatten is not None and normalized_bbox_targets is not None:
            pos_mask = (labels < p_pred.shape[-1]) & torch.isfinite(normalized_bbox_targets).all(dim=-1)
            if pos_mask.any():
                code_weights = pred_bboxes_flatten.new_tensor(self.code_weights[:10])
                l1_error = torch.abs(
                    pred_bboxes_flatten[pos_mask, :10].detach() - normalized_bbox_targets[pos_mask, :10]
                )
                quality_error = (l1_error * code_weights[None, :]).mean(dim=-1, keepdim=True)
                quality_weight = torch.exp(-self.cls_quality_tau * quality_error).clamp(min=1e-3, max=1.0)
                query_quality[pos_mask] = quality_weight.squeeze(-1)

        class_weights = base_weights * (
            neg_weights * (1.0 - y_true) + query_quality[:, None] * y_true
        )
        instance_weights = class_weights.sum(dim=-1, keepdim=True)
        return query_quality, neg_weights, instance_weights

    def _compute_dynamic_kl_scale(self, y_true, p_pred, alpha, beta):
        if not self.use_dynamic_kl:
            return torch.ones_like(p_pred)
        return torch.abs(y_true - p_pred).detach()

    def _edl_cls_loss(self,
                      labels,
                      labels_weights,
                      cls_alpha,
                      cls_beta,
                      cls_avg_factor,
                      task_id,
                      pred_bboxes_flatten=None,
                      normalized_bbox_targets=None,
                      bbox_weights=None):
        alpha_flat = cls_alpha.flatten(0, -2)
        beta_flat = cls_beta.flatten(0, -2)
        num_cls = alpha_flat.shape[-1]
        neg_gamma = self._get_task_class_param(self.cls_neg_gamma, task_id, alpha_flat)
        neg_balance = self._get_task_class_param(self.cls_neg_balance, task_id, alpha_flat)

        y_true = F.one_hot(labels.clamp(0, num_cls - 1).long(), num_classes=num_cls).float()
        background_mask = labels >= num_cls
        y_true[background_mask] = 0

        S = alpha_flat + beta_flat
        p_pred = alpha_flat / S

        query_quality, neg_weights, instance_weights = self._build_edl_class_weights(
            labels,
            labels_weights,
            y_true,
            p_pred,
            neg_gamma,
            neg_balance,
            pred_bboxes_flatten=pred_bboxes_flatten,
            normalized_bbox_targets=normalized_bbox_targets,
        )

        mse = (y_true - p_pred) ** 2
        var = (p_pred * (1 - p_pred)) / (S + 1)
        cls_terms = mse + var
        loss_mse = (cls_terms.sum(dim=-1, keepdim=True) * instance_weights).sum() / cls_avg_factor

        alpha_tilde = y_true * alpha_flat + (1 - y_true) * 1.0
        beta_tilde = y_true * 1.0 + (1 - y_true) * beta_flat
        S_tilde = alpha_tilde + beta_tilde
        kl = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde) + \
             (alpha_tilde - 1) * torch.digamma(alpha_tilde) + \
             (beta_tilde - 1) * torch.digamma(beta_tilde) - \
             (S_tilde - 2) * torch.digamma(S_tilde)

        kl_scale = self._compute_dynamic_kl_scale(y_true, p_pred, alpha_flat, beta_flat)
        kl_warmup = min(1.0, self.current_iter / max(self.cls_warmup_iters, 1))
        kl_terms = kl * kl_scale
        loss_kl = kl_warmup * (kl_terms.sum(dim=-1, keepdim=True) * instance_weights).sum() / cls_avg_factor

        if self.debug and self.training:
            pos_mask = labels < num_cls
            neg_entry_mask = ~y_true.bool()
            neg_values = neg_weights[neg_entry_mask]
            debug_stats = dict(
                pos_q_mean=query_quality[pos_mask].mean().item()
                if pos_mask.any() else float('nan'),
                neg_w_mean=neg_values.mean().item() if neg_values.numel() > 0 else float('nan'),
                neg_w_p95=torch.quantile(neg_values, 0.95).item() if neg_values.numel() > 0 else float('nan'),
                inst_w_mean=instance_weights.mean().item(),
                cal_mean=kl_scale.mean().item(),
                max_p_mean=p_pred.max(dim=-1).values.mean().item(),
            )
            self._edl_debug_stats.append(debug_stats)

        return self.cls_edl_weight * loss_mse + self.cls_kl_weight * loss_kl
    
    def lipschitz_l1_loss(self, target, gamma, v, alpha, beta, eps=1e-6):
        """
        Faithful MT-ENet Lipschitz-modified L1 Loss (Paper: Proposition 2)
        
        Three design points carried over from the original MT-ENet:
        1. v, alpha, beta are detached, so L_LMSE only produces a gradient for gamma (Section 5.1)
        2. The U_l threshold follows from Proposition 2 (it involves the digamma function)
        3. Lipschitz L1 is used only when chi^2 > U_l; otherwise it falls back to plain L1
        
        Behaviour:
        - chi^2 > U_l: LL1 = |y - gamma| / L, a bounded gradient that compensates when the NLL gradient vanishes
        - chi^2 <= U_l: plain L1 = |y - gamma|; the prediction is accurate enough not to disturb NLL uncertainty learning
        
        where L = sqrt(2*beta*(1+v)/(alpha*v)) is the Lipschitz constant.
        
        Args:
            target: Ground truth values [N, D]
            gamma: NIG predicted mean (γ) [N, D]
            v, alpha, beta: NIG distribution parameters [N, D]
            eps: Numerical stability constant
        
        Returns:
            Lipschitz L1 loss (element-wise, unreduced) [N, D]
        """
        error = torch.abs(target - gamma)
        chi_sq = (target - gamma).detach().pow(2)  # detached: used only for the threshold comparison
        
        # --- Modification: v/alpha/beta are not detached, so the LL1 gradient reaches v ---
        # The original MT-ENet detach leaves v to the NIG NLL alone, which causes evidence contraction in 3D detection
        # Keeping LL1's gradient w.r.t. v, d(LL1)/dv = |y - gamma| * d(1/L)/dv, prevents v from collapsing
        
        # --- [Proposition 2] compute the U_l threshold (detached so U_l itself carries no gradient) ---
        v_d = v.detach()
        alpha_d = alpha.detach()
        beta_d = beta.detach()
        psi_diff = torch.digamma(alpha_d + 0.5) - torch.digamma(alpha_d)
        log_v_ratio = torch.log(v_d / (v_d + 1.0) + eps)
        C_v = (1.0 / (2.0 * (v_d + 1.0))
               - log_v_ratio
               - psi_diff / 2.0)
        C_v = C_v.clamp(min=0.01)
        base_var_for_threshold = 2.0 * beta_d * (1.0 + v_d) / (alpha_d * v_d + eps)
        U_l = (base_var_for_threshold * C_v).clamp(min=1e-8)
        
        # --- Lipschitz constant L = sqrt(2*beta*(1+v)/(alpha*v)); not detached, so gradients reach v/alpha/beta ---
        L_sq = (2.0 * beta * (1.0 + v) / (alpha * v + eps)).clamp(min=0.01, max=100.0)
        L = torch.sqrt(L_sq)
        
        # --- [Proposition 2] conditional: Lipschitz L1 only where chi^2 > U_l ---
        # chi^2 > U_l: the error is large and the NLL gradient may vanish, so LL1 compensates
        # chi^2 <= U_l: accurate enough, so plain L1 is used and NLL uncertainty learning is left alone
        mask = (chi_sq > U_l).float()
        
        # LL1 behaviour depends on ll1_zero_below_threshold:
        #   True:  original MT-ENet - loss = 0 when chi^2 <= U_l
        #   False: fall back to plain L1 - loss = |y - gamma| when chi^2 <= U_l
        if self.ll1_zero_below_threshold:
            # Faithful to the original: a loss only where chi^2 > U_l
            loss = mask * (error / L)
        else:
            # Pragmatic compromise: fall back to plain L1 when chi^2 <= U_l
            loss = mask * (error / L) + (1.0 - mask) * error
        
        return loss

    
    def _extract_gamma_and_prob(self, pred_bboxes_hybrid, pred_logits_beta):
        """
        Extract Gamma (Mean) from Hybrid NIG/Standard Tensor.
        Handles mixed dimensions: EDL (4x) for Reg, Standard (1x) for Rot.
        
        Args:
            pred_bboxes_hybrid: [..., 34] (Mixed layout)
            pred_logits_beta: [..., num_cls * 2]
        Returns:
            pred_gamma: [..., 10] (Standard bbox format for matcher)
            probs: [..., num_cls] (Expected probability)
        """
        # 1. Define the layout (must match the head output and loss order exactly)
        # Center(8) + Height(4) + Dim(3) + Rot(2) + Vel(2) = 19 channels
        task_layout = [
            {'name': 'center', 'dim': 2, 'use_edl': True},
            {'name': 'height', 'dim': 1, 'use_edl': True},
            {'name': 'dim',    'dim': 3, 'use_edl': False},  # reverted to plain L1
            {'name': 'rot',    'dim': 2, 'use_edl': False}, # [Standard]
            {'name': 'vel',    'dim': 2, 'use_edl': False},
        ]
        
        gammas = []
        ptr = 0
        
        # 2. Parse in order and extract gamma
        for task in task_layout:
            if task['use_edl']:
                # EDL tasks: 4 channels per dim, [gamma, v, alpha, beta]
                width = task['dim'] * 4
                slice_data = pred_bboxes_hybrid[..., ptr : ptr + width]
                
                # Reshape to [..., dim, 4] and take index 0 (Gamma)
                # slice_data shape: [..., width] -> [..., dim, 4]
                gamma = slice_data.view(*slice_data.shape[:-1], task['dim'], 4)[..., 0]
                gammas.append(gamma)
                
                ptr += width
            else:
                # Plain tasks (rotation): one value per dimension, taken directly
                width = task['dim']
                gamma = pred_bboxes_hybrid[..., ptr : ptr + width]
                gammas.append(gamma)
                
                ptr += width
                
        # 3. Reassemble into the standard bbox format [..., 10] used for assigner cost
        pred_gamma = torch.cat(gammas, dim=-1)
        
        # 4. Classification probabilities - plain sigmoid (the Beta distribution is gone)
        probs = pred_logits_beta.sigmoid()
        
        return pred_gamma, probs

    def init_weights(self):
        super(EDLHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0)  # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :]) \
                    / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None,
                      None, :]
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(
            self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        proj_points = torch.einsum('bnd, bvcd -> bvnc',
                                   torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1),
                                   lidars2imgs)

        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (
                proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape)

        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (
                proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        proj_points_clone = torch.cat(
            [proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None,
                                                      :]) \
                          / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(
            self.pc_range[:3]))[None, None, None, :]

        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        rv_embeds = self._rv_query_embed(ref_points, img_metas)
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas, return_distill_queries=False):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        x = self.shared_conv(x)

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)

        modalities = copy.deepcopy(self.modalities["train" if self.training else "test"])
        outs_dec = self.transformer(
            x, x_img, bev_query_embeds, rv_query_embeds, bev_pos_embeds, rv_pos_embeds, img_metas,
            attn_masks=attn_mask, modalities=modalities
        )
        num_queries_per_modality = [m.shape[2] for m in outs_dec]
        outs_dec = torch.cat(outs_dec, dim=2)
        outs_dec = torch.nan_to_num(outs_dec)
        distill_queries = None
        if return_distill_queries:
            # Training uses DN queries while eval teachers do not. Strip DN
            # padding per modality so student and teacher query indices align.
            query_parts = []
            start = 0
            pad_size = int(mask_dict.get('pad_size', 0)) if mask_dict else 0
            for width in num_queries_per_modality:
                part = outs_dec[-1, :, start:start + width, :]
                if pad_size > 0:
                    part = part[:, pad_size:, :]
                query_parts.append(part)
                start += width
            distill_queries = torch.cat(query_parts, dim=1)

        reference = inverse_sigmoid(reference_points.clone())
        reference = reference.repeat(1, len(modalities), 1)

        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)

            # [EDL Modification] NIG Parameter Processing
            # Adapted for the transformer's deep supervision (one output per decoder layer)
            # Strategy: keep the full NIG parameters for the loss while emitting the same absolute coordinates as upstream
            
            # --- Center Processing ---
            center_raw = outs['center']
            nb_dec, bs, nq, c_dim = center_raw.shape
            center_nig = center_raw.view(nb_dec, bs, nq, 2, 4) 
            
            # Extract gamma, add the reference, then normalize with sigmoid
            gamma_center_norm = (center_nig[..., 0] + reference[None, :, :, :2]).sigmoid()
            
            # Denormalize gamma to the point-cloud range (absolute coordinates), as upstream does
            gamma_center_abs = gamma_center_norm.new_zeros(gamma_center_norm.shape)
            gamma_center_abs[..., 0:1] = gamma_center_norm[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            gamma_center_abs[..., 1:2] = gamma_center_norm[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            
            # [KEY FIX] keep the full NIG parameters for the loss (gamma stays normalized)
            # Layout: [nb_dec, bs, nq, 8] = [gamma_norm, v, alpha, beta] x 2
            outs['center_nig'] = torch.cat([gamma_center_norm.unsqueeze(-1), center_nig[..., 1:]], dim=-1).flatten(-2, -1)
            # [KEY FIX] the main output is in absolute coordinates, matching upstream MEFormer
            outs['center'] = gamma_center_abs

            # --- Height Processing ---
            height_raw = outs['height']
            nb_dec, bs, nq, h_dim = height_raw.shape
            height_nig = height_raw.view(nb_dec, bs, nq, 1, 4)
            
            gamma_height_norm = (height_nig[..., 0] + reference[None, :, :, 2:3]).sigmoid()
            gamma_height_abs = gamma_height_norm.new_zeros(gamma_height_norm.shape)
            gamma_height_abs[..., 0:1] = gamma_height_norm[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            # [KEY FIX] keep the full NIG parameters for the loss
            outs['height_nig'] = torch.cat([gamma_height_norm.unsqueeze(-1), height_nig[..., 1:]], dim=-1).flatten(-2, -1)
            outs['height'] = gamma_height_abs
            
            # --- dim processing (plain regression like rot/vel; no EDL) ---
            # dim is already [nb_dec, bs, nq, 3] and needs no further processing
            # The 'dim_nig' key is no longer produced

            
            # --- Classification Processing (EDL Beta Distribution) ---
            # cls_raw shape: [nb_dec, bs, nq, num_cls * 2]
            cls_raw = outs['cls_logits']
            nb_dec, bs, nq, c_cls = cls_raw.shape
            num_cls = c_cls // 2
            
            # Split the alpha and beta evidence
            # Layout: [alpha_1, beta_1, alpha_2, beta_2, ...] -> reshape to [nb_dec, bs, nq, num_cls, 2]
            cls_reshaped = cls_raw.view(nb_dec, bs, nq, num_cls, 2)
            alpha_evidence = cls_reshaped[..., 0]  # [nb_dec, bs, nq, num_cls]
            beta_evidence = cls_reshaped[..., 1]   # [nb_dec, bs, nq, num_cls]
            
            # Convert to alpha and beta (softplus keeps them positive; +1 keeps alpha > 1)
            alpha = F.softplus(alpha_evidence) + 1.0
            beta = F.softplus(beta_evidence) + 1.0
            
            # Expected probability, used for matching and inference
            prob = alpha / (alpha + beta)
            
            # Keep the EDL parameters for the loss
            outs['cls_alpha'] = alpha
            outs['cls_beta'] = beta
            # The main output is a logit, for compatibility with the sigmoid CE loss
            # Use log-odds: log(p / (1 - p)) = log(alpha / beta)
            outs['cls_logits'] = torch.log(alpha / beta + 1e-8)
            
            # --- vel processing (plain regression like rot; no EDL) ---
            # vel is already [nb_dec, bs, nq, 2] and needs no further processing
            # vel_nig is no longer created, because vel is a plain regression output

            if distill_queries is not None:
                outs['rev19_distill_queries'] = distill_queries

            if self.use_ensemble and "fused" in modalities:
                # [EDL-PME] pass each modality's NIG parameters in for the uncertainty computation
                pc_range = torch.Tensor(self.pc_range).to(reference.device)
                
                # Collect each modality's center/height NIG parameters for the PME uncertainty computation
                nig_params_dict = {}
                center_nig_last = outs['center_nig'][-1]  # [bs, total_nq, 8]
                height_nig_last = outs['height_nig'][-1]  # [bs, total_nq, 4]
                center_nig_split = center_nig_last.split(
                    num_queries_per_modality, dim=1)
                height_nig_split = height_nig_last.split(
                    num_queries_per_modality, dim=1)
                for idx, m in enumerate(modalities):
                    nig_params_dict[m] = {
                        'center_nig': center_nig_split[idx],
                        'height_nig': height_nig_split[idx],
                    }
                
                outs = self.ensemble(
                    outs_dec,
                    reference,
                    outs,
                    nig_params_dict,
                    modalities,
                    num_queries_per_modality,
                    task_id,
                    pc_range,
                    attn_mask,
                    img_metas=img_metas
                )
                modalities = ["ensemble"]
                num_queries_per_modality = [outs["center"].shape[2]]

            if distill_queries is not None:
                outs['rev19_distill_queries'] = distill_queries

            if self.training:
                # The upstream split logic
                # No change needed here: the split is on dim=2, the query dimension
                # [LOGGING] skip the PME side-information keys - they are dicts, not tensors
                _skip_keys = {
                    'modalities', 'per_modality_unc', 'modality_weights',
                    'selected_modality', 'fused_weight', 'selector_entropy',
                    'selector_weight_stats', 'selector_top1_ratio',
                    'selector_fused_weight_mean', 'selector_per_modality_unc',
                    'selector_modalities', 'selector_probs_last',
                    'selector_distance_prior_probs_last',
                    'selector_alpha_res_last',
                    'selector_source_cls_logits', 'selector_source_center',
                    'selector_source_height', 'selector_relative_unc_summary',
                    'selector_absolute_unc_summary',
                    'selector_disagreement_summary',
                    'selector_score_component_stats',
                    'selector_quality_top1_ratio',
                    'selector_distance_prior_top1_ratio',
                    'selector_alpha_res_mean',
                    'selector_alpha_res_hard_query_mean',
                    'selector_residual_usage_loss',
                    'selector_residual_override_ratio',
                    'selector_uncertainty_tail_query_ratio',
                    'selector_distance_bin_abnormality_stats',
                    'selector_teacher_top1_ratio', 'selector_pred_top1_ratio',
                    'selector_agreement_ratio',
                    'selector_source_cls_unc',
                    'v5_query_switch_triggered',
                    'v5_query_selected_modality_idx',
                    'v5_query_candidate_modality_idx',
                    'v5_query_fused_cls_anom',
                    'v5_query_fused_center_anom',
                    'v5_query_fused_height_anom',
                    'v5_query_fused_anomaly_score',
                    'v5_query_reference_distance_bin',
                    'v5_query_reference_fallback_level',
                    'v5_query_pred_class',
                    'v5_fused_cls_unc_last',
                    'v5_fused_center_unc_last',
                    'v5_fused_height_unc_last',
                    'v5_switch_ratio',
                    'v5_keep_fused_ratio',
                    'v5_switch_to_bev_ratio',
                    'v5_switch_to_img_ratio',
                    'v5_fused_cls_anom_mean',
                    'v5_fused_center_anom_mean',
                    'v5_fused_height_anom_mean',
                    'v5_fused_combined_anom_mean',
                    'v5_trigger_ratio_over_threshold',
                    'v5_bin_coverage_ratio',
                    'v5_fallback_class_ratio',
                    'v5_fallback_global_ratio',
                    'v5_distance_prior_candidate_ratio',
                    'v5_distance_prior_switched_ratio',
                    'v5_switch_agreement_with_prior',
                    'v5_head_shift',
                    'v7_img_router_logits_last',
                    'v7_bev_router_logits_last',
                    'v7_ref_fused_center_unc',
                    'v7_ref_bev_center_unc',
                    'v7_ref_img_center_unc',
                    'v7_ref_fused_distance',
                    'v7_ref_bev_distance',
                    'v7_ref_img_distance',
                    'v7_ref_pred_class',
                    'v8_fused_keep',
                    'v8_center_fused_keep',
                    'v8_damage_gate',
                    'v8_router_variant',
                    'v8_bev_damage',
                    'v8_img_damage',
                    'v8_dual_damage',
                    'v8_context_bev_damage',
                    'v8_context_img_damage',
                    'v8_a4_bev_damage_z',
                    'v8_a4_img_damage_z',
                    'v8_a4_cls_unc_z_bev',
                    'v8_a4_cls_unc_z_img',
                    'v8_a4_route_score_bev',
                    'v8_a4_route_score_img',
                    'v8_gscrg_raw_unc_fused',
                    'v8_gscrg_raw_unc_bev',
                    'v8_gscrg_raw_unc_img',
                    'v8_bev_reliability',
                    'v8_img_reliability',
                    'v8_reliability_margin',
                    'v8_non_fused_probs',
                    'v8_non_fused_choice',
                    'v8_center_source',
                    'v8_gate_reason',
                    'v8_damage_state',
                    'v8_reliability_state',
                    'v8_high_damage_query',
                    'v8_center_route_allowed',
                    'v8_soft_weight_fused',
                    'v8_soft_weight_bev',
                    'v8_soft_weight_img',
                    'v8_output_delta_cls',
                    'v8_output_delta_center',
                    'v8_output_delta_height',
                    'v8_fused_keep_mean',
                    'v8_center_fused_keep_mean',
                    'v8_damage_gate_mean',
                    'v8_high_damage_query_ratio',
                    'v8_center_route_allowed_ratio',
                    'v8_bev_damage_mean',
                    'v8_img_damage_mean',
                    'v8_dual_damage_mean',
                    'v8_context_bev_damage_mean',
                    'v8_context_img_damage_mean',
                    'v8_a4_anchor_ready',
                    'v8_a4_bev_damage_z_mean',
                    'v8_a4_img_damage_z_mean',
                    'v8_a4_cls_unc_z_bev_mean',
                    'v8_a4_cls_unc_z_img_mean',
                    'v8_a4_route_score_bev_mean',
                    'v8_a4_route_score_img_mean',
                    'v8_a4_clean_anchor_update_count',
                    'v8_bev_reliability_mean',
                    'v8_img_reliability_mean',
                    'v8_reliability_margin_mean',
                    'v8_non_fused_weight_ratio',
                    'v8_soft_weight_ratio',
                    'v8_damage_state_ratio',
                    'v8_reliability_state_ratio',
                    'v8_selected_source_ratio',
                    'v8_fused_keep_by_state',
                    'v8_non_fused_entropy_mean',
                    'v8_output_delta_cls_mean',
                    'v8_output_delta_center_mean',
                    'v8_output_delta_height_mean',
                    'v8_train_high_damage_ratio',
                    'v8_train_clean_fused_keep',
                    'v8_train_damage_fused_keep',
                    'v8_train_single_damage_keep',
                    'v8_train_dual_damage_keep',
                    'v8_train_domain_correct_ratio',
                    'v8_train_domain_reliability_loss',
                }
                total_queries = sum(num_queries_per_modality)
                for key in list(outs.keys()):
                    if key in _skip_keys:
                        continue
                    if not isinstance(outs[key], torch.Tensor):
                        continue
                    if outs[key].dim() < 3 or outs[key].shape[2] != total_queries:
                        continue
                    outs[key] = list(outs[key].split(num_queries_per_modality, dim=2))

            outs["modalities"] = modalities

            if mask_dict and mask_dict['pad_size'] > 0:
                # Denoising branch
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    if key in [
                        "modalities",
                        "per_modality_unc",
                        "modality_weights",
                        "selected_modality",
                        "fused_weight",
                        "selector_entropy",
                        "selector_weight_stats",
                        "selector_top1_ratio",
                        "selector_fused_weight_mean",
                        "selector_per_modality_unc",
                        "selector_modalities",
                        "selector_probs_last",
                        "selector_distance_prior_probs_last",
                        "selector_alpha_res_last",
                        "selector_source_cls_logits",
                        "selector_source_center",
                        "selector_source_height",
                        "selector_relative_unc_summary",
                        "selector_absolute_unc_summary",
                        "selector_disagreement_summary",
                        "selector_score_component_stats",
                        "selector_quality_top1_ratio",
                        "selector_distance_prior_top1_ratio",
                        "selector_alpha_res_mean",
                        "selector_alpha_res_hard_query_mean",
                        "selector_residual_usage_loss",
                        "selector_residual_override_ratio",
                        "selector_uncertainty_tail_query_ratio",
                        "selector_distance_bin_abnormality_stats",
                        "selector_teacher_top1_ratio",
                        "selector_pred_top1_ratio",
                        "selector_agreement_ratio",
                        "selector_source_cls_unc",
                        "v5_query_switch_triggered",
                        "v5_query_selected_modality_idx",
                        "v5_query_candidate_modality_idx",
                        "v5_query_fused_cls_anom",
                        "v5_query_fused_center_anom",
                        "v5_query_fused_height_anom",
                        "v5_query_fused_anomaly_score",
                        "v5_query_reference_distance_bin",
                        "v5_query_reference_fallback_level",
                        "v5_query_pred_class",
                        "v5_fused_cls_unc_last",
                        "v5_fused_center_unc_last",
                        "v5_fused_height_unc_last",
                        "v5_switch_ratio",
                        "v5_keep_fused_ratio",
                        "v5_switch_to_bev_ratio",
                        "v5_switch_to_img_ratio",
                        "v5_fused_cls_anom_mean",
                        "v5_fused_center_anom_mean",
                        "v5_fused_height_anom_mean",
                        "v5_fused_combined_anom_mean",
                        "v5_trigger_ratio_over_threshold",
                        "v5_bin_coverage_ratio",
                        "v5_fallback_class_ratio",
                        "v5_fallback_global_ratio",
                        "v5_distance_prior_candidate_ratio",
                        "v5_distance_prior_switched_ratio",
                        "v5_switch_agreement_with_prior",
                        "v5_head_shift",
                        "v7_img_router_logits_last",
                        "v7_bev_router_logits_last",
                        "v7_ref_fused_center_unc",
                        "v7_ref_bev_center_unc",
                        "v7_ref_img_center_unc",
                        "v7_ref_fused_distance",
                        "v7_ref_bev_distance",
                        "v7_ref_img_distance",
                        "v7_ref_pred_class",
                        "v8_fused_keep",
                        "v8_center_fused_keep",
                        "v8_damage_gate",
                        "v8_router_variant",
                        "v8_bev_damage",
                        "v8_img_damage",
                        "v8_dual_damage",
                        "v8_context_bev_damage",
                        "v8_context_img_damage",
                        "v8_a4_bev_damage_z",
                        "v8_a4_img_damage_z",
                        "v8_a4_cls_unc_z_bev",
                        "v8_a4_cls_unc_z_img",
                        "v8_a4_route_score_bev",
                        "v8_a4_route_score_img",
                        "v8_gscrg_raw_unc_fused",
                        "v8_gscrg_raw_unc_bev",
                        "v8_gscrg_raw_unc_img",
                        "v8_bev_reliability",
                        "v8_img_reliability",
                        "v8_reliability_margin",
                        "v8_non_fused_probs",
                        "v8_non_fused_choice",
                        "v8_center_source",
                        "v8_gate_reason",
                        "v8_damage_state",
                        "v8_reliability_state",
                        "v8_high_damage_query",
                        "v8_center_route_allowed",
                        "v8_soft_weight_fused",
                        "v8_soft_weight_bev",
                        "v8_soft_weight_img",
                        "v8_output_delta_cls",
                        "v8_output_delta_center",
                        "v8_output_delta_height",
                        "v8_fused_keep_mean",
                        "v8_center_fused_keep_mean",
                        "v8_damage_gate_mean",
                        "v8_high_damage_query_ratio",
                        "v8_center_route_allowed_ratio",
                        "v8_bev_damage_mean",
                        "v8_img_damage_mean",
                        "v8_dual_damage_mean",
                        "v8_context_bev_damage_mean",
                        "v8_context_img_damage_mean",
                        "v8_a4_anchor_ready",
                        "v8_a4_bev_damage_z_mean",
                        "v8_a4_img_damage_z_mean",
                        "v8_a4_cls_unc_z_bev_mean",
                        "v8_a4_cls_unc_z_img_mean",
                        "v8_a4_route_score_bev_mean",
                        "v8_a4_route_score_img_mean",
                        "v8_a4_clean_anchor_update_count",
                        "v8_bev_reliability_mean",
                        "v8_img_reliability_mean",
                        "v8_reliability_margin_mean",
                        "v8_non_fused_weight_ratio",
                        "v8_soft_weight_ratio",
                        "v8_damage_state_ratio",
                        "v8_reliability_state_ratio",
                        "v8_selected_source_ratio",
                        "v8_fused_keep_by_state",
                        "v8_non_fused_entropy_mean",
                        "v8_output_delta_cls_mean",
                        "v8_output_delta_center_mean",
                        "v8_output_delta_height_mean",
                        "v8_train_high_damage_ratio",
                        "v8_train_clean_fused_keep",
                        "v8_train_damage_fused_keep",
                        "v8_train_single_damage_keep",
                        "v8_train_dual_damage_keep",
                        "v8_train_domain_correct_ratio",
                        "v8_train_domain_reliability_loss",
                    ]:
                        continue
                    if not isinstance(outs[key], list):
                        continue
                    outs["dn_" + key] = []
                    for i in range(len(outs[key])):
                        outs['dn_' + key].append(outs[key][i][:, :, :mask_dict['pad_size'], :])
                        outs[key][i] = outs[key][i][:, :, mask_dict['pad_size']:, :]

                outs['dn_mask_dict'] = task_mask_dict

            ret_dicts.append(outs)

        if self.training and self.debug:
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            if is_main_process:
                self.debug_counter += 1

        if return_distill_queries:
            return ret_dicts, distill_queries
        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None, return_distill_queries=False):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        if return_distill_queries:
            results = multi_apply(
                self.forward_single,
                pts_feats,
                img_feats,
                img_metas,
                [return_distill_queries for _ in range(len(pts_feats))])
            return tuple(results[0]), results[1]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas)

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:

            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes,),
                                        num_classes,
                                        dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            
            # bbox_targets - uses the gt_bboxes dimensionality (usually 9)
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            
            # [KEY FIX] bbox_weights is fixed at 10 dims, matching upstream MEFormerHead
            # so that _loss_single_task can apply code_weights directly
            bbox_weights = torch.zeros(num_bboxes, 10, device=bbox_pred.device, dtype=bbox_pred.dtype)
            bbox_weights[pos_inds] = 1.0

            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks \
            = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)

        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits, return_pos_inds=False):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])

        targets = (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                   task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)
        if not return_pos_inds:
            return targets

        task_pos_inds_list = [
            [pos_inds_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))]
            for task_id in range(task_num)
        ]
        return targets + (task_pos_inds_list,)

    def _prepare_assign_outputs_single_sample(self, pred_bboxes, pred_logits):
        """Convert one-sample predictions into assigner-ready absolute boxes/logits."""
        assign_bboxes = []
        assign_logits = []

        for pred_nig, pred_cls in zip(pred_bboxes, pred_logits):
            ptr = 0

            center_nig = pred_nig[..., ptr:ptr + 8].view(-1, 2, 4)
            gamma_center_norm = center_nig[..., 0]
            ptr += 8

            height_nig = pred_nig[..., ptr:ptr + 4].view(-1, 1, 4)
            gamma_height_norm = height_nig[..., 0]
            ptr += 4

            gamma_dim = pred_nig[..., ptr:ptr + 3]
            ptr += 3

            gamma_rot = pred_nig[..., ptr:ptr + 2]
            ptr += 2

            gamma_vel = pred_nig[..., ptr:ptr + 2]

            gamma_center_abs = gamma_center_norm.clone()
            gamma_center_abs[..., 0] = (
                gamma_center_norm[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            gamma_center_abs[..., 1] = (
                gamma_center_norm[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            gamma_height_abs = gamma_height_norm * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            gamma_abs = torch.cat([
                gamma_center_abs,
                gamma_height_abs,
                gamma_dim,
                gamma_rot,
                gamma_vel,
            ], dim=-1)

            assign_bboxes.append(gamma_abs)
            assign_logits.append(pred_cls)

        return assign_bboxes, assign_logits
        
    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          cls_alpha,
                          cls_beta,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg,
                          task_id):
        """
        Compute Loss for a single task.
        
        EDL bbox loss structure:
        loss_bbox = l1_weight * L1_Loss + nig_nll_weight * NIG_NLL + nig_reg_weight * NIG_Reg
        
        EDL classification loss structure:
        loss_cls = cls_edl_weight * Digamma_NLL + cls_kl_weight * KL_Reg
        """
        
        # -----------------------------------------------------------
        # 1. Data preparation (identical to upstream)
        # -----------------------------------------------------------
        labels = torch.cat(labels_list, dim=0) 
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0) 
        bbox_weights = torch.cat(bbox_weights_list, dim=0) 

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)
        
        # -----------------------------------------------------------
        # 2. Parse the NIG parameters and extract gamma (the predicted mean)
        # -----------------------------------------------------------
        task_configs = [
            {'name': 'center', 'dim': 2, 'use_edl': True},
            {'name': 'height', 'dim': 1, 'use_edl': True},
            {'name': 'dim',    'dim': 3, 'use_edl': False},
            {'name': 'rot',    'dim': 2, 'use_edl': False},
            {'name': 'vel',    'dim': 2, 'use_edl': False},
        ]
        
        all_gammas = []
        nig_params_list = []  # NIG parameters kept for the EDL loss
        ptr_pred = 0
        
        for task in task_configs:
            width_pred = task['dim'] * 4 if task['use_edl'] else task['dim']
            pred_slice = pred_bboxes_flatten[:, ptr_pred : ptr_pred + width_pred]
            ptr_pred += width_pred
            
            if task['use_edl']:
                # Extract gamma, v, alpha and beta from the NIG parameters
                # [KEY FIX] EDLNIGHead already applied the activations - do not apply them again
                # EDLNIGHead emits:
                #   - gamma: raw logits (no activation)
                #   - v: softplus(v_raw) + epsilon (already activated, v > 0)
                #   - alpha: softplus(alpha_raw) + 1.0 + epsilon (already activated, alpha > 1)
                #   - beta: softplus(beta_raw) + epsilon (already activated, beta > 0)
                pred_reshaped = pred_slice.view(-1, task['dim'], 4)
                gamma = pred_reshaped[..., 0]  # gamma: unconstrained, use as-is
                v = pred_reshaped[..., 1]      # v: already activated, use as-is
                alpha = pred_reshaped[..., 2]  # alpha: already activated, use as-is
                beta = pred_reshaped[..., 3]   # beta: already activated, use as-is
                
                # [SAFETY] clamp only, for numerical stability - no activation here
                v = v.clamp(min=1e-6)
                alpha = alpha.clamp(min=1.001)  # keep alpha > 1
                beta = beta.clamp(min=1e-6)
                
                all_gammas.append(gamma)
                nig_params_list.append({
                    'name': task['name'],
                    'gamma': gamma, 'v': v, 'alpha': alpha, 'beta': beta
                })
            else:
                # rot is used directly; it has no NIG parameters
                all_gammas.append(pred_slice)
                nig_params_list.append({
                    'name': task['name'],
                    'gamma': pred_slice, 'v': None, 'alpha': None, 'beta': None
                })
        
        # Assemble into the standard [N, 10] bbox format
        # Network output order: [center(2), height(1), dim(3), rot(2), vel(2)]
        # i.e. [cx, cy, cz, w, l, h, sin, cos, vx, vy] (prediction order)
        pred_gamma_full = torch.cat(all_gammas, dim=-1)
        
        # -----------------------------------------------------------
        # [CRITICAL FIX] reorder to match the target layout normalize_bbox expects
        # target layout:     [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        # prediction layout: [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        # so cz (index 2) and w, l (indices 3:5) swap places
        # -----------------------------------------------------------
        pred_gamma_reordered = torch.cat([
            pred_gamma_full[:, 0:2],   # cx, cy (unchanged)
            pred_gamma_full[:, 3:5],   # w, l (first two of dim, moved to index 2:4)
            pred_gamma_full[:, 2:3],   # cz (from height, moved to index 4:5)
            pred_gamma_full[:, 5:6],   # h (third of dim, moved to index 5:6)
            pred_gamma_full[:, 6:10],  # rot, vel (unchanged)
        ], dim=-1)
        
        # -----------------------------------------------------------
        # [KEY FIX] convert pred_gamma to absolute coordinates / log scale
        # The order is now [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        # -----------------------------------------------------------
        pred_gamma_abs = pred_gamma_reordered.clone()
        pc_range = self.pc_range
        
        # center (0:2): normalized [0,1] -> absolute PC range
        pred_gamma_abs[:, 0] = pred_gamma_reordered[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        pred_gamma_abs[:, 1] = pred_gamma_reordered[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        
        # dim w, l (2:4): use the raw gamma directly (the network learns the log scale)
        # No conversion needed
        
        # height cz (4:5): normalized [0,1] -> absolute PC range
        pred_gamma_abs[:, 4] = pred_gamma_reordered[:, 4] * (pc_range[5] - pc_range[2]) + pc_range[2]
        
        # dim h (5:6): use the raw gamma directly
        # No conversion needed
        
        # rot (6:8) and vel (8:10) are left unchanged
        
        # -----------------------------------------------------------
        # Normalize the target bbox (as upstream)
        # -----------------------------------------------------------
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        
        # [FIX] align the dimensions
        if normalized_bbox_targets.shape[-1] < pred_gamma_abs.shape[-1]:
            padding_size = pred_gamma_abs.shape[-1] - normalized_bbox_targets.shape[-1]
            padding = torch.zeros(
                (*normalized_bbox_targets.shape[:-1], padding_size),
                device=normalized_bbox_targets.device,
                dtype=normalized_bbox_targets.dtype
            )
            normalized_bbox_targets = torch.cat([normalized_bbox_targets, padding], dim=-1)
        
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self._edl_cls_loss(
            labels,
            labels_weights,
            cls_alpha,
            cls_beta,
            cls_avg_factor,
            task_id,
            pred_bboxes_flatten=pred_gamma_abs,
            normalized_bbox_targets=normalized_bbox_targets,
            bbox_weights=bbox_weights,
        )

        # -----------------------------------------------------------
        # 4. BBox Regression Loss (EDL + L1)
        # -----------------------------------------------------------
        
        # Filter NaN (as upstream)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        
        # [KEY FIX] handle bbox_weights correctly
        # bbox_weights comes from get_targets: pos_inds -> 1.0, neg_inds -> 0.0
        # The shape should be [N, 10], with weight 1.0 only on the positives
        if bbox_weights.shape[-1] != 10:
            bbox_weights_padded = torch.zeros(
                bbox_weights.shape[0], 10,
                device=bbox_weights.device,
                dtype=bbox_weights.dtype
            )
            bbox_weights_padded[:, :bbox_weights.shape[-1]] = bbox_weights
            bbox_weights = bbox_weights_padded
        
        # [KEY FIX] positive-sample mask, used by both the L1 and NIG losses
        # bbox_weights[pos_inds] = 1.0, so sum > 0 marks a positive
        pos_mask_for_bbox = bbox_weights.sum(dim=-1) > 0
        
        # [KEY FIX] apply code_weights, matching upstream MEFormerHead
        # code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        code_weights = pred_gamma_abs.new_tensor(self.code_weights)
        bbox_weights_coded = bbox_weights * code_weights[None, :]
        
        # [KEY FIX] a valid sample is a positive AND numerically finite
        # This keeps the L1 loss on positives only, matching upstream MEFormer
        valid_mask = pos_mask_for_bbox & isnotnan
        num_valid = valid_mask.sum().item()
        
        # Fetch the EDL loss weights (with the iteration-based warmup factor applied)
        l1_weight = self.l1_weight
        l1_weight_standard = self.l1_weight_standard  # separate weight for rot/vel
        nig_nll_weight = self.nig_nll_weight
        nig_reg_weight = self.nig_reg_weight
        
        # ============================================================
        # 4.1 L1 loss - EDL tasks and plain regression handled separately
        # [KEY FIX] the EDL tasks (center/height) and plain regression (dim/rot/vel) are computed separately
        # pred_gamma_abs layout: [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        # EDL indices: 0:2 (center) + 4:5 (height) = 3 dims (non-contiguous)
        # Plain indices: 2:4 (dim w, l) + 5:10 (dim h + rot + vel) = 7 dims (non-contiguous)
        # [MT-ENet] when use_lipschitz_l1=True, the adaptive-gradient Lipschitz L1 is used
        # ============================================================
        if num_valid > 0:
            # Plain regression tasks: dim(2:4, 5:6) + rot(6:8) + vel(8:10), always plain L1
            std_pred = torch.cat([pred_gamma_abs[valid_mask, 2:4], pred_gamma_abs[valid_mask, 5:10]], dim=-1)
            std_target = torch.cat([normalized_bbox_targets[valid_mask, 2:4], normalized_bbox_targets[valid_mask, 5:10]], dim=-1)
            std_weights = torch.cat([bbox_weights_coded[valid_mask, 2:4], bbox_weights_coded[valid_mask, 5:10]], dim=-1)
            loss_l1_std = self.loss_bbox(
                std_pred, std_target, std_weights,
                avg_factor=max(num_total_pos, 1)
            )
            
            # EDL tasks: center(0:2) + height(4:5) (non-contiguous)
            if self.use_lipschitz_l1:
                # [MT-ENet] Lipschitz L1 adapts the gradient to the NIG uncertainty
                # Prepare normalized targets for Lipschitz L1, on the same scale as gamma
                # [CRITICAL FIX] the target layout is [cx, cy, w, l, cz, h, sin, cos, vx, vy]
                target_for_ll1 = normalized_bbox_targets[valid_mask].clone()
                
                # Re-normalize center to [0,1] (index 0:2 = cx, cy)
                target_for_ll1[:, 0] = (normalized_bbox_targets[valid_mask, 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
                target_for_ll1[:, 1] = (normalized_bbox_targets[valid_mask, 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
                
                # dim w, l (index 2:4): already log scale, no re-normalization needed
                # The network learns the log scale on its own
                
                # Re-normalize height to [0,1] (index 4:5 = cz)
                target_for_ll1[:, 4] = (normalized_bbox_targets[valid_mask, 4] - pc_range[2]) / (pc_range[5] - pc_range[2])
                
                # dim h (index 5:6): already log scale, no re-normalization needed

                
                # Compute the Lipschitz L1 for each EDL task
                ll1_losses = []
                for task_info in nig_params_list:
                    if task_info['v'] is None:  # skip the non-EDL tasks (dim/rot/vel)
                        continue
                    
                    task_name = task_info['name']
                    gamma_pos = task_info['gamma'][valid_mask]  # normalized gamma
                    v_pos = task_info['v'][valid_mask]
                    alpha_pos = task_info['alpha'][valid_mask]
                    beta_pos = task_info['beta'][valid_mask]
                    
                    # Determine the target slice and code_weights
                    # [CRITICAL FIX] use the correct target indices
                    if task_name == 'center':
                        # center gamma corresponds to target[0:2] = cx, cy
                        target_slice = target_for_ll1[:, 0:2]
                        weight_slice = bbox_weights_coded[valid_mask, 0:2]
                    elif task_name == 'height':
                        # height gamma corresponds to target[4:5] = cz
                        target_slice = target_for_ll1[:, 4:5]
                        weight_slice = bbox_weights_coded[valid_mask, 4:5]
                    else:
                        continue
                    
                    # [MT-ENet] Faithful Lipschitz L1 Loss (Proposition 2)
                    ll1 = self.lipschitz_l1_loss(
                        target_slice, gamma_pos, v_pos, alpha_pos, beta_pos
                    )
                    
                    # Apply code_weights and avg_factor
                    ll1_weighted = (ll1 * weight_slice).sum() / max(num_total_pos, 1)
                    ll1_losses.append(ll1_weighted)
                
                # Combine the Lipschitz L1 loss over all EDL tasks
                if len(ll1_losses) > 0:
                    loss_l1_edl = sum(ll1_losses) / len(ll1_losses)
                else:
                    loss_l1_edl = pred_gamma_abs.sum() * 0.0
            else:
                # Plain (upstream) L1 loss for the EDL tasks: center(0:2) + height(4:5)
                edl_pred = torch.cat([pred_gamma_abs[valid_mask, 0:2], pred_gamma_abs[valid_mask, 4:5]], dim=-1)
                edl_target = torch.cat([normalized_bbox_targets[valid_mask, 0:2], normalized_bbox_targets[valid_mask, 4:5]], dim=-1)
                edl_weights = torch.cat([bbox_weights_coded[valid_mask, 0:2], bbox_weights_coded[valid_mask, 4:5]], dim=-1)
                loss_l1_edl = self.loss_bbox(
                    edl_pred, edl_target, edl_weights,
                    avg_factor=max(num_total_pos, 1)
                )
            
            # Combine the L1 losses (each with its own weight)
            loss_l1 = l1_weight * loss_l1_edl + l1_weight_standard * loss_l1_std
        else:
            loss_l1 = pred_gamma_abs.sum() * 0.0

        
        # ============================================================
        # 4.2 EDL NIG loss (positives only)
        # ============================================================
        pos_mask_final = valid_mask  # the same mask
        num_pos = num_valid
        
        loss_nig_nll = torch.tensor(0.0, device=pred_bboxes.device)
        loss_nig_reg = torch.tensor(0.0, device=pred_bboxes.device)
        
        if num_pos > 0 and (nig_nll_weight > 0 or nig_reg_weight > 0):
            # Prepare the normalized target for the NIG loss
            # The NIG loss is computed in normalized space, on the same scale as gamma
            # [CRITICAL FIX] the target layout is [cx, cy, w, l, cz, h, sin, cos, vx, vy]
            target_for_nig = normalized_bbox_targets.clone()
            
            # Re-normalize center to [0,1] (index 0:2 = cx, cy)
            target_for_nig[:, 0] = (normalized_bbox_targets[:, 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
            target_for_nig[:, 1] = (normalized_bbox_targets[:, 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
            
            # Re-normalize height to [0,1] (index 4:5 = cz)
            target_for_nig[:, 4] = (normalized_bbox_targets[:, 4] - pc_range[2]) / (pc_range[5] - pc_range[2])
            
            # Compute the NIG loss for each EDL task
            nig_nll_total = 0.0
            nig_reg_total = 0.0
            edl_task_count = 0
            
            for task_info in nig_params_list:
                task_name = task_info['name']
                gamma = task_info['gamma']
                v = task_info['v']
                alpha = task_info['alpha']
                beta = task_info['beta']
                
                if v is None:
                    # Not an EDL task (dim/rot/vel): skip
                    continue
                
                # Determine the target slice (EDL tasks only: center/height)
                # dim, rot and vel were already skipped above by the `v is None` branch
                # [CRITICAL FIX] the target layout is [cx, cy, w, l, cz, h, sin, cos, vx, vy]
                if task_name == 'center':
                    # center: target[0:2] = cx, cy
                    target_slice = target_for_nig[pos_mask_final, 0:2]
                elif task_name == 'height':
                    # height: target[4:5] = cz (not 2:3)
                    target_slice = target_for_nig[pos_mask_final, 4:5]
                else:
                    continue  # skip unknown tasks safely
                
                # NIG parameters of the positive samples
                # Note: v, alpha and beta were already activated during parsing
                gamma_pos = gamma[pos_mask_final]
                v_pos = v[pos_mask_final]  # already softplus + 1e-6
                alpha_pos = alpha[pos_mask_final]  # already softplus + 1.0
                beta_pos = beta[pos_mask_final]  # already softplus + 1e-6
                
                # NIG NLL Loss (E-DETR Paper Equation 13)
                # L_NLL = 0.5 * log(π/v) - α * log(Ω) + (α + 0.5) * log((y - γ)² * v + Ω) + log(Γ(α)/Γ(α+0.5))
                # where Ω = 2β(1 + v)
                # NOTE: The Gamma function term is CRITICAL for preventing α → 1
                #       Its gradient ψ(α) - ψ(α+0.5) < 0 pushes α to INCREASE
                Omega = 2.0 * beta_pos * (1.0 + v_pos)
                
                # detach_gamma control:
                # - True:  L1 supervises gamma, NIG supervises only v, alpha, beta
                # - False: NIG fully supervises gamma, v, alpha, beta (original E-DETR)
                if self.detach_gamma:
                    # gamma receives no gradient from the NIG loss; the L1 loss supervises it alone
                    error = target_slice - gamma_pos.detach()
                else:
                    # Original E-DETR: the NIG loss supervises gamma together with v, alpha and beta
                    error = target_slice - gamma_pos
                
                nll = 0.5 * torch.log(torch.pi / v_pos.clamp(min=1e-6)) \
                      - alpha_pos * torch.log(Omega.clamp(min=1e-6)) \
                      + (alpha_pos + 0.5) * torch.log(error ** 2 * v_pos + Omega + 1e-6) \
                      + torch.lgamma(alpha_pos) - torch.lgamma(alpha_pos + 0.5)
                
                nig_nll_total = nig_nll_total + nll.mean()
                
                # NIG Regularization (Evidence Regularization)
                # L_reg = |y - γ| * (2v + α)
                # detach_gamma likewise decides whether gamma's gradient is cut here
                evidence = 2.0 * v_pos + alpha_pos
                reg = torch.abs(error) * evidence
                
                nig_reg_total = nig_reg_total + reg.mean()
                
                edl_task_count += 1
            
            if edl_task_count > 0:
                loss_nig_nll = nig_nll_total / edl_task_count
                loss_nig_reg = nig_reg_total / edl_task_count
        
        # ============================================================
        # 4.3 Combine the bbox loss
        # [KEY FIX] loss_l1 already includes l1_weight and l1_weight_standard; do not scale it again
        # [KEY FIX] the NIG loss is scaled by loss_bbox.loss_weight (0.25) to stay in proportion with the L1 loss
        # ============================================================
        loss_bbox_weight = self.loss_bbox.loss_weight
        loss_bbox = loss_l1 \
                    + loss_bbox_weight * self.edl_warmup_factor * nig_nll_weight * loss_nig_nll \
                    + loss_bbox_weight * self.edl_warmup_factor * nig_reg_weight * loss_nig_reg

        return torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox)
    
    def loss_single(self, pred_bboxes, pred_logits, cls_alpha, cls_beta, gt_bboxes_3d, gt_labels_3d):
        """
        Compute the loss for a single decoder layer.
        
        [CRITICAL] Upstream MEFormerHead.loss_single passes absolute coordinates to the
        assigner, because forward_single has already denormalized center/height.
        
        For EDL, pred_bboxes holds NIG parameters laid out as:
        center_nig(8) + height_nig(4) + dim(3) + rot(2) + vel(2) = 19
        
        where center_nig's gamma is already normalized (handled in forward_single).
        """
        batch_size = pred_bboxes[0].shape[0]
        
        # [EDL adaptation] prepare gamma (absolute coordinates) and probabilities for assignment
        assign_bboxes_list, assign_logits_list = [], []
        
        for idx in range(batch_size):
            # Take one sample's NIG parameters
            # pred_bboxes: list of [bs, nq, 19] tensors (per task)
            pb = [task_pred_bbox[idx] for task_pred_bbox in pred_bboxes]
            pl = [task_pred_logits[idx] for task_pred_logits in pred_logits]

            # Convert to gamma (absolute coordinates) and raw logits
            ab, al = self._prepare_assign_outputs_single_sample(pb, pl)
            
            assign_bboxes_list.append(ab)
            assign_logits_list.append(al)

        # Call get_targets (the assigner works in absolute coordinates)
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, assign_bboxes_list, assign_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        # Compute the loss, passing the raw NIG and EDL classification parameters
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task,
            pred_bboxes, # NIG Params [bs, nq, 19]
            pred_logits, # [bs, nq, num_cls] - log-odds logits
            cls_alpha,   # [bs, nq, num_cls] - α parameters
            cls_beta,    # [bs, nq, num_cls] - β parameters
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
            list(range(len(pred_bboxes))),
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             cls_alpha,
                             cls_beta,
                             mask_dict,
                             task_id):
        """
        Denoise loss - EDL variant.
        
        EDL bbox loss structure:
        loss_bbox = l1_weight * L1_Loss + nig_nll_weight * NIG_NLL + nig_reg_weight * NIG_Reg
        
        EDL classification loss structure:
        loss_cls = cls_edl_weight * Digamma_NLL + cls_kl_weight * KL_Reg
        """
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long().to(known_indice.device)
        bid = batch_idx[known_indice]

        # Get DN Queries
        pred_logits = pred_logits[(bid, map_known_indice)]
        cls_alpha_dn = cls_alpha[(bid, map_known_indice)]
        cls_beta_dn = cls_beta[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # Task masking - exactly as in upstream MEFormer
        # [KEY FIX] upstream filters only pred_bboxes and known_bboxs, never pred_logits or known_labels
        known_labels_raw = mask_dict['known_labels_raw']
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()
        
        # [KEY FIX] as upstream: filter the bbox only when task_mask_sum > 0, and never return early
        if task_mask_sum > 0:
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]
        
        # [KEY FIX] do not filter pred_logits or known_labels, and do not update num_tgt

        pc_range = self.pc_range

        # -----------------------------------------------------------
        # 1. Parse the NIG parameters and extract gamma
        # -----------------------------------------------------------
        task_configs = [
            {'name': 'center', 'dim': 2, 'use_edl': True},
            {'name': 'height', 'dim': 1, 'use_edl': True},
            {'name': 'dim',    'dim': 3, 'use_edl': False},
            {'name': 'rot',    'dim': 2, 'use_edl': False},
            {'name': 'vel',    'dim': 2, 'use_edl': False},
        ]
        
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        
        # [FIX] align the dimensions
        if normalized_bbox_targets.shape[-1] < 10:
            padding_size = 10 - normalized_bbox_targets.shape[-1]
            padding = torch.zeros(
                (*normalized_bbox_targets.shape[:-1], padding_size),
                device=normalized_bbox_targets.device,
                dtype=normalized_bbox_targets.dtype
            )
            normalized_bbox_targets = torch.cat([normalized_bbox_targets, padding], dim=-1)
        
        all_gammas = []
        nig_params_list = []  # NIG parameters kept for the EDL loss
        ptr_pred = 0
        
        for task in task_configs:
            width_pred = task['dim'] * 4 if task['use_edl'] else task['dim']
            pred_slice = pred_bboxes[:, ptr_pred : ptr_pred + width_pred]
            ptr_pred += width_pred
            
            if task['use_edl']:
                # [KEY FIX] EDLNIGHead already applied the activations - do not apply them again
                pred_reshaped = pred_slice.view(-1, task['dim'], 4)
                gamma = pred_reshaped[..., 0]  # gamma: unconstrained, use as-is
                v = pred_reshaped[..., 1]      # v: already activated, use as-is
                alpha = pred_reshaped[..., 2]  # alpha: already activated, use as-is
                beta = pred_reshaped[..., 3]   # beta: already activated, use as-is
                
                # [SAFETY] clamp only, for numerical stability - no activation here
                v = v.clamp(min=1e-6)
                alpha = alpha.clamp(min=1.001)  # keep alpha > 1
                beta = beta.clamp(min=1e-6)
                
                all_gammas.append(gamma)
                nig_params_list.append({
                    'name': task['name'],
                    'gamma': gamma, 'v': v, 'alpha': alpha, 'beta': beta
                })
            else:
                all_gammas.append(pred_slice)
                nig_params_list.append({
                    'name': task['name'],
                    'gamma': pred_slice, 'v': None, 'alpha': None, 'beta': None
                })
                
        pred_gamma_full = torch.cat(all_gammas, dim=-1)
        
        # -----------------------------------------------------------
        # [CRITICAL FIX] reorder to match the target layout normalize_bbox expects
        # target layout:     [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        # prediction layout: [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        # so cz (index 2) and w, l (indices 3:5) swap places
        # -----------------------------------------------------------
        pred_gamma_reordered = torch.cat([
            pred_gamma_full[:, 0:2],   # cx, cy (unchanged)
            pred_gamma_full[:, 3:5],   # w, l (first two of dim, moved to index 2:4)
            pred_gamma_full[:, 2:3],   # cz (from height, moved to index 4:5)
            pred_gamma_full[:, 5:6],   # h (third of dim, moved to index 5:6)
            pred_gamma_full[:, 6:10],  # rot, vel (unchanged)
        ], dim=-1)
        
        # -----------------------------------------------------------
        # [KEY FIX] convert pred_gamma to absolute coordinates / log scale
        # The order is now [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        # -----------------------------------------------------------
        pred_gamma_abs = pred_gamma_reordered.clone()
        
        # center (0:2): normalized [0,1] -> absolute PC range
        pred_gamma_abs[:, 0] = pred_gamma_reordered[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        pred_gamma_abs[:, 1] = pred_gamma_reordered[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        
        # dim w, l (2:4): use the raw gamma directly (the network learns the log scale)
        # No conversion needed
        
        # height cz (4:5): normalized [0,1] -> absolute PC range
        pred_gamma_abs[:, 4] = pred_gamma_reordered[:, 4] * (pc_range[5] - pc_range[2]) + pc_range[2]
        
        # dim h (5:6): use the raw gamma directly
        # No conversion needed
        
        # rot (6:8) and vel (8:10) are left unchanged

        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split
        cls_avg_factor = max(cls_avg_factor, 1)
        label_weights = torch.ones_like(known_labels)

        if task_mask_sum > 0:
            known_labels_cls = known_labels[task_mask]
            label_weights_cls = label_weights[task_mask]
            cls_alpha_cls = cls_alpha_dn[task_mask]
            cls_beta_cls = cls_beta_dn[task_mask]
            dn_bbox_weights = torch.ones_like(pred_gamma_abs)
            loss_cls = self._edl_cls_loss(
                known_labels_cls,
                label_weights_cls,
                cls_alpha_cls,
                cls_beta_cls,
                cls_avg_factor=max(known_labels_cls.numel(), 1),
                task_id=task_id,
                pred_bboxes_flatten=pred_gamma_abs,
                normalized_bbox_targets=normalized_bbox_targets,
                bbox_weights=dn_bbox_weights,
            )
        else:
            loss_cls = pred_logits.sum() * 0.0
        
        # [FIX] as upstream: average the gt box count across GPUs for the bbox loss
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # -----------------------------------------------------------
        # 3. BBox Regression Loss (EDL + L1)
        # -----------------------------------------------------------
        
        # Filter NaN (as upstream)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        
        # Fetch the EDL loss weights (with the iteration-based warmup factor applied)
        l1_weight = self.l1_weight
        l1_weight_standard = self.l1_weight_standard  # separate weight for rot/vel
        nig_nll_weight = self.nig_nll_weight
        nig_reg_weight = self.nig_reg_weight
        
        # ============================================================
        # 3.1 L1 loss - EDL tasks and plain regression handled separately
        # [MT-ENet] the DN loss uses Lipschitz L1 as well, matching the main loss
        # pred_gamma_abs layout: [cx, cy, w, l, cz, h, sin, cos, vx, vy]
        # EDL indices: 0:2 (center) + 4:5 (height) = 3 dims (non-contiguous)
        # Plain indices: 2:4 (dim w, l) + 5:10 (dim h + rot + vel) = 7 dims (non-contiguous)
        # ============================================================
        bbox_weights = torch.ones_like(normalized_bbox_targets)
        
        # [KEY FIX] apply code_weights, matching upstream MEFormerHead
        code_weights = normalized_bbox_targets.new_tensor(self.code_weights)
        bbox_weights = bbox_weights * code_weights[None, :]
        
        # EDL tasks: center(0:2) + height(4:5) (non-contiguous)
        if self.use_lipschitz_l1:
            # [MT-ENet] Lipschitz L1, same logic as in _loss_single_task
            pc_range = self.pc_range
            target_for_ll1_dn = normalized_bbox_targets[isnotnan].clone()
            
            # Re-normalize center to [0,1]
            target_for_ll1_dn[:, 0] = (normalized_bbox_targets[isnotnan, 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
            target_for_ll1_dn[:, 1] = (normalized_bbox_targets[isnotnan, 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
            # Re-normalize height to [0,1]
            target_for_ll1_dn[:, 4] = (normalized_bbox_targets[isnotnan, 4] - pc_range[2]) / (pc_range[5] - pc_range[2])
            
            ll1_losses_dn = []
            for task_info in nig_params_list:
                if task_info['v'] is None:
                    continue
                task_name = task_info['name']
                gamma_pos = task_info['gamma'][isnotnan]
                v_pos = task_info['v'][isnotnan]
                alpha_pos = task_info['alpha'][isnotnan]
                beta_pos = task_info['beta'][isnotnan]
                
                if task_name == 'center':
                    target_slice = target_for_ll1_dn[:, 0:2]
                    weight_slice = bbox_weights[isnotnan, 0:2]
                elif task_name == 'height':
                    target_slice = target_for_ll1_dn[:, 4:5]
                    weight_slice = bbox_weights[isnotnan, 4:5]
                else:
                    continue
                
                ll1 = self.lipschitz_l1_loss(
                    target_slice, gamma_pos, v_pos, alpha_pos, beta_pos
                )
                ll1_weighted = (ll1 * weight_slice).sum() / max(num_tgt, 1)
                ll1_losses_dn.append(ll1_weighted)
            
            if len(ll1_losses_dn) > 0:
                loss_l1_edl = sum(ll1_losses_dn) / len(ll1_losses_dn)
            else:
                loss_l1_edl = pred_gamma_abs.sum() * 0.0
        else:
            # Plain L1 loss for the EDL tasks: center(0:2) + height(4:5)
            edl_pred = torch.cat([pred_gamma_abs[isnotnan, 0:2], pred_gamma_abs[isnotnan, 4:5]], dim=-1)
            edl_target = torch.cat([normalized_bbox_targets[isnotnan, 0:2], normalized_bbox_targets[isnotnan, 4:5]], dim=-1)
            edl_weights = torch.cat([bbox_weights[isnotnan, 0:2], bbox_weights[isnotnan, 4:5]], dim=-1)
            loss_l1_edl = self.loss_bbox(
                edl_pred, edl_target, edl_weights,
                avg_factor=max(num_tgt, 1)
            )
        
        # Plain regression tasks: dim(2:4, 5:6) + rot(6:8) + vel(8:10) (non-contiguous)
        std_pred = torch.cat([pred_gamma_abs[isnotnan, 2:4], pred_gamma_abs[isnotnan, 5:10]], dim=-1)
        std_target = torch.cat([normalized_bbox_targets[isnotnan, 2:4], normalized_bbox_targets[isnotnan, 5:10]], dim=-1)
        std_weights = torch.cat([bbox_weights[isnotnan, 2:4], bbox_weights[isnotnan, 5:10]], dim=-1)
        loss_l1_std = self.loss_bbox(
            std_pred, std_target, std_weights,
            avg_factor=max(num_tgt, 1)
        )
        
        # Combine the L1 losses (each with its own weight)
        loss_l1 = l1_weight * loss_l1_edl + l1_weight_standard * loss_l1_std
        
        # ============================================================
        # 3.2 EDL NIG loss (every DN sample is a positive)
        # ============================================================
        num_pos = isnotnan.sum().item()
        
        loss_nig_nll = torch.tensor(0.0, device=pred_bboxes.device)
        loss_nig_reg = torch.tensor(0.0, device=pred_bboxes.device)
        
        if num_pos > 0 and (nig_nll_weight > 0 or nig_reg_weight > 0):
            # Prepare the normalized target for the NIG loss
            # [CRITICAL FIX] the target layout is [cx, cy, w, l, cz, h, sin, cos, vx, vy]
            target_for_nig = normalized_bbox_targets.clone()
            
            # Re-normalize center to [0,1] (index 0:2 = cx, cy)
            target_for_nig[:, 0] = (normalized_bbox_targets[:, 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
            target_for_nig[:, 1] = (normalized_bbox_targets[:, 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
            
            # Re-normalize height to [0,1] (index 4:5 = cz)
            target_for_nig[:, 4] = (normalized_bbox_targets[:, 4] - pc_range[2]) / (pc_range[5] - pc_range[2])
            
            # Compute the NIG loss for each EDL task
            nig_nll_total = 0.0
            nig_reg_total = 0.0
            edl_task_count = 0
            
            for task_info in nig_params_list:
                task_name = task_info['name']
                gamma = task_info['gamma']
                v = task_info['v']
                alpha = task_info['alpha']
                beta = task_info['beta']
                
                if v is None:
                    # Not an EDL task (dim/rot/vel): skip
                    continue
                
                # Determine the target slice (EDL tasks only: center/height)
                # dim, rot and vel were already skipped above by the `v is None` branch
                # [CRITICAL FIX] the target layout is [cx, cy, w, l, cz, h, sin, cos, vx, vy]
                if task_name == 'center':
                    # center: target[0:2] = cx, cy
                    target_slice = target_for_nig[isnotnan, 0:2]
                elif task_name == 'height':
                    # height: target[4:5] = cz (not 2:3)
                    target_slice = target_for_nig[isnotnan, 4:5]
                else:
                    continue  # skip unknown tasks safely
                
                # NIG parameters of the valid samples
                # Note: v, alpha and beta were already activated during parsing
                gamma_pos = gamma[isnotnan]
                v_pos = v[isnotnan]  # already softplus + 1e-6
                alpha_pos = alpha[isnotnan]  # already softplus + 1.0
                beta_pos = beta[isnotnan]  # already softplus + 1e-6
                
                # NIG NLL Loss (E-DETR Paper Equation 13)
                # NOTE: The Gamma function term is CRITICAL for preventing α → 1
                Omega = 2.0 * beta_pos * (1.0 + v_pos)
                
                # detach_gamma control
                if self.detach_gamma:
                    error = target_slice - gamma_pos.detach()
                else:
                    error = target_slice - gamma_pos
                
                nll = 0.5 * torch.log(torch.pi / v_pos.clamp(min=1e-6)) \
                      - alpha_pos * torch.log(Omega.clamp(min=1e-6)) \
                      + (alpha_pos + 0.5) * torch.log(error ** 2 * v_pos + Omega + 1e-6) \
                      + torch.lgamma(alpha_pos) - torch.lgamma(alpha_pos + 0.5)
                
                nig_nll_total = nig_nll_total + nll.mean()
                
                # NIG Regularization
                evidence = 2.0 * v_pos + alpha_pos
                reg = torch.abs(error) * evidence
                
                nig_reg_total = nig_reg_total + reg.mean()
                
                edl_task_count += 1
            
            if edl_task_count > 0:
                loss_nig_nll = nig_nll_total / edl_task_count
                loss_nig_reg = nig_reg_total / edl_task_count
        
        # ============================================================
        # 3.3 Combine the bbox loss
        # [KEY FIX] loss_l1 already includes l1_weight and l1_weight_standard; do not scale it again
        # [KEY FIX] the NIG loss is scaled by loss_bbox.loss_weight (0.25) as well
        # ============================================================
        loss_bbox_weight = self.loss_bbox.loss_weight
        loss_bbox = loss_l1 \
                    + loss_bbox_weight * self.edl_warmup_factor * nig_nll_weight * loss_nig_nll \
                    + loss_bbox_weight * self.edl_warmup_factor * nig_reg_weight * loss_nig_reg

        # [FIX] as upstream: handle NaN and the task_mask_sum == 0 case
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        if task_mask_sum == 0:
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_cls_alpha,
                       dn_cls_beta,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, 
            dn_cls_alpha, dn_cls_beta, dn_mask_dict,
            list(range(len(pred_bboxes)))
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _selector_regularization_loss(self, preds_dicts, modality_idx):
        loss_cls_terms = []
        loss_box_terms = []
        loss_prior_terms = []

        for task_preds in preds_dicts:
            if isinstance(task_preds, list):
                task_preds = task_preds[0]

            required_keys = {
                'selector_teacher_cls_logits',
                'selector_teacher_center',
                'selector_teacher_height',
                'selector_fused_weight',
            }
            if not required_keys.issubset(task_preds.keys()):
                continue

            final_cls = task_preds['cls_logits'][modality_idx]
            final_center = task_preds['center'][modality_idx]
            final_height = task_preds['height'][modality_idx]

            teacher_cls = task_preds['selector_teacher_cls_logits'][modality_idx].detach()
            teacher_center = task_preds['selector_teacher_center'][modality_idx].detach()
            teacher_height = task_preds['selector_teacher_height'][modality_idx].detach()
            fused_weight = task_preds['selector_fused_weight'][modality_idx]

            loss_cls_terms.append(F.mse_loss(final_cls, teacher_cls))
            loss_box_terms.append(
                F.l1_loss(final_center, teacher_center) +
                F.l1_loss(final_height, teacher_height)
            )
            loss_prior_terms.append(fused_weight.mean() * 0.0)

        if loss_cls_terms:
            return (
                torch.stack(loss_cls_terms).mean(),
                torch.stack(loss_box_terms).mean(),
                torch.stack(loss_prior_terms).mean(),
            )

        zero_ref = None
        for task_preds in preds_dicts:
            if isinstance(task_preds, list):
                task_preds = task_preds[0]
            if 'cls_logits' in task_preds:
                zero_ref = task_preds['cls_logits'][0][0]
                break
        if zero_ref is None:
            zero = torch.tensor(0.0)
        else:
            zero = zero_ref.new_tensor(0.0)
        return zero, zero, zero

    def _selector_target_loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, pred_bboxes, pred_logits):
        assign_bboxes_list = []
        assign_logits_list = []
        batch_size = pred_bboxes[0].shape[0]

        for batch_idx in range(batch_size):
            sample_bboxes = [task_pred_bbox[batch_idx] for task_pred_bbox in pred_bboxes]
            sample_logits = [task_pred_logits[batch_idx] for task_pred_logits in pred_logits]
            assign_bboxes, assign_logits = self._prepare_assign_outputs_single_sample(
                sample_bboxes, sample_logits)
            assign_bboxes_list.append(assign_bboxes)
            assign_logits_list.append(assign_logits)

        target_outputs = self.get_targets(
            gt_bboxes_3d,
            gt_labels_3d,
            assign_bboxes_list,
            assign_logits_list,
            return_pos_inds=True,
        )
        (
            task_labels_list,
            _task_labels_weight_list,
            task_bbox_targets_list,
            _task_bbox_weights_list,
            _num_total_pos_tasks,
            _num_total_neg_tasks,
            task_pos_inds_list,
        ) = target_outputs

        base_loss_terms = []
        residual_loss_terms = []
        zero_ref = None
        teacher_counts = {modality: 0.0 for modality in ('fused', 'bev', 'img')}
        pred_counts = {modality: 0.0 for modality in ('fused', 'bev', 'img')}
        prior_counts = {modality: 0.0 for modality in ('fused', 'bev', 'img')}
        total_agree = 0.0
        total_pos = 0.0
        alpha_pos_sum = 0.0
        alpha_hard_sum = 0.0
        alpha_easy_sum = 0.0
        hard_pos = 0.0
        easy_pos = 0.0
        residual_override_sum = 0.0
        second_opinion_loss_terms = []
        entropy_hard_terms = []
        alpha_sep_terms = []
        hard_entropy_sum = 0.0
        hard_entropy_count = 0.0
        prior_top2_match_sum = 0.0
        second_opinion_teacher_match_sum = 0.0
        second_opinion_count = 0.0

        for task_id, task_preds in enumerate(preds_dicts):
            if isinstance(task_preds, list):
                task_preds = task_preds[0]

            required_keys = {
                'selector_probs_last',
                'selector_distance_prior_probs_last',
                'selector_alpha_res_last',
                'selector_source_cls_logits',
                'selector_source_center',
                'selector_source_height',
            }
            if not required_keys.issubset(task_preds.keys()):
                continue

            selector_probs = task_preds['selector_probs_last']
            base_selector_probs = task_preds['selector_distance_prior_probs_last']
            alpha_res = task_preds['selector_alpha_res_last']
            source_cls = task_preds['selector_source_cls_logits']
            source_center = task_preds['selector_source_center']
            source_height = task_preds['selector_source_height']
            tail_ratio = task_preds.get('selector_uncertainty_tail_query_ratio', {})
            modalities = task_preds.get('selector_modalities', ['fused', 'bev', 'img'])
            if isinstance(source_center, dict) and 'fused' in source_center:
                fused_modality = 'fused'
            elif isinstance(source_height, dict) and 'fused' in source_height:
                fused_modality = 'fused'
            elif isinstance(source_cls, dict) and 'fused' in source_cls:
                fused_modality = 'fused'
            else:
                fused_modality = 'fused' if 'fused' in modalities else modalities[0]

            zero_ref = selector_probs

            for batch_idx in range(batch_size):
                pos_inds = task_pos_inds_list[task_id][batch_idx]
                if pos_inds.numel() == 0:
                    continue

                labels = task_labels_list[task_id][batch_idx][pos_inds].long()
                bbox_targets = task_bbox_targets_list[task_id][batch_idx][pos_inds]
                target_center = bbox_targets[:, 0:2]
                target_height = bbox_targets[:, 2:3]

                teacher_scores = []
                for modality in modalities:
                    center_pred = source_center[modality][batch_idx, pos_inds]
                    height_pred = source_height[modality][batch_idx, pos_inds]
                    cls_logits = source_cls[modality][batch_idx, pos_inds]
                    pos_cls_logits = cls_logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)

                    err_center = F.l1_loss(center_pred, target_center, reduction='none').mean(dim=-1)
                    err_height = F.l1_loss(height_pred, target_height, reduction='none').mean(dim=-1)
                    cls_cost = F.binary_cross_entropy_with_logits(
                        pos_cls_logits, torch.ones_like(pos_cls_logits), reduction='none')
                    teacher_scores.append(-(err_center + err_height + cls_cost))

                teacher_scores = torch.stack(teacher_scores, dim=-1)
                teacher_probs = F.softmax(
                    teacher_scores / self.selector_teacher_temperature, dim=-1
                )
                pred_probs = selector_probs[batch_idx, pos_inds].clamp(min=1e-8)
                base_probs = base_selector_probs[batch_idx, pos_inds].clamp(min=1e-8)
                alpha_vals = alpha_res[batch_idx, pos_inds].squeeze(-1)

                base_kl = (
                    base_probs * (base_probs.log() - teacher_probs.clamp(min=1e-8).log())
                ).sum(dim=-1)
                pred_kl = (
                    pred_probs * (pred_probs.log() - teacher_probs.clamp(min=1e-8).log())
                ).sum(dim=-1)

                teacher_top1 = teacher_probs.argmax(dim=-1)
                pred_top1 = pred_probs.argmax(dim=-1)
                base_top1 = base_probs.argmax(dim=-1)
                pred_topk = pred_probs.topk(k=min(2, pred_probs.shape[-1]), dim=-1).indices
                base_topk = base_probs.topk(k=min(2, base_probs.shape[-1]), dim=-1).indices
                if base_topk.shape[-1] == 1:
                    base_top2 = base_top1
                else:
                    base_top2 = base_topk[:, 1]

                teacher_margin = teacher_probs.topk(k=min(2, teacher_probs.shape[-1]), dim=-1).values
                if teacher_margin.shape[-1] == 1:
                    teacher_margin = teacher_margin[..., 0]
                else:
                    teacher_margin = teacher_margin[..., 0] - teacher_margin[..., 1]

                fused_center = source_center[fused_modality][batch_idx, pos_inds]
                fused_height = source_height[fused_modality][batch_idx, pos_inds]
                fused_cls = torch.sigmoid(source_cls[fused_modality][batch_idx, pos_inds]).max(dim=-1)[0]
                disagreement_terms = []
                for modality in modalities:
                    if modality == fused_modality:
                        continue
                    center_dis = torch.abs(
                        source_center[modality][batch_idx, pos_inds] - fused_center
                    ).mean(dim=-1)
                    height_dis = torch.abs(
                        source_height[modality][batch_idx, pos_inds] - fused_height
                    ).mean(dim=-1)
                    cls_dis = torch.abs(
                        torch.sigmoid(source_cls[modality][batch_idx, pos_inds]).max(dim=-1)[0] - fused_cls
                    )
                    disagreement_terms.append(center_dis + height_dis + cls_dis)
                if disagreement_terms:
                    disagreement_score = torch.stack(disagreement_terms, dim=-1).amax(dim=-1)
                    disagreement_thr = disagreement_score.median() + disagreement_score.std(unbiased=False)
                else:
                    disagreement_score = teacher_margin.new_zeros(teacher_margin.shape)
                    disagreement_thr = disagreement_score.new_tensor(float('inf'))

                hard_mask = (
                    (teacher_top1 != base_top1) |
                    (teacher_margin > 0.10) |
                    (disagreement_score > disagreement_thr)
                )
                hard_weight = 1.0 + teacher_margin + alpha_vals.detach()
                pred_entropy = -(pred_probs * pred_probs.log()).sum(dim=-1)

                base_loss_terms.append(base_kl.mean())
                if hard_mask.any():
                    residual_loss_terms.append((pred_kl[hard_mask] * hard_weight[hard_mask]).mean())
                    alpha_hard_sum += alpha_vals[hard_mask].sum().item()
                    hard_pos += float(hard_mask.float().sum().item())
                    hard_entropy_sum += pred_entropy[hard_mask].sum().item()
                    hard_entropy_count += float(hard_mask.float().sum().item())
                    entropy_hard_terms.append(
                        F.relu(
                            pred_entropy.new_tensor(self.selector_entropy_target_hard) - pred_entropy[hard_mask]
                        ).mean()
                    )
                else:
                    residual_loss_terms.append((pred_kl * hard_weight).mean())

                easy_mask = ~hard_mask
                if easy_mask.any():
                    alpha_easy_sum += alpha_vals[easy_mask].sum().item()
                    easy_pos += float(easy_mask.float().sum().item())
                    alpha_sep_terms.append(
                        F.relu(
                            alpha_vals.new_tensor(self.alpha_res_separation_margin) -
                            (alpha_vals[hard_mask].mean() - alpha_vals[easy_mask].mean())
                        ) if hard_mask.any() else alpha_vals.new_tensor(0.0)
                    )
                elif hard_mask.any():
                    alpha_sep_terms.append(
                        F.relu(
                            alpha_vals.new_tensor(self.alpha_res_separation_margin) -
                            (alpha_vals[hard_mask].mean() - alpha_vals.new_tensor(self.residual_floor))
                        )
                    )

                need_second = hard_mask & (teacher_top1 != base_top1)
                if need_second.any():
                    teacher_prob = pred_probs.gather(1, teacher_top1.unsqueeze(-1)).squeeze(-1)
                    prior2_prob = pred_probs.gather(1, base_top2.unsqueeze(-1)).squeeze(-1)
                    second_mask = need_second & (teacher_top1 != base_top2)
                    if second_mask.any():
                        second_opinion_loss_terms.append(
                            F.relu(
                                teacher_prob.new_tensor(0.05) -
                                (teacher_prob[second_mask] - prior2_prob[second_mask])
                            ).mean()
                        )
                    second_opinion_count += float(need_second.float().sum().item())
                    prior_top2_match_sum += (teacher_top1[need_second] == base_top2[need_second]).float().sum().item()
                    second_opinion_teacher_match_sum += (
                        (pred_topk[need_second] == teacher_top1[need_second].unsqueeze(-1)).any(dim=-1).float().sum().item()
                    )

                alpha_pos_sum += alpha_vals.sum().item()
                residual_override_sum += (pred_top1 != base_top1).float().sum().item()

                total_agree += (teacher_top1 == pred_top1).float().sum().item()
                total_pos += float(pos_inds.numel())
                for idx, modality in enumerate(modalities):
                    teacher_counts[modality] += (teacher_top1 == idx).float().sum().item()
                    pred_counts[modality] += (pred_top1 == idx).float().sum().item()
                    prior_counts[modality] += (base_top1 == idx).float().sum().item()

        if base_loss_terms:
            base_loss = torch.stack(base_loss_terms).mean()
            residual_loss = torch.stack(residual_loss_terms).mean()
        else:
            if zero_ref is None:
                zero = torch.tensor(0.0)
            else:
                zero = zero_ref.new_tensor(0.0)
            return (
                zero, zero, zero, zero, zero, zero,
                {}, {}, {}, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            )

        denom = max(total_pos, 1.0)
        teacher_ratio = {modality: teacher_counts[modality] / denom for modality in teacher_counts}
        pred_ratio = {modality: pred_counts[modality] / denom for modality in pred_counts}
        prior_ratio = {modality: prior_counts[modality] / denom for modality in prior_counts}
        agreement_ratio = total_agree / denom
        alpha_res_mean = alpha_pos_sum / denom
        alpha_res_hard_mean = alpha_hard_sum / max(hard_pos, 1.0)
        alpha_res_easy_mean = alpha_easy_sum / max(easy_pos, 1.0)
        residual_override_ratio = residual_override_sum / denom
        hard_entropy_mean = hard_entropy_sum / max(hard_entropy_count, 1.0)
        prior_top2_match_ratio = prior_top2_match_sum / max(second_opinion_count, 1.0)
        second_opinion_teacher_match_ratio = (
            second_opinion_teacher_match_sum / max(second_opinion_count, 1.0)
        )
        second_opinion_ratio = second_opinion_count / denom

        if zero_ref is None:
            usage_ref = torch.tensor(0.0)
        else:
            usage_ref = zero_ref.new_tensor(0.0)
        usage_loss = F.relu(
            usage_ref.new_tensor(self.residual_usage_target) - usage_ref.new_tensor(alpha_res_mean)
        )
        second_opinion_loss = (
            torch.stack(second_opinion_loss_terms).mean()
            if second_opinion_loss_terms else usage_ref.new_tensor(0.0)
        )
        entropy_hard_loss = (
            torch.stack(entropy_hard_terms).mean()
            if entropy_hard_terms else usage_ref.new_tensor(0.0)
        )
        alpha_sep_loss = (
            torch.stack(alpha_sep_terms).mean()
            if alpha_sep_terms else usage_ref.new_tensor(0.0)
        )

        return (
            base_loss,
            residual_loss,
            usage_loss,
            second_opinion_loss,
            entropy_hard_loss,
            alpha_sep_loss,
            teacher_ratio,
            pred_ratio,
            prior_ratio,
            agreement_ratio,
            alpha_res_mean,
            alpha_res_hard_mean,
            alpha_res_easy_mean,
            residual_override_ratio,
            second_opinion_ratio,
            prior_top2_match_ratio,
            second_opinion_teacher_match_ratio,
            hard_entropy_mean,
        )

    def _balanced_binary_router_loss(self, logits, labels):
        valid_mask = labels >= 0
        if not valid_mask.any():
            return None

        logits = logits[valid_mask]
        labels = labels[valid_mask].long()
        pos_inds = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
        if pos_inds.numel() == 0:
            return None

        neg_inds = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
        max_neg = int(pos_inds.numel()) * 3
        if neg_inds.numel() > max_neg:
            perm = torch.randperm(neg_inds.numel(), device=logits.device)[:max_neg]
            neg_inds = neg_inds[perm]

        keep_inds = torch.cat([pos_inds, neg_inds], dim=0)
        if keep_inds.numel() == 0:
            return None

        return F.cross_entropy(logits[keep_inds], labels[keep_inds])

    def _v7_router_loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, pred_bboxes, pred_logits):
        assign_bboxes_list = []
        assign_logits_list = []
        batch_size = pred_bboxes[0].shape[0]

        for batch_idx in range(batch_size):
            sample_bboxes = [task_pred_bbox[batch_idx] for task_pred_bbox in pred_bboxes]
            sample_logits = [task_pred_logits[batch_idx] for task_pred_logits in pred_logits]
            assign_bboxes, assign_logits = self._prepare_assign_outputs_single_sample(
                sample_bboxes, sample_logits)
            assign_bboxes_list.append(assign_bboxes)
            assign_logits_list.append(assign_logits)

        target_outputs = self.get_targets(
            gt_bboxes_3d,
            gt_labels_3d,
            assign_bboxes_list,
            assign_logits_list,
            return_pos_inds=True,
        )
        (
            _task_labels_list,
            _task_labels_weight_list,
            task_bbox_targets_list,
            _task_bbox_weights_list,
            _num_total_pos_tasks,
            _num_total_neg_tasks,
            task_pos_inds_list,
        ) = target_outputs

        img_router_terms = []
        bev_router_terms = []
        zero_ref = None

        for task_id, task_preds in enumerate(preds_dicts):
            if isinstance(task_preds, list):
                task_preds = task_preds[0]

            required_keys = {
                'center',
                'selector_source_center',
                'v7_img_router_logits_last',
                'v7_bev_router_logits_last',
            }
            if not required_keys.issubset(task_preds.keys()):
                continue

            source_center = task_preds['selector_source_center']
            if not isinstance(source_center, dict):
                continue
            if 'img' not in source_center or 'bev' not in source_center:
                continue

            baseline_center = task_preds['center'][0][-1]
            img_center = source_center['img']
            bev_center = source_center['bev']
            img_router_logits = task_preds['v7_img_router_logits_last']
            bev_router_logits = task_preds['v7_bev_router_logits_last']
            zero_ref = baseline_center

            for batch_idx in range(batch_size):
                pos_inds = task_pos_inds_list[task_id][batch_idx]
                if pos_inds.numel() == 0:
                    continue

                target_center = task_bbox_targets_list[task_id][batch_idx][pos_inds][:, 0:2]
                blend_center_pos = baseline_center[batch_idx, pos_inds]
                img_center_pos = img_center[batch_idx, pos_inds]
                bev_center_pos = bev_center[batch_idx, pos_inds]

                blend_err = torch.abs(blend_center_pos - target_center).mean(dim=-1)
                img_err = torch.abs(img_center_pos - target_center).mean(dim=-1)
                bev_err = torch.abs(bev_center_pos - target_center).mean(dim=-1)

                ignore_label = -torch.ones_like(blend_err, dtype=torch.long)

                img_labels = ignore_label.clone()
                img_pos = (img_err + self.v7_router_label_margin) < blend_err
                img_neg = (blend_err + self.v7_router_label_margin) < img_err
                img_labels[img_pos] = 1
                img_labels[img_neg] = 0

                bev_labels = ignore_label.clone()
                bev_pos = (bev_err + self.v7_router_label_margin) < blend_err
                bev_neg = (blend_err + self.v7_router_label_margin) < bev_err
                bev_labels[bev_pos] = 1
                bev_labels[bev_neg] = 0

                img_loss = self._balanced_binary_router_loss(
                    img_router_logits[batch_idx, pos_inds], img_labels)
                if img_loss is not None:
                    img_router_terms.append(img_loss)

                bev_loss = self._balanced_binary_router_loss(
                    bev_router_logits[batch_idx, pos_inds], bev_labels)
                if bev_loss is not None:
                    bev_router_terms.append(bev_loss)

        if zero_ref is None:
            zero = torch.tensor(0.0)
        else:
            zero = zero_ref.new_tensor(0.0)

        img_router_loss = torch.stack(img_router_terms).mean() if img_router_terms else zero
        bev_router_loss = torch.stack(bev_router_terms).mean() if bev_router_terms else zero
        return img_router_loss, bev_router_loss

    @staticmethod
    def _v8_meta_corruption_type(meta):
        if not isinstance(meta, dict):
            return 'clean', False, 0
        corruption_type = str(meta.get('corruption_type', 'clean') or 'clean').lower()
        severity = meta.get('corruption_severity', 0)
        try:
            severity = int(severity)
        except (TypeError, ValueError):
            severity = 0
        synthetic = bool(meta.get('synthetic_corruption', False))
        if corruption_type in ('', 'none', 'clean'):
            corruption_type = 'clean'
        return corruption_type, synthetic and corruption_type != 'clean', severity

    @staticmethod
    def _v8_corruption_domain(corruption_type):
        lidar_corruptions = {
            'beamsreducing',
            'pointsreducing',
            'spatialmisalignment',
            'temporalmisalignment',
        }
        image_corruptions = {
            'brightness',
            'dark',
            'motionblur',
            'missingcamera',
        }
        dual_corruptions = {'fog', 'snow'}
        if corruption_type in lidar_corruptions:
            return 'lidar'
        if corruption_type in image_corruptions:
            return 'image'
        if corruption_type in dual_corruptions:
            return 'dual'
        return 'clean'

    def _v8_damage_gate_loss(self, preds_dicts, img_metas=None):
        gate_terms = []
        modality_terms = []
        domain_reliability_terms = []
        high_damage_sum = 0.0
        high_damage_count = 0.0
        clean_keep_sum = 0.0
        clean_keep_count = 0.0
        damage_keep_sum = 0.0
        damage_keep_count = 0.0
        single_damage_keep_sum = 0.0
        single_damage_keep_count = 0.0
        dual_damage_keep_sum = 0.0
        dual_damage_keep_count = 0.0
        domain_correct_sum = 0.0
        domain_correct_count = 0.0
        zero_ref = None

        if img_metas is None:
            img_metas = []

        def _weighted_bce(values, target_value, weight):
            target = values.new_full(values.shape, float(target_value))
            loss = F.binary_cross_entropy(values, target, reduction='none')
            while weight.dim() < loss.dim():
                weight = weight.unsqueeze(-1)
            weight = weight.to(dtype=loss.dtype)
            return (loss * weight).sum() / weight.sum().clamp(min=1.0)

        def _top_cls_prob(task_preds, batch_size, num_queries, device):
            cls_logits = task_preds.get('cls_logits')
            if isinstance(cls_logits, (list, tuple)):
                cls_logits = cls_logits[0] if cls_logits else None
            if isinstance(cls_logits, torch.Tensor):
                if cls_logits.dim() == 4:
                    logits = cls_logits[-1]
                elif cls_logits.dim() == 3:
                    logits = cls_logits
                else:
                    logits = None
                if logits is not None and logits.shape[0] == batch_size and logits.shape[1] == num_queries:
                    return torch.sigmoid(logits).amax(dim=-1).detach()
            return torch.ones(batch_size, num_queries, device=device)

        for task_preds in preds_dicts:
            if isinstance(task_preds, list):
                task_preds = task_preds[0]
            required_keys = {
                'v8_fused_keep',
                'v8_high_damage_query',
                'v8_non_fused_probs',
                'v8_bev_damage',
                'v8_img_damage',
            }
            if not required_keys.issubset(task_preds.keys()):
                continue

            router_variant = task_preds.get('v8_router_variant', 'a1_damage')
            use_v8_a2 = router_variant == 'a2_reliability_soft'
            use_v8_a4 = router_variant == 'a4_clean_anchor_domain_reliability'
            use_v8_reliability = (
                use_v8_a2 or
                router_variant == 'a3_balanced_norm_reliability' or
                use_v8_a4
            )
            if use_v8_reliability and not {
                    'v8_bev_reliability',
                    'v8_img_reliability',
                    'v8_reliability_state',
            }.issubset(task_preds.keys()):
                continue

            fused_keep = task_preds['v8_fused_keep'].clamp(min=1e-6, max=1.0 - 1e-6)
            high_damage = task_preds['v8_high_damage_query'] > 0.5
            non_fused_probs = task_preds['v8_non_fused_probs'].clamp(min=1e-6, max=1.0)
            bev_damage = task_preds['v8_bev_damage']
            img_damage = task_preds['v8_img_damage']
            bev_reliability = task_preds.get('v8_bev_reliability')
            img_reliability = task_preds.get('v8_img_reliability')
            reliability_state = task_preds.get('v8_reliability_state')
            if bev_reliability is not None:
                bev_reliability = bev_reliability.clamp(min=1e-6, max=1.0 - 1e-6)
            if img_reliability is not None:
                img_reliability = img_reliability.clamp(min=1e-6, max=1.0 - 1e-6)
            zero_ref = fused_keep

            batch_size = fused_keep.shape[0]
            top_cls_prob = _top_cls_prob(
                task_preds, batch_size, fused_keep.shape[1], fused_keep.device)
            for batch_idx in range(batch_size):
                meta = img_metas[batch_idx] if batch_idx < len(img_metas) else {}
                corruption_type, synthetic, _severity = self._v8_meta_corruption_type(meta)
                domain = self._v8_corruption_domain(corruption_type)

                keep_values = fused_keep[batch_idx, :, 0]
                high_mask = high_damage[batch_idx]
                query_weight = 0.25 + 0.75 * top_cls_prob[batch_idx]
                domain_multiplier = 1.0
                if domain == 'image':
                    domain_multiplier *= 1.5
                if corruption_type == 'motionblur':
                    domain_multiplier *= 2.0
                query_weight = query_weight * domain_multiplier
                clean_target = keep_values.new_full(keep_values.shape, self.v8_clean_fused_keep_target)
                if use_v8_reliability:
                    bev_rel_values = bev_reliability[batch_idx, :, 0]
                    img_rel_values = img_reliability[batch_idx, :, 0]
                    rel_clean_target = keep_values.new_full(
                        keep_values.shape, self.v8_clean_reliability_target)

                if use_v8_a4:
                    if not synthetic or domain == 'clean':
                        bev_rel_target = 0.93
                        img_rel_target = 0.93
                    elif domain == 'lidar':
                        bev_rel_target = 0.35
                        img_rel_target = 0.88
                    elif domain == 'image':
                        bev_rel_target = 0.88
                        img_rel_target = 0.35
                    elif domain == 'dual':
                        bev_rel_target = 0.55
                        img_rel_target = 0.55
                    else:
                        bev_rel_target = 0.93
                        img_rel_target = 0.93
                    domain_reliability_terms.append(0.5 * (
                        _weighted_bce(bev_rel_values, bev_rel_target, query_weight) +
                        _weighted_bce(img_rel_values, img_rel_target, query_weight)
                    ))

                if not synthetic or domain == 'clean':
                    gate_terms.append(F.binary_cross_entropy(
                        keep_values, clean_target, reduction='mean'))
                    if use_v8_reliability and not use_v8_a4:
                        gate_terms.append(0.5 * (
                            F.binary_cross_entropy(bev_rel_values, rel_clean_target, reduction='mean') +
                            F.binary_cross_entropy(img_rel_values, rel_clean_target, reduction='mean')
                        ))
                    clean_keep_sum += keep_values.detach().sum().item()
                    clean_keep_count += float(keep_values.numel())
                    continue

                if use_v8_a4 and domain in ('lidar', 'image'):
                    target_idx = (
                        torch.ones_like(high_mask.long()) if domain == 'lidar'
                        else torch.zeros_like(high_mask.long())
                    )
                    nll = F.nll_loss(
                        non_fused_probs[batch_idx].log(),
                        target_idx,
                        reduction='none')
                    modality_terms.append(
                        (nll * query_weight).sum() / query_weight.sum().clamp(min=1.0))
                    pred_idx = non_fused_probs[batch_idx].argmax(dim=-1)
                    domain_correct_sum += (pred_idx == target_idx).float().sum().item()
                    domain_correct_count += float(target_idx.numel())

                if high_mask.any():
                    state_values = reliability_state[batch_idx] if reliability_state is not None else None
                    dual_mask = (
                        (state_values == 3) if state_values is not None
                        else torch.zeros_like(high_mask)
                    )
                    single_mask = high_mask & (~dual_mask)
                    damage_target = keep_values.new_full(
                        keep_values[high_mask].shape, self.v8_damage_fused_keep_target)
                    if use_v8_reliability and dual_mask.any():
                        per_query_target = keep_values.new_full(
                            keep_values.shape, self.v8_damage_fused_keep_target)
                        per_query_target = torch.where(
                            dual_mask,
                            per_query_target.new_full(
                                per_query_target.shape, self.v8_dual_damage_fused_keep_target),
                            per_query_target)
                        damage_target = per_query_target[high_mask]
                    gate_terms.append(F.binary_cross_entropy(
                        keep_values[high_mask], damage_target, reduction='mean'))
                    damage_keep_sum += keep_values[high_mask].detach().sum().item()
                    damage_keep_count += float(high_mask.float().sum().item())
                    if single_mask.any():
                        single_damage_keep_sum += keep_values[single_mask].detach().sum().item()
                        single_damage_keep_count += float(single_mask.float().sum().item())
                    if dual_mask.any():
                        dual_damage_keep_sum += keep_values[dual_mask].detach().sum().item()
                        dual_damage_keep_count += float(dual_mask.float().sum().item())

                    if (~high_mask).any() and self.v8_low_damage_fused_keep_weight > 0:
                        low_target = clean_target[~high_mask]
                        low_loss = F.binary_cross_entropy(
                            keep_values[~high_mask], low_target, reduction='mean')
                        gate_terms.append(low_loss * self.v8_low_damage_fused_keep_weight)
                        if use_v8_reliability and not use_v8_a4:
                            low_rel_target = rel_clean_target[~high_mask]
                            gate_terms.append(0.25 * self.v8_low_damage_fused_keep_weight * (
                                F.binary_cross_entropy(
                                    bev_rel_values[~high_mask], low_rel_target, reduction='mean') +
                                F.binary_cross_entropy(
                                    img_rel_values[~high_mask], low_rel_target, reduction='mean')
                            ))

                    if (not use_v8_a4) and domain in ('lidar', 'image', 'dual'):
                        if domain == 'lidar':
                            target_idx = torch.ones_like(high_mask[high_mask].long())
                            if use_v8_reliability:
                                damage_target_rel = keep_values.new_full(
                                    bev_rel_values[high_mask].shape, self.v8_damage_reliability_target)
                                safe_target_rel = keep_values.new_full(
                                    img_rel_values[high_mask].shape, self.v8_safe_reliability_target)
                                gate_terms.append(0.5 * (
                                    F.binary_cross_entropy(
                                        bev_rel_values[high_mask], damage_target_rel, reduction='mean') +
                                    F.binary_cross_entropy(
                                        img_rel_values[high_mask], safe_target_rel, reduction='mean')
                                ))
                        elif domain == 'image':
                            target_idx = torch.zeros_like(high_mask[high_mask].long())
                            if use_v8_reliability:
                                safe_target_rel = keep_values.new_full(
                                    bev_rel_values[high_mask].shape, self.v8_safe_reliability_target)
                                damage_target_rel = keep_values.new_full(
                                    img_rel_values[high_mask].shape, self.v8_damage_reliability_target)
                                gate_terms.append(0.5 * (
                                    F.binary_cross_entropy(
                                        bev_rel_values[high_mask], safe_target_rel, reduction='mean') +
                                    F.binary_cross_entropy(
                                        img_rel_values[high_mask], damage_target_rel, reduction='mean')
                                ))
                        else:
                            if use_v8_reliability:
                                target_idx = torch.where(
                                    bev_rel_values[high_mask] >= img_rel_values[high_mask],
                                    torch.zeros_like(high_mask[high_mask].long()),
                                    torch.ones_like(high_mask[high_mask].long()))
                            else:
                                target_idx = torch.where(
                                    bev_damage[batch_idx, high_mask, 0] <= img_damage[batch_idx, high_mask, 0],
                                    torch.zeros_like(high_mask[high_mask].long()),
                                    torch.ones_like(high_mask[high_mask].long()))
                        modality_terms.append(F.nll_loss(
                            non_fused_probs[batch_idx, high_mask].log(),
                            target_idx,
                            reduction='mean'))
                        pred_idx = non_fused_probs[batch_idx, high_mask].argmax(dim=-1)
                        domain_correct_sum += (pred_idx == target_idx).float().sum().item()
                        domain_correct_count += float(target_idx.numel())
                else:
                    gate_terms.append(
                        F.binary_cross_entropy(
                            keep_values,
                            clean_target,
                            reduction='mean') * self.v8_low_damage_fused_keep_weight)
                    if use_v8_reliability and not use_v8_a4:
                        gate_terms.append(0.5 * self.v8_low_damage_fused_keep_weight * (
                            F.binary_cross_entropy(bev_rel_values, rel_clean_target, reduction='mean') +
                            F.binary_cross_entropy(img_rel_values, rel_clean_target, reduction='mean')
                        ))

                high_damage_sum += high_mask.float().sum().item()
                high_damage_count += float(high_mask.numel())

        if zero_ref is None:
            zero = torch.tensor(0.0)
        else:
            zero = zero_ref.new_tensor(0.0)

        gate_loss = torch.stack(gate_terms).mean() if gate_terms else zero
        modality_loss = torch.stack(modality_terms).mean() if modality_terms else zero
        domain_reliability_loss = (
            torch.stack(domain_reliability_terms).mean()
            if domain_reliability_terms else zero
        )
        stats = {
            'v8_train_high_damage_ratio': high_damage_sum / max(high_damage_count, 1.0),
            'v8_train_clean_fused_keep': clean_keep_sum / max(clean_keep_count, 1.0),
            'v8_train_damage_fused_keep': damage_keep_sum / max(damage_keep_count, 1.0),
            'v8_train_single_damage_keep': single_damage_keep_sum / max(single_damage_keep_count, 1.0),
            'v8_train_dual_damage_keep': dual_damage_keep_sum / max(dual_damage_keep_count, 1.0),
            'v8_train_domain_correct_ratio': domain_correct_sum / max(domain_correct_count, 1.0),
            'v8_train_domain_reliability_loss': float(domain_reliability_loss.detach().item()),
        }
        return gate_loss, modality_loss, domain_reliability_loss, stats

    def _v9_domain_targets(self, corruption_type, domain, synthetic, severity, ref):
        severity = int(max(1, min(3, severity if severity is not None else 1)))
        if (not synthetic) or domain == 'clean':
            return 0.0, 0.0, 0.95, None
        if domain == 'dual':
            anomaly = {1: 0.4, 2: 0.6, 3: 0.7}[severity]
            keep = {1: 0.90, 2: 0.84, 3: 0.78}[severity]
            return anomaly, anomaly, keep, None
        if domain == 'lidar':
            if corruption_type in ('spatialmisalignment',):
                bev_tgt = {1: 0.3, 2: 0.5, 3: 0.7}[severity]
                keep = {1: 0.85, 2: 0.80, 3: 0.72}[severity]
                route = ref.new_tensor([0.40, 0.60])
            elif corruption_type in ('temporalmisalignment',):
                bev_tgt = {1: 0.3, 2: 0.5, 3: 0.7}[severity]
                keep = {1: 0.85, 2: 0.80, 3: 0.72}[severity]
                route = ref.new_tensor([0.50, 0.50])
                return bev_tgt, 0.1, keep, route
            else:
                bev_tgt = {1: 0.4, 2: 0.7, 3: 1.0}[severity]
                keep = {1: 0.85, 2: 0.78, 3: 0.65}[severity]
                route = {
                    1: ref.new_tensor([0.30, 0.70]),
                    2: ref.new_tensor([0.25, 0.75]),
                    3: ref.new_tensor([0.15, 0.85]),
                }[severity]
            return bev_tgt, 0.0, keep, route
        if domain == 'image':
            if corruption_type == 'missingcamera':
                img_tgt = {1: 0.5, 2: 0.7, 3: 0.9}[severity]
                keep = {1: 0.82, 2: 0.75, 3: 0.68}[severity]
                route = ref.new_tensor([0.80, 0.20])
            else:
                img_tgt = {1: 0.4, 2: 0.7, 3: 1.0}[severity]
                keep = {1: 0.85, 2: 0.78, 3: 0.65}[severity]
                route = {
                    1: ref.new_tensor([0.70, 0.30]),
                    2: ref.new_tensor([0.75, 0.25]),
                    3: ref.new_tensor([0.85, 0.15]),
                }[severity]
            return 0.0, img_tgt, keep, route
        return 0.0, 0.0, 0.95, None

    @staticmethod
    def _v9_bce_debug_key(corruption_type, modality):
        mapping = {
            ('motionblur', 'img'): 'v9_motionblur_img_bce_mean',
            ('beamsreducing', 'bev'): 'v9_beams_bev_bce_mean',
            ('pointsreducing', 'bev'): 'v9_points_bev_bce_mean',
            ('dark', 'img'): 'v9_dark_img_bce_mean',
            ('brightness', 'img'): 'v9_brightness_img_bce_mean',
            ('missingcamera', 'img'): 'v9_missingcamera_img_bce_mean',
            ('fog', 'bev'): 'v9_fog_bev_bce_mean',
            ('fog', 'img'): 'v9_fog_img_bce_mean',
        }
        return mapping.get((corruption_type, modality))

    def _v9_damage_gate_loss(self, preds_dicts, img_metas=None):
        anomaly_terms = []
        gate_terms = []
        route_terms = []
        route_match_sum = 0.0
        route_match_count = 0.0
        debug_bce_terms = collections.defaultdict(list)
        zero_ref = None
        if img_metas is None:
            img_metas = []

        def _top_cls_prob(task_preds, batch_size, num_queries, device):
            cls_logits = task_preds.get('cls_logits')
            if isinstance(cls_logits, (list, tuple)):
                cls_logits = cls_logits[0] if cls_logits else None
            if isinstance(cls_logits, torch.Tensor):
                logits = cls_logits[-1] if cls_logits.dim() == 4 else cls_logits
                if logits.dim() == 3 and logits.shape[0] == batch_size and logits.shape[1] == num_queries:
                    return torch.sigmoid(logits).amax(dim=-1).detach()
            return torch.ones(batch_size, num_queries, device=device)

        def _weighted_bce(values, target_value, weight):
            target = values.new_full(values.shape, float(target_value))
            loss = F.binary_cross_entropy(values, target, reduction='none')
            while weight.dim() < loss.dim():
                weight = weight.unsqueeze(-1)
            return (loss * weight.to(loss.dtype)).sum() / weight.sum().clamp(min=1.0)

        for task_preds in preds_dicts:
            if isinstance(task_preds, list):
                task_preds = task_preds[0]
            if task_preds.get('v8_router_variant', None) != 'v9_lite_domain_anchored':
                continue
            required = {
                'v9_a_bev', 'v9_a_img', 'v8_fused_keep', 'v8_non_fused_probs'
            }
            if not required.issubset(task_preds.keys()):
                continue
            a_bev = task_preds['v9_a_bev'].clamp(min=1e-6, max=1.0 - 1e-6)
            a_img = task_preds['v9_a_img'].clamp(min=1e-6, max=1.0 - 1e-6)
            fused_keep = task_preds['v8_fused_keep'].clamp(min=1e-6, max=1.0 - 1e-6)
            non_fused_probs = task_preds['v8_non_fused_probs'].clamp(min=1e-6, max=1.0)
            zero_ref = fused_keep
            batch_size, num_queries = fused_keep.shape[:2]
            top_cls_prob = _top_cls_prob(task_preds, batch_size, num_queries, fused_keep.device)
            for batch_idx in range(batch_size):
                meta = img_metas[batch_idx] if batch_idx < len(img_metas) else {}
                corruption_type, synthetic, severity = self._v8_meta_corruption_type(meta)
                domain = self._v8_corruption_domain(corruption_type)
                severity_weight = self.v9_severity_weights.get(int(max(0, min(3, severity))), 1.0)
                query_weight = (0.25 + 0.75 * top_cls_prob[batch_idx]) * severity_weight
                bev_tgt, img_tgt, keep_tgt, route_tgt = self._v9_domain_targets(
                    corruption_type, domain, synthetic, severity, fused_keep)
                bev_bce = _weighted_bce(a_bev[batch_idx], bev_tgt, query_weight)
                img_bce = _weighted_bce(a_img[batch_idx], img_tgt, query_weight)
                anomaly_terms.append(0.5 * (bev_bce + img_bce))
                for modality, loss_value in (('bev', bev_bce), ('img', img_bce)):
                    debug_key = self._v9_bce_debug_key(corruption_type, modality)
                    if debug_key is not None:
                        debug_bce_terms[debug_key].append(loss_value.detach())
                keep_target = fused_keep.new_full(fused_keep[batch_idx].shape, float(keep_tgt))
                keep_loss = F.mse_loss(fused_keep[batch_idx], keep_target, reduction='none').squeeze(-1)
                gate_terms.append((keep_loss * query_weight).sum() / query_weight.sum().clamp(min=1.0))
                if route_tgt is not None:
                    route_tgt = route_tgt.to(device=non_fused_probs.device, dtype=non_fused_probs.dtype)
                    soft_ce = -(route_tgt.view(1, 2) * non_fused_probs[batch_idx].log()).sum(dim=-1)
                    route_terms.append((soft_ce * query_weight).sum() / query_weight.sum().clamp(min=1.0))
                    pred_idx = non_fused_probs[batch_idx].argmax(dim=-1)
                    target_idx = route_tgt.argmax().expand_as(pred_idx)
                    route_match_sum += (pred_idx == target_idx).float().sum().item()
                    route_match_count += float(pred_idx.numel())

        zero = zero_ref.new_tensor(0.0) if zero_ref is not None else torch.tensor(0.0)
        anomaly_loss = torch.stack(anomaly_terms).mean() if anomaly_terms else zero
        gate_loss = torch.stack(gate_terms).mean() if gate_terms else zero
        route_loss = torch.stack(route_terms).mean() if route_terms else zero
        stats = {
            'v9_train_anomaly_loss': float(anomaly_loss.detach().item()),
            'v9_train_gate_domain_loss': float(gate_loss.detach().item()),
            'v9_train_route_consistency_loss': float(route_loss.detach().item()),
            'v9_train_route_target_match': route_match_sum / max(route_match_count, 1.0),
        }
        for key in (
                'v9_motionblur_img_bce_mean',
                'v9_beams_bev_bce_mean',
                'v9_points_bev_bce_mean',
                'v9_dark_img_bce_mean',
                'v9_brightness_img_bce_mean',
                'v9_missingcamera_img_bce_mean',
                'v9_fog_bev_bce_mean',
                'v9_fog_img_bce_mean'):
            values = debug_bce_terms.get(key, [])
            if values:
                local_sum = torch.stack(values).sum().to(device=zero.device, dtype=torch.float32)
                local_count = local_sum.new_tensor(float(len(values)))
            else:
                local_sum = zero.detach().new_tensor(0.0, dtype=torch.float32)
                local_count = local_sum.new_tensor(0.0)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            if float(local_count.item()) > 0.0:
                stats[key] = float((local_sum / local_count.clamp(min=1.0)).item())
            else:
                stats[key] = -1.0
            stats[key.replace('_mean', '_count')] = float(local_count.item())
        return anomaly_loss, gate_loss, route_loss, stats

    def _v10_zero_loss(self, ref_tensor):
        return ref_tensor.float().sum() * 0.0

    def _v10_all_reduce_count(self, count, device):
        value = torch.tensor(float(count), device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value.clamp(min=1.0)

    def _v10_local_task_gt(self, gt_bboxes_3d, gt_labels_3d, task_id, device):
        start = sum(len(names) for names in self.class_names[:task_id])
        end = start + len(self.class_names[task_id])
        boxes, labels = [], []
        max_gt = 0
        for sample_boxes, sample_labels in zip(gt_bboxes_3d, gt_labels_3d):
            sample_labels = sample_labels.to(device)
            mask = (sample_labels >= start) & (sample_labels < end)
            local_labels = (sample_labels[mask] - start).long()
            if hasattr(sample_boxes, 'gravity_center'):
                box_tensor = torch.cat(
                    (sample_boxes.gravity_center, sample_boxes.tensor[:, 3:]), dim=1)
            else:
                box_tensor = sample_boxes
            box_tensor = box_tensor.to(device)[mask]
            boxes.append(box_tensor)
            labels.append(local_labels)
            max_gt = max(max_gt, int(local_labels.numel()))
        if max_gt == 0:
            max_gt = 1
        padded_boxes = []
        padded_labels = []
        for box_tensor, label_tensor in zip(boxes, labels):
            box_pad = box_tensor.new_zeros((max_gt, 10))
            label_pad = label_tensor.new_full((max_gt,), -1)
            if box_tensor.numel() > 0:
                width = min(box_tensor.shape[-1], 10)
                box_pad[:box_tensor.shape[0], :width] = box_tensor[:, :width]
                label_pad[:label_tensor.shape[0]] = label_tensor
            padded_boxes.append(box_pad)
            padded_labels.append(label_pad)
        return {'gt_box': torch.stack(padded_boxes, dim=0), 'gt_cls': torch.stack(padded_labels, dim=0)}

    @staticmethod
    def _v10_last_tensor(value):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, torch.Tensor) and value.dim() == 4:
            value = value[-1]
        return value

    @staticmethod
    def _v10_align_query_tensor(tensor, main_nq, pad_size):
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.dim() < 3:
            return tensor
        if tensor.shape[1] == main_nq:
            return tensor
        if pad_size > 0 and tensor.shape[1] >= pad_size + main_nq:
            return tensor[:, pad_size:pad_size + main_nq]
        return tensor[:, :main_nq]

    def _build_branch_outputs_for_v10(self, task_preds):
        modalities = task_preds.get('modalities', [])
        if 'ensemble' not in modalities:
            return None
        main_center = self._v10_last_tensor(task_preds.get('center'))
        if main_center is None:
            return None
        main_nq = main_center.shape[1]
        pad_size = int(task_preds.get('dn_mask_dict', {}).get('pad_size', 0))
        source_cls = task_preds.get('selector_source_cls_logits')
        source_center = task_preds.get('selector_source_center')
        source_height = task_preds.get('selector_source_height')
        if not isinstance(source_cls, dict) or not isinstance(source_center, dict) or not isinstance(source_height, dict):
            return None
        fallback_dim = self._v10_align_query_tensor(
            self._v10_last_tensor(task_preds.get('dim')), main_nq, 0)
        fallback_rot = self._v10_align_query_tensor(
            self._v10_last_tensor(task_preds.get('rot')), main_nq, 0)
        fallback_vel = self._v10_align_query_tensor(
            self._v10_last_tensor(task_preds.get('vel')), main_nq, 0)
        if fallback_dim is None or fallback_rot is None or fallback_vel is None:
            return None
        branch_outputs = {}
        for branch in BRANCH_ORDER:
            if branch not in source_cls or branch not in source_center or branch not in source_height:
                return None
            branch_outputs[branch] = {
                'cls_logits': self._v10_align_query_tensor(source_cls[branch], main_nq, pad_size),
                'center': self._v10_align_query_tensor(source_center[branch], main_nq, pad_size),
                'height': self._v10_align_query_tensor(source_height[branch], main_nq, pad_size),
                # Step 1 did not expose per-branch dim/rot/vel source tensors.
                # Use the routed task output as a neutral fallback; Step 5 still
                # gets full formula numerics, while cls/center/height drive branch
                # discrimination until source dim/rot/vel are exposed.
                'dim': fallback_dim,
                'rot': fallback_rot,
                'vel': fallback_vel,
            }
        return branch_outputs

    def _get_v10_mask_clean(self, img_metas, batch_size, device):
        mask = torch.ones(batch_size, device=device, dtype=torch.float32)
        if img_metas is None:
            return mask
        for idx in range(batch_size):
            meta = img_metas[idx] if idx < len(img_metas) else {}
            corruption = str(meta.get('corruption_type', 'clean')).lower() if isinstance(meta, dict) else 'clean'
            applied = bool(meta.get('synthetic_corruption', False)) if isinstance(meta, dict) else False
            mask[idx] = 1.0 if (corruption in ('', 'none', 'clean') and not applied) else 0.0
        return mask

    def _v10_gather_query_values(self, tensor, pairs, pad_size=0, main_nq=None):
        if not isinstance(tensor, torch.Tensor) or pairs.numel() == 0:
            return None
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        if main_nq is not None:
            tensor = self._v10_align_query_tensor(tensor, main_nq, pad_size)
        b_idx = pairs[:, 0].long()
        q_idx = pairs[:, 1].long().clamp(min=0, max=tensor.shape[1] - 1)
        gathered = tensor[b_idx, q_idx]
        return gathered.squeeze(-1) if gathered.shape[-1] == 1 else gathered

    def _gather_v10_router_probs(self, task_preds, pairs, main_nq, pad_size):
        probs = []
        for key in ('v10_w_fused_cls', 'v10_w_bev_cls', 'v10_w_img_cls'):
            value = self._v10_gather_query_values(task_preds.get(key), pairs, pad_size, main_nq)
            if value is None:
                return None
            probs.append(value.float())
        return torch.stack(probs, dim=0).clamp(min=1e-8, max=1.0)

    def _gather_v10_dd_probs(self, task_preds, pairs, main_nq, pad_size):
        probs = []
        for key in ('v10_p_dmg_bev', 'v10_p_dmg_img', 'v10_p_dmg_dual'):
            value = self._v10_gather_query_values(task_preds.get(key), pairs, pad_size, main_nq)
            if value is None:
                return None
            probs.append(value.float())
        return torch.stack(probs, dim=0).clamp(min=1e-6, max=1.0 - 1e-6)

    def _gather_v10_class_values(self, tensor, pairs, gt_cls, main_nq, pad_size):
        tensor = self._v10_last_tensor(tensor)
        if not isinstance(tensor, torch.Tensor) or pairs.numel() == 0:
            return None
        if tensor.dim() != 3:
            return None
        tensor = self._v10_align_query_tensor(tensor, main_nq, pad_size)
        b_idx = pairs[:, 0].long()
        q_idx = pairs[:, 1].long().clamp(min=0, max=tensor.shape[1] - 1)
        g_idx = pairs[:, 2].long()
        cls_idx = gt_cls[b_idx, g_idx].long().clamp(min=0, max=tensor.shape[-1] - 1)
        return tensor[b_idx, q_idx, cls_idx].float()

    def _compute_v10_dd_consistency(self, p_dmg, pairs, img_metas, denom):
        if p_dmg is None or pairs.numel() == 0:
            return p_dmg.sum() * 0.0
        terms = []
        image_corr = {'motionblur', 'dark', 'brightness', 'missingcamera'}
        bev_corr = {'beamsreducing', 'pointsreducing'}
        for idx in range(pairs.shape[0]):
            batch_idx = int(pairs[idx, 0].item())
            meta = img_metas[batch_idx] if img_metas is not None and batch_idx < len(img_metas) else {}
            corruption = str(meta.get('corruption_type', '')).lower() if isinstance(meta, dict) else ''
            if corruption in bev_corr:
                terms.append(F.binary_cross_entropy(p_dmg[0, idx], p_dmg.new_tensor(1.0), reduction='sum'))
            elif corruption in image_corr:
                terms.append(F.binary_cross_entropy(p_dmg[1, idx], p_dmg.new_tensor(1.0), reduction='sum'))
        if not terms:
            return p_dmg.sum() * 0.0
        return torch.stack(terms).sum() / denom

    def _v10_loss_for_task(self, task_preds, task_id, gt_bboxes_3d, gt_labels_3d, img_metas=None):
        branch_outputs = self._build_branch_outputs_for_v10(task_preds)
        if branch_outputs is None:
            ref = self._v10_last_tensor(task_preds.get('center'))
            zero = ref.float().sum() * 0.0 if isinstance(ref, torch.Tensor) else torch.tensor(0.0)
            return None, zero, {}
        ref_tensor = branch_outputs['fused']['center']
        device = ref_tensor.device
        main_nq = ref_tensor.shape[1]
        pad_size = int(task_preds.get('dn_mask_dict', {}).get('pad_size', 0))
        if bool(task_preds.get('force_modality_active', False)):
            zero = self._v10_zero_loss(ref_tensor)
            return zero, (zero, zero, zero, zero, zero, zero, zero, zero), {
                'v10_n_pairs_local': 0.0,
                'v10_n_pairs_global': 0.0,
                'v10_skipped_force_modality': 1.0,
            }
        gt = self._v10_local_task_gt(gt_bboxes_3d, gt_labels_3d, task_id, device)
        matching_info = union_matching(
            branch_outputs, gt, assigner=self.assigner,
            matching_mode=self.v10_matching_mode)
        pairs = matching_info['pairs']
        denom = self._v10_all_reduce_count(int(pairs.shape[0]), device)
        if pairs.numel() == 0:
            zero = self._v10_zero_loss(ref_tensor)
            return zero, zero, {
                'v10_n_pairs_local': 0.0,
                'v10_n_pairs_global': float(denom.detach().item()),
            }
        with torch.cuda.amp.autocast(enabled=False):
            quality_full = compute_quality_per_branch(
                branch_outputs, gt, matching_info['matching'])
            quality = quality_full[:, pairs[:, 0], pairs[:, 2]]
            p_oracle = oracle_label_from_quality(
                quality, T_oracle=float(self.v10_oracle_thresholds.get('T_oracle', 0.5)))
            router_probs = self._gather_v10_router_probs(task_preds, pairs, main_nq, pad_size)
            dd_probs = self._gather_v10_dd_probs(task_preds, pairs, main_nq, pad_size)
            if router_probs is None or dd_probs is None:
                zero = self._v10_zero_loss(ref_tensor)
                return zero, zero, {'v10_n_pairs_local': float(pairs.shape[0])}
            log_router = router_probs.clamp(min=1e-8).log()
            route_kl = (
                p_oracle * (p_oracle.clamp(min=1e-8).log() - log_router)
            ).sum(dim=0).sum() / denom

            sorted_quality, sorted_idx = quality.sort(dim=0, descending=True)
            quality_gap = sorted_quality[0] - sorted_quality[1]
            tau_margin = float(self.v10_oracle_thresholds.get('tau_margin', 0.05))
            hard_mask = (quality_gap > tau_margin).float()
            oracle_idx = sorted_idx[0]
            runner_up_idx = sorted_idx[1]
            w_oracle = router_probs.gather(0, oracle_idx.unsqueeze(0)).squeeze(0)
            w_runner_up = router_probs.gather(0, runner_up_idx.unsqueeze(0)).squeeze(0)
            route_margin = (
                (w_runner_up - w_oracle + self.v10_route_margin_value).clamp(min=0.0) *
                hard_mask
            ).sum() / denom

            y_dmg = damage_label_from_quality(
                quality, tau_dmg=float(self.v10_oracle_thresholds.get('tau_dmg', 0.15)))
            dd_anomaly = F.binary_cross_entropy(dd_probs, y_dmg, reduction='sum') / denom
            dd_consistency = self._compute_v10_dd_consistency(
                dd_probs, pairs, img_metas, denom)

            center_floor = task_preds.get('v10_center_floor')
            if isinstance(center_floor, torch.Tensor):
                center_floor = self._v10_align_query_tensor(center_floor, main_nq, pad_size).float()
                mask_clean = self._get_v10_mask_clean(img_metas, center_floor.shape[0], center_floor.device)
                csg = ((center_floor - self.v10_csg_clean_target) ** 2)
                csg = (csg * mask_clean.view(-1, 1, 1)).sum() / (
                    mask_clean.sum().clamp(min=1.0) * center_floor.shape[1])
            else:
                csg = self._v10_zero_loss(ref_tensor)

            edlcue_kl = self._v10_zero_loss(ref_tensor)
            edlcue_rank = self._v10_zero_loss(ref_tensor)
            edlcue_delta = self._v10_zero_loss(ref_tensor)
            admission_logits = self._v10_last_tensor(
                task_preds.get('v10_edlcue_admission_logits'))
            delta_logits = self._v10_last_tensor(task_preds.get('v10_edlcue_delta'))
            calib_gt_logits = self._gather_v10_class_values(
                admission_logits, pairs, gt['gt_cls'], main_nq, pad_size)
            if calib_gt_logits is not None and calib_gt_logits.numel() > 0:
                quality_target = sorted_quality[0].detach().float()
                if calib_gt_logits.numel() > 1:
                    temperature = float(self.v10_oracle_thresholds.get(
                        'T_edlcue', self.v10_oracle_thresholds.get('T_oracle', 0.5)))
                    temperature = max(temperature, 1e-4)
                    target_dist = torch.softmax(quality_target / temperature, dim=0)
                    pred_log_dist = torch.log_softmax(calib_gt_logits / temperature, dim=0)
                    edlcue_kl = (
                        target_dist *
                        (target_dist.clamp(min=1e-8).log() - pred_log_dist)
                    ).sum()

                    qdiff = quality_target.unsqueeze(1) - quality_target.unsqueeze(0)
                    ldiff = calib_gt_logits.unsqueeze(1) - calib_gt_logits.unsqueeze(0)
                    rank_mask = (qdiff > tau_margin).float()
                    if rank_mask.sum() > 0:
                        edlcue_rank = (
                            (self.v10_edlcue_margin_value - ldiff).clamp(min=0.0) *
                            rank_mask
                        ).sum() / rank_mask.sum().clamp(min=1.0)

                gathered_delta = self._gather_v10_class_values(
                    delta_logits, pairs, gt['gt_cls'], main_nq, pad_size)
                if gathered_delta is not None and gathered_delta.numel() > 0:
                    edlcue_delta = gathered_delta.pow(2).mean()

            edlcue_warmup = float(getattr(self, 'v10_edlcue_warmup_factor', 1.0))
            total = (
                self.lambda_v10_route_kl * route_kl +
                self.lambda_v10_route_margin * route_margin +
                self.lambda_v10_dd_anomaly * dd_anomaly +
                self.lambda_v10_dd_consistency * dd_consistency +
                self.lambda_v10_csg * csg +
                self.lambda_v10_edlcue_kl * edlcue_warmup * edlcue_kl +
                self.lambda_v10_edlcue_rank * edlcue_warmup * edlcue_rank +
                self.lambda_v10_edlcue_delta * edlcue_delta
            )
            entropy = -(p_oracle * p_oracle.clamp(min=1e-8).log()).sum(dim=0)
            stats = {
                'v10_route_kl_raw': float(route_kl.detach().item()),
                'v10_route_margin_raw': float(route_margin.detach().item()),
                'v10_dd_anomaly_raw': float(dd_anomaly.detach().item()),
                'v10_dd_consistency_raw': float(dd_consistency.detach().item()),
                'v10_csg_raw': float(csg.detach().item()),
                'v10_edlcue_kl_raw': float(edlcue_kl.detach().item()),
                'v10_edlcue_rank_raw': float(edlcue_rank.detach().item()),
                'v10_edlcue_delta_raw': float(edlcue_delta.detach().item()),
                'v10_edlcue_warmup': edlcue_warmup,
                'v10_n_pairs_local': float(pairs.shape[0]),
                'v10_n_pairs_global': float(denom.detach().item()),
                'v10_quality_gap_p50': float(torch.quantile(quality_gap.detach().cpu(), 0.50).item()),
                'v10_quality_gap_p95': float(torch.quantile(quality_gap.detach().cpu(), 0.95).item()),
                'v10_oracle_entropy_mean': float(entropy.detach().mean().item()),
                'v10_matching_mode_is_single_best': (
                    1.0 if self.v10_matching_mode == 'single_best' else 0.0),
            }
        return total, (
            route_kl, route_margin, dd_anomaly, dd_consistency, csg,
            edlcue_kl, edlcue_rank, edlcue_delta), stats

    def _rev19_query_distill_loss(
            self,
            preds_dicts,
            teacher_queries_a4=None,
            teacher_queries_v3=None,
            weights_a4=None,
            weights_v3=None):
        if not preds_dicts:
            return None, {}
        task_preds = preds_dicts[0]
        if isinstance(task_preds, list):
            task_preds = task_preds[0]
        student_queries = task_preds.get('rev19_distill_queries')
        if student_queries is None:
            return None, {}

        def _as_weight_tensor(weights):
            if weights is None:
                return None
            if not isinstance(weights, torch.Tensor):
                weights = student_queries.new_tensor(weights)
            return weights.to(device=student_queries.device, dtype=student_queries.dtype).view(-1)

        def _one_teacher_loss(teacher_queries, weights):
            if teacher_queries is None or not isinstance(teacher_queries, torch.Tensor):
                return None, 0.0
            teacher_queries = teacher_queries.to(
                device=student_queries.device, dtype=student_queries.dtype)
            if teacher_queries.shape != student_queries.shape:
                raise ValueError(
                    "Rev19 query distillation shape mismatch: "
                    f"student={tuple(student_queries.shape)}, "
                    f"teacher={tuple(teacher_queries.shape)}")
            weights = _as_weight_tensor(weights)
            if weights is None:
                weights = student_queries.new_ones(student_queries.shape[0])
            if weights.numel() != student_queries.shape[0]:
                raise ValueError(
                    "Rev19 distillation weights must match batch size: "
                    f"weights={weights.numel()}, batch={student_queries.shape[0]}")
            active = weights > 0
            active_count = float(active.float().sum().detach().item())
            if not bool(active.any()):
                return None, 0.0
            q_student = F.normalize(student_queries[active], dim=-1)
            q_teacher = F.normalize(teacher_queries[active].detach(), dim=-1)
            per_sample = 1.0 - (q_student * q_teacher).sum(dim=-1).mean(dim=1)
            active_weights = weights[active]
            loss = (
                per_sample * active_weights
            ).sum() / active_weights.sum().clamp(min=1e-6)
            return loss, active_count

        terms = []
        stats = {}
        a4_loss, a4_count = _one_teacher_loss(teacher_queries_a4, weights_a4)
        if a4_loss is not None:
            terms.append(a4_loss)
            stats['rev19_query_distill_a4_mean'] = float(a4_loss.detach().item())
        stats['rev19_query_distill_a4_count'] = a4_count
        v3_loss, v3_count = _one_teacher_loss(teacher_queries_v3, weights_v3)
        if v3_loss is not None:
            terms.append(v3_loss)
            stats['rev19_query_distill_v3_mean'] = float(v3_loss.detach().item())
        stats['rev19_query_distill_v3_count'] = v3_count
        if not terms:
            return None, stats
        return torch.stack(terms).mean(), stats

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # [EDL] update the iteration-based warmup factor automatically
        if self.training:
            self._update_warmup_factor()
        
        loss_dict = dict()
        if self.lambda_rev19_query_distill > 0:
            rev19_loss, rev19_stats = self._rev19_query_distill_loss(
                preds_dicts,
                teacher_queries_a4=kwargs.get('rev19_teacher_queries_a4'),
                teacher_queries_v3=kwargs.get('rev19_teacher_queries_v3'),
                weights_a4=kwargs.get('rev19_distill_weights_a4'),
                weights_v3=kwargs.get('rev19_distill_weights_v3'))
            if rev19_loss is None:
                task_preds = preds_dicts[0]
                if isinstance(task_preds, list):
                    task_preds = task_preds[0]
                zero_ref = task_preds.get('rev19_distill_queries')
                if zero_ref is None:
                    zero_ref = task_preds.get('center')[0][-1]
                rev19_loss = zero_ref.new_tensor(0.0)
            loss_dict['loss_rev19_query_distill'] = (
                rev19_loss * self.lambda_rev19_query_distill)
            for key in (
                    'rev19_query_distill_a4_mean',
                    'rev19_query_distill_v3_mean',
                    'rev19_query_distill_a4_count',
                    'rev19_query_distill_v3_count'):
                loss_dict[key] = rev19_loss.new_tensor(
                    float(rev19_stats.get(key, -1.0)))
        for i, modality in enumerate(preds_dicts[0][0]["modalities"]):
            num_decoder = preds_dicts[0][0]['center'][i].shape[0]
            all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
            all_cls_alpha, all_cls_beta = collections.defaultdict(list), collections.defaultdict(list)

            for task_id, preds_dict in enumerate(preds_dicts, 0):
                for dec_id in range(num_decoder):
                    # [KEY FIX] compute the loss from the NIG-parameter form
                    # center_nig and height_nig carry the full [gamma, v, alpha, beta]
                    # dim, rot and vel are plain regression
                    pred_bbox = [
                        preds_dict[0]['center_nig'][i][dec_id] if 'center_nig' in preds_dict[0] else preds_dict[0]['center'][i][dec_id],
                        preds_dict[0]['height_nig'][i][dec_id] if 'height_nig' in preds_dict[0] else preds_dict[0]['height'][i][dec_id],
                        preds_dict[0]['dim'][i][dec_id],        # plain regression; dim_nig no longer exists
                        preds_dict[0]['rot'][i][dec_id]         # plain regression
                    ]
                    if 'vel' in preds_dict[0]:
                        pred_bbox.append(preds_dict[0]['vel'][i][dec_id])  # vel is plain regression, so there is no vel_nig
                    pred_bbox = torch.cat(pred_bbox, dim=-1)
                    all_pred_bboxes[dec_id].append(pred_bbox)
                    all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][i][dec_id])
                    # [EDL CLS] collect the alpha and beta parameters
                    all_cls_alpha[dec_id].append(preds_dict[0]['cls_alpha'][i][dec_id])
                    all_cls_beta[dec_id].append(preds_dict[0]['cls_beta'][i][dec_id])
            all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
            all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]
            all_cls_alpha = [all_cls_alpha[idx] for idx in range(num_decoder)]
            all_cls_beta = [all_cls_beta[idx] for idx in range(num_decoder)]

            loss_cls, loss_bbox = multi_apply(
                self.loss_single, all_pred_bboxes, all_pred_logits,
                all_cls_alpha, all_cls_beta,
                [gt_bboxes_3d for _ in range(num_decoder)],
                [gt_labels_3d for _ in range(num_decoder)],
            )

            loss_dict[f'loss_cls_{modality}'] = loss_cls[-1]
            loss_dict[f'loss_bbox_{modality}'] = loss_bbox[-1]

            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                               loss_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls_{modality}'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.loss_bbox_{modality}'] = loss_bbox_i
                num_dec_layer += 1

            dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
            dn_cls_alpha, dn_cls_beta = collections.defaultdict(list), collections.defaultdict(list)
            dn_mask_dicts = collections.defaultdict(list)
            for task_id, preds_dict in enumerate(preds_dicts, 0):
                for dec_id in range(num_decoder):
                    # [KEY FIX] the DN loss uses the same layout
                    # center_nig and height_nig are NIG; dim, rot and vel are plain regression
                    pred_bbox = [
                        preds_dict[0]['dn_center_nig'][i][dec_id] if 'dn_center_nig' in preds_dict[0] else preds_dict[0]['dn_center'][i][dec_id],
                        preds_dict[0]['dn_height_nig'][i][dec_id] if 'dn_height_nig' in preds_dict[0] else preds_dict[0]['dn_height'][i][dec_id],
                        preds_dict[0]['dn_dim'][i][dec_id],        # plain regression; dn_dim_nig no longer exists
                        preds_dict[0]['dn_rot'][i][dec_id]
                    ]
                    if 'vel' in preds_dict[0]:
                        pred_bbox.append(preds_dict[0]['dn_vel'][i][dec_id])  # vel is plain regression, so there is no dn_vel_nig
                    pred_bbox = torch.cat(pred_bbox, dim=-1)
                    dn_pred_bboxes[dec_id].append(pred_bbox)
                    dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][i][dec_id])
                    # [EDL CLS] the DN branch collects alpha and beta too
                    dn_cls_alpha[dec_id].append(preds_dict[0]['dn_cls_alpha'][i][dec_id])
                    dn_cls_beta[dec_id].append(preds_dict[0]['dn_cls_beta'][i][dec_id])
                    dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
            dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
            dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
            dn_cls_alpha = [dn_cls_alpha[idx] for idx in range(num_decoder)]
            dn_cls_beta = [dn_cls_beta[idx] for idx in range(num_decoder)]
            dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
            dn_loss_cls, dn_loss_bbox = multi_apply(
                self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, 
                dn_cls_alpha, dn_cls_beta, dn_mask_dicts
            )

            loss_dict[f'dn_loss_cls_{modality}'] = dn_loss_cls[-1]
            loss_dict[f'dn_loss_bbox_{modality}'] = dn_loss_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                               dn_loss_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls_{modality}'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox_{modality}'] = loss_bbox_i
                num_dec_layer += 1

            if modality == 'ensemble':
                reg_cls, reg_box, reg_prior = self._selector_regularization_loss(preds_dicts, i)
                if self.lambda_consistency_cls > 0:
                    loss_dict['loss_selector_consistency_cls'] = (
                        reg_cls * self.lambda_consistency_cls
                    )
                if self.lambda_consistency_box > 0:
                    loss_dict['loss_selector_consistency_box'] = (
                        reg_box * self.lambda_consistency_box
                    )
                if self.lambda_fused_prior > 0:
                    loss_dict['loss_selector_fused_prior'] = (
                        reg_prior * self.lambda_fused_prior
                    )
                if self.lambda_selector_target > 0:
                    (
                        selector_target_base,
                        selector_target_residual,
                        residual_usage_loss,
                        selector_second_opinion_loss,
                        selector_entropy_hard_loss,
                        alpha_res_separation_loss,
                        teacher_ratio,
                        pred_ratio,
                        prior_ratio,
                        agreement_ratio,
                        alpha_res_mean,
                        alpha_res_hard_mean,
                        alpha_res_easy_mean,
                        residual_override_ratio,
                        second_opinion_ratio,
                        prior_top2_match_ratio,
                        second_opinion_teacher_match_ratio,
                        hard_entropy_mean,
                    ) = self._selector_target_loss(
                        gt_bboxes_3d,
                        gt_labels_3d,
                        preds_dicts,
                        all_pred_bboxes[-1],
                        all_pred_logits[-1],
                    )
                    loss_dict['loss_selector_target_base'] = (
                        selector_target_base * self.lambda_selector_target * self.lambda_selector_target_base
                    )
                    loss_dict['loss_selector_target_residual'] = (
                        selector_target_residual * self.lambda_selector_target * self.lambda_selector_target_residual
                    )
                    loss_dict['loss_residual_usage'] = (
                        residual_usage_loss * self.lambda_residual_usage
                    )
                    loss_dict['loss_selector_second_opinion'] = (
                        selector_second_opinion_loss * self.lambda_selector_second_opinion
                    )
                    loss_dict['loss_selector_entropy_hard'] = (
                        selector_entropy_hard_loss * self.lambda_selector_entropy_hard
                    )
                    loss_dict['loss_alpha_res_separation'] = (
                        alpha_res_separation_loss * self.lambda_alpha_res_separation
                    )
                    loss_dict['loss_selector_target'] = (
                        loss_dict['loss_selector_target_base'] +
                        loss_dict['loss_selector_target_residual'] +
                        loss_dict['loss_selector_second_opinion'] +
                        loss_dict['loss_selector_entropy_hard'] +
                        loss_dict['loss_alpha_res_separation']
                    )

                    for task_preds in preds_dicts:
                        if isinstance(task_preds, list):
                            task_preds = task_preds[0]
                        if 'selector_weight_stats' not in task_preds:
                            continue
                        task_preds['selector_teacher_top1_ratio'] = teacher_ratio
                        task_preds['selector_pred_top1_ratio'] = pred_ratio
                        task_preds['selector_distance_prior_top1_ratio'] = prior_ratio
                        task_preds['selector_agreement_ratio'] = agreement_ratio
                        task_preds['selector_alpha_res_mean'] = alpha_res_mean
                        task_preds['selector_alpha_res_hard_query_mean'] = alpha_res_hard_mean
                        task_preds['selector_alpha_res_easy_query_mean'] = alpha_res_easy_mean
                        task_preds['selector_residual_override_ratio'] = residual_override_ratio
                        task_preds['selector_residual_usage_loss'] = float(residual_usage_loss.item())
                        task_preds['selector_second_opinion_ratio'] = second_opinion_ratio
                        task_preds['selector_prior_top2_match_ratio'] = prior_top2_match_ratio
                        task_preds['selector_second_opinion_teacher_match_ratio'] = second_opinion_teacher_match_ratio
                        task_preds['selector_hard_entropy'] = hard_entropy_mean
                        break

                if self.v7_img_router_loss_weight > 0 or self.v7_bev_router_loss_weight > 0:
                    v7_img_router_loss, v7_bev_router_loss = self._v7_router_loss(
                        gt_bboxes_3d,
                        gt_labels_3d,
                        preds_dicts,
                        all_pred_bboxes[-1],
                        all_pred_logits[-1],
                    )
                    if self.v7_img_router_loss_weight > 0:
                        loss_dict['loss_v7_img_router'] = (
                            v7_img_router_loss * self.v7_img_router_loss_weight
                        )
                    if self.v7_bev_router_loss_weight > 0:
                        loss_dict['loss_v7_bev_router'] = (
                            v7_bev_router_loss * self.v7_bev_router_loss_weight
                        )
                    if 'loss_v7_img_router' in loss_dict or 'loss_v7_bev_router' in loss_dict:
                        loss_dict['loss_v7_router'] = (
                            loss_dict.get('loss_v7_img_router', v7_img_router_loss.new_tensor(0.0)) +
                            loss_dict.get('loss_v7_bev_router', v7_bev_router_loss.new_tensor(0.0))
                        )

                if (
                    self.lambda_v8_gate_supervision > 0 or
                    self.lambda_v8_modality_supervision > 0 or
                    self.lambda_v8_domain_reliability_supervision > 0
                ):
                    v8_gate_loss, v8_modality_loss, v8_domain_loss, v8_stats = self._v8_damage_gate_loss(
                        preds_dicts,
                        img_metas=kwargs.get('img_metas'))
                    if self.lambda_v8_gate_supervision > 0:
                        loss_dict['loss_v8_gate_supervision'] = (
                            v8_gate_loss * self.lambda_v8_gate_supervision
                        )
                    if self.lambda_v8_modality_supervision > 0:
                        loss_dict['loss_v8_modality_supervision'] = (
                            v8_modality_loss * self.lambda_v8_modality_supervision
                        )
                    if self.lambda_v8_domain_reliability_supervision > 0:
                        loss_dict['loss_v8_domain_reliability_supervision'] = (
                            v8_domain_loss * self.lambda_v8_domain_reliability_supervision
                        )
                    if (
                        'loss_v8_gate_supervision' in loss_dict or
                        'loss_v8_modality_supervision' in loss_dict or
                        'loss_v8_domain_reliability_supervision' in loss_dict
                    ):
                        loss_dict['loss_v8_damage_router'] = (
                            loss_dict.get('loss_v8_gate_supervision', v8_gate_loss.new_tensor(0.0)) +
                            loss_dict.get('loss_v8_modality_supervision', v8_modality_loss.new_tensor(0.0)) +
                            loss_dict.get('loss_v8_domain_reliability_supervision', v8_domain_loss.new_tensor(0.0))
                        )
                        for task_preds in preds_dicts:
                            if isinstance(task_preds, list):
                                task_preds = task_preds[0]
                            if 'v8_fused_keep_mean' not in task_preds:
                                continue
                            task_preds.update(v8_stats)
                            break

                if (
                    self.lambda_v9_anomaly_supervision > 0 or
                    self.lambda_v9_gate_domain > 0 or
                    self.lambda_v9_route_consistency > 0
                ):
                    v9_anomaly_loss, v9_gate_loss, v9_route_loss, v9_stats = self._v9_damage_gate_loss(
                        preds_dicts,
                        img_metas=kwargs.get('img_metas'))
                    anomaly_lambda = self.lambda_v9_anomaly_supervision
                    if self.v9_anomaly_head_lambda_warmup:
                        warm = min(max(float(self.current_epoch), 0.0), 1.0)
                        anomaly_lambda *= (0.1 + 0.9 * warm)
                    if self.lambda_v9_anomaly_supervision > 0:
                        loss_dict['loss_v9_anomaly_supervision'] = (
                            v9_anomaly_loss * anomaly_lambda
                        )
                    if self.lambda_v9_gate_domain > 0:
                        loss_dict['loss_v9_gate_domain'] = (
                            v9_gate_loss * self.lambda_v9_gate_domain
                        )
                    if self.lambda_v9_route_consistency > 0:
                        loss_dict['loss_v9_route_consistency'] = (
                            v9_route_loss * self.lambda_v9_route_consistency
                        )
                    for metric_key in (
                            'v9_motionblur_img_bce_mean',
                            'v9_beams_bev_bce_mean',
                            'v9_points_bev_bce_mean',
                            'v9_dark_img_bce_mean',
                            'v9_brightness_img_bce_mean',
                            'v9_missingcamera_img_bce_mean',
                            'v9_fog_bev_bce_mean',
                            'v9_fog_img_bce_mean'):
                        metric_value = v9_stats.get(metric_key, float('nan'))
                        # Distributed MMDet requires identical log_var keys on
                        # every rank. `_v9_damage_gate_loss` all-reduces global
                        # per-corruption means and returns -1 only when the
                        # whole distributed batch has no matching corruption.
                        # These keys do not contain "loss", so they are logged
                        # but not added to the optimization loss.
                        if not math.isfinite(metric_value):
                            metric_value = -1.0
                        loss_dict[metric_key] = v9_anomaly_loss.new_tensor(metric_value)
                        count_key = metric_key.replace('_mean', '_count')
                        loss_dict[count_key] = v9_anomaly_loss.new_tensor(
                            float(v9_stats.get(count_key, 0.0))
                        )
                    for task_preds in preds_dicts:
                        if isinstance(task_preds, list):
                            task_preds = task_preds[0]
                        if task_preds.get('v8_router_variant', None) != 'v9_lite_domain_anchored':
                            continue
                        task_preds.update(v9_stats)
                        break

                if self.v10_loss_enabled:
                    v10_components = []
                    v10_stats_accum = collections.defaultdict(list)
                    for v10_task_id, v10_task_preds in enumerate(preds_dicts):
                        if isinstance(v10_task_preds, list):
                            v10_task_preds = v10_task_preds[0]
                        _v10_total, v10_raw, v10_stats = self._v10_loss_for_task(
                            v10_task_preds,
                            v10_task_id,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            img_metas=kwargs.get('img_metas'))
                        if isinstance(v10_raw, tuple):
                            v10_components.append(v10_raw)
                        for key, value in v10_stats.items():
                            v10_stats_accum[key].append(float(value))
                    if v10_components:
                        route_kl = torch.stack([item[0] for item in v10_components]).mean()
                        route_margin = torch.stack([item[1] for item in v10_components]).mean()
                        dd_anomaly = torch.stack([item[2] for item in v10_components]).mean()
                        dd_consistency = torch.stack([item[3] for item in v10_components]).mean()
                        csg = torch.stack([item[4] for item in v10_components]).mean()
                        edlcue_kl = torch.stack([item[5] for item in v10_components]).mean()
                        edlcue_rank = torch.stack([item[6] for item in v10_components]).mean()
                        edlcue_delta = torch.stack([item[7] for item in v10_components]).mean()
                        edlcue_warmup = float(getattr(self, 'v10_edlcue_warmup_factor', 1.0))
                        loss_dict['loss_v10_route_kl'] = route_kl * self.lambda_v10_route_kl
                        loss_dict['loss_v10_route_margin'] = route_margin * self.lambda_v10_route_margin
                        loss_dict['loss_v10_dd_anomaly'] = dd_anomaly * self.lambda_v10_dd_anomaly
                        loss_dict['loss_v10_dd_consistency'] = dd_consistency * self.lambda_v10_dd_consistency
                        loss_dict['loss_v10_csg'] = csg * self.lambda_v10_csg
                        loss_dict['loss_v10_edlcue_kl'] = (
                            edlcue_kl * self.lambda_v10_edlcue_kl * edlcue_warmup)
                        loss_dict['loss_v10_edlcue_rank'] = (
                            edlcue_rank * self.lambda_v10_edlcue_rank * edlcue_warmup)
                        loss_dict['loss_v10_edlcue_delta'] = edlcue_delta * self.lambda_v10_edlcue_delta
                        loss_dict['v10_edlcue_warmup'] = route_kl.new_tensor(edlcue_warmup)
                        for key, values in v10_stats_accum.items():
                            if not values:
                                continue
                            metric_value = sum(values) / len(values)
                            loss_dict[key] = route_kl.new_tensor(metric_value)
                        for task_preds in preds_dicts:
                            if isinstance(task_preds, list):
                                task_preds = task_preds[0]
                            task_preds.update({
                                key: (sum(values) / len(values))
                                for key, values in v10_stats_accum.items()
                                if values
                            })
                            task_preds['loss_v10_route_kl'] = float(route_kl.detach().item())
                            task_preds['loss_v10_route_margin'] = float(route_margin.detach().item())
                            task_preds['loss_v10_dd_anomaly'] = float(dd_anomaly.detach().item())
                            task_preds['loss_v10_dd_consistency'] = float(dd_consistency.detach().item())
                            task_preds['loss_v10_csg'] = float(csg.detach().item())
                            task_preds['loss_v10_edlcue_kl'] = float(edlcue_kl.detach().item())
                            task_preds['loss_v10_edlcue_rank'] = float(edlcue_rank.detach().item())
                            task_preds['loss_v10_edlcue_delta'] = float(edlcue_delta.detach().item())
                            break

                if self.lambda_v8_center_delta > 0:
                    center_delta_terms = []
                    for task_preds in preds_dicts:
                        if isinstance(task_preds, list):
                            task_preds = task_preds[0]
                        center_delta = task_preds.get('v8_output_delta_center')
                        if center_delta is not None:
                            center_delta_terms.append(center_delta.clamp(max=10.0).mean())
                    if center_delta_terms:
                        loss_dict['loss_v8_center_delta'] = (
                            torch.stack(center_delta_terms).mean() * self.lambda_v8_center_delta
                        )

        # -----------------------------------------------------------
        # [DEBUG] EDL Training Monitor (once per iteration, rank 0 only)
        # NIG parameters: every 50 iterations (aligned with the torch epoch info)
        # Verbose debug: every 250 iterations
        # -----------------------------------------------------------
        if self.training and self.debug:
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            if is_main_process:
                def _debug_task_preds(task_preds):
                    if isinstance(task_preds, (list, tuple)):
                        task_preds = task_preds[0] if task_preds else {}
                    return task_preds if isinstance(task_preds, dict) else {}

                def _debug_scalar(value, default=float('nan')):
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 0:
                            return default
                        return float(value.detach().float().mean().item())
                    if isinstance(value, Number):
                        return float(value)
                    return default

                def _debug_flatten_tensor(value):
                    value = self._v10_last_tensor(value)
                    if not isinstance(value, torch.Tensor) or value.numel() == 0:
                        return None
                    flat = value.detach().float().reshape(-1)
                    finite = torch.isfinite(flat)
                    if not finite.any():
                        return None
                    return flat[finite]

                def _debug_cat_key(preds_dicts, key):
                    chunks = []
                    for _task_preds in preds_dicts:
                        flat = _debug_flatten_tensor(_debug_task_preds(_task_preds).get(key))
                        if flat is not None:
                            chunks.append(flat)
                    return torch.cat(chunks, dim=0) if chunks else None

                def _debug_tensor_stats(value):
                    flat = value if isinstance(value, torch.Tensor) else _debug_flatten_tensor(value)
                    if flat is None or flat.numel() == 0:
                        return None
                    abs_flat = flat.abs()
                    return {
                        'mean': float(flat.mean().item()),
                        'abs_mean': float(abs_flat.mean().item()),
                        'p95_abs': float(torch.quantile(abs_flat, 0.95).item()),
                        'max_abs': float(abs_flat.max().item()),
                        'pos_ratio': float((flat > 0).float().mean().item()),
                        'neg_ratio': float((flat < 0).float().mean().item()),
                    }

                def _debug_edlcue_task(preds_dicts):
                    for _task_preds in preds_dicts:
                        _task_preds = _debug_task_preds(_task_preds)
                        if 'v10_edlcue_delta' in _task_preds:
                            return _task_preds
                    return None

                def _debug_edlcue_score_stats(preds_dicts, topk=50):
                    raw_chunks = []
                    admission_chunks = []
                    for _task_preds in preds_dicts:
                        _task_preds = _debug_task_preds(_task_preds)
                        raw_logits = self._v10_last_tensor(_task_preds.get('cls_logits'))
                        admission_logits = self._v10_last_tensor(
                            _task_preds.get('v10_edlcue_admission_logits'))
                        if (
                                isinstance(raw_logits, torch.Tensor) and
                                isinstance(admission_logits, torch.Tensor) and
                                raw_logits.shape == admission_logits.shape):
                            raw_chunks.append(raw_logits.detach().float().sigmoid().reshape(-1))
                            admission_chunks.append(
                                admission_logits.detach().float().sigmoid().reshape(-1))
                    if not raw_chunks:
                        return None
                    raw_scores = torch.cat(raw_chunks, dim=0)
                    admission_scores = torch.cat(admission_chunks, dim=0)
                    finite = torch.isfinite(raw_scores) & torch.isfinite(admission_scores)
                    if not finite.any():
                        return None
                    raw_scores = raw_scores[finite]
                    admission_scores = admission_scores[finite]
                    score_delta = admission_scores - raw_scores
                    k = min(int(topk), raw_scores.numel())
                    if k <= 0:
                        return None
                    raw_top_vals, raw_top_idx = raw_scores.topk(k)
                    admission_top_vals, admission_top_idx = admission_scores.topk(k)
                    raw_top_set = set(raw_top_idx.detach().cpu().tolist())
                    admission_top_set = set(admission_top_idx.detach().cpu().tolist())
                    overlap = len(raw_top_set.intersection(admission_top_set)) / float(k)
                    return {
                        'k': k,
                        'raw_top_mean': float(raw_top_vals.mean().item()),
                        'admission_top_mean': float(admission_top_vals.mean().item()),
                        'admission_delta_top_mean': float(score_delta[admission_top_idx].mean().item()),
                        'score_delta_abs_mean': float(score_delta.abs().mean().item()),
                        'topk_overlap': float(overlap),
                    }

                # NIG parameter summary (every 50 iterations, aligned with the mmdet logger)
                if self.debug_counter % self.nig_debug_interval == 0:
                    print(f"\n[EDL] Iter={self.debug_counter} | warmup={self.edl_warmup_factor:.4f}")
                    
                    warnings = []
                    # Order of the heads to report (dim and vel do not use EDL)
                    head_order = ['center', 'height', 'reg']
                    head_dims = {'center': 2, 'height': 1, 'reg': 2}
                    
                    # === EDL Cls Debug Info ===
                    for preds_dict in preds_dicts:
                        if 'cls_alpha' in preds_dict[0] and 'cls_beta' in preds_dict[0]:
                            # Take the last decoder layer
                            cls_alpha = preds_dict[0]['cls_alpha'][0][-1]  # [bs, nq, num_cls]
                            cls_beta = preds_dict[0]['cls_beta'][0][-1]
                            
                            alpha_mean = cls_alpha.mean().item()
                            alpha_min = cls_alpha.min().item()
                            alpha_max = cls_alpha.max().item()
                            
                            beta_mean = cls_beta.mean().item()
                            beta_min = cls_beta.min().item()
                            beta_max = cls_beta.max().item()
                            
                            probs = cls_alpha / (cls_alpha + cls_beta)
                            
                            # [DEBUG] report foreground (top-k) and background statistics separately
                            # Most queries are background, which drags the mean down
                            # The 50 highest-probability queries are used to observe how much evidence "likely objects" carry
                            
                            # Flatten [bs, nq, num_cls] -> [N] using each query's most probable class
                            max_probs, _ = probs.max(dim=-1)
                            flat_probs = max_probs.flatten()
                            
                            # Top-k: assume each batch has objects and take the 50 most confident queries
                            k = min(50, flat_probs.numel())
                            topk_vals, topk_inds = torch.topk(flat_probs, k)
                            
                            # Extract the top-k alpha/beta. This is a simplification: alpha is taken at the max-probability class
                            # Strictly, alpha should be taken at the query's predicted class
                            # For simplicity the max alpha is used as the representative value
                            alpha_max_val, _ = cls_alpha.max(dim=-1)
                            beta_mean_val = cls_beta.mean(dim=-1)  # beta is usually similar across classes
                            
                            alpha_flat = alpha_max_val.flatten()
                            beta_flat = beta_mean_val.flatten()
                            
                            alpha_topk_vals = alpha_flat[topk_inds]
                            beta_topk_vals = beta_flat[topk_inds]
                            
                            alpha_topk_mean = alpha_topk_vals.mean().item()
                            alpha_topk_min = alpha_topk_vals.min().item()
                            alpha_topk_max = alpha_topk_vals.max().item()
                            
                            beta_topk_mean = beta_topk_vals.mean().item()
                            beta_topk_min = beta_topk_vals.min().item()
                            beta_topk_max = beta_topk_vals.max().item()
                            
                            p_topk_mean = topk_vals.mean().item()
                            p_topk_min = topk_vals.min().item()
                            p_topk_max = topk_vals.max().item()
                            
                            print(f"cls(Top{k}): α = {alpha_topk_mean:.2f} [{alpha_topk_min:.1f}, {alpha_topk_max:.1f}], β = {beta_topk_mean:.2f} [{beta_topk_min:.1f}, {beta_topk_max:.1f}], p = {p_topk_mean:.3f} [{p_topk_min:.2f}, {p_topk_max:.2f}]")
                            
                            # Warning checks; alpha is checked on the top-k only, so background does not drag the mean down
                            if alpha_topk_mean < 1.5:
                                warnings.append(f"cls (Top{k}): α={alpha_topk_mean:.2f} too low (foreground evidence weak)!")
                        break  # first task only

                    # === EDL Bbox Debug Info ===
                    for preds_dict in preds_dicts:
                        for head_name in head_order:
                            nig_key = f'{head_name}_nig'
                            if nig_key in preds_dict[0]:
                                # Take the last decoder layer
                                nig_data = preds_dict[0][nig_key][0][-1]  # [bs, nq, dim*4]
                                dim = head_dims.get(head_name, 2)
                                nig_reshaped = nig_data.reshape(-1, dim, 4)
                                
                                # Extract the NIG parameters (v = index 1, alpha = index 2, beta = index 3)
                                v_vals = nig_reshaped[..., 1]
                                alpha_vals = nig_reshaped[..., 2]
                                beta_vals = nig_reshaped[..., 3]
                                
                                alpha_mean = alpha_vals.mean().item()
                                alpha_min = alpha_vals.min().item()
                                alpha_max = alpha_vals.max().item()
                                v_mean = v_vals.mean().item()
                                v_min = v_vals.min().item()
                                v_max = v_vals.max().item()
                                
                                # Epistemic uncertainty: eps = beta / ((alpha - 1) * v)
                                epsilon = (beta_vals / ((alpha_vals - 1.0).clamp(min=1e-6) * v_vals.clamp(min=1e-6))).mean().item()
                                
                                print(f"    {head_name:>6}: α = {alpha_mean:.2f} [{alpha_min:.1f}, {alpha_max:.1f}], v = {v_mean:.2f} [{v_min:.1f}, {v_max:.1f}], ε = {epsilon:.6f}")
                                
                                # Warning checks
                                if alpha_mean < 1.5:
                                    warnings.append(f"{head_name}: α={alpha_mean:.2f} too close to 1!")
                                if v_mean < 1.0:
                                    warnings.append(f"{head_name}: v={v_mean:.2f} too low!")
                        break  # first task only
                    
                    cue_task_preds = _debug_edlcue_task(preds_dicts)
                    cue_delta_flat = _debug_cat_key(preds_dicts, 'v10_edlcue_delta')
                    cue_delta_stats = _debug_tensor_stats(cue_delta_flat)
                    if cue_task_preds is not None and cue_delta_stats is not None:
                        trainable_stats = _debug_tensor_stats(
                            _debug_cat_key(preds_dicts, 'v10_edlcue_trainable_delta'))
                        manual_stats = _debug_tensor_stats(
                            _debug_cat_key(preds_dicts, 'v10_edlcue_manual_delta'))
                        max_delta = _debug_scalar(
                            cue_task_preds.get('v10_edlcue_max_delta'), default=float('nan'))
                        active = _debug_scalar(
                            cue_task_preds.get('v10_edlcue_active'), default=0.0)
                        train_abs = (
                            trainable_stats['abs_mean']
                            if trainable_stats is not None else float('nan'))
                        manual_abs = (
                            manual_stats['abs_mean']
                            if manual_stats is not None else float('nan'))
                        print(
                            "v10_edlcue(delta):"
                            f" active={active:.0f},"
                            f" mean={cue_delta_stats['mean']:.4f},"
                            f" abs={cue_delta_stats['abs_mean']:.4f},"
                            f" p95_abs={cue_delta_stats['p95_abs']:.4f},"
                            f" max_abs={cue_delta_stats['max_abs']:.4f},"
                            f" train_abs={train_abs:.4f},"
                            f" manual_abs={manual_abs:.4f},"
                            f" bound={max_delta:.3f},"
                            f" warm={getattr(self, 'v10_edlcue_warmup_factor', 1.0):.3f}"
                        )
                        if (
                                math.isfinite(max_delta) and
                                cue_delta_stats['max_abs'] > max_delta + 1e-5):
                            warnings.append(
                                f"v10_edlcue: |delta|max={cue_delta_stats['max_abs']:.4f} "
                                f"exceeds bound {max_delta:.4f}!")

                    # Emit the warning
                    for warn in warnings:
                        print(f"\n  ⚠️ Warning: {warn}\n")
                
                # Verbose debug (every 250 iterations)
                if self.debug_counter % self.debug_interval == 0:
                    total_cls = sum(v.item() for k, v in loss_dict.items() if 'loss_cls' in k and 'd0.' not in k and 'd1.' not in k and 'd2.' not in k and 'd3.' not in k and 'd4.' not in k and 'dn_' not in k)
                    total_bbox = sum(v.item() for k, v in loss_dict.items() if 'loss_bbox' in k and 'd0.' not in k and 'd1.' not in k and 'd2.' not in k and 'd3.' not in k and 'd4.' not in k and 'dn_' not in k)
                    
                    print(
                        f"\n[EDL Monitor] Iter={self.debug_counter}"
                        f" | warmup={self.edl_warmup_factor:.3f}"
                        f" | edlcue_warm={getattr(self, 'v10_edlcue_warmup_factor', 1.0):.3f}"
                        f" | detach_γ={self.detach_gamma}")
                    print(f"  Loss: cls_total≈{total_cls:.4f}, bbox_total≈{total_bbox:.4f}\n")

                    cue_task_preds = _debug_edlcue_task(preds_dicts)
                    cue_delta_flat = _debug_cat_key(preds_dicts, 'v10_edlcue_delta')
                    cue_delta_stats = _debug_tensor_stats(cue_delta_flat)
                    if cue_task_preds is not None and cue_delta_stats is not None:
                        trainable_stats = _debug_tensor_stats(
                            _debug_cat_key(preds_dicts, 'v10_edlcue_trainable_delta'))
                        manual_stats = _debug_tensor_stats(
                            _debug_cat_key(preds_dicts, 'v10_edlcue_manual_delta'))
                        score_stats = _debug_edlcue_score_stats(preds_dicts, topk=50)
                        max_delta = _debug_scalar(
                            cue_task_preds.get('v10_edlcue_max_delta'), default=float('nan'))
                        active = _debug_scalar(
                            cue_task_preds.get('v10_edlcue_active'), default=0.0)
                        near_bound = float('nan')
                        bounded_ok = float('nan')
                        if math.isfinite(max_delta) and max_delta > 0:
                            near_bound = float(
                                (cue_delta_flat.abs() >= max_delta * 0.95).float().mean().item())
                            bounded_ok = float(cue_delta_stats['max_abs'] <= max_delta + 1e-5)
                        cue_mode = getattr(
                            getattr(self, 'ensemble', None), 'v10_edlcue_mode', 'unknown')
                        train_mean = (
                            trainable_stats['mean']
                            if trainable_stats is not None else float('nan'))
                        train_abs = (
                            trainable_stats['abs_mean']
                            if trainable_stats is not None else float('nan'))
                        manual_mean = (
                            manual_stats['mean']
                            if manual_stats is not None else float('nan'))
                        manual_abs = (
                            manual_stats['abs_mean']
                            if manual_stats is not None else float('nan'))
                        print(
                            "  V10 EDL Cue:"
                            f" mode={cue_mode},"
                            f" active={active:.0f},"
                            f" bound={max_delta:.3f},"
                            f" bounded_ok={bounded_ok:.0f},"
                            f" mean={cue_delta_stats['mean']:.4f},"
                            f" abs={cue_delta_stats['abs_mean']:.4f},"
                            f" p95_abs={cue_delta_stats['p95_abs']:.4f},"
                            f" max_abs={cue_delta_stats['max_abs']:.4f},"
                            f" pos={cue_delta_stats['pos_ratio']:.3f},"
                            f" neg={cue_delta_stats['neg_ratio']:.3f},"
                            f" near_bound={near_bound:.3f}"
                        )
                        print(
                            "  V10 EDL Cue Components:"
                            f" train_mean={train_mean:.4f},"
                            f" train_abs={train_abs:.4f},"
                            f" manual_mean={manual_mean:.4f},"
                            f" manual_abs={manual_abs:.4f}"
                        )
                        if score_stats is not None:
                            print(
                                "  V10 EDL Cue Scores:"
                                f" top{score_stats['k']}_raw={score_stats['raw_top_mean']:.4f},"
                                f" top{score_stats['k']}_admit={score_stats['admission_top_mean']:.4f},"
                                f" admit_delta_top={score_stats['admission_delta_top_mean']:.5f},"
                                f" delta_abs_all={score_stats['score_delta_abs_mean']:.5f},"
                                f" topk_overlap={score_stats['topk_overlap']:.3f}"
                            )
                        print(
                            "  V10 EDL Cue Loss:"
                            f" kl_raw={cue_task_preds.get('v10_edlcue_kl_raw', float('nan')):.4f},"
                            f" rank_raw={cue_task_preds.get('v10_edlcue_rank_raw', float('nan')):.4f},"
                            f" delta_raw={cue_task_preds.get('v10_edlcue_delta_raw', float('nan')):.4f},"
                            f" warm={cue_task_preds.get('v10_edlcue_warmup', getattr(self, 'v10_edlcue_warmup_factor', 1.0)):.3f},"
                            f" kl_scaled={_debug_scalar(loss_dict.get('loss_v10_edlcue_kl'), 0.0):.4f},"
                            f" rank_scaled={_debug_scalar(loss_dict.get('loss_v10_edlcue_rank'), 0.0):.4f},"
                            f" delta_scaled={_debug_scalar(loss_dict.get('loss_v10_edlcue_delta'), 0.0):.4f},"
                            f" pairs={cue_task_preds.get('v10_n_pairs_global', float('nan')):.0f},"
                            f" gap_p50={cue_task_preds.get('v10_quality_gap_p50', float('nan')):.4f},"
                            f" gap_p95={cue_task_preds.get('v10_quality_gap_p95', float('nan')):.4f},"
                            f" oracle_H={cue_task_preds.get('v10_oracle_entropy_mean', float('nan')):.4f}"
                        )

                    is_v8_reliability_batch = False
                    for _task_preds in preds_dicts:
                        if isinstance(_task_preds, list):
                            _task_preds = _task_preds[0]
                        if _task_preds.get('v8_router_variant', None) in (
                            'a2_reliability_soft',
                            'a3_balanced_norm_reliability',
                            'a4_clean_anchor_domain_reliability',
                            'v9_lite_domain_anchored',
                        ):
                            is_v8_reliability_batch = True
                            break

                    if 'loss_selector_consistency_cls' in loss_dict or 'loss_selector_consistency_box' in loss_dict:
                        print(
                            "  Selector Reg:"
                            f" cls≈{loss_dict.get('loss_selector_consistency_cls', torch.tensor(0.0)).item():.4f},"
                            f" box≈{loss_dict.get('loss_selector_consistency_box', torch.tensor(0.0)).item():.4f},"
                            f" prior≈{loss_dict.get('loss_selector_fused_prior', torch.tensor(0.0)).item():.4f}"
                        )
                        if (not is_v8_reliability_batch) and 'loss_selector_target' in loss_dict:
                            print(
                                "  Selector Target:"
                                f" base≈{loss_dict.get('loss_selector_target_base', torch.tensor(0.0)).item():.4f},"
                                f" residual≈{loss_dict.get('loss_selector_target_residual', torch.tensor(0.0)).item():.4f},"
                                f" total≈{loss_dict.get('loss_selector_target', torch.tensor(0.0)).item():.4f}"
                            )
                        if (not is_v8_reliability_batch) and 'loss_residual_usage' in loss_dict:
                            print(
                                "  Residual Usage:"
                                f" loss≈{loss_dict.get('loss_residual_usage', torch.tensor(0.0)).item():.4f}"
                            )
                        if (not is_v8_reliability_batch) and (
                            'loss_selector_second_opinion' in loss_dict or
                            'loss_selector_entropy_hard' in loss_dict or
                            'loss_alpha_res_separation' in loss_dict
                        ):
                            print(
                                "  Selector V3.1:"
                                f" second≈{loss_dict.get('loss_selector_second_opinion', torch.tensor(0.0)).item():.4f},"
                                f" entropy_hard≈{loss_dict.get('loss_selector_entropy_hard', torch.tensor(0.0)).item():.4f},"
                                f" alpha_sep≈{loss_dict.get('loss_alpha_res_separation', torch.tensor(0.0)).item():.4f}"
                            )

                    for task_preds in preds_dicts:
                        if isinstance(task_preds, list):
                            task_preds = task_preds[0]
                        if 'selector_weight_stats' not in task_preds:
                            continue

                        weight_stats = task_preds['selector_weight_stats']
                        top1_ratio = task_preds.get('selector_top1_ratio', {})
                        per_mod_unc = task_preds.get('selector_per_modality_unc', {})
                        entropy = task_preds.get('selector_entropy', float('nan'))
                        fused_w = task_preds.get('selector_fused_weight_mean', float('nan'))
                        alpha_res_mean = task_preds.get('selector_alpha_res_mean', float('nan'))
                        alpha_res_hard_mean = task_preds.get('selector_alpha_res_hard_query_mean', float('nan'))
                        alpha_res_easy_mean = task_preds.get('selector_alpha_res_easy_query_mean', float('nan'))
                        override_ratio = task_preds.get('selector_residual_override_ratio', float('nan'))
                        prior_ratio = task_preds.get('selector_distance_prior_top1_ratio', {})
                        quality_top1_ratio = task_preds.get('selector_quality_top1_ratio', prior_ratio)
                        tail_ratio = task_preds.get('selector_uncertainty_tail_query_ratio', {})
                        second_opinion_ratio = task_preds.get('selector_second_opinion_ratio', float('nan'))
                        prior_top2_match_ratio = task_preds.get('selector_prior_top2_match_ratio', float('nan'))
                        second_opinion_teacher_match_ratio = task_preds.get(
                            'selector_second_opinion_teacher_match_ratio', float('nan'))
                        hard_entropy = task_preds.get('selector_hard_entropy', float('nan'))
                        score_components = task_preds.get('selector_score_component_stats', {})
                        router_variant = task_preds.get('v8_router_variant', None)
                        is_v8_reliability = router_variant in (
                            'a2_reliability_soft',
                            'a3_balanced_norm_reliability',
                            'a4_clean_anchor_domain_reliability',
                            'v9_lite_domain_anchored',
                        )

                        if is_v8_reliability and 'v8_fused_keep_mean' in task_preds:
                            v8_rel_state = task_preds.get('v8_reliability_state_ratio', {})
                            v8_keep_state = task_preds.get('v8_fused_keep_by_state', {})
                            v8_soft = task_preds.get('v8_soft_weight_ratio', {})
                            v8_non_fused = task_preds.get('v8_non_fused_weight_ratio', {})
                            single_keep_vals = [
                                v8_keep_state.get('bev_unreliable', float('nan')),
                                v8_keep_state.get('img_unreliable', float('nan')),
                            ]
                            single_keep_vals = [
                                v for v in single_keep_vals
                                if not (isinstance(v, float) and math.isnan(v))
                            ]
                            single_keep = (
                                sum(single_keep_vals) / max(len(single_keep_vals), 1)
                                if single_keep_vals else float('nan')
                            )
                            print(
                                "  V8 Reliability:"
                                f" r_bev≈{task_preds.get('v8_bev_reliability_mean', float('nan')):.3f},"
                                f" r_img≈{task_preds.get('v8_img_reliability_mean', float('nan')):.3f},"
                                f" margin≈{task_preds.get('v8_reliability_margin_mean', float('nan')):.3f},"
                                f" both≈{v8_rel_state.get('both_reliable', float('nan')):.3f},"
                                f" bev_dmg≈{v8_rel_state.get('bev_unreliable', float('nan')):.3f},"
                                f" img_dmg≈{v8_rel_state.get('img_unreliable', float('nan')):.3f},"
                                f" dual≈{v8_rel_state.get('dual_unreliable', float('nan')):.3f},"
                                f" amb≈{v8_rel_state.get('ambiguous', float('nan')):.3f}"
                            )
                            print(
                                "  V8 Gate:"
                                f" keep≈{task_preds.get('v8_fused_keep_mean', float('nan')):.3f},"
                                f" center_keep≈{task_preds.get('v8_center_fused_keep_mean', float('nan')):.3f},"
                                f" clean_keep≈{v8_keep_state.get('both_reliable', float('nan')):.3f},"
                                f" single_keep≈{single_keep:.3f},"
                                f" dual_keep≈{v8_keep_state.get('dual_unreliable', float('nan')):.3f},"
                                f" non_fused_mass≈{task_preds.get('v8_damage_gate_mean', float('nan')):.3f},"
                                f" high≈{task_preds.get('v8_high_damage_query_ratio', float('nan')):.3f}"
                            )
                            print(
                                "  V8 Route:"
                                f" w_fused={v8_soft.get('fused', float('nan')):.3f},"
                                f" w_bev={v8_soft.get('bev', float('nan')):.3f},"
                                f" w_img={v8_soft.get('img', float('nan')):.3f},"
                                f" nf_bev={v8_non_fused.get('bev', float('nan')):.3f},"
                                f" nf_img={v8_non_fused.get('img', float('nan')):.3f},"
                                f" nf_entropy≈{task_preds.get('v8_non_fused_entropy_mean', float('nan')):.3f},"
                                f" center_allowed≈{task_preds.get('v8_center_route_allowed_ratio', float('nan')):.3f},"
                                f" domain_ok≈{task_preds.get('v8_train_domain_correct_ratio', float('nan')):.3f}"
                            )
                            print(
                                "  V8 Context Damage:"
                                f" bev≈{task_preds.get('v8_context_bev_damage_mean', float('nan')):.3f},"
                                f" img≈{task_preds.get('v8_context_img_damage_mean', float('nan')):.3f}"
                            )
                            if router_variant == 'a4_clean_anchor_domain_reliability':
                                print(
                                    "  V8 A4 Anchor:"
                                    f" ready≈{task_preds.get('v8_a4_anchor_ready', float('nan')):.0f},"
                                    f" updates≈{task_preds.get('v8_a4_clean_anchor_update_count', float('nan')):.0f},"
                                    f" z_bev≈{task_preds.get('v8_a4_bev_damage_z_mean', float('nan')):.3f},"
                                    f" z_img≈{task_preds.get('v8_a4_img_damage_z_mean', float('nan')):.3f},"
                                    f" cls_z_bev≈{task_preds.get('v8_a4_cls_unc_z_bev_mean', float('nan')):.3f},"
                                    f" cls_z_img≈{task_preds.get('v8_a4_cls_unc_z_img_mean', float('nan')):.3f},"
                                    f" route_bev≈{task_preds.get('v8_a4_route_score_bev_mean', float('nan')):.3f},"
                                    f" route_img≈{task_preds.get('v8_a4_route_score_img_mean', float('nan')):.3f},"
                                    f" dom_rel≈{task_preds.get('v8_train_domain_reliability_loss', float('nan')):.4f}"
                                )
                            if router_variant == 'v9_lite_domain_anchored':
                                print(
                                    "  V9 Anomaly:"
                                    f" a_bev≈{task_preds.get('v9_a_bev_mean', float('nan')):.3f},"
                                    f" a_img≈{task_preds.get('v9_a_img_mean', float('nan')):.3f},"
                                    f" bev>tau≈{task_preds.get('v9_a_bev_above_tau_mean', float('nan')):.3f},"
                                    f" img>tau≈{task_preds.get('v9_a_img_above_tau_mean', float('nan')):.3f},"
                                    f" clean_like≈{task_preds.get('v9_clean_like_mask_mean', float('nan')):.3f},"
                                    f" tau=({task_preds.get('v9_tau_bev_current', float('nan')):.3f},"
                                    f"{task_preds.get('v9_tau_img_current', float('nan')):.3f})"
                                )
                                print(
                                    "  V9 Tail/Route:"
                                    f" d_bev≈{task_preds.get('v9_anchor_distance_bev_mean', float('nan')):.3f},"
                                    f" d_img≈{task_preds.get('v9_anchor_distance_img_mean', float('nan')):.3f},"
                                    f" tail=({task_preds.get('v9_tail_excess_bev_mean', float('nan')):.3f},"
                                    f"{task_preds.get('v9_tail_excess_img_mean', float('nan')):.3f}),"
                                    f" q95=({task_preds.get('v9_clean_q95_z_bev', float('nan')):.3f},"
                                    f"{task_preds.get('v9_clean_q95_z_img', float('nan')):.3f}),"
                                    f" rscore=({task_preds.get('v9_route_score_bev_mean', float('nan')):.3f},"
                                    f"{task_preds.get('v9_route_score_img_mean', float('nan')):.3f}),"
                                    f" route_tgt≈{task_preds.get('v9_train_route_target_match', float('nan')):.3f}"
                                )
                                print(
                                    "  V9 Online/Guard:"
                                    f" dist_thr=({task_preds.get('v9_clean_distance_threshold_bev', float('nan')):.3f},"
                                    f"{task_preds.get('v9_clean_distance_threshold_img', float('nan')):.3f}),"
                                    f" updates=(tau:{task_preds.get('v9_tau_update_count', float('nan')):.0f},"
                                    f"dist:{task_preds.get('v9_clean_distance_update_count', float('nan')):.0f}),"
                                    f" frozen≈{task_preds.get('v9_clean_anchor_frozen', float('nan')):.0f},"
                                    f" vel_l1≈{task_preds.get('v9_velocity_l1_mean', float('nan')):.4f},"
                                    f" keep_min≈{task_preds.get('v9_current_keep_min', float('nan')):.2f},"
                                    f" gate_unit_q50≈{task_preds.get('v9_fused_keep_unit_q50', float('nan')):.3f},"
                                    f" at_min≈{task_preds.get('v9_fused_keep_at_min_mean', float('nan')):.3f}"
                                )
                                print(
                                    "  V9 Loss:"
                                    f" anomaly≈{task_preds.get('v9_train_anomaly_loss', float('nan')):.4f},"
                                    f" gate≈{task_preds.get('v9_train_gate_domain_loss', float('nan')):.4f},"
                                    f" route≈{task_preds.get('v9_train_route_consistency_loss', float('nan')):.4f},"
                                    f" mblur_img_bce≈{task_preds.get('v9_motionblur_img_bce_mean', float('nan')):.4f},"
                                    f" beams_bev_bce≈{task_preds.get('v9_beams_bev_bce_mean', float('nan')):.4f}"
                                )
                            print(
                                "  V8 Output Delta:"
                                f" cls_l1={task_preds.get('v8_output_delta_cls_mean', float('nan')):.5f},"
                                f" center_l1={task_preds.get('v8_output_delta_center_mean', float('nan')):.5f},"
                                f" height_l1={task_preds.get('v8_output_delta_height_mean', float('nan')):.5f}\n"
                            )
                            break

                        print(
                            "  Selector Weights:"
                            f" fused={weight_stats.get('fused', float('nan')):.3f},"
                            f" bev={weight_stats.get('bev', float('nan')):.3f},"
                            f" img={weight_stats.get('img', float('nan')):.3f},"
                            f" entropy={entropy:.3f},"
                            f" fused_w={fused_w:.3f}"
                        )
                        print(
                            "  Selector Top1:"
                            f" fused={top1_ratio.get('fused', float('nan')):.3f},"
                            f" bev={top1_ratio.get('bev', float('nan')):.3f},"
                            f" img={top1_ratio.get('img', float('nan')):.3f}"
                        )
                        if 'v8_fused_keep_mean' in task_preds:
                            v8_non_fused = task_preds.get('v8_non_fused_weight_ratio', {})
                            v8_state = task_preds.get('v8_damage_state_ratio', {})
                            v8_selected = task_preds.get('v8_selected_source_ratio', {})
                            print(
                                "  V8 Gate:"
                                f" keep≈{task_preds.get('v8_fused_keep_mean', float('nan')):.3f},"
                                f" damage_gate≈{task_preds.get('v8_damage_gate_mean', float('nan')):.3f},"
                                f" high_damage≈{task_preds.get('v8_high_damage_query_ratio', float('nan')):.3f},"
                                f" loss_keep_clean≈{task_preds.get('v8_train_clean_fused_keep', float('nan')):.3f},"
                                f" loss_keep_damage≈{task_preds.get('v8_train_damage_fused_keep', float('nan')):.3f}"
                            )
                            print(
                                "  V8 Damage:"
                                f" bev≈{task_preds.get('v8_bev_damage_mean', float('nan')):.3f},"
                                f" img≈{task_preds.get('v8_img_damage_mean', float('nan')):.3f},"
                                f" dual≈{task_preds.get('v8_dual_damage_mean', float('nan')):.3f},"
                                f" nf_bev={v8_non_fused.get('bev', float('nan')):.3f},"
                                f" nf_img={v8_non_fused.get('img', float('nan')):.3f}"
                            )
                            print(
                                "  V8 State:"
                                f" clean_like={v8_state.get('clean_like', float('nan')):.3f},"
                                f" bev_damage={v8_state.get('bev_damage', float('nan')):.3f},"
                                f" img_damage={v8_state.get('img_damage', float('nan')):.3f},"
                                f" dual={v8_state.get('dual_damage', float('nan')):.3f},"
                                f" sel_fused={v8_selected.get('fused', float('nan')):.3f},"
                                f" sel_bev={v8_selected.get('bev', float('nan')):.3f},"
                                f" sel_img={v8_selected.get('img', float('nan')):.3f}"
                            )
                        if 'selector_teacher_top1_ratio' in task_preds:
                            teacher_ratio = task_preds['selector_teacher_top1_ratio']
                            pred_ratio = task_preds.get('selector_pred_top1_ratio', top1_ratio)
                            agreement_ratio = task_preds.get('selector_agreement_ratio', float('nan'))
                            print(
                                "  Selector Prior:"
                                f" fused={quality_top1_ratio.get('fused', float('nan')):.3f},"
                                f" bev={quality_top1_ratio.get('bev', float('nan')):.3f},"
                                f" img={quality_top1_ratio.get('img', float('nan')):.3f}"
                            )
                            print(
                                "  Selector Teacher:"
                                f" fused={teacher_ratio.get('fused', float('nan')):.3f},"
                                f" bev={teacher_ratio.get('bev', float('nan')):.3f},"
                                f" img={teacher_ratio.get('img', float('nan')):.3f},"
                                f" agree={agreement_ratio:.3f}"
                            )
                            print(
                                "  Selector Pred:"
                                f" fused={pred_ratio.get('fused', float('nan')):.3f},"
                                f" bev={pred_ratio.get('bev', float('nan')):.3f},"
                                f" img={pred_ratio.get('img', float('nan')):.3f}"
                            )
                            print(
                                "  Residual Router:"
                                f" alpha≈{alpha_res_mean:.3f},"
                                f" alpha_hard≈{alpha_res_hard_mean:.3f},"
                                f" override≈{override_ratio:.3f}"
                            )
                        if 'v5_switch_ratio' in task_preds:
                            v5_candidate_ratio = task_preds.get('v5_distance_prior_candidate_ratio', {})
                            v5_switched_ratio = task_preds.get('v5_distance_prior_switched_ratio', {})
                            v5_head_shift = task_preds.get('v5_head_shift', {})
                            print(
                                "  V5 Gate:"
                                f" switch={task_preds.get('v5_switch_ratio', float('nan')):.3f},"
                                f" keep_fused={task_preds.get('v5_keep_fused_ratio', float('nan')):.3f},"
                                f" to_bev={task_preds.get('v5_switch_to_bev_ratio', float('nan')):.3f},"
                                f" to_img={task_preds.get('v5_switch_to_img_ratio', float('nan')):.3f}"
                            )
                            print(
                                "  V5 Anomaly:"
                                f" cls≈{task_preds.get('v5_fused_cls_anom_mean', float('nan')):.3f},"
                                f" center≈{task_preds.get('v5_fused_center_anom_mean', float('nan')):.3f},"
                                f" height≈{task_preds.get('v5_fused_height_anom_mean', float('nan')):.3f},"
                                f" combined≈{task_preds.get('v5_fused_combined_anom_mean', float('nan')):.3f},"
                                f" trigger≈{task_preds.get('v5_trigger_ratio_over_threshold', float('nan')):.3f}"
                            )
                            print(
                                "  V5 Reference:"
                                f" bin_cov≈{task_preds.get('v5_bin_coverage_ratio', float('nan')):.3f},"
                                f" fallback_cls≈{task_preds.get('v5_fallback_class_ratio', float('nan')):.3f},"
                                f" fallback_global≈{task_preds.get('v5_fallback_global_ratio', float('nan')):.3f}"
                            )
                            print(
                                "  V5 Selection:"
                                f" cand_bev={v5_candidate_ratio.get('bev', float('nan')):.3f},"
                                f" cand_img={v5_candidate_ratio.get('img', float('nan')):.3f},"
                                f" switched_bev={v5_switched_ratio.get('bev', float('nan')):.3f},"
                                f" switched_img={v5_switched_ratio.get('img', float('nan')):.3f},"
                                f" agree≈{task_preds.get('v5_switch_agreement_with_prior', float('nan')):.3f}"
                            )
                            print(
                                "  V5 Head Shift:"
                                f" cls≈{v5_head_shift.get('cls', float('nan')):.5f},"
                                f" center≈{v5_head_shift.get('center', float('nan')):.5f},"
                                f" height≈{v5_head_shift.get('height', float('nan')):.5f}"
                            )
                            print(
                                "  Second Opinion:"
                                f" ratio≈{second_opinion_ratio:.3f},"
                                f" prior_top2≈{prior_top2_match_ratio:.3f},"
                                f" teacher_top2≈{second_opinion_teacher_match_ratio:.3f},"
                                f" hard_entropy≈{hard_entropy:.3f}"
                            )
                            print(
                                "  Alpha Split:"
                                f" easy≈{alpha_res_easy_mean:.3f},"
                                f" hard≈{alpha_res_hard_mean:.3f}"
                            )
                        print(
                            "  Selector Unc:"
                            f" fused={per_mod_unc.get('fused', float('nan')):.4f},"
                            f" bev={per_mod_unc.get('bev', float('nan')):.4f},"
                            f" img={per_mod_unc.get('img', float('nan')):.4f}"
                        )
                        print(
                            "  Selector Tail:"
                            f" fused={tail_ratio.get('fused', float('nan')):.3f},"
                            f" bev={tail_ratio.get('bev', float('nan')):.3f},"
                            f" img={tail_ratio.get('img', float('nan')):.3f}"
                        )
                        if 'selector_relative_unc_summary' in task_preds:
                            rel_unc = task_preds['selector_relative_unc_summary']
                            print(
                                "  Selector RelUnc:"
                                f" fused={rel_unc.get('fused', float('nan')):.4f},"
                                f" bev={rel_unc.get('bev', float('nan')):.4f},"
                                f" img={rel_unc.get('img', float('nan')):.4f}"
                            )
                        if score_components:
                            print(
                                "  Selector Score:"
                                f" fused={score_components.get('fused', {}).get('final_score', float('nan')):.4f},"
                                f" bev={score_components.get('bev', {}).get('final_score', float('nan')):.4f},"
                                f" img={score_components.get('img', {}).get('final_score', float('nan')):.4f}"
                            )

                        if 'selector_teacher_cls_logits' in task_preds:
                            final_cls = task_preds['cls_logits'][0][-1]
                            teacher_cls = task_preds['selector_teacher_cls_logits'][0][-1]
                            final_center = task_preds['center'][0][-1]
                            teacher_center = task_preds['selector_teacher_center'][0][-1]
                            final_height = task_preds['height'][0][-1]
                            teacher_height = task_preds['selector_teacher_height'][0][-1]

                            cls_shift = (final_cls - teacher_cls).abs().mean().item()
                            center_shift = (final_center - teacher_center).abs().mean().item()
                            height_shift = (final_height - teacher_height).abs().mean().item()
                            print(
                                "  Selector Shift:"
                                f" cls_l1={cls_shift:.5f},"
                                f" center_l1={center_shift:.5f},"
                                f" height_l1={height_shift:.5f}\n"
                            )
                        break

        return loss_dict

    @staticmethod
    def _branch_candidate_gather(value, gather_idx):
        if not isinstance(value, torch.Tensor):
            return None
        if value.dim() == 2:
            return value.gather(1, gather_idx)
        if value.dim() == 3:
            idx = gather_idx.unsqueeze(-1).expand(-1, -1, value.shape[-1])
            return value.gather(1, idx)
        return None

    def _merge_branch_candidate_pme_info(self, per_task_pme_info):
        """Merge task-local branch candidates while preserving task identity."""
        candidate_infos = [
            item for item in per_task_pme_info
            if isinstance(item, dict) and isinstance(item.get('branch_candidate_score'), torch.Tensor)
        ]
        if not candidate_infos:
            return {}

        tensor_keys = set()
        scalar_values = {}
        total_available = 0.0
        for info in candidate_infos:
            for key, value in info.items():
                if not (isinstance(key, str) and key.startswith('branch_candidate_')):
                    continue
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    tensor_keys.add(key)
                elif key not in scalar_values:
                    scalar_values[key] = value
            try:
                total_available += float(info.get('branch_candidate_total_available', 0.0))
            except Exception:
                pass

        merged = {}
        for key in sorted(tensor_keys):
            values = []
            ref_tail = None
            for info in candidate_infos:
                value = info.get(key)
                if not isinstance(value, torch.Tensor) or value.dim() < 2:
                    continue
                tail = tuple(value.shape[2:])
                if ref_tail is None:
                    ref_tail = tail
                if tail != ref_tail:
                    continue
                values.append(value)
            if values:
                merged[key] = torch.cat(values, dim=1)

        score = merged.get('branch_candidate_score')
        branch_idx = merged.get('branch_candidate_branch_idx')
        if not isinstance(score, torch.Tensor) or not isinstance(branch_idx, torch.Tensor):
            return merged

        cap = scalar_values.get('branch_candidate_limit_per_branch')
        if cap is None:
            cap = scalar_values.get('branch_candidate_limit')
        try:
            cap = int(cap)
        except Exception:
            cap = int(score.shape[1])
        cap = max(1, min(cap, int(score.shape[1])))

        branch_ids = sorted(
            int(item) for item in branch_idx.detach().cpu().unique().tolist()
            if int(item) >= 0)
        selected = []
        branch_ranks = []
        for branch_id in branch_ids:
            branch_mask = branch_idx == int(branch_id)
            branch_score = torch.where(
                branch_mask, score, score.new_full(score.shape, -1.0))
            k = min(cap, int(branch_score.shape[1]))
            _, idx = branch_score.topk(k, dim=1)
            selected.append(idx)
            rank = torch.arange(
                k, device=score.device, dtype=torch.long).view(1, k).expand(score.shape[0], -1)
            branch_ranks.append(rank)

        if not selected:
            return merged
        gather_idx = torch.cat(selected, dim=1)
        out = {}
        for key, value in merged.items():
            gathered = self._branch_candidate_gather(value, gather_idx)
            if gathered is not None:
                out[key] = gathered

        total_logged = int(gather_idx.shape[1])
        logging_rank = torch.arange(
            total_logged, device=score.device, dtype=torch.long).view(1, total_logged).expand(score.shape[0], -1)
        out['branch_candidate_rank'] = logging_rank
        out['branch_candidate_logging_rank'] = logging_rank
        out['branch_candidate_branch_rank'] = torch.cat(branch_ranks, dim=1)
        out['branch_candidate_schema_version'] = 2.0
        out['branch_candidate_limit'] = float(total_logged)
        out['branch_candidate_limit_per_branch'] = float(cap)
        out['branch_candidate_total_available'] = float(total_available)
        if 'branch_candidate_score_threshold' in scalar_values:
            out['branch_candidate_score_threshold'] = scalar_values['branch_candidate_score_threshold']
        return out

    def _merge_query_semantic_pme_info(self, per_task_pme_info):
        """Merge task-local query semantic diagnostics while preserving queries."""
        semantic_infos = [
            item for item in per_task_pme_info
            if isinstance(item, dict) and isinstance(item.get('qsem_img_score'), torch.Tensor)
        ]
        if not semantic_infos:
            return {}

        tensor_keys = set()
        scalar_values = {}
        for info in semantic_infos:
            for key, value in info.items():
                if not (isinstance(key, str) and key.startswith('qsem_')):
                    continue
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    tensor_keys.add(key)
                else:
                    try:
                        scalar_values.setdefault(key, []).append(float(value))
                    except Exception:
                        pass

        merged = {}
        for key in sorted(tensor_keys):
            values = []
            ref_tail = None
            for info in semantic_infos:
                value = info.get(key)
                if not isinstance(value, torch.Tensor) or value.dim() < 2:
                    continue
                tail = tuple(value.shape[2:])
                if ref_tail is None:
                    ref_tail = tail
                if tail != ref_tail:
                    continue
                values.append(value)
            if values:
                merged[key] = torch.cat(values, dim=1)
        for key, values in scalar_values.items():
            if values:
                merged[key] = float(sum(values) / max(len(values), 1))
        return merged

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """
        EDL-aware bbox decoding with structured uncertainty extraction.
        Returns [bboxes, scores, labels, edl_info] per sample.
        
        edl_info keys (always present):
            - center_unc: [nq, 1] center epistemic uncertainty per box
            - height_unc: [nq, 1] height epistemic uncertainty per box
            - cls_unc: [nq, 1] classification uncertainty per box
            - combined_unc: [nq, 1] averaged uncertainty per box
        edl_info keys (PME only, Stage 2):
            - per_modality_unc: dict {modality_name: float}
            - modality_weights: dict {modality_name: float}
            - selected_modality: str
            - fused_weight: float
            - selector_entropy: float
            - quality_top1_ratio: dict {modality_name: float}
            - distance_prior_top1_ratio: dict {modality_name: float}
            - residual_override_ratio: float
            - alpha_res_mean: float
            - second_opinion_ratio: float
            - second_opinion_teacher_match_ratio: float
            - uncertainty_tail_query_ratio: dict {modality_name: float}
            - absolute_unc_summary: dict {modality_name: float}
            - disagreement_summary: dict {modality_name: float}
            - score_component_stats: nested dict
            - distance_bin_abnormality_stats: nested dict
        """
        # -----------------------------------------------------------
        # 1. Extract the structured uncertainty (computed from the NIG parameters)
        # -----------------------------------------------------------
        # preds_dicts structure: tuple of (list of dicts per task)
        if isinstance(preds_dicts, tuple):
            preds_dicts_for_decode = preds_dicts[0]
        else:
            preds_dicts_for_decode = preds_dicts
        
        # Collect the uncertainty for each task
        per_task_unc = []
        per_task_pme_info = []
        
        for task_preds in preds_dicts_for_decode:
            if isinstance(task_preds, list):
                task_preds = task_preds[0]
            
            task_edl = {}
            
            # --- Regression Uncertainty (center, height) ---
            for nig_key, unc_name in [('center_nig', 'center_unc'), 
                                       ('height_nig', 'height_unc')]:
                if nig_key in task_preds:
                    nig_tensor = task_preds[nig_key]
                    # Take the last decoder layer and the first modality
                    if isinstance(nig_tensor, list):
                        nig_tensor = nig_tensor[0]
                    if nig_tensor.dim() == 4:  # [nb_dec, bs, nq, dim*4]
                        nig_tensor = nig_tensor[-1]  # last layer
                    
                    dim = nig_tensor.shape[-1] // 4
                    if dim > 0:
                        nig_reshaped = nig_tensor.view(*nig_tensor.shape[:-1], dim, 4)
                        # [FIX] activate the raw NIG parameters with softplus
                        v = F.softplus(nig_reshaped[..., 1]).clamp(min=1e-6)
                        alpha = (F.softplus(nig_reshaped[..., 2]) + 1.0).clamp(min=1.0 + 1e-6)
                        beta = F.softplus(nig_reshaped[..., 3]).clamp(min=1e-6)
                        
                        epistemic = beta / (v * (alpha - 1.0).clamp(min=1e-6))
                        task_edl[unc_name] = epistemic.mean(dim=-1, keepdim=True)  # [bs, nq, 1]
            
            # --- Classification Uncertainty ---
            if 'cls_alpha' in task_preds and 'cls_beta' in task_preds:
                cls_alpha = task_preds['cls_alpha']
                cls_beta = task_preds['cls_beta']
                if isinstance(cls_alpha, list):
                    cls_alpha = cls_alpha[0]
                if isinstance(cls_beta, list):
                    cls_beta = cls_beta[0]
                if cls_alpha.dim() == 4:
                    cls_alpha = cls_alpha[-1]
                    cls_beta = cls_beta[-1]
                # Total uncertainty for Beta distribution = 1 / (alpha + beta)
                # Average across classes for per-box scalar
                cls_unc = (1.0 / (cls_alpha + cls_beta)).mean(dim=-1, keepdim=True)
                task_edl['cls_unc'] = cls_unc  # [bs, nq, 1]
            
            # --- Combined Uncertainty ---
            all_uncs = [v for k, v in task_edl.items() if k.endswith('_unc')]
            if len(all_uncs) > 0:
                task_edl['combined_unc'] = torch.cat(all_uncs, dim=-1).mean(dim=-1, keepdim=True)
            
            per_task_unc.append(task_edl)
            
            # --- PME-specific info (Stage 2 only) ---
            pme_info = {}
            if 'per_modality_unc' in task_preds:
                pme_info['per_modality_unc'] = task_preds['per_modality_unc']
            if 'modality_weights' in task_preds:
                pme_info['modality_weights'] = task_preds['modality_weights']
            if 'selected_modality' in task_preds:
                pme_info['selected_modality'] = task_preds['selected_modality']
            if 'fused_weight' in task_preds:
                pme_info['fused_weight'] = task_preds['fused_weight']
            if 'selector_entropy' in task_preds:
                pme_info['selector_entropy'] = task_preds['selector_entropy']
            if 'quality_top1_ratio' in task_preds:
                pme_info['quality_top1_ratio'] = task_preds['quality_top1_ratio']
            if 'distance_prior_top1_ratio' in task_preds:
                pme_info['distance_prior_top1_ratio'] = task_preds['distance_prior_top1_ratio']
            if 'residual_override_ratio' in task_preds:
                pme_info['residual_override_ratio'] = task_preds['residual_override_ratio']
            if 'alpha_res_mean' in task_preds:
                pme_info['alpha_res_mean'] = task_preds['alpha_res_mean']
            if 'selector_second_opinion_ratio' in task_preds:
                pme_info['second_opinion_ratio'] = task_preds['selector_second_opinion_ratio']
            if 'selector_second_opinion_teacher_match_ratio' in task_preds:
                pme_info['second_opinion_teacher_match_ratio'] = (
                    task_preds['selector_second_opinion_teacher_match_ratio']
                )
            if 'uncertainty_tail_query_ratio' in task_preds:
                pme_info['uncertainty_tail_query_ratio'] = task_preds['uncertainty_tail_query_ratio']
            if 'absolute_unc_summary' in task_preds:
                pme_info['absolute_unc_summary'] = task_preds['absolute_unc_summary']
            if 'disagreement_summary' in task_preds:
                pme_info['disagreement_summary'] = task_preds['disagreement_summary']
            if 'score_component_stats' in task_preds:
                pme_info['score_component_stats'] = task_preds['score_component_stats']
            if 'distance_bin_abnormality_stats' in task_preds:
                pme_info['distance_bin_abnormality_stats'] = task_preds['distance_bin_abnormality_stats']
            for key in [
                'v5_switch_ratio',
                'v5_keep_fused_ratio',
                'v5_switch_to_bev_ratio',
                'v5_switch_to_img_ratio',
                'v5_fused_cls_anom_mean',
                'v5_fused_center_anom_mean',
                'v5_fused_height_anom_mean',
                'v5_fused_combined_anom_mean',
                'v5_trigger_ratio_over_threshold',
                'v5_bin_coverage_ratio',
                'v5_fallback_class_ratio',
                'v5_fallback_global_ratio',
                'v5_distance_prior_candidate_ratio',
                'v5_distance_prior_switched_ratio',
                'v5_switch_agreement_with_prior',
                'v5_head_shift',
                'v5_query_switch_triggered',
                'v5_query_selected_modality_idx',
                'v5_query_candidate_modality_idx',
                'v5_query_fused_cls_anom',
                'v5_query_fused_center_anom',
                'v5_query_fused_height_anom',
                'v5_query_fused_anomaly_score',
                'v5_query_reference_distance_bin',
                'v5_query_reference_fallback_level',
                'v5_query_pred_class',
                'v5_fused_cls_unc',
                'v5_fused_center_unc',
                'v5_fused_height_unc',
                'v6_switch_ratio',
                'v6_gate_anomaly_mean',
                'v6_gate_pass_anomaly_ratio',
                'v6_gate_pass_candidate_prob_ratio',
                'v6_gate_pass_candidate_margin_ratio',
                'v6_gate_pass_final_ratio',
                'v6_candidate_modality_ratio',
                'v6_center_source_ratio',
                'v6_height_source_ratio',
                'v6_reference_fallback_ratio',
                'v6_switch_triggered',
                'v6_candidate_modality_idx',
                'v6_gate_anomaly_score',
                'v6_gate_passed_confidence',
                'v6_gate_passed_margin',
                'v6_gate_pass_anomaly',
                'v6_gate_pass_candidate_prob',
                'v6_gate_pass_candidate_margin',
                'v6_gate_pass_final',
                'v6_reference_distance_bin',
                'v6_reference_fallback_level',
                'v6_reference_pred_class',
                'v6_baseline_selected_modality_idx',
                'v6_center_source_idx',
                'v6_height_source_idx',
                'v6_cls_source_idx',
                'v6_non_fused_selector_probs',
                'v7_switch_ratio',
                'v7_damage_state_ratio',
                'v7_center_source_ratio',
                'v7_bev_center_anom_mean',
                'v7_img_center_anom_mean',
                'v7_fused_center_anom_mean',
                'v7_switch_triggered',
                'v7_damage_state_idx',
                'v7_center_source_idx',
                'v7_bev_center_anom',
                'v7_img_center_anom',
                'v7_fused_center_anom',
                'v7_img_fallback_prob',
                'v7_img_fallback_margin',
                'v7_bev_fallback_prob',
                'v7_bev_fallback_margin',
                'v7_reference_distance_bin',
                'v7_pred_class',
                'v7_query_bev_reference_distance_bin',
                'v7_query_img_reference_distance_bin',
                'v7_query_fused_reference_distance_bin',
                'v7_query_bev_reference_fallback_level',
                'v7_query_img_reference_fallback_level',
                'v7_query_fused_reference_fallback_level',
                'v7_ref_fused_center_unc',
                'v7_ref_bev_center_unc',
                'v7_ref_img_center_unc',
                'v7_ref_fused_distance',
                'v7_ref_bev_distance',
                'v7_ref_img_distance',
                'v7_ref_pred_class',
                'v8_fused_keep_mean',
                'v8_center_fused_keep_mean',
                'v8_damage_gate_mean',
                'v8_non_fused_weight_ratio',
                'v8_soft_weight_ratio',
                'v8_damage_state_ratio',
                'v8_reliability_state_ratio',
                'v8_selected_source_ratio',
                'v8_high_damage_query_ratio',
                'v8_center_route_allowed_ratio',
                'v8_bev_damage_mean',
                'v8_img_damage_mean',
                'v8_dual_damage_mean',
                'v8_context_bev_damage_mean',
                'v8_context_img_damage_mean',
                'v8_a4_anchor_ready',
                'v8_a4_bev_damage_z_mean',
                'v8_a4_img_damage_z_mean',
                'v8_a4_cls_unc_z_bev_mean',
                'v8_a4_cls_unc_z_img_mean',
                'v8_a4_route_score_bev_mean',
                'v8_a4_route_score_img_mean',
                'v8_a4_clean_anchor_update_count',
                'v8_bev_reliability_mean',
                'v8_img_reliability_mean',
                'v8_reliability_margin_mean',
                'v8_fused_keep_by_state',
                'v8_non_fused_entropy_mean',
                'v8_output_delta_cls_mean',
                'v8_output_delta_center_mean',
                'v8_output_delta_height_mean',
                'v8_router_variant',
                'v8_fused_keep',
                'v8_center_fused_keep',
                'v8_damage_gate',
                'v8_bev_damage',
                'v8_img_damage',
                'v8_dual_damage',
                'v8_context_bev_damage',
                'v8_context_img_damage',
                'v8_a4_bev_damage_z',
                'v8_a4_img_damage_z',
                'v8_a4_cls_unc_z_bev',
                'v8_a4_cls_unc_z_img',
                'v8_a4_route_score_bev',
                'v8_a4_route_score_img',
                'v8_gscrg_raw_unc_fused',
                'v8_gscrg_raw_unc_bev',
                'v8_gscrg_raw_unc_img',
                'v8_bev_reliability',
                'v8_img_reliability',
                'v8_reliability_margin',
                'v8_non_fused_probs',
                'v8_non_fused_choice',
                'v8_center_source',
                'v8_gate_reason',
                'v8_damage_state',
                'v8_reliability_state',
                'v8_high_damage_query',
                'v8_center_route_allowed',
                'v8_soft_weight_fused',
                'v8_soft_weight_bev',
                'v8_soft_weight_img',
                'v8_output_delta_cls',
                'v8_output_delta_center',
                'v8_output_delta_height',
            ]:
                if key in task_preds:
                    pme_info[key] = task_preds[key]
            for key, value in task_preds.items():
                if key in (
                        'v10_edlcue_admission_logits',
                        'v10_edlcue_delta',
                        'v10_edlcue_trainable_delta',
                        'v10_edlcue_manual_delta'):
                    continue
                if (
                    isinstance(key, str) and
                    (key.startswith('v9_') or key.startswith('v10_') or
                     key.startswith('supp_') or
                     key.startswith('qsem_') or
                     key.startswith('branch_candidate_') or
                     key.startswith('loss_v10_') or
                     key in ('force_modality_active', 'forced_branch'))
                ):
                    pme_info[key] = value
            per_task_pme_info.append(pme_info)
        
        merged_branch_candidate_info = self._merge_branch_candidate_pme_info(per_task_pme_info)
        merged_query_semantic_info = self._merge_query_semantic_pme_info(per_task_pme_info)

        # -----------------------------------------------------------
        # 2. Standard decoding (absolute coordinates, identical to upstream)
        # -----------------------------------------------------------
        decoded_results = self.bbox_coder.decode(preds_dicts)
        
        # -----------------------------------------------------------
        # 3. Post-processing and uncertainty injection
        # -----------------------------------------------------------
        ret_list = []
        num_samples = len(decoded_results)
        
        for i in range(num_samples):
            preds = decoded_results[i]
            
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            
            # z-axis adjustment (as upstream)
            if bboxes.shape[-1] >= 6 and bboxes.shape[0] > 0:
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] *  0.5
            
            # --- Build the per-sample edl_info ---
            edl_info = {}
            if 'raw_scores' in preds:
                edl_info['v10_edlcue_raw_scores'] = preds['raw_scores'].unsqueeze(-1)
            if 'admission_scores' in preds:
                edl_info['v10_edlcue_admission_scores'] = preds['admission_scores'].unsqueeze(-1)
            for src_key, dst_key in (
                    ('decode_query_idx', 'final_decode_query_idx'),
                    ('decode_task_idx', 'final_decode_task_idx'),
                    ('decode_final_bbox_idx', 'final_decode_bbox_idx'),
                    ('decode_topk_rank', 'final_decode_topk_rank')):
                if src_key in preds and isinstance(preds[src_key], torch.Tensor):
                    value = preds[src_key].view(-1, 1)
                    if value.shape[0] == bboxes.shape[0]:
                        edl_info[dst_key] = value

            def _gather_final_query_telemetry(src_key):
                """Gather per-query telemetry onto decoded prediction order."""
                task_idx = preds.get('decode_task_idx')
                query_idx = preds.get('decode_query_idx')
                if (
                        not isinstance(task_idx, torch.Tensor) or
                        not isinstance(query_idx, torch.Tensor) or
                        task_idx.numel() != bboxes.shape[0] or
                        query_idx.numel() != bboxes.shape[0]):
                    return None
                task_idx = task_idx.long().view(-1)
                query_idx = query_idx.long().view(-1)
                task_values = []
                for task_pme in per_task_pme_info:
                    if not isinstance(task_pme, dict) or src_key not in task_pme:
                        return None
                    value = task_pme[src_key]
                    if not isinstance(value, torch.Tensor):
                        return None
                    if value.dim() < 2 or value.shape[0] <= i:
                        return None
                    task_values.append(value[i])
                if not task_values:
                    return None
                base_shape = tuple(task_values[0].shape)
                if any(tuple(value.shape) != base_shape for value in task_values):
                    return None
                stacked = torch.stack(task_values, dim=0)
                if stacked.dim() < 2:
                    return None
                valid = (
                    (task_idx >= 0) & (task_idx < stacked.shape[0]) &
                    (query_idx >= 0) & (query_idx < stacked.shape[1])
                )
                out_shape = (bboxes.shape[0],) + tuple(stacked.shape[2:])
                gathered = torch.zeros(
                    out_shape, device=stacked.device, dtype=stacked.dtype)
                if bool(valid.any().item()):
                    gathered[valid] = stacked[task_idx[valid], query_idx[valid]]
                return gathered
            
            # Combine the uncertainty over all tasks by averaging
            for unc_key in ['center_unc', 'height_unc', 'cls_unc', 'combined_unc']:
                task_vals = []
                for task_edl in per_task_unc:
                    if unc_key in task_edl and i < task_edl[unc_key].shape[0]:
                        task_vals.append(task_edl[unc_key][i])  # [nq, 1]
                if len(task_vals) > 0:
                    # Average across tasks
                    merged = torch.stack(task_vals, dim=0).mean(dim=0)  # [nq, 1]
                    # Align with the number of decoded boxes
                    if merged.shape[0] > bboxes.shape[0]:
                        merged = merged[:bboxes.shape[0]]
                    elif merged.shape[0] < bboxes.shape[0]:
                        pad = torch.zeros(bboxes.shape[0] - merged.shape[0], 1, 
                                          device=merged.device)
                        merged = torch.cat([merged, pad], dim=0)
                    edl_info[unc_key] = merged
            
            # PME side information (stage 2 only), averaged across tasks
            for pme_info in per_task_pme_info:
                if 'per_modality_unc' in pme_info and 'per_modality_unc' not in edl_info:
                    edl_info['per_modality_unc'] = pme_info['per_modality_unc']
                if 'modality_weights' in pme_info and 'modality_weights' not in edl_info:
                    edl_info['modality_weights'] = pme_info['modality_weights']
                if 'selected_modality' in pme_info and 'selected_modality' not in edl_info:
                    edl_info['selected_modality'] = pme_info['selected_modality']
                if 'fused_weight' in pme_info and 'fused_weight' not in edl_info:
                    edl_info['fused_weight'] = pme_info['fused_weight']
                if 'selector_entropy' in pme_info and 'selector_entropy' not in edl_info:
                    edl_info['selector_entropy'] = pme_info['selector_entropy']
                if 'quality_top1_ratio' in pme_info and 'quality_top1_ratio' not in edl_info:
                    edl_info['quality_top1_ratio'] = pme_info['quality_top1_ratio']
                if 'distance_prior_top1_ratio' in pme_info and 'distance_prior_top1_ratio' not in edl_info:
                    edl_info['distance_prior_top1_ratio'] = pme_info['distance_prior_top1_ratio']
                if 'residual_override_ratio' in pme_info and 'residual_override_ratio' not in edl_info:
                    edl_info['residual_override_ratio'] = pme_info['residual_override_ratio']
                if 'alpha_res_mean' in pme_info and 'alpha_res_mean' not in edl_info:
                    edl_info['alpha_res_mean'] = pme_info['alpha_res_mean']
                if 'second_opinion_ratio' in pme_info and 'second_opinion_ratio' not in edl_info:
                    edl_info['second_opinion_ratio'] = pme_info['second_opinion_ratio']
                if (
                    'second_opinion_teacher_match_ratio' in pme_info and
                    'second_opinion_teacher_match_ratio' not in edl_info
                ):
                    edl_info['second_opinion_teacher_match_ratio'] = (
                        pme_info['second_opinion_teacher_match_ratio']
                    )
                if 'uncertainty_tail_query_ratio' in pme_info and 'uncertainty_tail_query_ratio' not in edl_info:
                    edl_info['uncertainty_tail_query_ratio'] = pme_info['uncertainty_tail_query_ratio']
                if 'absolute_unc_summary' in pme_info and 'absolute_unc_summary' not in edl_info:
                    edl_info['absolute_unc_summary'] = pme_info['absolute_unc_summary']
                if 'disagreement_summary' in pme_info and 'disagreement_summary' not in edl_info:
                    edl_info['disagreement_summary'] = pme_info['disagreement_summary']
                if 'score_component_stats' in pme_info and 'score_component_stats' not in edl_info:
                    edl_info['score_component_stats'] = pme_info['score_component_stats']
                if 'distance_bin_abnormality_stats' in pme_info and 'distance_bin_abnormality_stats' not in edl_info:
                    edl_info['distance_bin_abnormality_stats'] = pme_info['distance_bin_abnormality_stats']
                for key in [
                    'v6_switch_ratio',
                    'v6_gate_anomaly_mean',
                    'v6_gate_pass_anomaly_ratio',
                    'v6_gate_pass_candidate_prob_ratio',
                    'v6_gate_pass_candidate_margin_ratio',
                    'v6_gate_pass_final_ratio',
                    'v6_candidate_modality_ratio',
                    'v6_center_source_ratio',
                    'v6_height_source_ratio',
                    'v6_reference_fallback_ratio',
                    'v6_switch_triggered',
                    'v6_candidate_modality_idx',
                    'v6_gate_anomaly_score',
                    'v6_gate_passed_confidence',
                    'v6_gate_passed_margin',
                    'v6_gate_pass_anomaly',
                    'v6_gate_pass_candidate_prob',
                    'v6_gate_pass_candidate_margin',
                    'v6_gate_pass_final',
                    'v6_reference_distance_bin',
                    'v6_reference_fallback_level',
                    'v6_reference_pred_class',
                    'v6_baseline_selected_modality_idx',
                    'v6_center_source_idx',
                    'v6_height_source_idx',
                    'v6_cls_source_idx',
                    'v6_non_fused_selector_probs',
                    'v7_switch_ratio',
                    'v7_damage_state_ratio',
                    'v7_center_source_ratio',
                    'v7_bev_center_anom_mean',
                    'v7_img_center_anom_mean',
                    'v7_fused_center_anom_mean',
                    'v7_switch_triggered',
                    'v7_damage_state_idx',
                    'v7_center_source_idx',
                    'v7_bev_center_anom',
                    'v7_img_center_anom',
                    'v7_fused_center_anom',
                    'v7_img_fallback_prob',
                    'v7_img_fallback_margin',
                    'v7_bev_fallback_prob',
                    'v7_bev_fallback_margin',
                    'v7_reference_distance_bin',
                    'v7_pred_class',
                    'v7_query_bev_reference_distance_bin',
                    'v7_query_img_reference_distance_bin',
                    'v7_query_fused_reference_distance_bin',
                    'v7_query_bev_reference_fallback_level',
                    'v7_query_img_reference_fallback_level',
                    'v7_query_fused_reference_fallback_level',
                    'v7_ref_fused_center_unc',
                    'v7_ref_bev_center_unc',
                    'v7_ref_img_center_unc',
	                    'v7_ref_fused_distance',
	                    'v7_ref_bev_distance',
		                    'v7_ref_img_distance',
		                    'v7_ref_pred_class',
		                    'v8_fused_keep_mean',
		                    'v8_center_fused_keep_mean',
		                    'v8_damage_gate_mean',
		                    'v8_non_fused_weight_ratio',
	                    'v8_soft_weight_ratio',
	                    'v8_damage_state_ratio',
	                    'v8_reliability_state_ratio',
		                    'v8_selected_source_ratio',
		                    'v8_high_damage_query_ratio',
		                    'v8_center_route_allowed_ratio',
		                    'v8_bev_damage_mean',
		                    'v8_img_damage_mean',
		                    'v8_dual_damage_mean',
		                    'v8_context_bev_damage_mean',
		                    'v8_context_img_damage_mean',
		                    'v8_a4_anchor_ready',
		                    'v8_a4_bev_damage_z_mean',
		                    'v8_a4_img_damage_z_mean',
		                    'v8_a4_cls_unc_z_bev_mean',
		                    'v8_a4_cls_unc_z_img_mean',
		                    'v8_a4_route_score_bev_mean',
		                    'v8_a4_route_score_img_mean',
		                    'v8_a4_clean_anchor_update_count',
		                    'v8_bev_reliability_mean',
	                    'v8_img_reliability_mean',
	                    'v8_reliability_margin_mean',
	                    'v8_fused_keep_by_state',
	                    'v8_non_fused_entropy_mean',
	                    'v8_output_delta_cls_mean',
	                    'v8_output_delta_center_mean',
		                    'v8_output_delta_height_mean',
		                    'v8_router_variant',
		                    'v8_fused_keep',
		                    'v8_center_fused_keep',
		                    'v8_damage_gate',
		                    'v8_bev_damage',
		                    'v8_img_damage',
		                    'v8_dual_damage',
		                    'v8_context_bev_damage',
		                    'v8_context_img_damage',
		                    'v8_a4_bev_damage_z',
		                    'v8_a4_img_damage_z',
		                    'v8_a4_cls_unc_z_bev',
		                    'v8_a4_cls_unc_z_img',
		                    'v8_a4_route_score_bev',
		                    'v8_a4_route_score_img',
		                    'v8_gscrg_raw_unc_fused',
		                    'v8_gscrg_raw_unc_bev',
		                    'v8_gscrg_raw_unc_img',
		                    'v8_bev_reliability',
	                    'v8_img_reliability',
	                    'v8_reliability_margin',
	                    'v8_non_fused_probs',
	                    'v8_non_fused_choice',
	                    'v8_center_source',
	                    'v8_gate_reason',
		                    'v8_damage_state',
		                    'v8_reliability_state',
		                    'v8_high_damage_query',
		                    'v8_center_route_allowed',
		                    'v8_soft_weight_fused',
	                    'v8_soft_weight_bev',
	                    'v8_soft_weight_img',
	                    'v8_output_delta_cls',
	                    'v8_output_delta_center',
	                    'v8_output_delta_height',
	                    'v5_switch_ratio',
                    'v5_keep_fused_ratio',
                    'v5_switch_to_bev_ratio',
                    'v5_switch_to_img_ratio',
                    'v5_fused_cls_anom_mean',
                    'v5_fused_center_anom_mean',
                    'v5_fused_height_anom_mean',
                    'v5_fused_combined_anom_mean',
                    'v5_trigger_ratio_over_threshold',
                    'v5_bin_coverage_ratio',
                    'v5_fallback_class_ratio',
                    'v5_fallback_global_ratio',
                    'v5_distance_prior_candidate_ratio',
                    'v5_distance_prior_switched_ratio',
                    'v5_switch_agreement_with_prior',
                    'v5_head_shift',
                    'v5_query_switch_triggered',
                    'v5_query_selected_modality_idx',
                    'v5_query_candidate_modality_idx',
                    'v5_query_fused_cls_anom',
                    'v5_query_fused_center_anom',
                    'v5_query_fused_height_anom',
                    'v5_query_fused_anomaly_score',
                    'v5_query_reference_distance_bin',
                    'v5_query_reference_fallback_level',
                    'v5_query_pred_class',
                    'v5_fused_cls_unc',
                    'v5_fused_center_unc',
                    'v5_fused_height_unc',
                ]:
                    if key in pme_info and key not in edl_info:
                        edl_info[key] = pme_info[key]
                for key, value in pme_info.items():
                    if (
                            isinstance(key, str) and
                            (key.startswith('v9_') or key.startswith('v10_') or
                             key.startswith('supp_') or
                             key.startswith('qsem_') or
                             key.startswith('loss_v10_') or
                             key in ('force_modality_active', 'forced_branch')) and
                        key not in edl_info
                    ):
                        edl_info[key] = value

                for key, value in merged_branch_candidate_info.items():
                    edl_info[key] = value
                for key, value in merged_query_semantic_info.items():
                    edl_info[key] = value

            for key in [
                'v6_switch_triggered',
                'v6_candidate_modality_idx',
                'v6_gate_anomaly_score',
                'v6_gate_passed_confidence',
                'v6_gate_passed_margin',
                'v6_gate_pass_anomaly',
                'v6_gate_pass_candidate_prob',
                'v6_gate_pass_candidate_margin',
                'v6_gate_pass_final',
                'v6_reference_distance_bin',
                'v6_reference_fallback_level',
                'v6_reference_pred_class',
                'v6_baseline_selected_modality_idx',
                'v6_center_source_idx',
                'v6_height_source_idx',
                'v6_cls_source_idx',
                'v6_non_fused_selector_probs',
                'v7_switch_triggered',
                'v7_damage_state_idx',
                'v7_center_source_idx',
                'v7_bev_center_anom',
                'v7_img_center_anom',
                'v7_fused_center_anom',
                'v7_img_fallback_prob',
                'v7_img_fallback_margin',
                'v7_bev_fallback_prob',
                'v7_bev_fallback_margin',
                'v7_reference_distance_bin',
                'v7_pred_class',
                'v7_query_bev_reference_distance_bin',
                'v7_query_img_reference_distance_bin',
                'v7_query_fused_reference_distance_bin',
                'v7_query_bev_reference_fallback_level',
                'v7_query_img_reference_fallback_level',
                'v7_query_fused_reference_fallback_level',
                'v7_ref_fused_center_unc',
                'v7_ref_bev_center_unc',
                'v7_ref_img_center_unc',
	                'v7_ref_fused_distance',
	                'v7_ref_bev_distance',
	                'v7_ref_img_distance',
	                'v7_ref_pred_class',
	                'v8_fused_keep',
	                'v8_center_fused_keep',
	                'v8_damage_gate',
	                'v8_bev_damage',
	                'v8_img_damage',
	                'v8_dual_damage',
	                'v8_context_bev_damage',
	                'v8_context_img_damage',
	                'v8_a4_bev_damage_z',
	                'v8_a4_img_damage_z',
	                'v8_a4_cls_unc_z_bev',
	                'v8_a4_cls_unc_z_img',
	                'v8_a4_route_score_bev',
	                'v8_a4_route_score_img',
	                'v8_gscrg_raw_unc_fused',
	                'v8_gscrg_raw_unc_bev',
	                'v8_gscrg_raw_unc_img',
	                'v8_bev_reliability',
	                'v8_img_reliability',
	                'v8_reliability_margin',
	                'v8_non_fused_probs',
	                'v8_non_fused_choice',
	                'v8_center_source',
	                'v8_gate_reason',
	                'v8_damage_state',
	                'v8_reliability_state',
	                'v8_high_damage_query',
	                'v8_center_route_allowed',
	                'v8_soft_weight_fused',
	                'v8_soft_weight_bev',
	                'v8_soft_weight_img',
	                'v8_output_delta_cls',
	                'v8_output_delta_center',
	                'v8_output_delta_height',
	                'v5_query_switch_triggered',
                'v5_query_selected_modality_idx',
                'v5_query_candidate_modality_idx',
                'v5_query_fused_cls_anom',
                'v5_query_fused_center_anom',
                'v5_query_fused_height_anom',
                'v5_query_fused_anomaly_score',
                'v5_query_reference_distance_bin',
                'v5_query_reference_fallback_level',
                'v5_query_pred_class',
                'v5_fused_cls_unc',
                'v5_fused_center_unc',
                'v5_fused_height_unc',
            ]:
                if key not in edl_info or not isinstance(edl_info[key], torch.Tensor):
                    continue
                tensor_val = edl_info[key]
                if tensor_val.dim() >= 2 and tensor_val.shape[0] > i:
                    tensor_val = tensor_val[i]
                if tensor_val.shape[0] > bboxes.shape[0]:
                    tensor_val = tensor_val[:bboxes.shape[0]]
                elif tensor_val.shape[0] < bboxes.shape[0]:
                    pad_shape = (bboxes.shape[0] - tensor_val.shape[0],) + tuple(tensor_val.shape[1:])
                    pad = torch.zeros(*pad_shape, device=tensor_val.device, dtype=tensor_val.dtype)
                    tensor_val = torch.cat([tensor_val, pad], dim=0)
                edl_info[key] = tensor_val

            # Branch-candidate logs are query/action candidates, not final
            # decoded predictions. Slice the batch dimension only; do not align
            # to final bbox count or the bounded top-k candidate set will be
            # silently truncated.
            for key in list(edl_info.keys()):
                if not (isinstance(key, str) and key.startswith('branch_candidate_')):
                    continue
                tensor_val = edl_info[key]
                if (
                        isinstance(tensor_val, torch.Tensor) and
                        tensor_val.dim() >= 1 and
                        tensor_val.shape[0] == num_samples
                ):
                    edl_info[key] = tensor_val[i]

            # Query semantic diagnostics are query-level sidecars. Slice only
            # the batch dimension and do not align them to decoded bbox count.
            for key in list(edl_info.keys()):
                if not (isinstance(key, str) and key.startswith('qsem_')):
                    continue
                tensor_val = edl_info[key]
                if (
                    isinstance(tensor_val, torch.Tensor) and
                    tensor_val.dim() >= 1 and
                    tensor_val.shape[0] == num_samples
                ):
                    edl_info[key] = tensor_val[i]

            # V9/V10/SUP tensors are copied dynamically to keep the logger schema extensible.
            # Align batch-level [bs, nq, ...] tensors to the current sample before
            # per-prediction sidecar logging consumes them.
            for key in list(edl_info.keys()):
                if not (
                    isinstance(key, str) and
                    (key.startswith('v9_') or key.startswith('v10_') or
                     key.startswith('supp_'))
                ):
                    continue
                tensor_val = edl_info[key]
                if (
                    not isinstance(tensor_val, torch.Tensor) or
                    tensor_val.dim() < 2 or
                    tensor_val.shape[0] != num_samples
                ):
                    continue
                tensor_val = tensor_val[i]
                if tensor_val.shape[0] > bboxes.shape[0]:
                    tensor_val = tensor_val[:bboxes.shape[0]]
                elif tensor_val.shape[0] < bboxes.shape[0]:
                    pad_shape = (bboxes.shape[0] - tensor_val.shape[0],) + tuple(tensor_val.shape[1:])
                    pad = torch.zeros(*pad_shape, device=tensor_val.device, dtype=tensor_val.dtype)
                    tensor_val = torch.cat([tensor_val, pad], dim=0)
                edl_info[key] = tensor_val

            final_supp_added = _gather_final_query_telemetry('supp_added')
            if final_supp_added is not None:
                if final_supp_added.dim() == 1:
                    final_supp_added = final_supp_added.unsqueeze(-1)
                final_supp_added = final_supp_added.float()
                final_source_is_supplement = (final_supp_added > 0.5).float()
                final_source_branch = _gather_final_query_telemetry('supp_source_branch')
                if final_source_branch is None:
                    final_source_branch = torch.full_like(final_source_is_supplement, 2.0)
                if final_source_branch.dim() == 1:
                    final_source_branch = final_source_branch.unsqueeze(-1)
                final_source_branch = final_source_branch.float()
                final_source_branch = torch.where(
                    final_source_is_supplement > 0.5,
                    final_source_branch,
                    torch.zeros_like(final_source_branch))
                edl_info['final_source_alignment_valid'] = torch.ones_like(final_source_is_supplement)
                edl_info['final_source_is_supplement'] = final_source_is_supplement
                edl_info['final_source_branch'] = final_source_branch

                for src_key, dst_key in (
                        ('supp_img_score', 'final_supp_img_score'),
                        ('supp_bev_score', 'final_supp_bev_score'),
                        ('supp_branch_score', 'final_supp_branch_score'),
                        ('supp_fused_score', 'final_supp_fused_score'),
                        ('supp_utility_score', 'final_supp_utility_score'),
                        ('supp_fused_slot_utility_score', 'final_supp_fused_slot_utility_score'),
                        ('supp_marginal_utility_score', 'final_supp_marginal_utility_score'),
                        ('supp_query_selected_candidate_idx', 'final_supp_candidate_query_idx'),
                        ('supp_img_class', 'final_supp_img_class'),
                        ('supp_branch_class', 'final_supp_branch_class')):
                    gathered = _gather_final_query_telemetry(src_key)
                    if gathered is not None:
                        if gathered.dim() == 1:
                            gathered = gathered.unsqueeze(-1)
                        edl_info[dst_key] = gathered

            if 'v9_a_bev' in edl_info and 'v9_a_img' in edl_info:
                corruption_type, synthetic, severity = self._v8_meta_corruption_type(img_metas[i])
                domain = self._v8_corruption_domain(corruption_type)
                target_synthetic = synthetic or corruption_type != 'clean'
                bev_tgt, img_tgt, _, _ = self._v9_domain_targets(
                    corruption_type, domain, target_synthetic, severity, edl_info['v9_a_bev'])
                for modality, values, target_value in (
                        ('bev', edl_info['v9_a_bev'], bev_tgt),
                        ('img', edl_info['v9_a_img'], img_tgt)):
                    debug_key = self._v9_bce_debug_key(corruption_type, modality)
                    if debug_key is None or not isinstance(values, torch.Tensor) or values.numel() == 0:
                        continue
                    values = values.float().clamp(min=1e-6, max=1.0 - 1e-6)
                    target = values.new_full(values.shape, float(target_value))
                    edl_info[debug_key] = F.binary_cross_entropy(
                        values, target, reduction='mean').detach()
            
            # --- Inject combined_unc into the bboxes (backwards compatible) ---
            if 'combined_unc' in edl_info:
                combined_unc = edl_info['combined_unc']
                if combined_unc.dim() == 1:
                    combined_unc = combined_unc.unsqueeze(-1)
                bboxes = torch.cat([bboxes, combined_unc], dim=-1)
            
            # Wrap as a 3D box
            if 'box_type_3d' in img_metas[i]:
                box_type = img_metas[i]['box_type_3d']
                box_dim = bboxes.shape[-1] - 1 if bboxes.shape[-1] > 9 else bboxes.shape[-1]
                bboxes_final = box_type(bboxes[..., :box_dim], box_dim=box_dim)
                if bboxes.shape[-1] > box_dim:
                    bboxes_final.uncertainty = bboxes[..., box_dim:]
            else:
                bboxes_final = bboxes
            
            ret_list.append([bboxes_final, scores, labels, edl_info])
            
        return ret_list
