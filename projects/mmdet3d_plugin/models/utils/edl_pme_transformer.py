# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from PMETransformer for EDL (Evidential Deep Learning) Support
# ------------------------------------------------------------------------
import copy
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmdet3d.models import builder

from projects.mmdet3d_plugin.models.dense_heads.meformer_head import pos2embed


@TRANSFORMER.register_module()
class EDLPMETransformer(BaseModule):
    """EDL uncertainty-aware PME.

    Two operation modes are supported:
    - ``detector``: keep the original PME behavior and predict a new ensemble
      branch with task heads.
    - ``soft_selector``: learn query-level modality weights and softly combine
      stage-1 modality outputs without rebuilding a new detector head.
    - ``primary_router``: route primarily between non-fused modalities and use
      fused features only as a residual fallback branch.
    - ``hard_selective_switch``: keep fused as default and switch only anomalous
      queries to a single non-fused modality during inference.
    - ``gated_primary_router``: keep the soft-selector blended output as the
      default and only override selected heads for anomalous queries using a
      non-fused router between ``bev`` and ``img``.
    - ``modality_damage_aware_router``: keep the soft-selector blended output
      as the default and only override ``center`` using branch-local modality
      damage detection with dedicated fallback heads.
    - ``clean_reference_gate_router``: keep fused/soft output as the default
      and apply bounded image fallback when clean-reference branch statistics
      indicate BEV damage while image remains clean-like.
    - ``damage_gated_soft_router``: keep fused as a query-level gated default
      and let ``bev``/``img`` compete only inside the non-fused residual path.
    """

    SUPP_RUNTIME_FEATURES = {
        'score',
        'distance_to_ego',
        'center_unc',
        'height_unc',
        'cls_unc',
        'combined_unc',
        'branch_idx',
        'branch_rank',
        'cls_weight',
        'center_weight',
        'height_weight',
        'non_fused_prob',
        'fused_keep',
        'damage_gate',
        'bev_damage',
        'img_damage',
        'dual_damage',
        'branch_is_fused',
        'branch_is_bev',
        'branch_is_img',
        'v8_a4_route_score_bev',
        'v8_a4_route_score_img',
        'v8_bev_reliability',
        'v8_img_reliability',
        'v8_reliability_margin',
        'v9_a_bev',
        'v9_a_img',
        'v9_tail_prob_bev',
        'v9_tail_prob_img',
        'v9_route_score_bev',
        'v9_route_score_img',
    }

    def __init__(
            self,
            decoder=None,
            heads=None,
            separate_head=None,
            num_classes=None,
            use_unc_bias=True,
            use_value_scaling=True,
            use_dim_edl=False,
            pme_unc_sources=('center', 'height'),
            ensemble_mode='detector',
            fused_prior_bias=0.0,
            selector_score_mode='legacy',
            selector_unc_score_weight=1.0,
            selector_rel_unc_score_weight=0.5,
            selector_disagreement_score_weight=0.5,
            selector_context_score_weight=1.0,
            selector_distance_score_weight=1.0,
            selector_fused_score_bias=None,
            selector_log_score_components=False,
            selector_softmax_temperature=1.0,
            selector_apply_heads=None,
            selector_frozen_heads=None,
            selector_unc_transform='none',
            selector_unc_norm='none',
            selector_use_relative_unc=False,
            residual_floor=0.1,
            residual_usage_target=0.2,
            distance_bin_edges=(20.0, 40.0),
            v5_reference_stats_path='',
            v5_stats_binning='class_distance',
            v5_distance_bins=(20.0, 40.0),
            v5_anomaly_score_mode='percentile',
            v5_anomaly_threshold=1.0,
            v5_anomaly_component_weights=None,
            v5_switch_heads=('cls', 'center', 'height'),
            v5_alt_pick_rule='distance_prior_first',
            v5_enable_fallback=True,
            v5_log_debug=True,
            v6_reference_stats_path='',
            v6_stats_binning='class_distance',
            v6_distance_bins=(20.0, 40.0),
            v6_anomaly_score_mode='percentile',
            v6_anomaly_threshold=1.0,
            v6_anomaly_component_weights=None,
            v6_switch_heads=('center', 'height'),
            v6_candidate_rule='non_fused_selector',
            v6_candidate_min_prob=0.60,
            v6_candidate_min_margin=0.10,
            v6_enable_fallback=True,
            v6_log_debug=True,
            v7_reference_stats_path='',
            v7_stats_binning='class_distance',
            v7_distance_bins=(20.0, 40.0),
            v7_bev_damage_threshold=1.25,
            v7_img_damage_threshold=1.75,
            v7_bev_healthy_ceiling=1.0,
            v7_img_healthy_ceiling=1.0,
            v7_img_fallback_prob_threshold=0.70,
            v7_img_fallback_margin_threshold=0.15,
            v7_bev_fallback_prob_threshold=0.80,
            v7_bev_fallback_margin_threshold=0.20,
            v7_router_label_margin=0.10,
            v7_img_router_loss_weight=0.10,
            v7_bev_router_loss_weight=0.05,
            v7_allow_missing_stats_fallback=True,
            v7_log_debug=True,
            crg_enable=False,
            crg_switch_heads=('center',),
            crg_bev_damage_threshold=1.25,
            crg_img_healthy_ceiling=1.0,
            crg_max_switch_ratio=0.02,
            crg_no_img_control=False,
            scrg_enable=False,
            scrg_mode='flag',
            scrg_switch_heads=('center',),
            scrg_scene_metric='top10',
            scrg_top_fraction=0.10,
            scrg_bev_threshold_key='p99',
            scrg_img_ceiling_key='p99',
            scrg_margin_threshold_key='p99',
            scrg_bev_threshold_scale=1.0,
            scrg_img_ceiling_scale=1.0,
            scrg_margin_threshold_scale=1.0,
            scrg_bev_metric_name='',
            scrg_img_metric_name='',
            scrg_aux_metric_name='',
            scrg_bev_signal='center_unc',
            scrg_img_signal='center_unc',
            scrg_aux_signal='margin_unc',
            scrg_max_switch_ratio=0.10,
            supp_enable=False,
            supp_mode='img_highscore',
            supp_max_add_queries=20,
            supp_img_score_threshold=0.05,
            supp_bev_score_threshold=0.05,
            supp_fused_replace_score_ceiling=1.01,
            supp_nms_center_dist=0.0,
            supp_per_class_cap=0,
            supp_expected_model_fingerprint='',
            supp_require_reference_stats=False,
            supp_utility_mapper_path='',
            supp_utility_threshold=0.0,
            supp_img_utility_mapper_path='',
            supp_bev_utility_mapper_path='',
            supp_img_utility_threshold=-1.0,
            supp_bev_utility_threshold=-1.0,
            supp_fused_slot_utility_mapper_path='',
            supp_marginal_utility_enable=False,
            supp_marginal_utility_margin=0.0,
            supp_dual_damage_marginal_margin=-1.0,
            supp_scene_mapper_path='',
            supp_scene_mapper_threshold=0.50,
            supp_scene_mapper_gate_mode='override',
            supp_scene_budget_min_ratio=0.0,
            supp_scene_budget_max_ratio=0.0,
            supp_scene_budget_full_threshold=1.0,
            supp_rank_mode='img_score',
            supp_utility_rank_weight=1.0,
            supp_img_damage_guard_enable=False,
            supp_img_damage_guard_mode='off',
            supp_img_damage_signal='',
            supp_img_damage_metric_name='',
            supp_img_damage_threshold_key='p99',
            supp_img_damage_threshold_scale=1.0,
            supp_fused_damage_signal='fused_center_unc',
            supp_fused_damage_metric_name='',
            supp_fused_damage_threshold_key='p99',
            supp_fused_damage_threshold_scale=2.0,
            supp_relative_damage_reference_key='median',
            supp_bev_fused_dominance_ratio=1.0,
            supp_bev_damage_upper_threshold_scale=0.0,
            supp_dual_abstain_signal='',
            supp_dual_abstain_threshold=-1.0,
            supp_img_abstain_signal='',
            supp_img_abstain_threshold=-1.0,
            supp_qsem_gate_enable=False,
            supp_qsem_disagree_signal='qsem_top300_img_bev_class_disagree_ratio',
            supp_qsem_disagree_threshold=-1.0,
            supp_qsem_margin_signal='qsem_top300_img_margin_mean',
            supp_qsem_margin_threshold=-1.0,
            supp_contextheight_gate_enable=False,
            supp_contextheight_context_signal='v8_context_img_damage_mean',
            supp_contextheight_context_threshold=-1.0,
            supp_contextheight_height_signal='distance_bin_abnormality_stats.height.bev.mid',
            supp_contextheight_height_threshold=-1.0,
            supp_contextheight_v9_bev_signal='v9_a_bev_above_tau_mean',
            supp_contextheight_v9_bev_threshold=-1.0,
            supp_dual_damage_budget_ratio=0.0,
            supp_dual_damage_utility_threshold=0.0,
            supp_dual_damage_gate_mode='strict_img',
            supp_dual_scene_mapper_threshold=0.90,
            supp_dual_bev_threshold_scale=1.0,
            supp_dual_bev_rank_boost=1.0,
            v8_fused_keep_init=0.82,
            v8_fused_keep_min=0.10,
            v8_fused_keep_max=0.95,
            v8_damage_temperature=1.0,
            v8_high_damage_threshold=0.65,
            v8_unc_score_weight=0.30,
            v8_rel_unc_score_weight=0.40,
            v8_conditioned_score_weight=0.25,
            v8_percentile_score_weight=0.35,
            v8_disagreement_score_weight=0.25,
            v8_cls_gap_score_weight=0.20,
            v8_router_variant='a1_damage',
            v8_reliability_init=0.90,
            v8_reliability_temperature=0.35,
            v8_reliable_threshold=0.70,
            v8_unreliable_quantile=0.75,
            v8_unreliable_min=0.35,
            v8_reliability_margin=0.15,
            v8_a3_raw_damage_weight=0.10,
            v8_a3_context_damage_scale=1.0,
            v8_a3_unreliable_min=0.60,
            v8_a3_center_safe_keep_floor=0.88,
            v8_a3_center_single_damage_keep_floor=0.70,
            v8_a3_center_route_margin=0.25,
            v8_a3_center_non_fused_margin=0.20,
            v8_a4_anchor_momentum=0.02,
            v8_a4_anchor_eps=1e-4,
            v8_a4_reliability_temperature=0.55,
            v8_a4_route_temperature=0.45,
            v8_a4_context_route_weight=0.25,
            v8_a4_distance_route_weight=0.10,
            v8_a4_center_disagreement_route_weight=0.15,
            v8_a4_clean_damage_threshold=0.45,
            v8_a4_single_damage_threshold=0.80,
            v8_a4_single_safe_threshold=0.65,
            v8_a4_damage_margin=0.25,
            v8_a4_dual_damage_threshold=0.65,
            v9_image_feature_enrichment=True,
            v9_use_tail_features=True,
            v9_tail_z_threshold=1.0,
            v9_tail_temperature=0.5,
            v9_use_per_query_tail_excess=True,
            v9_anomaly_head_hidden=32,
            v9_q95_z_reference=1.645,
            v9_tau_target_quantile=0.94,
            v9_tau_momentum=0.02,
            v9_tau_init=0.50,
            update_tau_online=True,
            v9_route_w_other_damage=1.5,
            v9_route_w_self_damage=0.5,
            v9_route_w_context=0.25,
            v9_route_w_distance=0.10,
            v9_route_w_consistency=0.30,
            v9_route_temperature=0.45,
            v9_disagreement_temperature=2.0,
            v9_use_modality_specific_consistency=True,
            v9_use_separate_v9_fused_gate_head=True,
            v9_fused_gate_input_dim=7,
            v9_clean_distance_metric='positive_rms',
            v9_clean_distance_threshold_init=0.5,
            v9_clean_distance_threshold_target_quantile=0.80,
            v9_clean_distance_threshold_momentum=0.02,
            v9_freeze_clean_anchor_epochs=0,
            v9_emit_clean_q95_z_diagnostic=True,
            v9_clean_fused_keep_floor=0.95,
            v9_center_domain_consistent_only=True,
            v9_route_w_warmup_epochs=1,
            v9_anomaly_head_detach_input=False,
            v9_anomaly_head_lambda_warmup=False,
            v9_emit_velocity_diagnostic=True,
            v9_enable_per_head_dispatch=False,
            v9_cap_mode='none',
            v9_cap_cls=0.95,
            v9_cap_height=0.95,
            v9_cap_vel=None,
            v9_cap_center=None,
            v9_cap_dual_scale=0.85,
            v10_enable=False,
            v10_input_detach=True,
            v10_cls_fused_keep_min=0.85,
            v10_use_conservative_schedule=False,
            v10_use_per_head_cls_dispatch=False,
            v10_dd_head_hidden=32,
            v10_router_head_hidden=64,
            v10_csg_head_hidden=16,
            v10_edlcue_enable=False,
            v10_edlcue_mode='trainable',
            v10_edlcue_input_dim=24,
            v10_edlcue_hidden=32,
            v10_edlcue_max_delta=0.25,
            v10_oracle_thresholds=None,
            force_modality=None,
            v8_log_debug=True,
            init_cfg=None):
        super(EDLPMETransformer, self).__init__(init_cfg=init_cfg)

        self.dist_scaler = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dist_bias = nn.Parameter(torch.randn(1), requires_grad=True)

        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_layers = decoder["num_layers"]
        self.num_heads = decoder["transformerlayers"]["attn_cfgs"][0]["num_heads"]

        self.box_pos_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))

        self.modality_proj = nn.ModuleDict({
            "fused": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)),
            "bev": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)),
            "img": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims))
        })

        self.use_unc_bias = use_unc_bias
        self.use_value_scaling = use_value_scaling
        self.use_dim_edl = use_dim_edl
        self.pme_unc_sources = tuple(pme_unc_sources)
        self.ensemble_mode = ensemble_mode
        self.fused_prior_bias = fused_prior_bias
        self.selector_score_mode = selector_score_mode
        self.selector_unc_score_weight = selector_unc_score_weight
        self.selector_rel_unc_score_weight = selector_rel_unc_score_weight
        self.selector_disagreement_score_weight = selector_disagreement_score_weight
        self.selector_context_score_weight = selector_context_score_weight
        self.selector_distance_score_weight = selector_distance_score_weight
        self.selector_fused_score_bias = (
            fused_prior_bias if selector_fused_score_bias is None else selector_fused_score_bias
        )
        self.selector_log_score_components = selector_log_score_components
        self.selector_softmax_temperature = max(selector_softmax_temperature, 1e-6)
        self.selector_apply_heads = set(selector_apply_heads or ['cls_logits', 'center', 'height'])
        self.selector_frozen_heads = set(selector_frozen_heads or ['dim', 'rot', 'vel'])
        self.selector_unc_transform = selector_unc_transform
        self.selector_unc_norm = selector_unc_norm
        self.selector_use_relative_unc = selector_use_relative_unc
        self.residual_floor = residual_floor
        self.residual_usage_target = residual_usage_target
        self.modality_names = ("fused", "bev", "img")
        self.v5_reference_stats_path = v5_reference_stats_path
        self.v5_stats_binning = v5_stats_binning
        self.v5_anomaly_score_mode = v5_anomaly_score_mode
        self.v5_anomaly_threshold = float(v5_anomaly_threshold)
        self.v5_switch_heads = set(v5_switch_heads or ['cls', 'center', 'height'])
        self.v5_alt_pick_rule = v5_alt_pick_rule
        self.v5_enable_fallback = v5_enable_fallback
        self.v5_log_debug = v5_log_debug
        component_weights = v5_anomaly_component_weights or {
            'cls': 1.0,
            'center': 1.0,
            'height': 1.0,
        }
        self.v5_anomaly_component_weights = {
            'cls': float(component_weights.get('cls', 1.0)),
            'center': float(component_weights.get('center', 1.0)),
            'height': float(component_weights.get('height', 1.0)),
        }
        self._v5_reference_stats_cache = None
        self._v5_reference_stats_cache_path = None
        self.v6_reference_stats_path = v6_reference_stats_path
        self.v6_stats_binning = v6_stats_binning
        self.v6_anomaly_score_mode = v6_anomaly_score_mode
        self.v6_anomaly_threshold = float(v6_anomaly_threshold)
        self.v6_switch_heads = set(v6_switch_heads or ['center', 'height'])
        self.v6_candidate_rule = v6_candidate_rule
        self.v6_candidate_min_prob = float(v6_candidate_min_prob)
        self.v6_candidate_min_margin = float(v6_candidate_min_margin)
        self.v6_enable_fallback = v6_enable_fallback
        self.v6_log_debug = v6_log_debug
        v6_component_weights = v6_anomaly_component_weights or {
            'cls': 1.0,
            'center': 1.0,
            'height': 1.0,
        }
        self.v6_anomaly_component_weights = {
            'cls': float(v6_component_weights.get('cls', 1.0)),
            'center': float(v6_component_weights.get('center', 1.0)),
            'height': float(v6_component_weights.get('height', 1.0)),
        }
        self._v6_reference_stats_cache = None
        self._v6_reference_stats_cache_path = None
        self.v7_reference_stats_path = v7_reference_stats_path
        self.v7_stats_binning = v7_stats_binning
        self.v7_bev_damage_threshold = float(v7_bev_damage_threshold)
        self.v7_img_damage_threshold = float(v7_img_damage_threshold)
        self.v7_bev_healthy_ceiling = float(v7_bev_healthy_ceiling)
        self.v7_img_healthy_ceiling = float(v7_img_healthy_ceiling)
        self.v7_img_fallback_prob_threshold = float(v7_img_fallback_prob_threshold)
        self.v7_img_fallback_margin_threshold = float(v7_img_fallback_margin_threshold)
        self.v7_bev_fallback_prob_threshold = float(v7_bev_fallback_prob_threshold)
        self.v7_bev_fallback_margin_threshold = float(v7_bev_fallback_margin_threshold)
        self.v7_router_label_margin = float(v7_router_label_margin)
        self.v7_img_router_loss_weight = float(v7_img_router_loss_weight)
        self.v7_bev_router_loss_weight = float(v7_bev_router_loss_weight)
        self.v7_allow_missing_stats_fallback = bool(v7_allow_missing_stats_fallback)
        self.v7_log_debug = v7_log_debug
        self.crg_enable = bool(crg_enable)
        self.crg_switch_heads = set(crg_switch_heads or ['center'])
        self.crg_bev_damage_threshold = float(crg_bev_damage_threshold)
        self.crg_img_healthy_ceiling = float(crg_img_healthy_ceiling)
        self.crg_max_switch_ratio = max(float(crg_max_switch_ratio), 0.0)
        self.crg_no_img_control = bool(crg_no_img_control)
        self.scrg_enable = bool(scrg_enable)
        self.scrg_mode = str(scrg_mode)
        self.scrg_switch_heads = set(scrg_switch_heads or ['center'])
        self.scrg_scene_metric = str(scrg_scene_metric)
        self.scrg_top_fraction = min(max(float(scrg_top_fraction), 1e-6), 1.0)
        self.scrg_bev_threshold_key = str(scrg_bev_threshold_key)
        self.scrg_img_ceiling_key = str(scrg_img_ceiling_key)
        self.scrg_margin_threshold_key = str(scrg_margin_threshold_key)
        self.scrg_bev_threshold_scale = float(scrg_bev_threshold_scale)
        self.scrg_img_ceiling_scale = float(scrg_img_ceiling_scale)
        self.scrg_margin_threshold_scale = float(scrg_margin_threshold_scale)
        self.scrg_bev_metric_name = str(scrg_bev_metric_name or '')
        self.scrg_img_metric_name = str(scrg_img_metric_name or '')
        self.scrg_aux_metric_name = str(scrg_aux_metric_name or '')
        self.scrg_bev_signal = str(scrg_bev_signal or 'center_unc')
        self.scrg_img_signal = str(scrg_img_signal or 'center_unc')
        self.scrg_aux_signal = str(scrg_aux_signal or 'margin_unc')
        self.scrg_max_switch_ratio = max(float(scrg_max_switch_ratio), 0.0)
        self.supp_enable = bool(supp_enable)
        self.supp_mode = str(supp_mode)
        self.supp_max_add_queries = max(int(supp_max_add_queries), 0)
        self.supp_img_score_threshold = float(supp_img_score_threshold)
        self.supp_bev_score_threshold = float(supp_bev_score_threshold)
        self.supp_fused_replace_score_ceiling = float(supp_fused_replace_score_ceiling)
        self.supp_nms_center_dist = max(float(supp_nms_center_dist), 0.0)
        self.supp_per_class_cap = max(int(supp_per_class_cap), 0)
        self.supp_expected_model_fingerprint = str(
            supp_expected_model_fingerprint or '')
        self.supp_require_reference_stats = bool(supp_require_reference_stats)
        self.supp_utility_mapper_path = str(supp_utility_mapper_path or '')
        self.supp_utility_threshold = float(supp_utility_threshold)
        self.supp_utility_mapper = self._load_supp_utility_mapper(
            self.supp_utility_mapper_path, expected_target='candidate_useful')
        if self.supp_utility_mapper_path and self.supp_utility_mapper is None:
            raise RuntimeError(
                f"Utility mapper was requested but could not be loaded safely: "
                f"{self.supp_utility_mapper_path}")
        self.supp_img_utility_mapper_path = str(supp_img_utility_mapper_path or '')
        self.supp_bev_utility_mapper_path = str(supp_bev_utility_mapper_path or '')
        self.supp_img_utility_threshold = float(supp_img_utility_threshold)
        self.supp_bev_utility_threshold = float(supp_bev_utility_threshold)
        self.supp_img_utility_mapper = self._load_supp_utility_mapper(
            self.supp_img_utility_mapper_path, expected_target='img_supp_opportunity')
        self.supp_bev_utility_mapper = self._load_supp_utility_mapper(
            self.supp_bev_utility_mapper_path, expected_target='bev_supp_opportunity')
        if self.supp_img_utility_mapper_path and self.supp_img_utility_mapper is None:
            raise RuntimeError(
                f"IMG utility mapper was requested but could not be loaded safely: "
                f"{self.supp_img_utility_mapper_path}")
        if self.supp_bev_utility_mapper_path and self.supp_bev_utility_mapper is None:
            raise RuntimeError(
                f"BEV utility mapper was requested but could not be loaded safely: "
                f"{self.supp_bev_utility_mapper_path}")
        self.supp_fused_slot_utility_mapper_path = str(supp_fused_slot_utility_mapper_path or '')
        self.supp_fused_slot_utility_mapper = self._load_supp_utility_mapper(
            self.supp_fused_slot_utility_mapper_path,
            expected_target='candidate_useful',
            expected_branch_filter='fused')
        if self.supp_fused_slot_utility_mapper_path and self.supp_fused_slot_utility_mapper is None:
            raise RuntimeError(
                f"Fused slot utility mapper was requested but could not be loaded safely: "
                f"{self.supp_fused_slot_utility_mapper_path}")
        self.supp_marginal_utility_enable = bool(supp_marginal_utility_enable)
        if self.supp_marginal_utility_enable and not self.supp_fused_slot_utility_mapper_path:
            raise RuntimeError(
                "supp_marginal_utility_enable=True requires "
                "supp_fused_slot_utility_mapper_path.")
        self.supp_marginal_utility_margin = float(supp_marginal_utility_margin)
        self.supp_dual_damage_marginal_margin = float(supp_dual_damage_marginal_margin)
        self.supp_scene_mapper_path = str(supp_scene_mapper_path or '')
        self.supp_scene_mapper_threshold = float(supp_scene_mapper_threshold)
        self.supp_scene_mapper_gate_mode = str(supp_scene_mapper_gate_mode or 'override')
        self.supp_scene_budget_min_ratio = max(float(supp_scene_budget_min_ratio), 0.0)
        self.supp_scene_budget_max_ratio = max(float(supp_scene_budget_max_ratio), 0.0)
        if self.supp_scene_budget_max_ratio > 0.0:
            self.supp_scene_budget_max_ratio = max(
                self.supp_scene_budget_max_ratio, self.supp_scene_budget_min_ratio)
        self.supp_scene_budget_full_threshold = float(supp_scene_budget_full_threshold)
        self.supp_rank_mode = str(supp_rank_mode or 'img_score')
        self.supp_utility_rank_weight = float(supp_utility_rank_weight)
        self.supp_img_damage_guard_enable = bool(supp_img_damage_guard_enable)
        self.supp_img_damage_guard_mode = str(supp_img_damage_guard_mode or 'off').lower()
        self.supp_img_damage_signal = str(supp_img_damage_signal or '')
        self.supp_img_damage_metric_name = str(supp_img_damage_metric_name or '')
        self.supp_img_damage_threshold_key = str(
            supp_img_damage_threshold_key or 'p99')
        self.supp_img_damage_threshold_scale = max(float(supp_img_damage_threshold_scale), 0.0)
        self.supp_fused_damage_signal = str(
            supp_fused_damage_signal or 'fused_center_unc')
        self.supp_fused_damage_metric_name = str(supp_fused_damage_metric_name or '')
        self.supp_fused_damage_threshold_key = str(
            supp_fused_damage_threshold_key or 'p99')
        self.supp_fused_damage_threshold_scale = max(
            float(supp_fused_damage_threshold_scale), 0.0)
        self.supp_relative_damage_reference_key = str(
            supp_relative_damage_reference_key or 'median')
        self.supp_bev_fused_dominance_ratio = max(
            float(supp_bev_fused_dominance_ratio), 0.0)
        self.supp_bev_damage_upper_threshold_scale = max(
            float(supp_bev_damage_upper_threshold_scale), 0.0)
        self.supp_dual_abstain_signal = str(supp_dual_abstain_signal or '')
        self.supp_dual_abstain_threshold = float(supp_dual_abstain_threshold)
        self.supp_img_abstain_signal = str(supp_img_abstain_signal or '')
        self.supp_img_abstain_threshold = float(supp_img_abstain_threshold)
        self.supp_qsem_gate_enable = bool(supp_qsem_gate_enable)
        self.supp_qsem_disagree_signal = str(
            supp_qsem_disagree_signal or
            'qsem_top300_img_bev_class_disagree_ratio')
        self.supp_qsem_disagree_threshold = float(supp_qsem_disagree_threshold)
        self.supp_qsem_margin_signal = str(
            supp_qsem_margin_signal or 'qsem_top300_img_margin_mean')
        self.supp_qsem_margin_threshold = float(supp_qsem_margin_threshold)
        self.supp_contextheight_gate_enable = bool(supp_contextheight_gate_enable)
        self.supp_contextheight_context_signal = str(
            supp_contextheight_context_signal or 'v8_context_img_damage_mean')
        self.supp_contextheight_context_threshold = float(
            supp_contextheight_context_threshold)
        self.supp_contextheight_height_signal = str(
            supp_contextheight_height_signal or
            'distance_bin_abnormality_stats.height.bev.mid')
        self.supp_contextheight_height_threshold = float(
            supp_contextheight_height_threshold)
        self.supp_contextheight_v9_bev_signal = str(
            supp_contextheight_v9_bev_signal or 'v9_a_bev_above_tau_mean')
        self.supp_contextheight_v9_bev_threshold = float(
            supp_contextheight_v9_bev_threshold)
        self.supp_dual_damage_budget_ratio = max(float(supp_dual_damage_budget_ratio), 0.0)
        self.supp_dual_damage_utility_threshold = float(supp_dual_damage_utility_threshold)
        self.supp_dual_damage_gate_mode = str(supp_dual_damage_gate_mode or 'strict_img').lower()
        self.supp_dual_scene_mapper_threshold = float(supp_dual_scene_mapper_threshold)
        self.supp_dual_bev_threshold_scale = max(float(supp_dual_bev_threshold_scale), 0.0)
        self.supp_dual_bev_rank_boost = max(float(supp_dual_bev_rank_boost), 0.0)
        self.supp_scene_mapper = self._load_supp_scene_mapper(self.supp_scene_mapper_path)
        if self.supp_scene_mapper_path and self.supp_scene_mapper is None:
            raise RuntimeError(
                f"Scene mapper was requested but could not be loaded safely: "
                f"{self.supp_scene_mapper_path}"
            )
        self._supp_scene_missing_features_warned = False
        self._v7_reference_stats_cache = None
        self._v7_reference_stats_cache_path = None
        self._v7_missing_stats_warned = False
        self.v8_fused_keep_init = float(v8_fused_keep_init)
        self.v8_fused_keep_min = float(v8_fused_keep_min)
        self.v8_fused_keep_max = float(v8_fused_keep_max)
        self.v8_damage_temperature = max(float(v8_damage_temperature), 1e-6)
        self.v8_high_damage_threshold = float(v8_high_damage_threshold)
        self.v8_unc_score_weight = float(v8_unc_score_weight)
        self.v8_rel_unc_score_weight = float(v8_rel_unc_score_weight)
        self.v8_conditioned_score_weight = float(v8_conditioned_score_weight)
        self.v8_percentile_score_weight = float(v8_percentile_score_weight)
        self.v8_disagreement_score_weight = float(v8_disagreement_score_weight)
        self.v8_cls_gap_score_weight = float(v8_cls_gap_score_weight)
        self.v8_router_variant = v8_router_variant
        self.v8_reliability_init = min(max(float(v8_reliability_init), 1e-4), 1.0 - 1e-4)
        self.v8_reliability_temperature = max(float(v8_reliability_temperature), 1e-6)
        self.v8_reliable_threshold = float(v8_reliable_threshold)
        self.v8_unreliable_quantile = min(max(float(v8_unreliable_quantile), 0.0), 1.0)
        self.v8_unreliable_min = float(v8_unreliable_min)
        self.v8_reliability_margin = float(v8_reliability_margin)
        self.v8_a3_raw_damage_weight = float(v8_a3_raw_damage_weight)
        self.v8_a3_context_damage_scale = float(v8_a3_context_damage_scale)
        self.v8_a3_unreliable_min = float(v8_a3_unreliable_min)
        self.v8_a3_center_safe_keep_floor = float(v8_a3_center_safe_keep_floor)
        self.v8_a3_center_single_damage_keep_floor = float(v8_a3_center_single_damage_keep_floor)
        self.v8_a3_center_route_margin = float(v8_a3_center_route_margin)
        self.v8_a3_center_non_fused_margin = float(v8_a3_center_non_fused_margin)
        self.v8_a4_anchor_momentum = min(max(float(v8_a4_anchor_momentum), 0.0), 1.0)
        self.v8_a4_anchor_eps = max(float(v8_a4_anchor_eps), 1e-8)
        self.v8_a4_reliability_temperature = max(float(v8_a4_reliability_temperature), 1e-6)
        self.v8_a4_route_temperature = max(float(v8_a4_route_temperature), 1e-6)
        self.v8_a4_context_route_weight = float(v8_a4_context_route_weight)
        self.v8_a4_distance_route_weight = float(v8_a4_distance_route_weight)
        self.v8_a4_center_disagreement_route_weight = float(v8_a4_center_disagreement_route_weight)
        self.v8_a4_clean_damage_threshold = float(v8_a4_clean_damage_threshold)
        self.v8_a4_single_damage_threshold = float(v8_a4_single_damage_threshold)
        self.v8_a4_single_safe_threshold = float(v8_a4_single_safe_threshold)
        self.v8_a4_damage_margin = float(v8_a4_damage_margin)
        self.v8_a4_dual_damage_threshold = float(v8_a4_dual_damage_threshold)
        self.v9_image_feature_enrichment = bool(v9_image_feature_enrichment)
        self.v9_use_tail_features = bool(v9_use_tail_features)
        self.v9_tail_z_threshold = float(v9_tail_z_threshold)
        self.v9_tail_temperature = max(float(v9_tail_temperature), 1e-6)
        self.v9_use_per_query_tail_excess = bool(v9_use_per_query_tail_excess)
        self.v9_anomaly_head_hidden = int(v9_anomaly_head_hidden)
        self.v9_q95_z_reference = float(v9_q95_z_reference)
        self.v9_tau_target_quantile = min(max(float(v9_tau_target_quantile), 0.0), 1.0)
        self.v9_tau_momentum = min(max(float(v9_tau_momentum), 0.0), 1.0)
        self.v9_tau_init = min(max(float(v9_tau_init), 1e-4), 1.0 - 1e-4)
        self.update_tau_online = bool(update_tau_online)
        self.v9_route_w_other_damage = float(v9_route_w_other_damage)
        self.v9_route_w_self_damage = float(v9_route_w_self_damage)
        self.v9_route_w_context = float(v9_route_w_context)
        self.v9_route_w_distance = float(v9_route_w_distance)
        self.v9_route_w_consistency = float(v9_route_w_consistency)
        self.v9_route_temperature = max(float(v9_route_temperature), 1e-6)
        self.v9_disagreement_temperature = max(float(v9_disagreement_temperature), 1e-6)
        self.v9_use_modality_specific_consistency = bool(v9_use_modality_specific_consistency)
        self.v9_use_separate_v9_fused_gate_head = bool(v9_use_separate_v9_fused_gate_head)
        self.v9_fused_gate_input_dim = int(v9_fused_gate_input_dim)
        self.v9_clean_distance_metric = str(v9_clean_distance_metric)
        self.v9_clean_distance_threshold_init = float(v9_clean_distance_threshold_init)
        self.v9_clean_distance_threshold_target_quantile = min(
            max(float(v9_clean_distance_threshold_target_quantile), 0.0), 1.0)
        self.v9_clean_distance_threshold_momentum = min(
            max(float(v9_clean_distance_threshold_momentum), 0.0), 1.0)
        self.v9_freeze_clean_anchor_epochs = int(v9_freeze_clean_anchor_epochs)
        self.v9_emit_clean_q95_z_diagnostic = bool(v9_emit_clean_q95_z_diagnostic)
        self.v9_clean_fused_keep_floor = float(v9_clean_fused_keep_floor)
        self.v9_center_domain_consistent_only = bool(v9_center_domain_consistent_only)
        self.v9_route_w_warmup_epochs = max(int(v9_route_w_warmup_epochs), 0)
        self.v9_anomaly_head_detach_input = bool(v9_anomaly_head_detach_input)
        self.v9_anomaly_head_lambda_warmup = bool(v9_anomaly_head_lambda_warmup)
        self.v9_emit_velocity_diagnostic = bool(v9_emit_velocity_diagnostic)
        self.v9_enable_per_head_dispatch = bool(v9_enable_per_head_dispatch)
        self.v9_cap_mode = str(v9_cap_mode)
        self.v9_cap_cls = min(max(float(v9_cap_cls), 0.0), 1.0)
        self.v9_cap_height = min(max(float(v9_cap_height), 0.0), 1.0)
        self.v9_cap_vel = (
            None if v9_cap_vel is None else min(max(float(v9_cap_vel), 0.0), 1.0))
        self.v9_cap_center = (
            None if v9_cap_center is None else min(max(float(v9_cap_center), 0.0), 1.0))
        self.v9_cap_dual_scale = min(max(float(v9_cap_dual_scale), 0.0), 1.0)
        self.v10_enable = bool(v10_enable)
        self.v10_input_detach = bool(v10_input_detach)
        self.v10_cls_fused_keep_min = min(max(float(v10_cls_fused_keep_min), 0.0), 1.0)
        self.v10_use_conservative_schedule = bool(v10_use_conservative_schedule)
        self.v10_use_per_head_cls_dispatch = bool(v10_use_per_head_cls_dispatch)
        self.v10_dd_head_hidden = int(v10_dd_head_hidden)
        self.v10_router_head_hidden = int(v10_router_head_hidden)
        self.v10_csg_head_hidden = int(v10_csg_head_hidden)
        self.v10_edlcue_enable = bool(v10_edlcue_enable)
        self.v10_edlcue_mode = str(v10_edlcue_mode)
        self.v10_edlcue_input_dim = int(v10_edlcue_input_dim)
        self.v10_edlcue_hidden = int(v10_edlcue_hidden)
        self.v10_edlcue_max_delta = max(float(v10_edlcue_max_delta), 0.0)
        self.v10_oracle_thresholds = dict(v10_oracle_thresholds or {})
        self.v10_dd_input_dim = 52
        self.v10_router_input_dim = 40
        self.v10_csg_input_dim = 8
        if force_modality == 'lidar':
            force_modality = 'bev'
        if force_modality not in (None, 'fused', 'bev', 'img'):
            raise ValueError(
                "force_modality must be one of None, 'fused', 'bev', 'img', or 'lidar'")
        self.force_modality = force_modality
        self.num_classes = list(num_classes or [])
        self._current_epoch = 0
        self.v8_log_debug = v8_log_debug
        self.register_buffer(
            'distance_bin_edges_tensor',
            torch.tensor(distance_bin_edges, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'v5_distance_bins_tensor',
            torch.tensor(v5_distance_bins, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'v6_distance_bins_tensor',
            torch.tensor(v6_distance_bins, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'v7_distance_bins_tensor',
            torch.tensor(v7_distance_bins, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'v8_a4_anchor_mean',
            torch.zeros(2, 12, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v8_a4_anchor_var',
            torch.ones(2, 12, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v8_a4_anchor_count',
            torch.zeros(1, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_tau_bev',
            torch.full((1,), self.v9_tau_init, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_tau_img',
            torch.full((1,), self.v9_tau_init, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_tau_update_count',
            torch.zeros(1, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_clean_distance_threshold_bev',
            torch.full((1,), self.v9_clean_distance_threshold_init, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_clean_distance_threshold_img',
            torch.full((1,), self.v9_clean_distance_threshold_init, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_clean_distance_update_count',
            torch.zeros(1, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_clean_q95_z_bev',
            torch.zeros(1, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'v9_clean_q95_z_img',
            torch.zeros(1, dtype=torch.float32),
            persistent=True)

        self.unc_scaler = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.unc_bias_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if self.ensemble_mode in (
                'soft_selector',
                'primary_router',
                'hard_selective_switch',
                'gated_primary_router',
                'modality_damage_aware_router',
                'clean_reference_gate_router',
                'damage_gated_soft_router'):
            self.selector_context_proj = nn.Linear(self.embed_dims, len(self.modality_names))
            self.distance_bin_embed = nn.Embedding(len(distance_bin_edges) + 1, 4)
            self.distance_prior_head = nn.Sequential(
                nn.Linear(3 + 4, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1))
            self.residual_router_head = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1))
            self.alpha_res_head = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1))
            if self.ensemble_mode == 'modality_damage_aware_router':
                self.v7_img_fallback_router = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, 2))
                self.v7_bev_fallback_router = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, 2))
            else:
                self.v7_img_fallback_router = None
                self.v7_bev_fallback_router = None
            if self.ensemble_mode == 'damage_gated_soft_router':
                self.v8_fused_gate_head = nn.Sequential(
                    nn.Linear(6, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1))
                self.v9_fused_gate_head = nn.Sequential(
                    nn.Linear(self.v9_fused_gate_input_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1))
                self.v8_a3_damage_context_head = nn.Sequential(
                    nn.Linear(self.embed_dims, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 2))
                self.v9_bev_anomaly_head = nn.Sequential(
                    nn.Linear(17, self.v9_anomaly_head_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.v9_anomaly_head_hidden, 1),
                    nn.Sigmoid())
                self.v9_img_anomaly_head = nn.Sequential(
                    nn.Linear(19, self.v9_anomaly_head_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.v9_anomaly_head_hidden, 1),
                    nn.Sigmoid())
                self.v10_dd_head = self._make_v10_mlp(
                    self.v10_dd_input_dim,
                    self.v10_dd_head_hidden,
                    3,
                    final_activation='sigmoid')
                self.v10_router_head_cls = self._make_v10_mlp(
                    self.v10_router_input_dim,
                    self.v10_router_head_hidden,
                    3)
                self.v10_router_head_height = self._make_v10_mlp(
                    self.v10_router_input_dim,
                    self.v10_router_head_hidden,
                    3)
                self.v10_router_head_center = self._make_v10_mlp(
                    self.v10_router_input_dim,
                    self.v10_router_head_hidden,
                    3)
                self.v10_center_safe_gate = self._make_v10_mlp(
                    self.v10_csg_input_dim,
                    self.v10_csg_head_hidden,
                    1)
                self.v10_edlcue_head = (
                    self._make_v10_mlp(
                        self.v10_edlcue_input_dim,
                        self.v10_edlcue_hidden,
                        1)
                    if self.v10_edlcue_enable else None)
                self.v8_damage_feature_weights_raw = nn.Parameter(torch.zeros(2, 6))
                self.v8_a4_damage_feature_weights_raw = nn.Parameter(torch.zeros(2, 12))
                self.v8_damage_feature_bias = nn.Parameter(torch.zeros(2))
                self.v8_a4_damage_feature_bias = nn.Parameter(torch.full((2,), -1.0))
            else:
                self.v8_fused_gate_head = None
                self.v9_fused_gate_head = None
                self.v8_a3_damage_context_head = None
                self.v9_bev_anomaly_head = None
                self.v9_img_anomaly_head = None
                self.v10_dd_head = None
                self.v10_router_head_cls = None
                self.v10_router_head_height = None
                self.v10_router_head_center = None
                self.v10_center_safe_gate = None
                self.v10_edlcue_head = None
                self.v8_damage_feature_weights_raw = None
                self.v8_a4_damage_feature_weights_raw = None
                self.v8_damage_feature_bias = None
                self.v8_a4_damage_feature_bias = None
            self.task_heads = None
        else:
            self.selector_context_proj = None
            self.distance_bin_embed = None
            self.distance_prior_head = None
            self.residual_router_head = None
            self.alpha_res_head = None
            self.v7_img_fallback_router = None
            self.v7_bev_fallback_router = None
            self.v8_fused_gate_head = None
            self.v9_fused_gate_head = None
            self.v8_a3_damage_context_head = None
            self.v9_bev_anomaly_head = None
            self.v9_img_anomaly_head = None
            self.v10_dd_head = None
            self.v10_router_head_cls = None
            self.v10_router_head_height = None
            self.v10_router_head_center = None
            self.v10_center_safe_gate = None
            self.v10_edlcue_head = None
            self.v8_damage_feature_weights_raw = None
            self.v8_a4_damage_feature_weights_raw = None
            self.v8_damage_feature_bias = None
            self.v8_a4_damage_feature_bias = None
            self.task_heads = nn.ModuleList()
            for num_cls in num_classes:
                task_heads = copy.deepcopy(heads)
                for key, value in task_heads.items():
                    if key in ('rot', 'vel') or (key == 'dim' and not self.use_dim_edl):
                        task_heads[key] = (value[0], value[1])
                    else:
                        task_heads[key] = (value[0] * 4, value[1])
                task_heads.update(dict(cls_logits=(num_cls * 2, 2)))
                separate_head_cfg = copy.deepcopy(separate_head)
                separate_head_cfg.update(
                    in_channels=self.embed_dims,
                    heads=task_heads,
                    num_cls=num_cls,
                    groups=decoder["num_layers"])
                self.task_heads.append(builder.build_head(separate_head_cfg))

    def init_weights(self):
        for module in self.decoder.modules():
            if hasattr(module, 'weight') and module.weight.dim() > 1:
                xavier_init(module, distribution='uniform')
        if self.selector_context_proj is not None:
            xavier_init(self.selector_context_proj, distribution='uniform')
            for module in self.distance_prior_head.modules():
                if hasattr(module, 'weight') and module.weight.dim() > 1:
                    xavier_init(module, distribution='uniform')
            for module in self.residual_router_head.modules():
                if hasattr(module, 'weight') and module.weight.dim() > 1:
                    xavier_init(module, distribution='uniform')
            for module in self.alpha_res_head.modules():
                if hasattr(module, 'weight') and module.weight.dim() > 1:
                    xavier_init(module, distribution='uniform')
            nn.init.zeros_(self.selector_context_proj.bias)
            nn.init.zeros_(self.distance_prior_head[-1].weight)
            nn.init.zeros_(self.distance_prior_head[-1].bias)
            nn.init.zeros_(self.residual_router_head[-1].weight)
            nn.init.zeros_(self.residual_router_head[-1].bias)
            nn.init.zeros_(self.alpha_res_head[-1].weight)
            alpha_bias = torch.logit(torch.tensor(
                (self.residual_usage_target - self.residual_floor) /
                max(1e-6, 1.0 - self.residual_floor)))
            nn.init.constant_(self.alpha_res_head[-1].bias, float(alpha_bias))
            for router_head in (self.v7_img_fallback_router, self.v7_bev_fallback_router):
                if router_head is None:
                    continue
                for module in router_head.modules():
                    if hasattr(module, 'weight') and module.weight.dim() > 1:
                        xavier_init(module, distribution='uniform')
                nn.init.zeros_(router_head[-1].weight)
                nn.init.zeros_(router_head[-1].bias)
            if self.v8_fused_gate_head is not None:
                for module in self.v8_fused_gate_head.modules():
                    if hasattr(module, 'weight') and module.weight.dim() > 1:
                        xavier_init(module, distribution='uniform')
                nn.init.zeros_(self.v8_fused_gate_head[-1].weight)
                keep_span = max(self.v8_fused_keep_max - self.v8_fused_keep_min, 1e-6)
                init_keep = min(max(self.v8_fused_keep_init, self.v8_fused_keep_min + 1e-4),
                                self.v8_fused_keep_max - 1e-4)
                init_unit = (init_keep - self.v8_fused_keep_min) / keep_span
                gate_bias = torch.logit(torch.tensor(init_unit))
                nn.init.constant_(self.v8_fused_gate_head[-1].bias, float(gate_bias))
            if self.v9_fused_gate_head is not None:
                for module in self.v9_fused_gate_head.modules():
                    if hasattr(module, 'weight') and module.weight.dim() > 1:
                        xavier_init(module, distribution='uniform')
                nn.init.zeros_(self.v9_fused_gate_head[-1].weight)
                keep_span = max(self.v8_fused_keep_max - self.v8_fused_keep_min, 1e-6)
                init_keep = min(max(self.v8_fused_keep_init, self.v8_fused_keep_min + 1e-4),
                                self.v8_fused_keep_max - 1e-4)
                init_unit = (init_keep - self.v8_fused_keep_min) / keep_span
                gate_bias = torch.logit(torch.tensor(init_unit))
                nn.init.constant_(self.v9_fused_gate_head[-1].bias, float(gate_bias))
            if self.v8_a3_damage_context_head is not None:
                for module in self.v8_a3_damage_context_head.modules():
                    if hasattr(module, 'weight') and module.weight.dim() > 1:
                        xavier_init(module, distribution='uniform')
                nn.init.zeros_(self.v8_a3_damage_context_head[-1].weight)
                nn.init.zeros_(self.v8_a3_damage_context_head[-1].bias)
            for anomaly_head in (self.v9_bev_anomaly_head, self.v9_img_anomaly_head):
                if anomaly_head is None:
                    continue
                for module in anomaly_head.modules():
                    if hasattr(module, 'weight') and module.weight.dim() > 1:
                        xavier_init(module, distribution='uniform')
                nn.init.zeros_(anomaly_head[-2].weight)
                nn.init.constant_(anomaly_head[-2].bias, -2.0)
            for v10_head in (
                    self.v10_dd_head,
                    self.v10_router_head_cls,
                    self.v10_router_head_height,
                    self.v10_router_head_center,
                    self.v10_center_safe_gate,
                    self.v10_edlcue_head):
                self._init_v10_mlp(v10_head)
            if self.v8_damage_feature_weights_raw is not None:
                init_weights = torch.tensor([
                    self.v8_unc_score_weight,
                    self.v8_rel_unc_score_weight,
                    self.v8_conditioned_score_weight,
                    self.v8_percentile_score_weight,
                    self.v8_disagreement_score_weight,
                    self.v8_cls_gap_score_weight,
                ], dtype=self.v8_damage_feature_weights_raw.dtype)
                init_weights = init_weights.clamp(min=1e-6)
                raw_init = torch.log(torch.expm1(init_weights))
                with torch.no_grad():
                    self.v8_damage_feature_weights_raw.copy_(
                        raw_init.unsqueeze(0).expand_as(self.v8_damage_feature_weights_raw))
                    self.v8_damage_feature_bias.zero_()
            if self.v8_a4_damage_feature_weights_raw is not None:
                init_weights = torch.tensor([
                    0.20, 0.20, 0.20,
                    0.25, 0.25, 0.25,
                    0.20, 0.20, 0.20,
                    0.15, 0.15, 0.20,
                ], dtype=self.v8_a4_damage_feature_weights_raw.dtype)
                init_weights = init_weights.clamp(min=1e-6)
                raw_init = torch.log(torch.expm1(init_weights))
                with torch.no_grad():
                    self.v8_a4_damage_feature_weights_raw.copy_(
                        raw_init.unsqueeze(0).expand_as(self.v8_a4_damage_feature_weights_raw))
                    self.v8_a4_damage_feature_bias.fill_(-1.0)
        self._is_init = True

    @staticmethod
    def _compute_single_nig_uncertainty(nig_tensor):
        dim = nig_tensor.shape[-1] // 4
        if dim == 0:
            return torch.zeros(*nig_tensor.shape[:-1], 1, device=nig_tensor.device)

        nig = nig_tensor.view(*nig_tensor.shape[:-1], dim, 4)
        raw_v = nig[..., 1]
        raw_alpha = nig[..., 2]
        raw_beta = nig[..., 3]

        v = F.softplus(raw_v).clamp(min=1e-6)
        alpha = (F.softplus(raw_alpha) + 1.0).clamp(min=1.0 + 1e-6)
        beta = F.softplus(raw_beta).clamp(min=1e-6)
        unc = beta / (v * (alpha - 1.0).clamp(min=1e-6))
        return unc.mean(dim=-1, keepdim=True)

    def compute_uncertainty(self, nig_inputs):
        if nig_inputs is None:
            return None

        if isinstance(nig_inputs, dict):
            unc_list = []
            for task_name in self.pme_unc_sources:
                nig_key = f'{task_name}_nig'
                if nig_key in nig_inputs and nig_inputs[nig_key] is not None:
                    unc_list.append(self._compute_single_nig_uncertainty(nig_inputs[nig_key]))
            if not unc_list:
                return None
            return torch.cat(unc_list, dim=-1).mean(dim=-1, keepdim=True)

        return self._compute_single_nig_uncertainty(nig_inputs)

    @staticmethod
    def _make_v10_mlp(input_dim, hidden_dim, output_dim, final_activation=None):
        layers = [
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(output_dim)),
        ]
        if final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    @staticmethod
    def _init_v10_mlp(module):
        if module is None:
            return
        linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            xavier_init(layer, distribution='uniform')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if linear_layers:
            nn.init.zeros_(linear_layers[-1].weight)
            if linear_layers[-1].bias is not None:
                nn.init.zeros_(linear_layers[-1].bias)

    @staticmethod
    def _v10_to_query_feature(value, bs, nq, device, dtype):
        if value is None or not isinstance(value, torch.Tensor):
            return torch.zeros(bs, nq, 1, device=device, dtype=dtype)
        feat = value
        if feat.dim() == 4:
            feat = feat[-1]
        elif feat.dim() == 1:
            feat = feat.view(1, 1, 1)
        elif feat.dim() == 2:
            feat = feat.unsqueeze(-1)
        if feat.dim() != 3:
            return torch.zeros(bs, nq, 1, device=device, dtype=dtype)
        feat = feat.to(device=device, dtype=dtype)
        if feat.shape[0] != bs:
            if feat.shape[0] == 1:
                feat = feat.expand(bs, -1, -1)
            else:
                return torch.zeros(bs, nq, feat.shape[-1], device=device, dtype=dtype)
        if feat.shape[1] != nq:
            if feat.shape[1] == 1:
                feat = feat.expand(-1, nq, -1)
            else:
                return torch.zeros(bs, nq, feat.shape[-1], device=device, dtype=dtype)
        return torch.nan_to_num(feat, nan=0.0, posinf=1e4, neginf=-1e4)

    def _v10_pack_features(self, tensors, target_dim, bs, nq, device, dtype, detach=True):
        query_tensors = [
            self._v10_to_query_feature(tensor, bs, nq, device, dtype)
            for tensor in tensors
        ]
        if query_tensors:
            features = torch.cat(query_tensors, dim=-1)
        else:
            features = torch.zeros(bs, nq, 0, device=device, dtype=dtype)
        if features.shape[-1] < target_dim:
            pad = torch.zeros(
                bs, nq, target_dim - features.shape[-1],
                device=device, dtype=dtype)
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > target_dim:
            features = features[..., :target_dim]
        return features.detach() if detach else features

    @staticmethod
    def _v10_last_decoder_tensor(value):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, torch.Tensor) and value.dim() == 4:
            return value[-1]
        return value

    def _v10_expand_class_feature(self, value, target, detach=True):
        bs, nq, num_cls = target.shape
        feat = self._v10_to_query_feature(
            self._v10_last_decoder_tensor(value),
            bs, nq, target.device, target.dtype)
        if detach:
            feat = feat.detach()
        if feat.shape[-1] == num_cls:
            return feat.unsqueeze(-1)
        if feat.shape[-1] != 1:
            feat = feat.mean(dim=-1, keepdim=True)
        return feat.unsqueeze(2).expand(-1, -1, num_cls, -1)

    def _v10_source_score_feature(self, source_cls_logits, branch, target):
        if not isinstance(source_cls_logits, dict) or branch not in source_cls_logits:
            return target.new_zeros(target.shape)
        logits = self._v10_last_decoder_tensor(source_cls_logits.get(branch))
        if not isinstance(logits, torch.Tensor):
            return target.new_zeros(target.shape)
        logits = self._v10_to_query_feature(
            logits, target.shape[0], target.shape[1],
            target.device, target.dtype)
        if logits.shape[-1] == target.shape[-1]:
            return logits.sigmoid()
        if logits.shape[-1] == 1:
            return logits.sigmoid().expand_as(target)
        return target.new_zeros(target.shape)

    @staticmethod
    def _ev_branch_candidate_gather(flat_tensor, topk_idx):
        if not isinstance(flat_tensor, torch.Tensor):
            return None
        if flat_tensor.dim() == 2:
            return flat_tensor.gather(1, topk_idx)
        if flat_tensor.dim() == 3:
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, flat_tensor.shape[-1])
            return flat_tensor.gather(1, gather_idx)
        return None

    def _maybe_export_ev_branch_candidates(
            self,
            blended_outs,
            source_outs,
            modalities,
            num_queries_per_modality,
            pc_range,
            task_id=0,
            raw_unc_by_source=None,
            source_cls_unc=None,
            v8_info=None):
        if not bool(getattr(self, 'ev_branch_candidate_log_enable', False)):
            return
        if not isinstance(source_outs, dict) or not modalities:
            return
        source_cls_logits = self._get_source_tensor_last(
            source_outs, 'cls_logits', modalities, num_queries_per_modality)
        if not isinstance(source_cls_logits, dict):
            return

        branches = getattr(self, 'ev_branch_candidate_branches', None)
        if isinstance(branches, str):
            branches = [item.strip() for item in branches.split(',') if item.strip()]
        if not branches:
            branches = ['fused', 'bev', 'img']
        branches = [branch for branch in branches if branch in modalities and branch in source_cls_logits]
        if not branches:
            return

        source_center = self._get_source_tensor_last(
            source_outs, 'center', modalities, num_queries_per_modality)
        source_height = self._get_source_tensor_last(
            source_outs, 'height', modalities, num_queries_per_modality)
        source_dim = self._get_source_tensor_last(
            source_outs, 'dim', modalities, num_queries_per_modality)
        source_rot = self._get_source_tensor_last(
            source_outs, 'rot', modalities, num_queries_per_modality)
        source_vel = self._get_source_tensor_last(
            source_outs, 'vel', modalities, num_queries_per_modality)
        raw_unc_by_source = raw_unc_by_source or {}
        source_cls_unc = source_cls_unc or {}

        score_list = []
        valid_list = []
        label_list = []
        local_label_list = []
        task_id_list = []
        global_label_list = []
        task_class_offset_list = []
        candidate_uid_list = []
        branch_idx_list = []
        query_idx_list = []
        branch_rank_list = []
        center_abs_list = []
        center_norm_list = []
        height_norm_list = []
        dim_list = []
        rot_list = []
        vel_list = []
        cls_unc_list = []
        center_unc_list = []
        height_unc_list = []
        combined_unc_list = []
        cls_weight_list = []
        center_weight_list = []
        height_weight_list = []
        non_fused_prob_list = []
        fused_keep_list = []
        damage_gate_list = []
        bev_damage_list = []
        img_damage_list = []
        dual_damage_list = []
        damage_state_list = []
        center_source_list = []
        admitted_policy_list = []
        action_label_list = []
        action_space_list = []
        selected_cls_list = []
        selected_center_list = []
        selected_height_list = []

        first_logits = source_cls_logits[branches[0]]
        bs, nq, _ = first_logits.shape
        device = first_logits.device
        dtype = first_logits.dtype
        pc = first_logits.new_tensor(pc_range if pc_range is not None else [0, 0, 0, 1, 1, 1])
        task_id_int = int(task_id or 0)
        class_offset = (
            int(sum(self.num_classes[:task_id_int]))
            if self.num_classes and task_id_int < len(self.num_classes) else 0)

        def zeros(width=1):
            return first_logits.new_zeros(bs, nq, width)

        def branch_feature(mapping, branch, width=1):
            value = mapping.get(branch) if isinstance(mapping, dict) else None
            if isinstance(value, torch.Tensor) and value.shape[0] == bs and value.shape[1] == nq:
                if value.dim() == 2:
                    value = value.unsqueeze(-1)
                return value.to(device=device, dtype=dtype)
            return zeros(width)

        def query_feature(value, branch, width=1, default=0.0):
            if isinstance(value, torch.Tensor):
                feat = self._v10_to_query_feature(value, bs, nq, device, dtype)
                if feat.shape[-1] == 1:
                    return feat
                if branch in modalities and feat.shape[-1] >= len(modalities):
                    idx = modalities.index(branch)
                    return feat[..., idx:idx + 1]
                return feat.mean(dim=-1, keepdim=True)
            return first_logits.new_full((bs, nq, 1), float(default))

        if isinstance(v8_info, dict):
            cls_weight_seq = v8_info.get('cls_weight_seq')
            center_weight_seq = v8_info.get('center_weight_seq')
            height_weight_seq = v8_info.get('height_weight_seq')
            non_fused_probs = v8_info.get('non_fused_probs')
            fused_keep = v8_info.get('fused_keep')
            damage_gate = v8_info.get('damage_gate')
            bev_damage = v8_info.get('bev_damage')
            img_damage = v8_info.get('img_damage')
            dual_damage = v8_info.get('dual_damage')
            damage_state = v8_info.get('damage_state_idx')
            center_source = v8_info.get('center_source_idx')
        else:
            cls_weight_seq = center_weight_seq = height_weight_seq = None
            non_fused_probs = fused_keep = damage_gate = None
            bev_damage = img_damage = dual_damage = None
            damage_state = center_source = None

        cls_weight_last = self._v10_last_decoder_tensor(cls_weight_seq)
        center_weight_last = self._v10_last_decoder_tensor(center_weight_seq)
        height_weight_last = self._v10_last_decoder_tensor(height_weight_seq)
        fused_keep_last = self._v10_last_decoder_tensor(fused_keep)
        if damage_gate is None and isinstance(fused_keep_last, torch.Tensor):
            damage_gate = 1.0 - fused_keep_last

        supp_added = blended_outs.get('supp_query_added')
        supp_source = blended_outs.get('supp_query_source_branch')
        supp_selected_idx = blended_outs.get('supp_query_selected_candidate_idx')
        if isinstance(supp_added, torch.Tensor):
            supp_added = self._v10_to_query_feature(supp_added, bs, nq, device, dtype)[..., 0] > 0.5
        else:
            supp_added = torch.zeros(bs, nq, device=device, dtype=torch.bool)
        if isinstance(supp_source, torch.Tensor):
            supp_source = self._v10_to_query_feature(supp_source, bs, nq, device, dtype)[..., 0].long()
        else:
            supp_source = torch.full((bs, nq), -1, device=device, dtype=torch.long)
        if isinstance(supp_selected_idx, torch.Tensor):
            supp_selected_idx = self._v10_to_query_feature(
                supp_selected_idx, bs, nq, device, dtype)[..., 0].long().clamp(min=0, max=nq - 1)
        else:
            supp_selected_idx = torch.zeros(bs, nq, device=device, dtype=torch.long)

        score_thr = float(getattr(self, 'ev_branch_candidate_score_thr', 0.0) or 0.0)
        per_branch_topk = int(getattr(self, 'ev_branch_candidate_topk_per_sample', 300) or 0)
        if per_branch_topk <= 0 or per_branch_topk > nq:
            per_branch_topk = nq

        for branch in branches:
            branch_global_idx = modalities.index(branch)
            logits = source_cls_logits[branch].float()
            num_cls = int(logits.shape[-1])
            scores, labels = logits.sigmoid().max(dim=-1)
            valid = scores >= score_thr
            ranking_score = torch.where(valid, scores, scores.new_full(scores.shape, -1.0))
            branch_score, branch_topidx = ranking_score.topk(per_branch_topk, dim=1)
            branch_rank = torch.arange(
                per_branch_topk, device=device, dtype=torch.long).view(1, per_branch_topk).expand(bs, -1)
            full_query_idx = torch.arange(
                nq, device=device, dtype=torch.long).view(1, nq).expand(bs, -1)

            score_list.append(branch_score)
            valid_list.append(self._ev_branch_candidate_gather(valid.float(), branch_topidx))
            local_label = self._ev_branch_candidate_gather(labels.float(), branch_topidx).long()
            label_list.append(local_label + class_offset)
            local_label_list.append(local_label)
            task_id_list.append(
                local_label.new_full(local_label.shape, int(task_id_int)))
            global_label_list.append(local_label + class_offset)
            task_class_offset_list.append(
                local_label.new_full(local_label.shape, int(class_offset)))
            branch_idx_list.append(
                local_label.new_full(local_label.shape, int(branch_global_idx)))
            query_idx = self._ev_branch_candidate_gather(full_query_idx.float(), branch_topidx).long()
            query_idx_list.append(query_idx)
            branch_rank_list.append(branch_rank)
            candidate_uid_list.append(
                task_id_list[-1].long() * 100000000 +
                branch_idx_list[-1].long() * 1000000 +
                query_idx.long() * max(num_cls, 1) +
                local_label.long())
            selected_query = torch.zeros(bs, nq, device=device, dtype=torch.float32)
            policy_mask = (supp_added & (supp_source == int(branch_global_idx))).float()
            selected_query.scatter_add_(1, supp_selected_idx, policy_mask)
            selected_for_policy = self._ev_branch_candidate_gather(
                (selected_query > 0.5).float(), branch_topidx).long()
            admitted_policy_list.append(selected_for_policy)
            action_label_list.append(
                torch.where(
                    selected_for_policy > 0,
                    selected_for_policy.new_full(selected_for_policy.shape, int(branch_global_idx)),
                    selected_for_policy.new_full(selected_for_policy.shape, -1)))
            action_space_list.append(
                torch.where(
                    selected_for_policy > 0,
                    selected_for_policy.new_ones(selected_for_policy.shape),
                    selected_for_policy.new_full(selected_for_policy.shape, -1)))
            selected_cls_list.append(selected_for_policy)
            selected_center_list.append(selected_for_policy)
            selected_height_list.append(selected_for_policy)

            # EDLHead has already converted center/height gamma to absolute
            # lidar coordinates before PME ensemble routing.
            center_xy_abs = branch_feature(source_center, branch, width=2)
            height_abs = branch_feature(source_height, branch, width=1)
            center_norm = torch.cat([
                (center_xy_abs[..., 0:1] - pc[0]) / torch.clamp(pc[3] - pc[0], min=1e-6),
                (center_xy_abs[..., 1:2] - pc[1]) / torch.clamp(pc[4] - pc[1], min=1e-6),
            ], dim=-1)
            height_norm = (
                (height_abs[..., 0:1] - pc[2]) / torch.clamp(pc[5] - pc[2], min=1e-6))
            center_abs = torch.cat([center_xy_abs, height_abs], dim=-1)
            center_abs_list.append(self._ev_branch_candidate_gather(center_abs, branch_topidx))
            center_norm_list.append(self._ev_branch_candidate_gather(center_norm, branch_topidx))
            height_norm_list.append(self._ev_branch_candidate_gather(height_norm, branch_topidx))
            dim_list.append(self._ev_branch_candidate_gather(
                branch_feature(source_dim, branch, width=3), branch_topidx))
            rot_list.append(self._ev_branch_candidate_gather(
                branch_feature(source_rot, branch, width=2), branch_topidx))
            vel_list.append(self._ev_branch_candidate_gather(
                branch_feature(source_vel, branch, width=2), branch_topidx))

            cls_unc = branch_feature(source_cls_unc, branch, width=1)
            center_unc = branch_feature(raw_unc_by_source.get('center', {}), branch, width=1)
            height_unc = branch_feature(raw_unc_by_source.get('height', {}), branch, width=1)
            cls_unc_list.append(self._ev_branch_candidate_gather(cls_unc, branch_topidx))
            center_unc_list.append(self._ev_branch_candidate_gather(center_unc, branch_topidx))
            height_unc_list.append(self._ev_branch_candidate_gather(height_unc, branch_topidx))
            combined_unc_list.append(self._ev_branch_candidate_gather(
                torch.cat([cls_unc, center_unc, height_unc], dim=-1).mean(dim=-1, keepdim=True),
                branch_topidx))

            cls_weight_list.append(self._ev_branch_candidate_gather(
                query_feature(cls_weight_last, branch, default=1.0 if branch == 'fused' else 0.0),
                branch_topidx))
            center_weight_list.append(self._ev_branch_candidate_gather(
                query_feature(center_weight_last, branch, default=1.0 if branch == 'fused' else 0.0),
                branch_topidx))
            height_weight_list.append(self._ev_branch_candidate_gather(
                query_feature(height_weight_last, branch, default=1.0 if branch == 'fused' else 0.0),
                branch_topidx))
            if isinstance(non_fused_probs, torch.Tensor) and branch in ('bev', 'img'):
                nf_idx = 0 if branch == 'bev' else 1
                non_fused_prob_list.append(
                    self._ev_branch_candidate_gather(
                        self._v10_to_query_feature(
                            non_fused_probs[..., nf_idx:nf_idx + 1], bs, nq, device, dtype),
                        branch_topidx))
            else:
                non_fused_prob_list.append(
                    self._ev_branch_candidate_gather(zeros(1), branch_topidx))
            fused_keep_list.append(self._ev_branch_candidate_gather(
                query_feature(fused_keep_last, branch, default=1.0 if branch == 'fused' else 0.0),
                branch_topidx))
            damage_gate_list.append(self._ev_branch_candidate_gather(
                query_feature(damage_gate, branch), branch_topidx))
            bev_damage_list.append(self._ev_branch_candidate_gather(
                query_feature(bev_damage, branch), branch_topidx))
            img_damage_list.append(self._ev_branch_candidate_gather(
                query_feature(img_damage, branch), branch_topidx))
            dual_damage_list.append(self._ev_branch_candidate_gather(
                query_feature(dual_damage, branch), branch_topidx))
            damage_state_list.append(self._ev_branch_candidate_gather(
                query_feature(damage_state, branch).to(torch.long).float(), branch_topidx).long())
            center_source_list.append(self._ev_branch_candidate_gather(
                query_feature(center_source, branch).to(torch.long).float(), branch_topidx).long())

        total_available = len(branches) * nq
        total_logged = sum(item.shape[1] for item in score_list)
        logging_rank = torch.arange(
            total_logged, device=device, dtype=torch.long).view(1, total_logged).expand(bs, -1)

        def flat_cat(values, dim=-1):
            return torch.cat(values, dim=1)

        blended_outs['branch_candidate_schema_version'] = 2.0
        blended_outs['branch_candidate_limit'] = float(total_logged)
        blended_outs['branch_candidate_limit_per_branch'] = float(per_branch_topk)
        blended_outs['branch_candidate_total_available'] = float(total_available)
        blended_outs['branch_candidate_score_threshold'] = float(score_thr)
        blended_outs['branch_candidate_rank'] = logging_rank
        blended_outs['branch_candidate_logging_rank'] = logging_rank
        blended_outs['branch_candidate_branch_rank'] = flat_cat(branch_rank_list).long()
        blended_outs['branch_candidate_score'] = flat_cat(score_list)
        blended_outs['branch_candidate_valid'] = flat_cat(valid_list)
        blended_outs['branch_candidate_label'] = flat_cat(label_list).long()
        blended_outs['branch_candidate_local_label'] = flat_cat(local_label_list).long()
        blended_outs['branch_candidate_global_label'] = flat_cat(global_label_list).long()
        blended_outs['branch_candidate_task_id'] = flat_cat(task_id_list).long()
        blended_outs['branch_candidate_task_class_offset'] = flat_cat(task_class_offset_list).long()
        blended_outs['branch_candidate_uid'] = flat_cat(candidate_uid_list).long()
        blended_outs['branch_candidate_branch_idx'] = flat_cat(branch_idx_list).long()
        blended_outs['branch_candidate_query_idx'] = flat_cat(query_idx_list).long()
        blended_outs['branch_candidate_center_abs'] = flat_cat(center_abs_list)
        blended_outs['branch_candidate_center_norm'] = flat_cat(center_norm_list)
        blended_outs['branch_candidate_height_norm'] = flat_cat(height_norm_list)
        blended_outs['branch_candidate_dim'] = flat_cat(dim_list)
        blended_outs['branch_candidate_rot'] = flat_cat(rot_list)
        blended_outs['branch_candidate_vel'] = flat_cat(vel_list)
        blended_outs['branch_candidate_cls_unc'] = flat_cat(cls_unc_list)
        blended_outs['branch_candidate_center_unc'] = flat_cat(center_unc_list)
        blended_outs['branch_candidate_height_unc'] = flat_cat(height_unc_list)
        blended_outs['branch_candidate_combined_unc'] = flat_cat(combined_unc_list)
        blended_outs['branch_candidate_cls_weight'] = flat_cat(cls_weight_list)
        blended_outs['branch_candidate_center_weight'] = flat_cat(center_weight_list)
        blended_outs['branch_candidate_height_weight'] = flat_cat(height_weight_list)
        blended_outs['branch_candidate_non_fused_prob'] = flat_cat(non_fused_prob_list)
        blended_outs['branch_candidate_fused_keep'] = flat_cat(fused_keep_list)
        blended_outs['branch_candidate_damage_gate'] = flat_cat(damage_gate_list)
        blended_outs['branch_candidate_bev_damage'] = flat_cat(bev_damage_list)
        blended_outs['branch_candidate_img_damage'] = flat_cat(img_damage_list)
        blended_outs['branch_candidate_dual_damage'] = flat_cat(dual_damage_list)
        blended_outs['branch_candidate_damage_state'] = flat_cat(damage_state_list).long()
        blended_outs['branch_candidate_center_source'] = flat_cat(center_source_list).long()
        blended_outs['branch_candidate_admitted_by_current_policy'] = flat_cat(
            admitted_policy_list).long()
        blended_outs['branch_candidate_action_label'] = flat_cat(action_label_list).long()
        blended_outs['branch_candidate_action_space'] = flat_cat(action_space_list).long()
        blended_outs['branch_candidate_selected_for_cls'] = flat_cat(selected_cls_list).long()
        blended_outs['branch_candidate_selected_for_center'] = flat_cat(selected_center_list).long()
        blended_outs['branch_candidate_selected_for_height'] = flat_cat(selected_height_list).long()

    def _v10_build_edlcue_info(
            self,
            blended_outs,
            source_cls_logits,
            source_cls_unc,
            v10_info,
            aux_terms):
        if not self.v10_edlcue_enable:
            return {}

        base_logits_seq = blended_outs.get('cls_logits')
        if not isinstance(base_logits_seq, torch.Tensor) or base_logits_seq.dim() != 4:
            return {}
        base_logits = base_logits_seq[-1].float()
        base_scores = base_logits.sigmoid()
        bs, nq, num_cls = base_scores.shape

        def scalar_feature(value):
            return self._v10_expand_class_feature(
                value, base_scores, detach=self.v10_input_detach)

        cls_alpha = blended_outs.get('cls_alpha')
        cls_beta = blended_outs.get('cls_beta')
        if isinstance(cls_alpha, torch.Tensor) and isinstance(cls_beta, torch.Tensor):
            cls_unc = 1.0 / (
                self._v10_last_decoder_tensor(cls_alpha).float() +
                self._v10_last_decoder_tensor(cls_beta).float()
            ).clamp(min=1e-6)
        else:
            cls_unc = base_scores.new_zeros(base_scores.shape)

        fused_score = self._v10_source_score_feature(source_cls_logits, 'fused', base_scores)
        bev_score = self._v10_source_score_feature(source_cls_logits, 'bev', base_scores)
        img_score = self._v10_source_score_feature(source_cls_logits, 'img', base_scores)
        non_fused_score = torch.maximum(bev_score, img_score)

        v10_info = v10_info or {}
        aux_terms = aux_terms or {}
        p_dmg_bev = v10_info.get('v10_p_dmg_bev')
        p_dmg_img = v10_info.get('v10_p_dmg_img')
        p_dmg_dual = v10_info.get('v10_p_dmg_dual')
        w_fused_cls = v10_info.get('v10_w_fused_cls')
        w_bev_cls = v10_info.get('v10_w_bev_cls')
        w_img_cls = v10_info.get('v10_w_img_cls')

        feature_terms = [
            base_scores.unsqueeze(-1),
            torch.tanh(base_logits / 4.0).unsqueeze(-1),
            cls_unc.clamp(min=0.0, max=1.0).unsqueeze(-1),
            fused_score.unsqueeze(-1),
            bev_score.unsqueeze(-1),
            img_score.unsqueeze(-1),
            non_fused_score.unsqueeze(-1),
            (bev_score - fused_score).abs().unsqueeze(-1),
            (img_score - fused_score).abs().unsqueeze(-1),
            scalar_feature(source_cls_unc.get('fused') if isinstance(source_cls_unc, dict) else None),
            scalar_feature(source_cls_unc.get('bev') if isinstance(source_cls_unc, dict) else None),
            scalar_feature(source_cls_unc.get('img') if isinstance(source_cls_unc, dict) else None),
            scalar_feature(p_dmg_bev),
            scalar_feature(p_dmg_img),
            scalar_feature(p_dmg_dual),
            scalar_feature(w_fused_cls),
            scalar_feature(w_bev_cls),
            scalar_feature(w_img_cls),
            scalar_feature(aux_terms.get('reliability_margin')),
            scalar_feature(aux_terms.get('v9_mean_disagree')),
            scalar_feature(aux_terms.get('v9_tail_prob_bev')),
            scalar_feature(aux_terms.get('v9_tail_prob_img')),
            scalar_feature(aux_terms.get('v9_tail_excess_bev')),
            scalar_feature(aux_terms.get('v9_tail_excess_img')),
        ]
        features = torch.cat(feature_terms, dim=-1)
        if features.shape[-1] < self.v10_edlcue_input_dim:
            pad = features.new_zeros(
                bs, nq, num_cls, self.v10_edlcue_input_dim - features.shape[-1])
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[-1] > self.v10_edlcue_input_dim:
            features = features[..., :self.v10_edlcue_input_dim]
        if self.v10_input_detach:
            features = features.detach()

        max_delta = base_scores.new_tensor(float(self.v10_edlcue_max_delta))
        if self.v10_edlcue_head is not None:
            trainable_raw = self.v10_edlcue_head(features).squeeze(-1)
            trainable_delta = torch.tanh(trainable_raw) * max_delta
        else:
            trainable_raw = base_scores.new_zeros(base_scores.shape)
            trainable_delta = base_scores.new_zeros(base_scores.shape)

        unc_risk = torch.tanh(cls_unc.abs() * 3.0)
        tail_risk = torch.maximum(
            scalar_feature(aux_terms.get('v9_tail_prob_bev')).squeeze(-1).clamp(min=0.0),
            scalar_feature(aux_terms.get('v9_tail_prob_img')).squeeze(-1).clamp(min=0.0))
        damage_risk = torch.maximum(
            torch.maximum(
                scalar_feature(p_dmg_bev).squeeze(-1),
                scalar_feature(p_dmg_img).squeeze(-1)),
            scalar_feature(p_dmg_dual).squeeze(-1)).clamp(min=0.0, max=1.0)
        reliability = scalar_feature(w_fused_cls).squeeze(-1).clamp(min=0.0, max=1.0)
        manual_raw = (
            -0.60 * unc_risk
            -0.60 * tail_risk
            -0.50 * damage_risk
            +0.25 * reliability)
        manual_delta = torch.tanh(manual_raw) * max_delta

        mode = self.v10_edlcue_mode.lower()
        if mode == 'manual':
            delta = manual_delta
        elif mode in ('trainable_plus_manual', 'manual_plus_trainable'):
            delta = (trainable_delta + manual_delta).clamp(
                min=-float(self.v10_edlcue_max_delta),
                max=float(self.v10_edlcue_max_delta))
        elif mode in ('inverse_control', 'inverse'):
            delta = -manual_delta
        elif mode in ('zero', 'off', 'disabled'):
            delta = base_scores.new_zeros(base_scores.shape)
        else:
            delta = trainable_delta

        delta = delta.to(dtype=base_logits_seq.dtype)
        delta_seq = delta.unsqueeze(0).expand_as(base_logits_seq)
        admission_logits = base_logits_seq + delta_seq
        info = {
            'v10_edlcue_admission_logits': admission_logits,
            'v10_edlcue_delta': delta_seq,
            'v10_edlcue_trainable_delta': trainable_delta.to(dtype=base_logits_seq.dtype).unsqueeze(0).expand_as(base_logits_seq),
            'v10_edlcue_manual_delta': manual_delta.to(dtype=base_logits_seq.dtype).unsqueeze(0).expand_as(base_logits_seq),
            'v10_edlcue_active': base_logits_seq.new_tensor(1.0),
            'v10_edlcue_max_delta': base_logits_seq.new_tensor(float(self.v10_edlcue_max_delta)),
            'v10_edlcue_delta_mean': delta.detach().mean(),
            'v10_edlcue_delta_abs_mean': delta.detach().abs().mean(),
            'v10_edlcue_manual_delta_mean': manual_delta.detach().mean(),
            'v10_edlcue_trainable_raw_mean': trainable_raw.detach().mean(),
        }
        return info

    def set_epoch(self, epoch):
        self._current_epoch = int(epoch)

    def _compute_source_uncertainty(self, modality_nig_inputs, source_name, shape_ref, device):
        if modality_nig_inputs is None:
            return torch.zeros(shape_ref[0], shape_ref[1], 1, device=device)

        nig_key = f'{source_name}_nig'
        if nig_key not in modality_nig_inputs or modality_nig_inputs[nig_key] is None:
            return torch.zeros(shape_ref[0], shape_ref[1], 1, device=device)

        return self._compute_single_nig_uncertainty(modality_nig_inputs[nig_key])

    def _transform_uncertainty(self, unc_tensor):
        if self.selector_unc_transform == 'log1p':
            return torch.log1p(unc_tensor.clamp(min=0.0))
        return unc_tensor

    def _normalize_uncertainty(self, unc_tensor):
        if self.selector_unc_norm == 'per_source_query_norm':
            mean = unc_tensor.mean(dim=1, keepdim=True)
            std = unc_tensor.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
            return (unc_tensor - mean) / std
        return unc_tensor

    def _normalize_uncertainty_dict(self, unc_tensors, modalities):
        if self.selector_unc_norm == 'per_query_modality_norm':
            stacked = torch.cat([unc_tensors[modality] for modality in modalities], dim=-1)
            mean = stacked.mean(dim=-1, keepdim=True)
            std = stacked.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-6)
            normalized = (stacked - mean) / std
            return {
                modality: normalized[..., idx:idx + 1]
                for idx, modality in enumerate(modalities)
            }
        return {
            modality: self._normalize_uncertainty(unc_tensors[modality])
            for modality in modalities
        }

    def _bucketize_distance(self, distance_xy):
        return torch.bucketize(distance_xy, self.distance_bin_edges_tensor.to(distance_xy.device))

    def _distance_bin_embed(self, distance_xy):
        bins = self._bucketize_distance(distance_xy)
        return self.distance_bin_embed(bins)

    def _bucketize_v5_distance(self, distance_xy):
        return torch.bucketize(distance_xy, self.v5_distance_bins_tensor.to(distance_xy.device))

    def _bucketize_v6_distance(self, distance_xy):
        return torch.bucketize(distance_xy, self.v6_distance_bins_tensor.to(distance_xy.device))

    def _bucketize_v7_distance(self, distance_xy):
        return torch.bucketize(distance_xy, self.v7_distance_bins_tensor.to(distance_xy.device))

    def _distance_conditioned_stats(self, value_map, distance_map, modalities):
        conditioned = {}
        percentile = {}
        distance_bin_stats = {
            modality: {'near': 0.0, 'mid': 0.0, 'far': 0.0}
            for modality in modalities
        }
        distance_bin_counts = {
            modality: {'near': 0, 'mid': 0, 'far': 0}
            for modality in modalities
        }
        bin_names = ['near', 'mid', 'far']
        for modality in modalities:
            values = value_map[modality]
            dists = distance_map[modality]
            bins = self._bucketize_distance(dists)
            cond = torch.zeros_like(values)
            pct = torch.zeros_like(values)
            for batch_idx in range(values.shape[0]):
                for bin_idx, bin_name in enumerate(bin_names):
                    mask = bins[batch_idx].squeeze(-1) == bin_idx
                    if mask.any():
                        vals = values[batch_idx, mask, 0]
                        mean = vals.mean()
                        std = vals.std(unbiased=False).clamp(min=1e-6)
                        cond[batch_idx, mask, 0] = (vals - mean) / std
                        rank = torch.argsort(torch.argsort(vals))
                        denom = max(vals.shape[0] - 1, 1)
                        pct_vals = rank.float() / denom
                        pct[batch_idx, mask, 0] = pct_vals
                        distance_bin_stats[modality][bin_name] += float(vals.mean().item())
                        distance_bin_counts[modality][bin_name] += 1
            for bin_name in bin_names:
                count = distance_bin_counts[modality][bin_name]
                if count > 0:
                    distance_bin_stats[modality][bin_name] /= count
            conditioned[modality] = cond
            percentile[modality] = pct
        return conditioned, percentile, distance_bin_stats

    def _split_source_tensor(self, source_outs, key, num_queries_per_modality):
        if key not in source_outs:
            return None
        tensor = source_outs[key]
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
            return None
        return list(tensor.split(num_queries_per_modality, dim=2))

    def _blend_source_tensor(
            self,
            source_outs,
            key,
            selector_probs,
            modalities,
            num_queries_per_modality,
            use_selector):
        split_tensors = self._split_source_tensor(source_outs, key, num_queries_per_modality)
        if split_tensors is None:
            return None

        fused_idx = modalities.index("fused")
        if not use_selector:
            return split_tensors[fused_idx]

        nq_fused = num_queries_per_modality[fused_idx]
        for split_tensor in split_tensors:
            if split_tensor.shape[2] != nq_fused:
                raise AssertionError(
                    f"Soft selector requires aligned query counts. "
                    f"Got {split_tensor.shape[2]} vs fused {nq_fused} for key {key}.")

        stacked = torch.stack(split_tensors, dim=-1)
        return (stacked * selector_probs.unsqueeze(-2)).sum(dim=-1)

    def _get_source_tensor_last(self, source_outs, key, modalities, num_queries_per_modality):
        split_tensors = self._split_source_tensor(source_outs, key, num_queries_per_modality)
        if split_tensors is None:
            return None
        return {
            modality: split_tensors[idx][-1]
            for idx, modality in enumerate(modalities)
        }

    def _get_source_cls_unc_last(self, source_outs, modalities, num_queries_per_modality):
        alpha_map = self._get_source_tensor_last(source_outs, 'cls_alpha', modalities, num_queries_per_modality)
        beta_map = self._get_source_tensor_last(source_outs, 'cls_beta', modalities, num_queries_per_modality)
        if alpha_map is None or beta_map is None:
            return None
        return {
            modality: (1.0 / (alpha_map[modality] + beta_map[modality]).clamp(min=1e-6)).mean(dim=-1, keepdim=True)
            for modality in modalities
        }

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

    def _v8_clean_sample_mask(self, img_metas, batch_size, device):
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        if img_metas is None:
            return mask
        for batch_idx in range(batch_size):
            meta = img_metas[batch_idx] if batch_idx < len(img_metas) else {}
            corruption_type, synthetic, _severity = self._v8_meta_corruption_type(meta)
            mask[batch_idx] = (not synthetic) or corruption_type == 'clean'
        return mask

    @torch.no_grad()
    def _v8_update_a4_clean_anchor(self, bev_features, img_features, clean_mask):
        if clean_mask is None or not clean_mask.any():
            return
        if (
            self.v8_router_variant == 'v9_lite_domain_anchored' and
            self.v9_freeze_clean_anchor_epochs > 0 and
            getattr(self, '_current_epoch', 0) < self.v9_freeze_clean_anchor_epochs
        ):
            return
        feature_stack = torch.stack([bev_features, img_features], dim=0).detach()
        feature_stack = torch.log1p(feature_stack.clamp(min=0.0)).float()
        clean_features = feature_stack[:, clean_mask]
        if clean_features.numel() == 0:
            return
        clean_features = clean_features.reshape(2, -1, clean_features.shape[-1])
        batch_mean = clean_features.mean(dim=1)
        batch_var = clean_features.var(dim=1, unbiased=False).clamp(min=self.v8_a4_anchor_eps)
        if float(self.v8_a4_anchor_count.item()) <= 0.0:
            self.v8_a4_anchor_mean.copy_(batch_mean)
            self.v8_a4_anchor_var.copy_(batch_var)
        else:
            momentum = self.v8_a4_anchor_momentum
            self.v8_a4_anchor_mean.mul_(1.0 - momentum).add_(batch_mean, alpha=momentum)
            self.v8_a4_anchor_var.mul_(1.0 - momentum).add_(batch_var, alpha=momentum)
        self.v8_a4_anchor_count.add_(float(clean_features.shape[1]))

    def _v8_a4_anchor_z(self, features, modality_idx):
        log_features = torch.log1p(features.clamp(min=0.0)).float()
        mean = self.v8_a4_anchor_mean[modality_idx].to(device=features.device, dtype=torch.float32)
        std = torch.sqrt(
            self.v8_a4_anchor_var[modality_idx].to(device=features.device, dtype=torch.float32) +
            self.v8_a4_anchor_eps)
        z = (log_features - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        return torch.nan_to_num(z, nan=0.0, posinf=6.0, neginf=-2.0).clamp(min=-2.0, max=6.0).to(features.dtype)

    @torch.no_grad()
    def _v9_update_clean_thresholds(self, a_bev, a_img, d_bev, d_img, bev_z, img_z, clean_mask):
        if not self.update_tau_online:
            return
        if clean_mask is None or not clean_mask.any():
            return
        clean_a_bev = a_bev.detach()[clean_mask].reshape(-1)
        clean_a_img = a_img.detach()[clean_mask].reshape(-1)
        clean_d_bev = d_bev.detach()[clean_mask].reshape(-1)
        clean_d_img = d_img.detach()[clean_mask].reshape(-1)
        if clean_a_bev.numel() > 0:
            q_bev = torch.quantile(clean_a_bev.float(), self.v9_tau_target_quantile)
            q_img = torch.quantile(clean_a_img.float(), self.v9_tau_target_quantile)
            momentum = self.v9_tau_momentum
            self.v9_tau_bev.mul_(1.0 - momentum).add_(q_bev.to(self.v9_tau_bev.device), alpha=momentum)
            self.v9_tau_img.mul_(1.0 - momentum).add_(q_img.to(self.v9_tau_img.device), alpha=momentum)
            # Count EMA update steps, not clean queries. The quantile is
            # computed once per forward over all clean queries in the batch.
            self.v9_tau_update_count.add_(1.0)
        if clean_d_bev.numel() > 0:
            qd_bev = torch.quantile(clean_d_bev.float(), self.v9_clean_distance_threshold_target_quantile)
            qd_img = torch.quantile(clean_d_img.float(), self.v9_clean_distance_threshold_target_quantile)
            momentum = self.v9_clean_distance_threshold_momentum
            self.v9_clean_distance_threshold_bev.mul_(1.0 - momentum).add_(
                qd_bev.to(self.v9_clean_distance_threshold_bev.device), alpha=momentum)
            self.v9_clean_distance_threshold_img.mul_(1.0 - momentum).add_(
                qd_img.to(self.v9_clean_distance_threshold_img.device), alpha=momentum)
            # Same convention as tau_update_count: one closed-loop EMA update
            # per forward that has clean samples.
            self.v9_clean_distance_update_count.add_(1.0)
        if self.v9_emit_clean_q95_z_diagnostic:
            clean_bev_z = torch.relu(bev_z.detach()[clean_mask]).reshape(-1)
            clean_img_z = torch.relu(img_z.detach()[clean_mask]).reshape(-1)
            if clean_bev_z.numel() > 0:
                q95_bev = torch.quantile(clean_bev_z.float(), 0.95)
                q95_img = torch.quantile(clean_img_z.float(), 0.95)
                momentum = self.v9_tau_momentum
                self.v9_clean_q95_z_bev.mul_(1.0 - momentum).add_(
                    q95_bev.to(self.v9_clean_q95_z_bev.device), alpha=momentum)
                self.v9_clean_q95_z_img.mul_(1.0 - momentum).add_(
                    q95_img.to(self.v9_clean_q95_z_img.device), alpha=momentum)

    def _v9_tail_features(self, z):
        z_float = torch.nan_to_num(z.float(), nan=0.0, posinf=6.0, neginf=-2.0)
        centered = (z_float - self.v9_tail_z_threshold) / self.v9_tail_temperature
        tail_prob = torch.sigmoid(centered).mean(dim=(1, 2), keepdim=True)
        tail_excess = (
            F.softplus(centered) * self.v9_tail_temperature
        ).mean(dim=(1, 2), keepdim=True)
        relu_z = torch.relu(z_float).reshape(z_float.shape[0], -1)
        k = max(1, int(relu_z.shape[1] * 0.05))
        cvar = torch.topk(relu_z, k=k, dim=1).values.mean(dim=1).view(-1, 1, 1)
        q95 = torch.quantile(z_float.reshape(z_float.shape[0], -1), 0.95, dim=1).view(-1, 1, 1)
        q95_shift = torch.relu(q95 - z_float.new_tensor(self.v9_q95_z_reference))
        scene_tail = torch.cat([tail_prob, tail_excess, cvar, q95_shift], dim=-1).to(z.dtype)
        scene_tail = scene_tail.expand(-1, z.shape[1], -1)
        query_tail = (
            F.softplus(centered) * self.v9_tail_temperature
        ).mean(dim=-1, keepdim=True).to(z.dtype)
        return scene_tail, query_tail, q95_shift.to(z.dtype), tail_prob.to(z.dtype), tail_excess.to(z.dtype), cvar.to(z.dtype)

    def _blend_selected_source_tensor(
            self,
            source_outs,
            key,
            selector_probs,
            selected_modalities,
            all_modalities,
            num_queries_per_modality,
            use_selector):
        split_tensors = self._split_source_tensor(source_outs, key, num_queries_per_modality)
        if split_tensors is None:
            return None

        tensor_map = {
            modality: split_tensors[idx]
            for idx, modality in enumerate(all_modalities)
        }
        selected_tensors = [tensor_map[modality] for modality in selected_modalities]
        ref_nq = selected_tensors[0].shape[2]
        for split_tensor in selected_tensors:
            if split_tensor.shape[2] != ref_nq:
                raise AssertionError(
                    f"Selector requires aligned query counts. "
                    f"Got {split_tensor.shape[2]} vs ref {ref_nq} for key {key}.")

        if not use_selector:
            return selected_tensors[0]

        stacked = torch.stack(selected_tensors, dim=-1)
        return (stacked * selector_probs.unsqueeze(-2)).sum(dim=-1)

    def _get_single_source_tensor(
            self,
            source_outs,
            key,
            modality,
            all_modalities,
            num_queries_per_modality):
        split_tensors = self._split_source_tensor(source_outs, key, num_queries_per_modality)
        if split_tensors is None:
            return None
        return split_tensors[all_modalities.index(modality)]

    def _blend_damage_gated_tensor(
            self,
            source_outs,
            key,
            non_fused_probs,
            fused_keep,
            non_fused_modalities,
            all_modalities,
            num_queries_per_modality,
            use_selector):
        fused_tensor = self._get_single_source_tensor(
            source_outs, key, 'fused', all_modalities, num_queries_per_modality)
        if not use_selector:
            return fused_tensor
        non_fused_tensor = self._blend_selected_source_tensor(
            source_outs, key, non_fused_probs, non_fused_modalities,
            all_modalities, num_queries_per_modality, use_selector=True)
        if fused_tensor is None:
            return non_fused_tensor
        if non_fused_tensor is None:
            return fused_tensor
        return fused_keep * fused_tensor + (1.0 - fused_keep) * non_fused_tensor

    def _hard_select_source_tensor(
            self,
            source_outs,
            key,
            selected_global_idx,
            all_modalities,
            num_queries_per_modality):
        split_tensors = self._split_source_tensor(source_outs, key, num_queries_per_modality)
        if split_tensors is None:
            return None
        ref_nq = split_tensors[0].shape[2]
        for split_tensor in split_tensors:
            if split_tensor.shape[2] != ref_nq:
                raise AssertionError(
                    f"Hard switch requires aligned query counts. "
                    f"Got {split_tensor.shape[2]} vs ref {ref_nq} for key {key}.")
        stacked = torch.stack(split_tensors, dim=-1)
        gather_idx = selected_global_idx.long().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        gather_idx = gather_idx.expand(stacked.shape[0], -1, -1, stacked.shape[3], 1)
        return torch.gather(stacked, dim=-1, index=gather_idx).squeeze(-1)

    def _load_v5_reference_stats(self, require_stats=False):
        if not self.v5_reference_stats_path:
            if require_stats:
                raise FileNotFoundError("v5_reference_stats_path is required for hard_selective_switch inference.")
            return None
        if (
            self._v5_reference_stats_cache is not None and
            self._v5_reference_stats_cache_path == self.v5_reference_stats_path
        ):
            return self._v5_reference_stats_cache
        if not os.path.isfile(self.v5_reference_stats_path):
            if require_stats:
                raise FileNotFoundError(
                    f"Missing v5 reference stats file: {self.v5_reference_stats_path}")
            return None
        with open(self.v5_reference_stats_path, 'r') as f:
            stats = json.load(f)
        self._v5_reference_stats_cache = stats
        self._v5_reference_stats_cache_path = self.v5_reference_stats_path
        return stats

    def _load_v6_reference_stats(self, require_stats=False):
        if not self.v6_reference_stats_path:
            if require_stats:
                raise FileNotFoundError("v6_reference_stats_path is required for gated_primary_router inference.")
            return None
        if (
            self._v6_reference_stats_cache is not None and
            self._v6_reference_stats_cache_path == self.v6_reference_stats_path
        ):
            return self._v6_reference_stats_cache
        if not os.path.isfile(self.v6_reference_stats_path):
            if require_stats:
                raise FileNotFoundError(
                    f"Missing v6 reference stats file: {self.v6_reference_stats_path}")
            return None
        with open(self.v6_reference_stats_path, 'r') as f:
            stats = json.load(f)
        self._v6_reference_stats_cache = stats
        self._v6_reference_stats_cache_path = self.v6_reference_stats_path
        return stats

    def _load_v7_reference_stats(self, require_stats=False):
        if not self.v7_reference_stats_path:
            if require_stats:
                raise FileNotFoundError(
                    "v7_reference_stats_path is required for modality_damage_aware_router inference.")
            return None
        if (
            self._v7_reference_stats_cache is not None and
            self._v7_reference_stats_cache_path == self.v7_reference_stats_path
        ):
            return self._v7_reference_stats_cache
        if not os.path.isfile(self.v7_reference_stats_path):
            if require_stats:
                raise FileNotFoundError(
                    f"Missing v7 reference stats file: {self.v7_reference_stats_path}")
            return None
        with open(self.v7_reference_stats_path, 'r') as f:
            stats = json.load(f)
        expected_fingerprint = str(
            getattr(self, 'supp_expected_model_fingerprint', '') or '')
        if expected_fingerprint:
            model_provenance = stats.get('model_provenance') or {}
            actual_fingerprint = str(model_provenance.get('fingerprint') or '')
            if actual_fingerprint != expected_fingerprint:
                message = (
                    'UMR supplement reference-stats model fingerprint mismatch: '
                    f'expected={expected_fingerprint}, '
                    f'got={actual_fingerprint or "missing"}, '
                    f'path={self.v7_reference_stats_path}')
                if require_stats or getattr(
                        self, 'supp_require_reference_stats', False):
                    raise RuntimeError(message)
                print(f'[UMR-SUP][WARN] {message}')
                return None
        self._v7_reference_stats_cache = stats
        self._v7_reference_stats_cache_path = self.v7_reference_stats_path
        return stats

    @staticmethod
    def _compute_v5_single_anomaly(value, stat_entry, mode):
        if stat_entry is None:
            return 0.0
        if mode == 'mad':
            median = float(stat_entry.get('median', 0.0))
            mad = max(float(stat_entry.get('mad', 0.0)), 1e-6)
            return max((float(value) - median) / mad, 0.0)
        p95 = float(stat_entry.get('p95', 0.0))
        p99 = max(float(stat_entry.get('p99', p95)), p95 + 1e-6)
        return max((float(value) - p95) / max(p99 - p95, 1e-6), 0.0)

    def _lookup_v5_reference_entry(self, stats, class_id, distance_bin_idx):
        if stats is None:
            return None, 3
        class_key = str(int(class_id))
        bin_key = str(int(distance_bin_idx))
        if self.v5_enable_fallback:
            class_distance = stats.get('class_distance', {})
            if class_key in class_distance and bin_key in class_distance[class_key]:
                return class_distance[class_key][bin_key], 0
            class_global = stats.get('class_global', {})
            if class_key in class_global:
                return class_global[class_key], 1
            if 'global' in stats:
                return stats['global'], 2
            return None, 3
        class_distance = stats.get('class_distance', {})
        if class_key in class_distance and bin_key in class_distance[class_key]:
            return class_distance[class_key][bin_key], 0
        return None, 3

    def _lookup_v6_reference_entry(self, stats, class_id, distance_bin_idx):
        if stats is None:
            return None, 3
        class_key = str(int(class_id))
        bin_key = str(int(distance_bin_idx))
        if self.v6_enable_fallback:
            class_distance = stats.get('class_distance', {})
            if class_key in class_distance and bin_key in class_distance[class_key]:
                return class_distance[class_key][bin_key], 0
            class_global = stats.get('class_global', {})
            if class_key in class_global:
                return class_global[class_key], 1
            if 'global' in stats:
                return stats['global'], 2
            return None, 3
        class_distance = stats.get('class_distance', {})
        if class_key in class_distance and bin_key in class_distance[class_key]:
            return class_distance[class_key][bin_key], 0
        return None, 3

    def _lookup_v7_reference_entry(self, stats, class_id, distance_bin_idx):
        if stats is None:
            return None, 3
        class_key = str(int(class_id))
        bin_key = str(int(distance_bin_idx))
        class_distance = stats.get('class_distance', {})
        if class_key in class_distance and bin_key in class_distance[class_key]:
            return class_distance[class_key][bin_key], 0
        class_global = stats.get('class_global', {})
        if class_key in class_global:
            return class_global[class_key], 1
        if 'global' in stats:
            return stats['global'], 2
        return None, 3

    def _compute_v7_center_anomaly(
            self,
            pred_class,
            fused_center,
            bev_center,
            img_center,
            fused_center_unc,
            bev_center_unc,
            img_center_unc):
        require_stats = (not self.training) and (not self.v7_allow_missing_stats_fallback)
        stats = self._load_v7_reference_stats(require_stats=require_stats)
        if stats is None:
            if (
                not self.training and
                self.v7_allow_missing_stats_fallback and
                not self._v7_missing_stats_warned
            ):
                print(
                    "[EDL-V7] Missing v7 reference stats; "
                    "falling back to soft-selector baseline for this eval run. "
                    "Export v7 reference stats before formal modality_damage_aware_router validation."
                )
                self._v7_missing_stats_warned = True
            return None

        bs, nq = pred_class.shape
        fused_distance_bin = self._bucketize_v7_distance(
            fused_center[..., :2].norm(dim=-1, keepdim=True)).squeeze(-1)
        bev_distance_bin = self._bucketize_v7_distance(
            bev_center[..., :2].norm(dim=-1, keepdim=True)).squeeze(-1)
        img_distance_bin = self._bucketize_v7_distance(
            img_center[..., :2].norm(dim=-1, keepdim=True)).squeeze(-1)

        fused_center_anom = fused_center_unc.new_zeros(bs, nq, 1)
        bev_center_anom = bev_center_unc.new_zeros(bs, nq, 1)
        img_center_anom = img_center_unc.new_zeros(bs, nq, 1)
        fused_fallback_level = pred_class.new_full((bs, nq), 3)
        bev_fallback_level = pred_class.new_full((bs, nq), 3)
        img_fallback_level = pred_class.new_full((bs, nq), 3)

        for batch_idx in range(bs):
            for query_idx in range(nq):
                class_id = pred_class[batch_idx, query_idx].item()
                fused_entry, fused_fallback = self._lookup_v7_reference_entry(
                    stats, class_id, fused_distance_bin[batch_idx, query_idx].item())
                bev_entry, bev_fallback = self._lookup_v7_reference_entry(
                    stats, class_id, bev_distance_bin[batch_idx, query_idx].item())
                img_entry, img_fallback = self._lookup_v7_reference_entry(
                    stats, class_id, img_distance_bin[batch_idx, query_idx].item())
                fused_fallback_level[batch_idx, query_idx] = fused_fallback
                bev_fallback_level[batch_idx, query_idx] = bev_fallback
                img_fallback_level[batch_idx, query_idx] = img_fallback
                if fused_entry is not None:
                    fused_center_anom[batch_idx, query_idx, 0] = self._compute_v5_single_anomaly(
                        fused_center_unc[batch_idx, query_idx, 0].item(),
                        fused_entry.get('fused_center_unc'),
                        'percentile')
                if bev_entry is not None:
                    bev_center_anom[batch_idx, query_idx, 0] = self._compute_v5_single_anomaly(
                        bev_center_unc[batch_idx, query_idx, 0].item(),
                        bev_entry.get('bev_center_unc'),
                        'percentile')
                if img_entry is not None:
                    img_center_anom[batch_idx, query_idx, 0] = self._compute_v5_single_anomaly(
                        img_center_unc[batch_idx, query_idx, 0].item(),
                        img_entry.get('img_center_unc'),
                        'percentile')

        return {
            'pred_class': pred_class,
            'fused_center_unc': fused_center_unc,
            'bev_center_unc': bev_center_unc,
            'img_center_unc': img_center_unc,
            'fused_center_anom': fused_center_anom,
            'bev_center_anom': bev_center_anom,
            'img_center_anom': img_center_anom,
            'fused_distance_bin': fused_distance_bin,
            'bev_distance_bin': bev_distance_bin,
            'img_distance_bin': img_distance_bin,
            'fused_fallback_level': fused_fallback_level,
            'bev_fallback_level': bev_fallback_level,
            'img_fallback_level': img_fallback_level,
        }

    @staticmethod
    def _apply_crg_query_budget(mask, score, max_ratio):
        """Keep only the top anomaly queries per sample under a fixed budget."""
        if max_ratio >= 1.0:
            return mask
        if max_ratio <= 0.0:
            return torch.zeros_like(mask)

        limited = torch.zeros_like(mask)
        num_queries = mask.shape[1]
        max_count = max(1, int(num_queries * max_ratio + 0.999999))
        for batch_idx in range(mask.shape[0]):
            candidate_idx = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)
            if candidate_idx.numel() == 0:
                continue
            if candidate_idx.numel() <= max_count:
                limited[batch_idx, candidate_idx] = True
                continue
            candidate_score = score[batch_idx, candidate_idx]
            top_idx = torch.topk(candidate_score, k=max_count, largest=True).indices
            limited[batch_idx, candidate_idx[top_idx]] = True
        return limited

    def _apply_crg_img_fallback(
            self,
            blended_outs,
            source_outs,
            v7_info,
            modalities,
            num_queries_per_modality,
            telemetry_prefix='crg'):
        bev_anom = v7_info['bev_center_anom'][..., 0]
        img_anom = v7_info['img_center_anom'][..., 0]
        bev_damaged = bev_anom > self.crg_bev_damage_threshold
        img_healthy = img_anom < self.crg_img_healthy_ceiling
        lidar_failure_state = bev_damaged & img_healthy
        img_switch_mask = lidar_failure_state
        if self.crg_no_img_control:
            img_switch_mask = torch.zeros_like(img_switch_mask)
        budget_score = bev_anom - img_anom
        img_switch_mask = self._apply_crg_query_budget(
            img_switch_mask, budget_score, self.crg_max_switch_ratio)
        switch_mask = img_switch_mask

        soft_blend_idx = torch.full_like(v7_info['pred_class'], -1)
        center_source_idx = soft_blend_idx.clone()
        center_source_idx = torch.where(
            switch_mask,
            center_source_idx.new_full(center_source_idx.shape, modalities.index('img')),
            center_source_idx)
        selected_global_idx = center_source_idx.new_full(
            center_source_idx.shape, modalities.index('img'))
        output_mask = switch_mask.unsqueeze(0).unsqueeze(-1)

        if 'cls' in self.crg_switch_heads:
            for key in ('cls_logits', 'cls_alpha', 'cls_beta'):
                candidate_tensor = self._hard_select_source_tensor(
                    source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
                if candidate_tensor is not None and blended_outs.get(key) is not None:
                    blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        if 'center' in self.crg_switch_heads:
            for key in ('center', 'center_nig'):
                candidate_tensor = self._hard_select_source_tensor(
                    source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
                if candidate_tensor is not None and blended_outs.get(key) is not None:
                    blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        if 'height' in self.crg_switch_heads:
            for key in ('height', 'height_nig'):
                candidate_tensor = self._hard_select_source_tensor(
                    source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
                if candidate_tensor is not None and blended_outs.get(key) is not None:
                    blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        for key in ('dim', 'dim_nig', 'rot', 'vel'):
            if key.replace('_nig', '') not in self.crg_switch_heads:
                continue
            candidate_tensor = self._hard_select_source_tensor(
                source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
            if candidate_tensor is not None and blended_outs.get(key) is not None:
                blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        damage_state_idx = torch.zeros_like(v7_info['pred_class'])
        damage_state_idx = torch.where(
            lidar_failure_state,
            damage_state_idx.new_full(damage_state_idx.shape, 1),
            damage_state_idx)

        blended_outs['v7_query_damage_state_idx'] = damage_state_idx
        blended_outs['v7_query_switch_triggered'] = switch_mask.float()
        blended_outs['v7_query_center_source_idx'] = center_source_idx
        blended_outs['v7_query_bev_center_anom'] = v7_info['bev_center_anom']
        blended_outs['v7_query_img_center_anom'] = v7_info['img_center_anom']
        blended_outs['v7_query_fused_center_anom'] = v7_info['fused_center_anom']
        blended_outs['v7_query_img_fallback_prob'] = img_healthy.float().unsqueeze(-1)
        blended_outs['v7_query_img_fallback_margin'] = budget_score.unsqueeze(-1)
        blended_outs['v7_query_bev_fallback_prob'] = torch.zeros_like(
            blended_outs['v7_query_img_fallback_prob'])
        blended_outs['v7_query_bev_fallback_margin'] = torch.zeros_like(
            blended_outs['v7_query_img_fallback_margin'])
        blended_outs['v7_query_reference_distance_bin'] = v7_info['bev_distance_bin']
        blended_outs['v7_query_pred_class'] = v7_info['pred_class']
        blended_outs['v7_query_bev_reference_distance_bin'] = v7_info['bev_distance_bin']
        blended_outs['v7_query_img_reference_distance_bin'] = v7_info['img_distance_bin']
        blended_outs['v7_query_fused_reference_distance_bin'] = v7_info['fused_distance_bin']
        blended_outs['v7_query_bev_reference_fallback_level'] = v7_info['bev_fallback_level']
        blended_outs['v7_query_img_reference_fallback_level'] = v7_info['img_fallback_level']
        blended_outs['v7_query_fused_reference_fallback_level'] = v7_info['fused_fallback_level']
        blended_outs[f'{telemetry_prefix}_bev_damage_ratio'] = float(bev_damaged.float().mean().item())
        blended_outs[f'{telemetry_prefix}_img_healthy_ratio'] = float(img_healthy.float().mean().item())
        blended_outs[f'{telemetry_prefix}_budget_ratio'] = float(self.crg_max_switch_ratio)
        blended_outs[f'{telemetry_prefix}_no_img_control'] = float(
            1.0 if self.crg_no_img_control else 0.0)
        blended_outs[f'{telemetry_prefix}_switch_ratio'] = float(switch_mask.float().mean().item())
        return blended_outs

    def _scrg_scene_score(self, values):
        if self.scrg_scene_metric == 'mean':
            return values.mean(dim=1)
        if self.scrg_scene_metric in ('p90', 'p95', 'p99'):
            quantile = {
                'p90': 0.90,
                'p95': 0.95,
                'p99': 0.99,
            }[self.scrg_scene_metric]
            return torch.quantile(values.float(), quantile, dim=1).to(values.dtype)
        # Default: robust top-k scene score. This is meant to detect scene-level
        # LiDAR degradation without letting a single outlier query dominate.
        k = max(1, int(values.shape[1] * self.scrg_top_fraction + 0.999999))
        return torch.topk(values, k=k, dim=1, largest=True).values.mean(dim=1)

    @staticmethod
    def _scrg_reference_value(scene_stats, metric_name, stat_key, fallback=None):
        if not metric_name or str(metric_name).lower() in ('none', 'null', 'off'):
            return fallback
        if not scene_stats:
            return fallback
        entry = scene_stats.get(metric_name)
        if not entry:
            return fallback
        if stat_key in entry:
            return float(entry[stat_key])
        if stat_key == 'p995' and 'p99' in entry:
            return float(entry['p99'])
        if stat_key == 'p99' and 'p95' in entry:
            return float(entry['p95'])
        return fallback

    @staticmethod
    def _scrg_tensor_value(signal_name, blended_outs, v7_info):
        signal_name = str(signal_name or 'center_unc')
        if signal_name == 'center_unc':
            return None
        if signal_name == 'margin_unc':
            bev = v7_info['bev_center_unc'][..., 0]
            img = v7_info['img_center_unc'][..., 0]
            return bev - img
        if signal_name == 'bev_center_unc':
            return v7_info['bev_center_unc'][..., 0]
        if signal_name == 'img_center_unc':
            return v7_info['img_center_unc'][..., 0]
        if signal_name == 'fused_center_unc':
            return v7_info['fused_center_unc'][..., 0]
        if signal_name in (
                'bev_raw_modality_unc',
                'img_raw_modality_unc',
                'fused_raw_modality_unc'):
            alias = {
                'bev_raw_modality_unc': 'v8_gscrg_raw_unc_bev',
                'img_raw_modality_unc': 'v8_gscrg_raw_unc_img',
                'fused_raw_modality_unc': 'v8_gscrg_raw_unc_fused',
            }[signal_name]
            value = blended_outs.get(alias)
            if isinstance(value, torch.Tensor):
                if value.dim() >= 3 and value.shape[-1] == 1:
                    value = value[..., 0]
                return value
        value = blended_outs.get(signal_name)
        if isinstance(value, torch.Tensor):
            if value.dim() >= 3 and value.shape[-1] == 1:
                value = value[..., 0]
            return value
        return None

    @staticmethod
    def _publish_raw_modality_uncertainty(blended_outs, raw_per_modality_unc):
        """Expose raw modality uncertainty before post-inference guards run."""
        for modality, value in (raw_per_modality_unc or {}).items():
            if isinstance(value, torch.Tensor):
                blended_outs[f'v8_gscrg_raw_unc_{modality}'] = value
        return blended_outs

    def _apply_scrg_img_fallback(
            self,
            blended_outs,
            source_outs,
            v7_info,
            modalities,
            num_queries_per_modality,
            telemetry_prefix='scrg'):
        stats = self._load_v7_reference_stats(require_stats=False)
        scene_stats = (stats or {}).get('scene_global', {})
        metric_suffix = 'mean' if self.scrg_scene_metric == 'mean' else 'top10'
        bev_metric_name = self.scrg_bev_metric_name or f'bev_{metric_suffix}_unc'
        img_metric_name = self.scrg_img_metric_name or f'img_{metric_suffix}_unc'
        aux_metric_name = self.scrg_aux_metric_name or f'margin_{metric_suffix}_unc'

        if self.scrg_bev_signal == 'center_unc':
            bev_raw = v7_info['bev_center_unc'][..., 0]
        else:
            bev_raw = self._scrg_tensor_value(self.scrg_bev_signal, blended_outs, v7_info)
        if self.scrg_img_signal == 'center_unc':
            img_raw = v7_info['img_center_unc'][..., 0]
        else:
            img_raw = self._scrg_tensor_value(self.scrg_img_signal, blended_outs, v7_info)
        aux_raw = self._scrg_tensor_value(self.scrg_aux_signal, blended_outs, v7_info)

        if bev_raw is None:
            bev_raw = v7_info['bev_center_unc'][..., 0]
        if img_raw is None:
            img_raw = v7_info['img_center_unc'][..., 0]
        if aux_raw is None:
            aux_raw = bev_raw - img_raw

        bev_scene_score = self._scrg_scene_score(bev_raw)
        img_scene_score = self._scrg_scene_score(img_raw)
        aux_scene_score = self._scrg_scene_score(aux_raw)

        bev_threshold = self._scrg_reference_value(
            scene_stats, bev_metric_name, self.scrg_bev_threshold_key)
        img_ceiling = self._scrg_reference_value(
            scene_stats, img_metric_name, self.scrg_img_ceiling_key)
        aux_threshold = self._scrg_reference_value(
            scene_stats, aux_metric_name, self.scrg_margin_threshold_key)

        aux_required = bool(aux_metric_name) and str(aux_metric_name).lower() not in ('none', 'null', 'off')
        if bev_threshold is None or img_ceiling is None or (aux_required and aux_threshold is None):
            scene_gate = torch.zeros_like(bev_scene_score, dtype=torch.bool)
            bev_threshold_value = float('nan')
            img_ceiling_value = float('nan')
            aux_threshold_value = float('nan')
        else:
            bev_threshold_value = bev_threshold * self.scrg_bev_threshold_scale
            img_ceiling_value = img_ceiling * self.scrg_img_ceiling_scale
            aux_threshold_value = (
                aux_threshold * self.scrg_margin_threshold_scale
                if aux_threshold is not None else float('nan'))
            scene_gate = (
                (bev_scene_score > bev_scene_score.new_tensor(bev_threshold_value)) &
                (img_scene_score < img_scene_score.new_tensor(img_ceiling_value))
            )
            if aux_required:
                scene_gate = scene_gate & (
                    aux_scene_score > aux_scene_score.new_tensor(aux_threshold_value))

        if aux_required and self.scrg_aux_signal == 'margin_unc':
            query_candidate = aux_raw > 0.0
            budget_score = aux_raw
        else:
            query_candidate = torch.ones_like(bev_raw, dtype=torch.bool)
            budget_score = bev_raw
        switch_mask = scene_gate.unsqueeze(1) & query_candidate
        switch_mask = self._apply_crg_query_budget(
            switch_mask, budget_score, self.scrg_max_switch_ratio)
        if self.scrg_mode == 'flag':
            switch_mask = torch.zeros_like(switch_mask)

        soft_blend_idx = torch.full_like(v7_info['pred_class'], -1)
        center_source_idx = soft_blend_idx.clone()
        center_source_idx = torch.where(
            switch_mask,
            center_source_idx.new_full(center_source_idx.shape, modalities.index('img')),
            center_source_idx)
        selected_global_idx = center_source_idx.new_full(
            center_source_idx.shape, modalities.index('img'))
        output_mask = switch_mask.unsqueeze(0).unsqueeze(-1)

        if 'cls' in self.scrg_switch_heads:
            for key in ('cls_logits', 'cls_alpha', 'cls_beta'):
                candidate_tensor = self._hard_select_source_tensor(
                    source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
                if candidate_tensor is not None and blended_outs.get(key) is not None:
                    blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        if 'center' in self.scrg_switch_heads:
            for key in ('center', 'center_nig'):
                candidate_tensor = self._hard_select_source_tensor(
                    source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
                if candidate_tensor is not None and blended_outs.get(key) is not None:
                    blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        if 'height' in self.scrg_switch_heads:
            for key in ('height', 'height_nig'):
                candidate_tensor = self._hard_select_source_tensor(
                    source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
                if candidate_tensor is not None and blended_outs.get(key) is not None:
                    blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        for key in ('dim', 'dim_nig', 'rot', 'vel'):
            if key.replace('_nig', '') not in self.scrg_switch_heads:
                continue
            candidate_tensor = self._hard_select_source_tensor(
                source_outs, key, selected_global_idx, modalities, num_queries_per_modality)
            if candidate_tensor is not None and blended_outs.get(key) is not None:
                blended_outs[key] = torch.where(output_mask, candidate_tensor, blended_outs[key])

        damage_state_idx = torch.zeros_like(v7_info['pred_class'])
        damage_state_idx = torch.where(
            scene_gate.unsqueeze(1),
            damage_state_idx.new_full(damage_state_idx.shape, 1),
            damage_state_idx)

        blended_outs['v7_query_damage_state_idx'] = damage_state_idx
        blended_outs['v7_query_switch_triggered'] = switch_mask.float()
        blended_outs['v7_query_center_source_idx'] = center_source_idx
        blended_outs['v7_query_bev_center_anom'] = v7_info['bev_center_anom']
        blended_outs['v7_query_img_center_anom'] = v7_info['img_center_anom']
        blended_outs['v7_query_fused_center_anom'] = v7_info['fused_center_anom']
        blended_outs['v7_query_img_fallback_prob'] = scene_gate.float().unsqueeze(-1).expand_as(
            v7_info['bev_center_anom'])
        blended_outs['v7_query_img_fallback_margin'] = budget_score.unsqueeze(-1)
        blended_outs['v7_query_bev_fallback_prob'] = torch.zeros_like(
            blended_outs['v7_query_img_fallback_prob'])
        blended_outs['v7_query_bev_fallback_margin'] = torch.zeros_like(
            blended_outs['v7_query_img_fallback_margin'])
        blended_outs['v7_query_reference_distance_bin'] = v7_info['bev_distance_bin']
        blended_outs['v7_query_pred_class'] = v7_info['pred_class']
        blended_outs['v7_query_bev_reference_distance_bin'] = v7_info['bev_distance_bin']
        blended_outs['v7_query_img_reference_distance_bin'] = v7_info['img_distance_bin']
        blended_outs['v7_query_fused_reference_distance_bin'] = v7_info['fused_distance_bin']
        blended_outs['v7_query_bev_reference_fallback_level'] = v7_info['bev_fallback_level']
        blended_outs['v7_query_img_reference_fallback_level'] = v7_info['img_fallback_level']
        blended_outs['v7_query_fused_reference_fallback_level'] = v7_info['fused_fallback_level']
        blended_outs[f'{telemetry_prefix}_scene_gate_ratio'] = float(scene_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_bev_score'] = float(bev_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_img_score'] = float(img_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_margin_score'] = float(aux_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_threshold'] = float(bev_threshold_value)
        blended_outs[f'{telemetry_prefix}_img_ceiling'] = float(img_ceiling_value)
        blended_outs[f'{telemetry_prefix}_margin_threshold'] = float(aux_threshold_value)
        blended_outs[f'{telemetry_prefix}_budget_ratio'] = float(self.scrg_max_switch_ratio)
        blended_outs[f'{telemetry_prefix}_switch_ratio'] = float(switch_mask.float().mean().item())
        return blended_outs

    def _compute_scrg_scene_gate(self, blended_outs, v7_info):
        stats = self._load_v7_reference_stats(require_stats=False)
        scene_stats = (stats or {}).get('scene_global', {})
        metric_suffix = 'mean' if self.scrg_scene_metric == 'mean' else 'top10'
        bev_metric_name = self.scrg_bev_metric_name or f'bev_{metric_suffix}_unc'
        img_metric_name = self.scrg_img_metric_name or f'img_{metric_suffix}_unc'
        aux_metric_name = self.scrg_aux_metric_name or f'margin_{metric_suffix}_unc'

        if self.scrg_bev_signal == 'center_unc':
            bev_raw = v7_info['bev_center_unc'][..., 0]
        else:
            bev_raw = self._scrg_tensor_value(self.scrg_bev_signal, blended_outs, v7_info)
        if self.scrg_img_signal == 'center_unc':
            img_raw = v7_info['img_center_unc'][..., 0]
        else:
            img_raw = self._scrg_tensor_value(self.scrg_img_signal, blended_outs, v7_info)
        aux_raw = self._scrg_tensor_value(self.scrg_aux_signal, blended_outs, v7_info)
        if bev_raw is None:
            bev_raw = v7_info['bev_center_unc'][..., 0]
        if img_raw is None:
            img_raw = v7_info['img_center_unc'][..., 0]
        if aux_raw is None:
            aux_raw = bev_raw - img_raw

        bev_scene_score = self._scrg_scene_score(bev_raw)
        img_scene_score = self._scrg_scene_score(img_raw)
        aux_scene_score = self._scrg_scene_score(aux_raw)
        bev_threshold = self._scrg_reference_value(
            scene_stats, bev_metric_name, self.scrg_bev_threshold_key)
        img_ceiling = self._scrg_reference_value(
            scene_stats, img_metric_name, self.scrg_img_ceiling_key)
        aux_threshold = self._scrg_reference_value(
            scene_stats, aux_metric_name, self.scrg_margin_threshold_key)
        aux_required = bool(aux_metric_name) and str(aux_metric_name).lower() not in ('none', 'null', 'off')
        if bev_threshold is None or img_ceiling is None or (aux_required and aux_threshold is None):
            scene_gate = torch.zeros_like(bev_scene_score, dtype=torch.bool)
            return scene_gate, bev_scene_score, img_scene_score, aux_scene_score, float('nan'), float('nan'), float('nan')

        bev_threshold_value = bev_threshold * self.scrg_bev_threshold_scale
        img_ceiling_value = img_ceiling * self.scrg_img_ceiling_scale
        aux_threshold_value = (
            aux_threshold * self.scrg_margin_threshold_scale
            if aux_threshold is not None else float('nan'))
        scene_gate = (
            (bev_scene_score > bev_scene_score.new_tensor(bev_threshold_value)) &
            (img_scene_score < img_scene_score.new_tensor(img_ceiling_value))
        )
        if aux_required:
            scene_gate = scene_gate & (
                aux_scene_score > aux_scene_score.new_tensor(aux_threshold_value))
        return (
            scene_gate,
            bev_scene_score,
            img_scene_score,
            aux_scene_score,
            float(bev_threshold_value),
            float(img_ceiling_value),
            float(aux_threshold_value),
        )

    def _load_supp_utility_mapper(self, path, expected_target=None, expected_branch_filter=None):
        if not path:
            return None
        def _forbidden_path_markers(value):
            lower = str(value or '').lower().replace('\\', '/')
            markers = []
            for marker in (
                    'multicorrupt_results', 'targeted_probe', 'nuscenes_infos_val',
                    'corrupted-val', 'corrupted_val', 'corrupted/val',
                    'val_corrupt', 'val_clean', 'validation'):
                if marker in lower:
                    markers.append(marker)
            parts = [part for part in lower.split('/') if part]
            if 'val' in parts:
                markers.append('val')
            return markers
        mapper_path_markers = _forbidden_path_markers(path)
        if mapper_path_markers:
            print(
                f"[UMR-SUP][WARN] Utility mapper path is validation/evaluation-derived; "
                f"disabling: {path}: {mapper_path_markers}")
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[UMR-SUP][WARN] Failed to load utility mapper {path}: {exc}")
            return None
        features = payload.get('features', [])
        weights = payload.get('model', {}).get('weights', [])
        means = payload.get('normalization', {}).get('mean', [])
        stds = payload.get('normalization', {}).get('std', [])
        if not features or not weights or len(features) != len(weights):
            print(f"[UMR-SUP][WARN] Invalid utility mapper schema: {path}")
            return None
        if len(means) != len(features) or len(stds) != len(features):
            print(f"[UMR-SUP][WARN] Invalid utility mapper normalization: {path}")
            return None
        if payload.get('debug_only') or not payload.get('formal_calibration_safe', False):
            print(f"[UMR-SUP][WARN] Utility mapper is not formal-safe; disabling: {path}")
            return None
        if payload.get('unsafe_features'):
            print(f"[UMR-SUP][WARN] Utility mapper uses unsafe features; disabling: {path}")
            return None
        target = str(payload.get('target', 'unknown'))
        if expected_target and target != expected_target:
            print(
                f"[UMR-SUP][WARN] Utility mapper target mismatch; disabling: "
                f"{path}: expected={expected_target}, got={target}")
            return None
        if expected_branch_filter is not None:
            branch_filter = str(payload.get('branch_filter', '') or '')
            provenance_branch_filter = str(
                (payload.get('label_provenance') or {}).get('branch_filter', '') or '')
            if branch_filter != expected_branch_filter:
                print(
                    f"[UMR-SUP][WARN] Utility mapper branch filter mismatch; disabling: "
                    f"{path}: expected={expected_branch_filter}, got={branch_filter or 'all'}")
                return None
            if provenance_branch_filter != expected_branch_filter:
                print(
                    f"[UMR-SUP][WARN] Utility mapper provenance branch filter mismatch; "
                    f"disabling: {path}: expected={expected_branch_filter}, "
                    f"got={provenance_branch_filter or 'all'}")
                return None
        if payload.get('forbidden_input_markers'):
            print(
                f"[UMR-SUP][WARN] Utility mapper has forbidden input markers; "
                f"disabling: {path}: {payload.get('forbidden_input_markers')}")
            return None
        provenance = payload.get('label_provenance') or {}
        if not isinstance(provenance, dict) or not provenance:
            print(f"[UMR-SUP][WARN] Utility mapper lacks label provenance; disabling: {path}")
            return None
        if provenance.get('formal_source_kind') != 'train_or_train_oc':
            print(
                f"[UMR-SUP][WARN] Utility mapper provenance is not train/train-O.C; "
                f"disabling: {path}: {provenance.get('formal_source_kind')}")
            return None
        if not provenance.get('labels_content_sha256'):
            print(f"[UMR-SUP][WARN] Utility mapper lacks label content hash; disabling: {path}")
            return None
        if provenance.get('forbidden_input_markers'):
            print(
                f"[UMR-SUP][WARN] Utility mapper provenance has forbidden markers; "
                f"disabling: {path}: {provenance.get('forbidden_input_markers')}")
            return None
        expected_fingerprint = str(
            getattr(self, 'supp_expected_model_fingerprint', '') or '')
        if expected_fingerprint:
            model_provenance = payload.get('model_provenance') or {}
            actual_fingerprint = str(model_provenance.get('fingerprint') or '')
            if actual_fingerprint != expected_fingerprint:
                print(
                    f"[UMR-SUP][WARN] Utility mapper model fingerprint mismatch; "
                    f"disabling: {path}: expected={expected_fingerprint}, "
                    f"got={actual_fingerprint or 'missing'}")
                return None
        for key in (
                'labels_path', 'labels_abspath', 'labels_realpath',
                'label_root', 'infos', 'prediction_log_root'):
            markers = _forbidden_path_markers(provenance.get(key))
            if markers:
                print(
                    f"[UMR-SUP][WARN] Utility mapper provenance path is forbidden; "
                    f"disabling: {path}: {key}={markers}")
                return None
        unsupported_features = [
            feature for feature in features
            if feature not in self.SUPP_RUNTIME_FEATURES
        ]
        if unsupported_features:
            print(
                f"[UMR-SUP][WARN] Utility mapper has unsupported runtime features; "
                f"disabling: {path}: {unsupported_features}")
            return None
        return {
            'features': list(features),
            'weights': [float(v) for v in weights],
            'bias': float(payload.get('model', {}).get('bias', 0.0)),
            'mean': [float(v) for v in means],
            'std': [float(v) if abs(float(v)) > 1e-8 else 1.0 for v in stds],
            'target': target,
            'path': path,
        }

    def _load_supp_scene_mapper(self, path):
        if not path:
            return None
        def _forbidden_path_markers(value):
            lower = str(value or '').lower().replace('\\', '/')
            markers = []
            for marker in (
                    'multicorrupt_results', 'targeted_probe', 'nuscenes_infos_val',
                    'corrupted-val', 'corrupted_val', 'corrupted/val',
                    'val_corrupt', 'val_clean', 'validation'):
                if marker in lower:
                    markers.append(marker)
            parts = [part for part in lower.split('/') if part]
            if 'val' in parts:
                markers.append('val')
            return markers
        if _forbidden_path_markers(path):
            print(
                f"[UMR-SUP][WARN] Scene mapper path is validation/evaluation-derived; "
                f"disabling: {path}")
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[UMR-SUP][WARN] Failed to load scene mapper {path}: {exc}")
            return None
        features = payload.get('features', [])
        weights = payload.get('model', {}).get('weights', [])
        means = payload.get('normalization', {}).get('mean', [])
        stds = payload.get('normalization', {}).get('std', [])
        if not features or not weights or len(features) != len(weights):
            print(f"[UMR-SUP][WARN] Invalid scene mapper schema: {path}")
            return None
        if len(means) != len(features) or len(stds) != len(features):
            print(f"[UMR-SUP][WARN] Invalid scene mapper normalization: {path}")
            return None
        if payload.get('debug_only') or not payload.get('formal_calibration_safe', False):
            print(f"[UMR-SUP][WARN] Scene mapper is not formal-safe; disabling: {path}")
            return None
        required_meta_keys = (
            'label_summary_path',
            'label_summary_missing',
            'feature_missing_rates',
            'bad_missing_features',
            'unsafe_features',
        )
        if any(key not in payload for key in required_meta_keys):
            print(f"[UMR-SUP][WARN] Scene mapper lacks required formal metadata; disabling: {path}")
            return None
        if not payload.get('label_summary_path') or payload.get('label_summary_missing'):
            print(f"[UMR-SUP][WARN] Scene mapper lacks formal label-summary provenance; disabling: {path}")
            return None
        if payload.get('unsafe_features'):
            print(f"[UMR-SUP][WARN] Scene mapper uses unsafe features; disabling: {path}")
            return None
        if payload.get('bad_missing_features'):
            print(f"[UMR-SUP][WARN] Scene mapper has missing training features; disabling: {path}")
            return None
        if payload.get('target') != 'scene_bev_damage_applicable':
            print(
                f"[UMR-SUP][WARN] Scene mapper target mismatch; disabling: {path}: "
                f"{payload.get('target')}")
            return None
        expected_fingerprint = str(
            getattr(self, 'supp_expected_model_fingerprint', '') or '')
        if expected_fingerprint:
            model_provenance = payload.get('model_provenance') or {}
            actual_fingerprint = str(model_provenance.get('fingerprint') or '')
            if actual_fingerprint != expected_fingerprint:
                print(
                    f"[UMR-SUP][WARN] Scene mapper model fingerprint mismatch; "
                    f"disabling: {path}: expected={expected_fingerprint}, "
                    f"got={actual_fingerprint or 'missing'}")
                return None
            formal_splits = set(model_provenance.get('formal_splits') or ())
            if not formal_splits or not formal_splits.issubset(
                    {'train', 'train_clean', 'train_oc', 'train_online_corrupt'}):
                print(
                    f"[UMR-SUP][WARN] Scene mapper provenance is not train-only; "
                    f"disabling: {path}: {sorted(formal_splits)}")
                return None
            for key in ('source_log_root', 'train_infos'):
                markers = _forbidden_path_markers(model_provenance.get(key))
                if markers:
                    print(
                        f"[UMR-SUP][WARN] Scene mapper provenance path is forbidden; "
                        f"disabling: {path}: {key}={markers}")
                    return None
        return {
            'features': list(features),
            'weights': [float(v) for v in weights],
            'bias': float(payload.get('model', {}).get('bias', 0.0)),
            'mean': [float(v) for v in means],
            'std': [float(v) if abs(float(v)) > 1e-8 else 1.0 for v in stds],
            'target': str(payload.get('target', 'unknown')),
            'path': path,
        }

    def _mapper_score_from_values(self, mapper, values, reference_tensor):
        if mapper is None:
            return None
        safe_values = [
            v if isinstance(v, torch.Tensor) else reference_tensor.new_zeros(reference_tensor.shape[:1])
            for v in values
        ]
        x = torch.stack(
            [v.to(device=reference_tensor.device, dtype=torch.float32) for v in safe_values],
            dim=-1)
        view_shape = [1] * (x.dim() - 1) + [-1]
        mean = x.new_tensor(mapper['mean']).view(*view_shape)
        std = x.new_tensor(mapper['std']).view(*view_shape).clamp(min=1e-8)
        weight = x.new_tensor(mapper['weights']).view(*view_shape)
        score = ((x - mean) / std * weight).sum(dim=-1) + float(mapper['bias'])
        return torch.sigmoid(score)

    def _supp_mapper_score(self, feature_map, reference_tensor, mapper=None):
        mapper = self.supp_utility_mapper if mapper is None else mapper
        if mapper is None:
            return None
        values = []
        for name in mapper['features']:
            tensor = feature_map.get(name)
            if tensor is None:
                tensor = reference_tensor.new_zeros(reference_tensor.shape)
            values.append(tensor.to(device=reference_tensor.device, dtype=torch.float32))
        return self._mapper_score_from_values(mapper, values, reference_tensor)

    def _scene_reduce_tensor(self, tensor, reference_tensor, mode='mean'):
        if tensor is None:
            return None
        quantiles = {
            'p95': 0.95,
            'q05': 0.05,
            'q50': 0.50,
        }
        if mode in quantiles:
            # A scene quantile must be reduced from query-level evidence for
            # each batch item.  Scalars and [B] tensors are already-reduced
            # values and must not be broadcast or returned as if they were a
            # query quantile.  This source-side contract prevents silent
            # train/serve skew in the finite-candidate scene mapper.
            if (
                    not isinstance(tensor, torch.Tensor) or
                    tensor.dim() < 2 or
                    tensor.shape[0] != reference_tensor.shape[0] or
                    tensor.numel() <= 0):
                return None
            out = tensor.to(
                device=reference_tensor.device, dtype=torch.float32)
            if not bool(torch.isfinite(out).all().item()):
                return None
            out = out.reshape(reference_tensor.shape[0], -1)
            if out.shape[1] <= 0:
                return None
            reduced = torch.quantile(
                out, quantiles[mode], dim=1)
            if not bool(torch.isfinite(reduced).all().item()):
                return None
            return reduced
        if not isinstance(tensor, torch.Tensor):
            try:
                value = float(tensor)
            except (TypeError, ValueError):
                return None
            return reference_tensor.new_full(reference_tensor.shape[:1], value)
        out = tensor.to(device=reference_tensor.device, dtype=torch.float32)
        while out.dim() > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.dim() == 0:
            return out.reshape(1).expand(reference_tensor.shape[0])
        if out.dim() == 1:
            if out.shape[0] == reference_tensor.shape[0]:
                return out
            return out.float().mean().reshape(1).expand(reference_tensor.shape[0])
        if out.dim() >= 3:
            out = out.reshape(out.shape[0], -1)
        if mode == 'ratio' or mode == 'mean':
            return out.float().mean(dim=1)
        return out.float().mean(dim=1)

    def _publish_query_semantic_telemetry(
            self, blended_outs, fused_cls, img_cls, bev_cls=None, task_id=0):
        """Expose query-level class confidence diagnostics without changing routing.

        These fields are used by offline diagnostics and by optional
        rule-based supplement guards to test whether image-side semantic
        degradation can separate image/dual corruption from pure BEV corruption.
        """
        def _last_logits(cls_logits):
            if not isinstance(cls_logits, torch.Tensor):
                return None
            logits = cls_logits
            if logits.dim() == 4:
                logits = logits[-1]
            if logits.dim() != 3 or logits.shape[-1] <= 0:
                return None
            return logits.detach().float()

        def _semantic_stats(cls_logits):
            logits = _last_logits(cls_logits)
            if logits is None:
                return None
            probs = logits.sigmoid()
            num_classes = probs.shape[-1]
            top_k = 2 if num_classes > 1 else 1
            top_values, top_classes = torch.topk(probs, k=top_k, dim=-1)
            top1 = top_values[..., 0]
            top2 = (
                top_values[..., 1]
                if top_k > 1 else torch.zeros_like(top1))
            margin = (top1 - top2).clamp(min=0.0)
            denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            class_prob = probs / denom
            entropy = -(
                class_prob.clamp(min=1e-8).log() * class_prob).sum(dim=-1)
            if num_classes > 1:
                entropy = entropy / probs.new_tensor(float(num_classes)).log()
            return {
                'score': top1,
                'margin': margin,
                'entropy': entropy.clamp(min=0.0),
                'class': top_classes[..., 0].to(dtype=torch.long),
            }

        stats = {
            'fused': _semantic_stats(fused_cls),
            'img': _semantic_stats(img_cls),
            'bev': _semantic_stats(bev_cls),
        }
        if stats['fused'] is None or stats['img'] is None:
            return

        reference = stats['fused']['score']
        bs, nq = reference.shape
        device = reference.device
        task_id_int = int(task_id or 0)
        class_offset = (
            int(sum(self.num_classes[:task_id_int]))
            if self.num_classes and task_id_int < len(self.num_classes) else 0)
        qidx = torch.arange(nq, device=device, dtype=reference.dtype).view(1, nq).expand(bs, -1)
        task_tensor = reference.new_full(reference.shape, float(task_id_int))
        offset_tensor = reference.new_full(reference.shape, float(class_offset))
        blended_outs['qsem_query_idx'] = qidx.unsqueeze(-1)
        blended_outs['qsem_task_id'] = task_tensor.unsqueeze(-1)
        blended_outs['qsem_task_class_offset'] = offset_tensor.unsqueeze(-1)

        def _put_query(prefix, key, tensor):
            if isinstance(tensor, torch.Tensor):
                blended_outs[f'qsem_{prefix}_{key}'] = tensor.unsqueeze(-1)

        def _put_scene(name, value):
            if isinstance(value, torch.Tensor):
                blended_outs[f'qsem_{name}'] = float(value.float().mean().item())

        def _put_scene_tensor(name, value):
            if not isinstance(value, torch.Tensor):
                return
            tensor = value.to(device=reference.device, dtype=torch.float32)
            while tensor.dim() > 1 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            if tensor.dim() == 0:
                tensor = tensor.reshape(1).expand(reference.shape[0])
            elif tensor.dim() > 1:
                tensor = tensor.reshape(tensor.shape[0], -1).float().mean(dim=1)
            if tensor.shape[0] != reference.shape[0]:
                tensor = tensor.float().mean().reshape(1).expand(reference.shape[0])
            blended_outs[f'qsem_{name}'] = tensor.unsqueeze(-1)

        def _put_reduced(prefix, key, tensor):
            if not isinstance(tensor, torch.Tensor):
                return
            for suffix, mode in (
                    ('mean', 'mean'),
                    ('q05', 'q05'),
                    ('q50', 'q50'),
                    ('p95', 'p95')):
                reduced = self._scene_reduce_tensor(tensor, reference, mode=mode)
                if reduced is not None:
                    _put_scene(f'{prefix}_{key}_{suffix}', reduced)

        for prefix, item in stats.items():
            if item is None:
                continue
            for key in ('score', 'margin', 'entropy'):
                _put_query(prefix, key, item[key])
                _put_reduced(prefix, key, item[key])
            _put_query(prefix, 'local_class', item['class'].to(dtype=reference.dtype))
            _put_query(prefix, 'global_class', item['class'].to(dtype=reference.dtype) + float(class_offset))

        img = stats['img']
        fused = stats['fused']
        low_score_020 = (img['score'] < 0.20).float()
        low_score_030 = (img['score'] < 0.30).float()
        low_margin_005 = (img['margin'] < 0.05).float()
        low_margin_010 = (img['margin'] < 0.10).float()
        high_entropy_080 = (img['entropy'] > 0.80).float()
        for name, tensor in (
                ('img_low_score020_ratio', low_score_020),
                ('img_low_score030_ratio', low_score_030),
                ('img_low_margin005_ratio', low_margin_005),
                ('img_low_margin010_ratio', low_margin_010),
                ('img_high_entropy080_ratio', high_entropy_080)):
            _put_query(name.rsplit('_ratio', 1)[0], 'flag', tensor)
            reduced = self._scene_reduce_tensor(tensor, reference, mode='mean')
            if reduced is not None:
                _put_scene(name, reduced)

        def _class_disagreement(lhs, rhs):
            if lhs is None or rhs is None:
                return None
            return (lhs['class'] != rhs['class']).float()

        def _highconf_disagreement(lhs, rhs):
            disagree = _class_disagreement(lhs, rhs)
            if disagree is None:
                return None
            highconf = (
                (lhs['score'] >= 0.20) &
                (rhs['score'] >= 0.20))
            return disagree * highconf.float()

        img_fused_disagree = _class_disagreement(img, fused)
        img_fused_highconf_disagree = _highconf_disagreement(img, fused)
        fused_bev_disagree = _class_disagreement(fused, stats['bev'])
        fused_bev_highconf_disagree = _highconf_disagreement(fused, stats['bev'])
        img_bev_disagree = _class_disagreement(img, stats['bev'])
        img_bev_highconf_disagree = _highconf_disagreement(img, stats['bev'])
        for name, tensor in (
                ('img_fused_class_disagree', img_fused_disagree),
                ('img_fused_highconf_disagree', img_fused_highconf_disagree),
                ('fused_bev_class_disagree', fused_bev_disagree),
                ('fused_bev_highconf_disagree', fused_bev_highconf_disagree),
                ('img_bev_class_disagree', img_bev_disagree),
                ('img_bev_highconf_disagree', img_bev_highconf_disagree)):
            if tensor is None:
                continue
            _put_query(name, 'flag', tensor)
            reduced = self._scene_reduce_tensor(tensor, reference, mode='mean')
            if reduced is not None:
                _put_scene(f'{name}_ratio', reduced)

        score_tensors = [
            item['score'] for item in (stats['fused'], stats['bev'], stats['img'])
            if item is not None and isinstance(item.get('score'), torch.Tensor)
        ]
        if score_tensors:
            ranking_score = torch.stack(score_tensors, dim=0).max(dim=0).values

            def _topk_mean(tensor, top_idx):
                if not isinstance(tensor, torch.Tensor):
                    return None
                src = tensor.to(device=reference.device, dtype=torch.float32)
                if src.dim() != 2:
                    src = src.reshape(src.shape[0], -1)
                return torch.gather(src, dim=1, index=top_idx).float().mean(dim=1)

            for topk in (100, 300):
                cap = min(int(topk), int(ranking_score.shape[1]))
                if cap <= 0:
                    continue
                _, top_idx = torch.topk(ranking_score, k=cap, dim=1)
                _put_scene_tensor(f'top{topk}_max_score_mean',
                                  _topk_mean(ranking_score, top_idx))
                _put_scene_tensor(f'top{topk}_query_count',
                                  ranking_score.new_full((bs,), float(cap)))
                for name, tensor in (
                        ('img_score_mean', img['score']),
                        ('img_margin_mean', img['margin']),
                        ('img_entropy_mean', img['entropy']),
                        ('img_low_score030_ratio', low_score_030),
                        ('img_low_margin010_ratio', low_margin_010),
                        ('img_high_entropy080_ratio', high_entropy_080),
                        ('img_fused_class_disagree_ratio', img_fused_disagree),
                        ('img_fused_highconf_disagree_ratio',
                         img_fused_highconf_disagree),
                        ('img_bev_class_disagree_ratio', img_bev_disagree),
                        ('img_bev_highconf_disagree_ratio',
                         img_bev_highconf_disagree)):
                    reduced = _topk_mean(tensor, top_idx)
                    if reduced is not None:
                        _put_scene_tensor(f'top{topk}_{name}', reduced)

        for name, tensor in (
                ('img_score_minus_fused_mean', img['score'] - fused['score']),
                ('img_margin_minus_fused_mean', img['margin'] - fused['margin'])):
            reduced = self._scene_reduce_tensor(tensor, reference, mode='mean')
            if reduced is not None:
                _put_scene(name, reduced)
        if stats['bev'] is not None:
            bev = stats['bev']
            for name, tensor in (
                    ('img_score_minus_bev_mean', img['score'] - bev['score']),
                    ('img_margin_minus_bev_mean', img['margin'] - bev['margin'])):
                reduced = self._scene_reduce_tensor(tensor, reference, mode='mean')
                if reduced is not None:
                    _put_scene(name, reduced)

    def _scene_feature_tensor(
            self, feature_name, blended_outs, v7_info, reference_tensor,
            return_available=False):
        name = str(feature_name or '')
        if name.startswith('scene_'):
            name = name[len('scene_'):]
        fallback = reference_tensor.new_zeros(reference_tensor.shape[:1])

        def _nested_get(container, dotted_name):
            if not isinstance(container, dict) or not dotted_name:
                return None
            current = container
            for part in str(dotted_name).split('.'):
                if not isinstance(current, dict) or part not in current:
                    return None
                current = current[part]
            return current

        aliases = {
            'fused_weight': ('selector_fused_weight_mean', 'mean'),
            'v8_high_damage_query_ratio': ('v8_high_damage_query', 'mean'),
            'v8_center_route_allowed_ratio': ('v8_center_route_allowed', 'mean'),
            'v8_output_delta_center_mean': ('v8_output_delta_center', 'mean'),
            'v8_output_delta_center_p95': ('v8_output_delta_center', 'p95'),
            'v9_center_l1_mean': ('v8_output_delta_center', 'mean'),
            'v9_center_l1_p95': ('v8_output_delta_center', 'p95'),
            # The formal scene mapper is trained from the per-scene q95 of
            # this query-level tensor.  The tensor exists before supplement
            # admission; the same-named scalar telemetry is published only
            # after admission and must never be used as a fallback.
            'v9_fused_keep_unit_q95': ('v9_fused_keep_unit', 'p95'),
            'v9_fused_keep_at_min_ratio': ('v9_fused_keep_at_min', 'mean'),
        }
        if name in aliases:
            src_name, mode = aliases[name]
            value = blended_outs.get(src_name)
            if value is None and isinstance(v7_info, dict):
                value = v7_info.get(src_name)
            if value is None and name == 'fused_weight':
                value = blended_outs.get('v8_fused_keep_mean')
            reduced = self._scene_reduce_tensor(value, reference_tensor, mode=mode)
            if reduced is not None:
                return (reduced, True) if return_available else reduced
            return (fallback, False) if return_available else fallback

        candidates = [(name, 'mean')]
        for suffix, mode in (
                ('_mean', 'mean'),
                ('_p95', 'p95'),
                ('_q05', 'q05'),
                ('_q50', 'q50'),
                ('_ratio', 'mean')):
            if name.endswith(suffix):
                candidates.insert(0, (name[:-len(suffix)], mode))
        for src_name, mode in candidates:
            lookup_names = [src_name]
            if src_name.startswith('distance_bin_abnormality_stats.'):
                lookup_names.append(f'selector_{src_name}')
            value = None
            for lookup_name in lookup_names:
                value = blended_outs.get(lookup_name)
                if value is None and '.' in lookup_name:
                    value = _nested_get(blended_outs, lookup_name)
                if value is None:
                    value = (
                        v7_info.get(lookup_name)
                        if isinstance(v7_info, dict) else None)
                if value is None and '.' in lookup_name:
                    value = _nested_get(v7_info, lookup_name)
                if value is not None:
                    break
            reduced = self._scene_reduce_tensor(value, reference_tensor, mode=mode)
            if reduced is not None:
                return (reduced, True) if return_available else reduced
        return (fallback, False) if return_available else fallback

    def _supp_scene_mapper_score(
            self, blended_outs, v7_info, reference_tensor,
            return_diagnostics=False):
        mapper = self.supp_scene_mapper
        diagnostics = {
            'configured': bool(mapper is not None),
            'expected_count': int(len((mapper or {}).get('features', ()))),
            'available_count': 0,
            'finite_count': 0,
            'missing_features': [],
            'contract_ready': False,
            'q95_available': False,
            'q95_runtime': reference_tensor.new_zeros(
                reference_tensor.shape[:1]),
        }
        if mapper is None:
            return (None, diagnostics) if return_diagnostics else None
        values = []
        missing = []
        batch_size = int(reference_tensor.shape[0])
        for name in mapper['features']:
            value, available = self._scene_feature_tensor(
                name, blended_outs, v7_info, reference_tensor,
                return_available=True)
            values.append(value)
            shape_ok = bool(
                isinstance(value, torch.Tensor) and
                value.dim() == 1 and
                value.shape[0] == batch_size and
                value.numel() == batch_size)
            if available and shape_ok:
                diagnostics['available_count'] += 1
            finite = bool(
                available and shape_ok and
                torch.isfinite(value).all().item())
            if finite:
                diagnostics['finite_count'] += 1
            else:
                missing.append(name)
            if name == 'scene_v9_fused_keep_unit_q95' and finite:
                diagnostics['q95_available'] = True
                diagnostics['q95_runtime'] = value
        diagnostics['missing_features'] = list(missing)
        if missing:
            if not getattr(self, '_supp_scene_missing_features_warned', False):
                print(
                    '[UMR-SUP][WARN] Scene mapper runtime feature contract is '
                    f'incomplete; failing gate closed: missing_or_invalid={missing}')
                self._supp_scene_missing_features_warned = True
            return (None, diagnostics) if return_diagnostics else None
        score = self._mapper_score_from_values(
            mapper, values, reference_tensor)
        score_ok = bool(
            isinstance(score, torch.Tensor) and
            score.dim() == 1 and score.shape[0] == batch_size and
            score.numel() == batch_size and
            torch.isfinite(score).all().item())
        diagnostics['contract_ready'] = score_ok
        if not score_ok:
            if not getattr(self, '_supp_scene_missing_features_warned', False):
                print(
                    '[UMR-SUP][WARN] Scene mapper produced a non-finite or '
                    'wrong-shaped score; failing gate closed')
                self._supp_scene_missing_features_warned = True
            score = None
        if return_diagnostics:
            return score, diagnostics
        return score

    def _supp_runtime_branch_head_weights(
            self, branch_name, modalities, v8_info, blended_outs, reference_tensor):
        """Return the head weights used by branch-candidate label logging.

        Action-utility mappers are trained from ``cls_weight_seq``,
        ``center_weight_seq``, and ``height_weight_seq``. Reconstructing all
        three from a shared fused gate changes the feature contract and can
        saturate a mapper when a trained feature has near-zero variance.
        """
        outputs = {}
        missing = []
        branch_idx = modalities.index(branch_name) if branch_name in modalities else None
        bs, nq = reference_tensor.shape[:2]
        v8_info = v8_info if isinstance(v8_info, dict) else {}
        blended_outs = blended_outs if isinstance(blended_outs, dict) else {}

        for feature_name, sequence_key, runtime_suffix in (
                ('cls_weight', 'cls_weight_seq', 'cls'),
                ('center_weight', 'center_weight_seq', 'center'),
                ('height_weight', 'height_weight_seq', 'height')):
            value = self._v10_last_decoder_tensor(v8_info.get(sequence_key))
            if (
                    isinstance(value, torch.Tensor) and
                    value.dim() == 3 and
                    value.shape[0] in (1, bs) and
                    value.shape[1] in (1, nq) and
                    branch_idx is not None and
                    value.shape[-1] > branch_idx):
                query_value = self._v10_to_query_feature(
                    value, bs, nq, reference_tensor.device, reference_tensor.dtype)
                outputs[feature_name] = query_value[..., branch_idx]
                continue

            runtime_key = f'v10_w_{branch_name}_{runtime_suffix}'
            runtime_value = blended_outs.get(runtime_key)
            if (
                    isinstance(runtime_value, torch.Tensor) and
                    runtime_value.dim() in (2, 3) and
                    runtime_value.shape[0] in (1, bs) and
                    runtime_value.shape[1] in (1, nq)):
                runtime_value = self._v10_to_query_feature(
                    runtime_value, bs, nq,
                    reference_tensor.device, reference_tensor.dtype)
                if runtime_value.shape[-1] >= 1:
                    outputs[feature_name] = runtime_value[..., 0]
                    continue

            outputs[feature_name] = reference_tensor.new_zeros(reference_tensor.shape)
            missing.append(feature_name)
        return outputs, missing

    def _supp_effective_max_add(self, scene_score, scene_gate, num_queries, device):
        base = torch.full(
            (scene_gate.shape[0],),
            float(self.supp_max_add_queries),
            dtype=torch.float32,
            device=device)
        if self.supp_scene_mapper is None or self.supp_scene_budget_max_ratio <= 0.0:
            return base.clamp(min=0.0, max=float(self.supp_max_add_queries))

        low = float(self.supp_scene_mapper_threshold)
        high = float(self.supp_scene_budget_full_threshold)
        if high <= low + 1e-6:
            ramp = scene_gate.float()
        else:
            ramp = ((scene_score - low) / (high - low)).clamp(min=0.0, max=1.0)
        min_ratio = float(self.supp_scene_budget_min_ratio)
        max_ratio = float(self.supp_scene_budget_max_ratio)
        ratio = (min_ratio + (max_ratio - min_ratio) * ramp) * scene_gate.float()
        budget = torch.ceil(ratio * float(num_queries))
        budget = torch.minimum(budget, base)
        return budget.clamp(min=0.0, max=float(self.supp_max_add_queries))

    @staticmethod
    def _supp_apply_scene_damage_guard(
            scene_gate, guard_enabled, guard_mode_code,
            img_damage_gate, bev_damage_gate,
            img_threshold_available, bev_threshold_available,
            fused_damage_gate=None, bev_fused_dominance_gate=None,
            bev_damage_upper_gate=None,
            dual_abstain_gate=None,
            img_abstain_gate=None,
            fused_threshold_available=False,
            dominance_reference_available=False,
            bev_upper_threshold_available=False,
            dual_abstain_available=False,
            img_abstain_available=False):
        """Apply a fail-closed modality-state guard to a scene gate."""
        if not guard_enabled:
            return scene_gate
        if not img_threshold_available:
            return scene_gate & torch.zeros_like(scene_gate)
        if guard_mode_code == 1:
            return scene_gate & (~img_damage_gate)
        if guard_mode_code == 2:
            return scene_gate
        if guard_mode_code == 3:
            if not bev_threshold_available:
                return scene_gate & torch.zeros_like(scene_gate)
            pure_bev_damage = bev_damage_gate & (~img_damage_gate)
            return scene_gate & pure_bev_damage
        if guard_mode_code == 4:
            if not fused_threshold_available or not dominance_reference_available:
                return scene_gate & torch.zeros_like(scene_gate)
            if fused_damage_gate is None or bev_fused_dominance_gate is None:
                return scene_gate & torch.zeros_like(scene_gate)
            return (
                scene_gate & (~img_damage_gate) & (~fused_damage_gate) &
                bev_fused_dominance_gate)
        if guard_mode_code == 5:
            if (
                    not bev_threshold_available or
                    not fused_threshold_available or
                    not dominance_reference_available):
                return scene_gate & torch.zeros_like(scene_gate)
            if (
                    fused_damage_gate is None or
                    bev_fused_dominance_gate is None or
                    bev_damage_upper_gate is None):
                return scene_gate & torch.zeros_like(scene_gate)
            if not bev_upper_threshold_available:
                return scene_gate & torch.zeros_like(scene_gate)
            return (
                scene_gate & bev_damage_gate & (~bev_damage_upper_gate) &
                (~fused_damage_gate) & bev_fused_dominance_gate)
        if guard_mode_code == 6:
            if not fused_threshold_available or not dominance_reference_available:
                return scene_gate & torch.zeros_like(scene_gate)
            if not dual_abstain_available or not img_abstain_available:
                return scene_gate & torch.zeros_like(scene_gate)
            if (
                    fused_damage_gate is None or
                    bev_fused_dominance_gate is None or
                    dual_abstain_gate is None or
                    img_abstain_gate is None):
                return scene_gate & torch.zeros_like(scene_gate)
            return (
                scene_gate & bev_fused_dominance_gate &
                (~fused_damage_gate) & (~dual_abstain_gate) &
                (~img_abstain_gate))
        return scene_gate

    @staticmethod
    def _supp_apply_contextheight_gate(
            scene_gate,
            context_gate,
            height_gate,
            v9_bev_gate,
            context_available,
            height_available,
            v9_bev_available):
        """Apply the replay-matched pure-BEV subset gate.

        The context and height clauses are alternatives. If one signal is
        absent at runtime, the other can still admit a scene; only a missing
        v9-BEV signal or both missing OR-branch signals fails closed.
        """
        if not v9_bev_available or not (context_available or height_available):
            return scene_gate & torch.zeros_like(scene_gate)
        return scene_gate & (context_gate | height_gate) & v9_bev_gate

    def _apply_img_supplement_admission(
            self,
            blended_outs,
            source_outs,
            v7_info,
            modalities,
            num_queries_per_modality,
            raw_unc_by_source=None,
            source_cls_unc=None,
            telemetry_prefix='supp',
            v8_info=None):
        if self.supp_require_reference_stats:
            self._load_v7_reference_stats(require_stats=True)
        if self.supp_max_add_queries <= 0:
            blended_outs[f'{telemetry_prefix}_status_code'] = 1.0
            blended_outs[f'{telemetry_prefix}_noadd_reason_code'] = 1.0
            return blended_outs
        fused_cls = blended_outs.get('cls_logits')
        img_cls = self._get_single_source_tensor(
            source_outs, 'cls_logits', 'img', modalities, num_queries_per_modality)
        fused_center = blended_outs.get('center')
        img_center = self._get_single_source_tensor(
            source_outs, 'center', 'img', modalities, num_queries_per_modality)
        if fused_cls is None or img_cls is None or fused_center is None or img_center is None:
            blended_outs[f'{telemetry_prefix}_status_code'] = 2.0
            blended_outs[f'{telemetry_prefix}_noadd_reason_code'] = 2.0
            return blended_outs
        supp_mode = str(self.supp_mode or 'img_highscore').lower()
        branch_aware_mode = supp_mode in (
            'dual_branch_highscore', 'dual_branch', 'branch_aware', 'best_branch')
        bev_cls = None
        bev_center = None
        if branch_aware_mode and 'bev' in modalities:
            bev_cls = self._get_single_source_tensor(
                source_outs, 'cls_logits', 'bev', modalities, num_queries_per_modality)
            bev_center = self._get_single_source_tensor(
                source_outs, 'center', 'bev', modalities, num_queries_per_modality)
        branch_aware_available = bool(
            branch_aware_mode and bev_cls is not None and bev_center is not None)

        (scene_gate, bev_scene_score, img_scene_score, aux_scene_score,
         bev_threshold_value, img_ceiling_value, aux_threshold_value) = (
            self._compute_scrg_scene_gate(blended_outs, v7_info))

        fused_score, fused_class = fused_cls[-1].sigmoid().max(dim=-1)
        img_score, img_class = img_cls[-1].sigmoid().max(dim=-1)
        batch_size, num_queries = fused_score.shape
        device = fused_score.device
        zero_query = fused_score.new_zeros(fused_score.shape)
        img_source_id = float(modalities.index('img')) if 'img' in modalities else 2.0
        bev_source_id = float(modalities.index('bev')) if 'bev' in modalities else 1.0
        if branch_aware_available:
            bev_score, bev_class = bev_cls[-1].sigmoid().max(dim=-1)
        else:
            bev_score = zero_query
            bev_class = img_class
        scene_mapper_score, scene_mapper_contract = (
            self._supp_scene_mapper_score(
                blended_outs, v7_info, fused_score,
                return_diagnostics=True))
        blended_outs[f'{telemetry_prefix}_scene_mapper_configured'] = float(
            1.0 if scene_mapper_contract['configured'] else 0.0)
        blended_outs[f'{telemetry_prefix}_scene_mapper_feature_expected_count'] = float(
            scene_mapper_contract['expected_count'])
        blended_outs[f'{telemetry_prefix}_scene_mapper_feature_available_count'] = float(
            scene_mapper_contract['available_count'])
        blended_outs[f'{telemetry_prefix}_scene_mapper_feature_finite_count'] = float(
            scene_mapper_contract['finite_count'])
        blended_outs[f'{telemetry_prefix}_scene_mapper_feature_missing_count'] = float(
            len(scene_mapper_contract['missing_features']))
        blended_outs[f'{telemetry_prefix}_scene_mapper_contract_ready'] = float(
            1.0 if scene_mapper_contract['contract_ready'] else 0.0)
        blended_outs[f'{telemetry_prefix}_scene_mapper_q95_available'] = float(
            1.0 if scene_mapper_contract['q95_available'] else 0.0)
        blended_outs[f'{telemetry_prefix}_scene_mapper_q95_runtime_mean'] = float(
            scene_mapper_contract['q95_runtime'].float().mean().item())
        scene_mapper_gate_ratio = float('nan')
        if scene_mapper_score is not None:
            mapper_gate = scene_mapper_score >= float(self.supp_scene_mapper_threshold)
            mode = self.supp_scene_mapper_gate_mode.lower()
            if mode in ('and', 'intersect'):
                scene_gate = scene_gate & mapper_gate
            elif mode in ('or', 'union'):
                scene_gate = scene_gate | mapper_gate
            else:
                scene_gate = mapper_gate
            scene_mapper_gate_ratio = float(mapper_gate.float().mean().item())
        else:
            scene_mapper_score = fused_score.new_zeros((batch_size,))
            if self.supp_scene_mapper is not None:
                # A configured mapper whose runtime feature contract is
                # incomplete must never fall back to the legacy SCRG gate.
                scene_gate = torch.zeros_like(scene_gate)

        guard_mode = str(self.supp_img_damage_guard_mode or 'off').lower()
        guard_mode_code = {
            'off': 0, 'none': 0, 'disabled': 0,
            'suppress': 1, 'drop': 1, 'block': 1,
            'dual_small': 2, 'dual_tiny': 2, 'cap_dual': 2,
            'bev_only_abstain': 3, 'pure_bev': 3, 'bev_only': 3,
            'relative_bev_fused': 4, 'bev_dominant_fused_safe': 4,
            'fused_safe_bev_dominant': 4,
            'relative_bev_fused_composite': 5, 'pure_bev_composite': 5,
            'bev_dominant_fused_safe_composite': 5,
            'relative_bev_fused_dual_abstain': 6,
            'pure_bev_dual_abstain': 6,
            'bev_dominant_dual_abstain': 6,
        }.get(guard_mode, 0)
        guard_enabled = bool(self.supp_img_damage_guard_enable) and guard_mode_code > 0
        scene_gate_pre_guard = scene_gate.clone()
        img_damage_threshold_value = float('nan')
        img_damage_gate = torch.zeros_like(scene_gate)
        img_damage_threshold_available = False
        bev_damage_threshold_value = float('nan')
        bev_damage_gate = torch.zeros_like(scene_gate)
        bev_damage_threshold_available = False
        bev_damage_upper_threshold_value = float('nan')
        bev_damage_upper_gate = torch.zeros_like(scene_gate)
        bev_damage_upper_threshold_available = False
        fused_damage_threshold_value = float('nan')
        fused_damage_gate = torch.zeros_like(scene_gate)
        fused_damage_threshold_available = False
        bev_fused_dominance_gate = torch.zeros_like(scene_gate)
        dominance_reference_available = False
        bev_fused_normalized_ratio = fused_score.new_zeros((batch_size,))
        dual_abstain_score = fused_score.new_zeros((batch_size,))
        dual_abstain_gate = torch.zeros_like(scene_gate)
        dual_abstain_available = False
        img_abstain_score = fused_score.new_zeros((batch_size,))
        img_abstain_gate = torch.zeros_like(scene_gate)
        img_abstain_available = False
        qsem_gate_enabled = bool(self.supp_qsem_gate_enable)
        qsem_disagree_score = fused_score.new_zeros((batch_size,))
        qsem_margin_score = fused_score.new_zeros((batch_size,))
        qsem_disagree_gate = torch.zeros_like(scene_gate)
        qsem_margin_gate = torch.zeros_like(scene_gate)
        qsem_gate = torch.ones_like(scene_gate)
        qsem_disagree_available = False
        qsem_margin_available = False
        contextheight_gate_enabled = bool(self.supp_contextheight_gate_enable)
        contextheight_context_score = fused_score.new_zeros((batch_size,))
        contextheight_height_score = fused_score.new_zeros((batch_size,))
        contextheight_v9_bev_score = fused_score.new_zeros((batch_size,))
        contextheight_context_gate = torch.zeros_like(scene_gate)
        contextheight_height_gate = torch.zeros_like(scene_gate)
        contextheight_v9_bev_gate = torch.zeros_like(scene_gate)
        contextheight_gate = torch.ones_like(scene_gate)
        contextheight_context_available = False
        contextheight_height_available = False
        contextheight_v9_bev_available = False
        scene_mapper_dual_gate = torch.zeros_like(scene_gate)
        img_ceiling_tensor = fused_score.new_tensor(float(img_ceiling_value))
        img_damage_score = img_scene_score
        if bool(torch.isfinite(img_ceiling_tensor).item()):
            img_damage_threshold_available = True
            img_damage_threshold_value = (
                float(img_ceiling_value) * float(self.supp_img_damage_threshold_scale))
            img_damage_gate = (
                img_scene_score >= img_scene_score.new_tensor(img_damage_threshold_value))
        else:
            if guard_enabled:
                img_damage_gate = torch.ones_like(scene_gate)
        if guard_enabled and self.supp_img_damage_signal:
            stats = self._load_v7_reference_stats(require_stats=False)
            scene_stats = (stats or {}).get('scene_global', {})
            img_damage_raw = self._scrg_tensor_value(
                self.supp_img_damage_signal, blended_outs, v7_info)
            if isinstance(img_damage_raw, torch.Tensor):
                if img_damage_raw.dim() >= 3 and img_damage_raw.shape[-1] == 1:
                    img_damage_raw = img_damage_raw[..., 0]
                img_damage_score = self._scrg_scene_score(img_damage_raw)
                img_damage_metric_name = (
                    self.supp_img_damage_metric_name or
                    f'{self.supp_img_damage_signal}_top10')
                img_damage_threshold = self._scrg_reference_value(
                    scene_stats,
                    img_damage_metric_name,
                    self.supp_img_damage_threshold_key)
                if img_damage_threshold is not None:
                    img_damage_threshold_available = True
                    img_damage_threshold_value = (
                        float(img_damage_threshold) *
                        float(self.supp_img_damage_threshold_scale))
                    img_damage_gate = (
                        img_damage_score >= img_damage_score.new_tensor(
                            img_damage_threshold_value))
                else:
                    img_damage_threshold_available = False
                    img_damage_threshold_value = float('nan')
                    img_damage_gate = torch.ones_like(scene_gate)
            else:
                img_damage_threshold_available = False
                img_damage_threshold_value = float('nan')
                img_damage_gate = torch.ones_like(scene_gate)
        bev_threshold_tensor = fused_score.new_tensor(float(bev_threshold_value))
        if bool(torch.isfinite(bev_threshold_tensor).item()):
            bev_damage_threshold_available = True
            bev_damage_threshold_value = (
                float(bev_threshold_value) * float(self.supp_dual_bev_threshold_scale))
            bev_damage_gate = (
                bev_scene_score >= bev_scene_score.new_tensor(bev_damage_threshold_value))
            if self.supp_bev_damage_upper_threshold_scale > 0.0:
                bev_damage_upper_threshold_available = True
                bev_damage_upper_threshold_value = (
                    float(bev_threshold_value) *
                    float(self.supp_bev_damage_upper_threshold_scale))
                bev_damage_upper_gate = (
                    bev_scene_score >= bev_scene_score.new_tensor(
                        bev_damage_upper_threshold_value))
        fused_scene_score = fused_score.new_zeros((batch_size,))
        if guard_enabled and guard_mode_code in (4, 5, 6):
            stats = self._load_v7_reference_stats(require_stats=False)
            scene_stats = (stats or {}).get('scene_global', {})
            metric_suffix = 'mean' if self.scrg_scene_metric == 'mean' else 'top10'
            bev_metric_name = self.scrg_bev_metric_name or f'bev_{metric_suffix}_unc'
            fused_metric_name = (
                self.supp_fused_damage_metric_name or f'fused_{metric_suffix}_unc')
            fused_raw = self._scrg_tensor_value(
                self.supp_fused_damage_signal, blended_outs, v7_info)
            if (
                    fused_raw is None and isinstance(v7_info, dict) and
                    self.supp_fused_damage_signal in ('fused_center_unc', 'center_unc')):
                fused_raw = v7_info.get('fused_center_unc')
        else:
            scene_stats = {}
            bev_metric_name = ''
            fused_metric_name = ''
            fused_raw = None
        fused_signal_available = isinstance(fused_raw, torch.Tensor)
        if fused_signal_available:
            if fused_raw.dim() >= 3 and fused_raw.shape[-1] == 1:
                fused_raw = fused_raw[..., 0]
            fused_scene_score = self._scrg_scene_score(fused_raw)
        fused_threshold = self._scrg_reference_value(
            scene_stats,
            fused_metric_name,
            self.supp_fused_damage_threshold_key)
        if fused_signal_available and fused_threshold is not None:
            fused_damage_threshold_value = (
                float(fused_threshold) * float(self.supp_fused_damage_threshold_scale))
            fused_damage_threshold_available = True
            fused_damage_gate = fused_scene_score >= fused_scene_score.new_tensor(
                fused_damage_threshold_value)
        bev_reference = self._scrg_reference_value(
            scene_stats,
            bev_metric_name,
            self.supp_relative_damage_reference_key)
        fused_reference = self._scrg_reference_value(
            scene_stats,
            fused_metric_name,
            self.supp_relative_damage_reference_key)
        if (
                fused_signal_available and
                bev_reference is not None and fused_reference is not None and
                float(bev_reference) > 1e-12 and float(fused_reference) > 1e-12):
            dominance_reference_available = True
            bev_normalized = bev_scene_score / float(bev_reference)
            fused_normalized = fused_scene_score / float(fused_reference)
            bev_fused_normalized_ratio = bev_normalized / fused_normalized.clamp(min=1e-12)
            bev_fused_dominance_gate = (
                bev_fused_normalized_ratio >= float(self.supp_bev_fused_dominance_ratio))
        if (
                guard_enabled and guard_mode_code == 6 and
                self.supp_dual_abstain_signal and
                self.supp_dual_abstain_threshold >= 0.0):
            dual_score, dual_available = self._scene_feature_tensor(
                self.supp_dual_abstain_signal,
                blended_outs,
                v7_info,
                fused_score,
                return_available=True)
            if dual_available and isinstance(dual_score, torch.Tensor):
                dual_abstain_available = True
                dual_abstain_score = dual_score.to(device=device, dtype=fused_score.dtype)
                dual_abstain_gate = (
                    dual_abstain_score >= dual_abstain_score.new_tensor(
                        float(self.supp_dual_abstain_threshold)))
        if (
                guard_enabled and guard_mode_code == 6 and
                self.supp_img_abstain_signal and
                self.supp_img_abstain_threshold >= 0.0):
            img_score_abstain, img_available = self._scene_feature_tensor(
                self.supp_img_abstain_signal,
                blended_outs,
                v7_info,
                fused_score,
                return_available=True)
            if img_available and isinstance(img_score_abstain, torch.Tensor):
                img_abstain_available = True
                img_abstain_score = img_score_abstain.to(device=device, dtype=fused_score.dtype)
                img_abstain_gate = (
                    img_abstain_score >= img_abstain_score.new_tensor(
                        float(self.supp_img_abstain_threshold)))
        scene_mapper_threshold_tensor = fused_score.new_tensor(
            float(self.supp_dual_scene_mapper_threshold))
        if bool(torch.isfinite(scene_mapper_threshold_tensor).item()):
            scene_mapper_dual_gate = (
                scene_mapper_score >= scene_mapper_threshold_tensor)
        dual_gate_mode = str(self.supp_dual_damage_gate_mode or 'strict_img').lower()
        dual_gate_mode_code = {
            'strict': 0, 'strict_img': 0, 'img': 0, 'img_only': 0,
            'scene_mapper': 1, 'mapper': 1, 'scene_mapper_or_img': 1,
            'mapper_or_img': 1,
            'bev_score': 2, 'bev': 2, 'bev_or_img': 2,
            'mapper_or_bev': 3, 'scene_mapper_or_bev': 3,
            'relaxed': 3, 'relaxed_or_img': 3,
        }.get(dual_gate_mode, 0)
        dual_damage_signal = img_damage_gate
        if dual_gate_mode_code == 1:
            dual_damage_signal = img_damage_gate | scene_mapper_dual_gate
        elif dual_gate_mode_code == 2:
            dual_damage_signal = img_damage_gate | bev_damage_gate
        elif dual_gate_mode_code == 3:
            dual_damage_signal = img_damage_gate | scene_mapper_dual_gate | bev_damage_gate
        dual_damage_gate = scene_gate_pre_guard & dual_damage_signal
        scene_gate = self._supp_apply_scene_damage_guard(
            scene_gate,
            guard_enabled,
            guard_mode_code,
            img_damage_gate,
            bev_damage_gate,
            img_damage_threshold_available,
            bev_damage_threshold_available,
            fused_damage_gate=fused_damage_gate,
            bev_fused_dominance_gate=bev_fused_dominance_gate,
            bev_damage_upper_gate=bev_damage_upper_gate,
            dual_abstain_gate=dual_abstain_gate,
            img_abstain_gate=img_abstain_gate,
            fused_threshold_available=fused_damage_threshold_available,
            dominance_reference_available=dominance_reference_available,
            bev_upper_threshold_available=bev_damage_upper_threshold_available,
            dual_abstain_available=dual_abstain_available,
            img_abstain_available=img_abstain_available)
        if qsem_gate_enabled:
            qsem_gate = torch.ones_like(scene_gate)
            qsem_active = False
            if (
                    self.supp_qsem_disagree_signal and
                    self.supp_qsem_disagree_threshold >= 0.0):
                qsem_active = True
                disagree_score, disagree_available = self._scene_feature_tensor(
                    self.supp_qsem_disagree_signal,
                    blended_outs,
                    v7_info,
                    fused_score,
                    return_available=True)
                if disagree_available and isinstance(disagree_score, torch.Tensor):
                    qsem_disagree_available = True
                    qsem_disagree_score = disagree_score.to(
                        device=device, dtype=fused_score.dtype)
                    qsem_disagree_gate = (
                        qsem_disagree_score >= qsem_disagree_score.new_tensor(
                            float(self.supp_qsem_disagree_threshold)))
                    qsem_gate = qsem_gate & qsem_disagree_gate
                else:
                    qsem_gate = qsem_gate & torch.zeros_like(scene_gate)
            if (
                    self.supp_qsem_margin_signal and
                    self.supp_qsem_margin_threshold >= 0.0):
                qsem_active = True
                margin_score, margin_available = self._scene_feature_tensor(
                    self.supp_qsem_margin_signal,
                    blended_outs,
                    v7_info,
                    fused_score,
                    return_available=True)
                if margin_available and isinstance(margin_score, torch.Tensor):
                    qsem_margin_available = True
                    qsem_margin_score = margin_score.to(
                        device=device, dtype=fused_score.dtype)
                    qsem_margin_gate = (
                        qsem_margin_score >= qsem_margin_score.new_tensor(
                            float(self.supp_qsem_margin_threshold)))
                    qsem_gate = qsem_gate & qsem_margin_gate
                else:
                    qsem_gate = qsem_gate & torch.zeros_like(scene_gate)
            if not qsem_active:
                qsem_gate = torch.zeros_like(scene_gate)
            scene_gate = scene_gate & qsem_gate
        if contextheight_gate_enabled:
            contextheight_gate = torch.zeros_like(scene_gate)
            if (
                    self.supp_contextheight_context_signal and
                    math.isfinite(self.supp_contextheight_context_threshold)):
                context_score, context_available = self._scene_feature_tensor(
                    self.supp_contextheight_context_signal,
                    blended_outs,
                    v7_info,
                    fused_score,
                    return_available=True)
                if context_available and isinstance(context_score, torch.Tensor):
                    contextheight_context_available = True
                    contextheight_context_score = context_score.to(
                        device=device, dtype=fused_score.dtype)
                    contextheight_context_gate = (
                        contextheight_context_score <
                        contextheight_context_score.new_tensor(
                            float(self.supp_contextheight_context_threshold)))
            if (
                    self.supp_contextheight_height_signal and
                    math.isfinite(self.supp_contextheight_height_threshold)):
                height_score, height_available = self._scene_feature_tensor(
                    self.supp_contextheight_height_signal,
                    blended_outs,
                    v7_info,
                    fused_score,
                    return_available=True)
                if height_available and isinstance(height_score, torch.Tensor):
                    contextheight_height_available = True
                    contextheight_height_score = height_score.to(
                        device=device, dtype=fused_score.dtype)
                    contextheight_height_gate = (
                        contextheight_height_score <
                        contextheight_height_score.new_tensor(
                            float(self.supp_contextheight_height_threshold)))
            if (
                    self.supp_contextheight_v9_bev_signal and
                    math.isfinite(self.supp_contextheight_v9_bev_threshold)):
                v9_bev_score, v9_bev_available = self._scene_feature_tensor(
                    self.supp_contextheight_v9_bev_signal,
                    blended_outs,
                    v7_info,
                    fused_score,
                    return_available=True)
                if v9_bev_available and isinstance(v9_bev_score, torch.Tensor):
                    contextheight_v9_bev_available = True
                    contextheight_v9_bev_score = v9_bev_score.to(
                        device=device, dtype=fused_score.dtype)
                    contextheight_v9_bev_gate = (
                        contextheight_v9_bev_score <
                        contextheight_v9_bev_score.new_tensor(
                            float(self.supp_contextheight_v9_bev_threshold)))
            contextheight_gate = self._supp_apply_contextheight_gate(
                scene_gate,
                contextheight_context_gate,
                contextheight_height_gate,
                contextheight_v9_bev_gate,
                contextheight_context_available,
                contextheight_height_available,
                contextheight_v9_bev_available)
            scene_gate = scene_gate & contextheight_gate
        guard_abstain_mask = scene_gate_pre_guard & (~scene_gate)

        effective_max_add = self._supp_effective_max_add(
            scene_mapper_score, scene_gate, num_queries, device)
        if (
            guard_enabled and guard_mode_code == 2 and
            self.supp_dual_damage_budget_ratio > 0.0
        ):
            dual_cap = torch.ceil(
                fused_score.new_full(
                    (batch_size,),
                    float(self.supp_dual_damage_budget_ratio) * float(num_queries)))
            dual_cap = dual_cap.clamp(min=0.0, max=float(self.supp_max_add_queries))
            effective_max_add = torch.where(
                dual_damage_gate,
                torch.minimum(effective_max_add, dual_cap),
                effective_max_add)
        if self.supp_scene_mapper is None:
            dynamic_budget_ratio = fused_score.new_full(
                (batch_size,),
                float(self.supp_max_add_queries) / max(float(num_queries), 1.0))
        else:
            dynamic_budget_ratio = effective_max_add / max(float(num_queries), 1.0)

        def _query_tensor(value, default=None):
            if value is None:
                return zero_query if default is None else default
            if isinstance(value, torch.Tensor):
                out = value
                if out.dim() == 3 and out.shape[-1] == 1:
                    out = out[..., 0]
                if out.dim() == 2:
                    return out.to(device=device, dtype=fused_score.dtype)
            return zero_query if default is None else default

        raw_unc_by_source = raw_unc_by_source or {}
        source_cls_unc = source_cls_unc or {}

        def _query_rank(score):
            if not isinstance(score, torch.Tensor) or score.dim() != 2:
                return zero_query
            order = torch.argsort(score, dim=1, descending=True)
            rank = torch.empty_like(score, dtype=torch.float32)
            rank_values = torch.arange(
                score.shape[1], device=score.device, dtype=torch.float32
            ).view(1, -1).expand_as(score)
            rank.scatter_(1, order, rank_values)
            return rank

        def _branch_scalar(branch_name, fused_value=0.0, bev_value=1.0, img_value=2.0):
            if branch_name == 'fused':
                value = fused_value
            elif branch_name == 'bev':
                value = bev_value
            elif branch_name == 'img':
                value = img_value
            else:
                value = 0.0
            return fused_score.new_full(fused_score.shape, float(value))

        def _non_fused_prob(branch_name):
            probs = blended_outs.get('v8_non_fused_probs')
            if not isinstance(probs, torch.Tensor) or probs.dim() < 3:
                return zero_query
            if branch_name not in ('bev', 'img'):
                return zero_query
            idx = 0 if branch_name == 'bev' else 1
            if probs.shape[-1] <= idx:
                return zero_query
            return probs[..., idx].to(device=device, dtype=fused_score.dtype)

        def _supp_feature_map(branch_name, branch_score, branch_center, mapper=None):
            center_unc = _query_tensor(raw_unc_by_source.get('center', {}).get(branch_name))
            height_unc = _query_tensor(raw_unc_by_source.get('height', {}).get(branch_name))
            cls_unc = _query_tensor(source_cls_unc.get(branch_name))
            combined_unc = (center_unc + height_unc + cls_unc) / 3.0
            branch_rank = _query_rank(branch_score)
            fused_keep = _query_tensor(blended_outs.get('v8_fused_keep'))
            if not bool((fused_keep != 0).any().item()):
                fused_keep = _query_tensor(blended_outs.get('v9_fused_keep_final_cls'))
            non_fused_prob = _non_fused_prob(branch_name)
            if branch_name == 'fused':
                branch_weight = fused_keep
            else:
                branch_weight = (1.0 - fused_keep).clamp(min=0.0, max=1.0) * non_fused_prob
            head_weights, missing_head_weights = self._supp_runtime_branch_head_weights(
                branch_name, modalities, v8_info, blended_outs, fused_score)
            mapper_features = set((mapper or {}).get('features', []))
            required_missing = sorted(mapper_features.intersection(missing_head_weights))
            if required_missing:
                raise RuntimeError(
                    '[UMR-SUP] Action-utility mapper runtime feature contract is '
                    f'incomplete for branch={branch_name}: missing={required_missing}. '
                    'The mapper was trained from v10 branch head weights; pass v8_info '
                    'instead of substituting a shared fused gate.')
            damage_gate = _query_tensor(blended_outs.get('v8_damage_gate'))
            if not bool((damage_gate != 0).any().item()):
                damage_gate = (1.0 - fused_keep).clamp(min=0.0, max=1.0)
            feature_map = {
                'score': branch_score,
                'distance_to_ego': branch_center[-1, ..., :2].norm(p=2, dim=-1),
                'center_unc': center_unc,
                'height_unc': height_unc,
                'cls_unc': cls_unc,
                'combined_unc': combined_unc,
                'branch_idx': _branch_scalar(branch_name),
                'candidate_rank': branch_rank,
                'branch_rank': branch_rank,
                'cls_weight': head_weights.get('cls_weight', branch_weight),
                'center_weight': head_weights.get('center_weight', branch_weight),
                'height_weight': head_weights.get('height_weight', branch_weight),
                'non_fused_prob': non_fused_prob,
                'fused_keep': fused_keep,
                'damage_gate': damage_gate,
                'bev_damage': _query_tensor(blended_outs.get('v8_bev_damage')),
                'img_damage': _query_tensor(blended_outs.get('v8_img_damage')),
                'dual_damage': _query_tensor(blended_outs.get('v8_dual_damage')),
                'branch_is_fused': _branch_scalar(branch_name, fused_value=1.0, bev_value=0.0, img_value=0.0),
                'branch_is_bev': _branch_scalar(branch_name, fused_value=0.0, bev_value=1.0, img_value=0.0),
                'branch_is_img': _branch_scalar(branch_name, fused_value=0.0, bev_value=0.0, img_value=1.0),
            }
            for feature_name in (
                    'v8_a4_route_score_bev', 'v8_a4_route_score_img',
                    'v8_bev_reliability', 'v8_img_reliability', 'v8_reliability_margin',
                    'v9_a_bev', 'v9_a_img',
                    'v9_tail_prob_bev', 'v9_tail_prob_img',
                    'v9_route_score_bev', 'v9_route_score_img'):
                feature_map[feature_name] = _query_tensor(blended_outs.get(feature_name))
            return feature_map

        def _branch_utility_config(branch_name):
            if branch_name == 'img' and self.supp_img_utility_mapper is not None:
                threshold = (
                    self.supp_img_utility_threshold
                    if self.supp_img_utility_threshold >= 0.0
                    else self.supp_utility_threshold
                )
                return self.supp_img_utility_mapper, threshold, self.supp_img_utility_mapper_path
            if branch_name == 'bev' and self.supp_bev_utility_mapper is not None:
                threshold = (
                    self.supp_bev_utility_threshold
                    if self.supp_bev_utility_threshold >= 0.0
                    else self.supp_utility_threshold
                )
                return self.supp_bev_utility_mapper, threshold, self.supp_bev_utility_mapper_path
            return self.supp_utility_mapper, self.supp_utility_threshold, self.supp_utility_mapper_path

        def _utility_gate(branch_name, branch_score, branch_center):
            mapper, threshold, mapper_path = _branch_utility_config(branch_name)
            feature_map = _supp_feature_map(
                branch_name, branch_score, branch_center, mapper=mapper)
            score = self._supp_mapper_score(feature_map, fused_score, mapper=mapper)
            utility_required = (
                bool(mapper_path) and
                threshold > 0.0
            )
            if score is None:
                score = zero_query
                ok = torch.ones_like(branch_score, dtype=torch.bool)
                if utility_required:
                    ok = torch.zeros_like(branch_score, dtype=torch.bool)
            else:
                ok = score >= threshold
            if (
                guard_enabled and guard_mode_code == 2 and
                self.supp_dual_damage_utility_threshold > threshold
            ):
                dual_ok = score >= float(self.supp_dual_damage_utility_threshold)
                ok = torch.where(dual_damage_gate.unsqueeze(1), dual_ok, ok)
            return score, ok

        img_utility_score, img_utility_ok = _utility_gate('img', img_score, img_center)
        if branch_aware_available:
            bev_utility_score, bev_utility_ok = _utility_gate('bev', bev_score, bev_center)
        else:
            bev_utility_score = zero_query
            bev_utility_ok = torch.zeros_like(img_utility_ok)
        fused_slot_utility_score = zero_query
        fused_slot_mapper_enabled = self.supp_fused_slot_utility_mapper is not None
        if fused_slot_mapper_enabled:
            fused_feature_map = _supp_feature_map(
                'fused', fused_score, fused_center,
                mapper=self.supp_fused_slot_utility_mapper)
            slot_score = self._supp_mapper_score(
                fused_feature_map, fused_score, mapper=self.supp_fused_slot_utility_mapper)
            if slot_score is not None:
                fused_slot_utility_score = slot_score
        marginal_enabled = bool(
            self.supp_marginal_utility_enable and fused_slot_mapper_enabled)
        marginal_margin = max(float(self.supp_marginal_utility_margin), 0.0)
        dual_marginal_margin = (
            float(self.supp_dual_damage_marginal_margin)
            if self.supp_dual_damage_marginal_margin >= 0.0
            else marginal_margin
        )
        img_utility_candidate_ratio = float(img_utility_ok.float().mean().item())
        bev_utility_candidate_ratio = float(bev_utility_ok.float().mean().item())
        img_utility_score_mean = float(img_utility_score.float().mean().item())
        bev_utility_score_mean = float(bev_utility_score.float().mean().item())
        fused_slot_utility_score_mean = float(
            fused_slot_utility_score.float().mean().item())
        branch_mapper_enabled_ratio = float(
            1.0 if (self.supp_img_utility_mapper is not None or self.supp_bev_utility_mapper is not None)
            else 0.0)
        utility_required = (
            (
                bool(self.supp_utility_mapper_path) or
                bool(self.supp_img_utility_mapper_path) or
                bool(self.supp_bev_utility_mapper_path)
            ) and
            max(
                float(self.supp_utility_threshold),
                float(self.supp_img_utility_threshold),
                float(self.supp_bev_utility_threshold),
            ) > 0.0
        )

        img_candidate_ok = (img_score >= self.supp_img_score_threshold) & img_utility_ok
        bev_candidate_ok = (bev_score >= self.supp_bev_score_threshold) & bev_utility_ok
        rank_mode = self.supp_rank_mode.lower()

        def _candidate_rank(score, utility):
            if rank_mode in ('utility_product', 'util_product'):
                return score * utility.clamp(min=0.0, max=1.0)
            if rank_mode in ('utility_sum', 'util_sum'):
                return score + float(self.supp_utility_rank_weight) * utility
            if rank_mode in ('semantic_margin', 'img_margin'):
                return score - fused_score
            return score

        img_rank_score = _candidate_rank(img_score, img_utility_score)
        bev_rank_score = _candidate_rank(bev_score, bev_utility_score)
        if branch_aware_available and self.supp_dual_bev_rank_boost > 0.0:
            bev_rank_score = torch.where(
                dual_damage_gate.unsqueeze(1),
                bev_rank_score * float(self.supp_dual_bev_rank_boost),
                bev_rank_score)
        choose_bev = torch.zeros_like(img_candidate_ok)
        if branch_aware_available:
            # Keep SGSUP_B semantics for ordinary BEV-damage scenes. BEV supplement is
            # only opened when the image branch is also scene-damaged, where IMG-only
            # supplement is no longer a clean action.
            choose_bev = (
                dual_damage_gate.unsqueeze(1) & bev_candidate_ok &
                ((bev_rank_score > img_rank_score) | (~img_candidate_ok))
            )
        candidate_score = torch.where(choose_bev, bev_score, img_score)
        candidate_class = torch.where(choose_bev, bev_class, img_class)
        candidate_center = torch.where(
            choose_bev.unsqueeze(0).unsqueeze(-1),
            bev_center if branch_aware_available else img_center,
            img_center)
        candidate_utility_score = torch.where(choose_bev, bev_utility_score, img_utility_score)
        utility_score = candidate_utility_score
        utility_ok = torch.where(choose_bev, bev_utility_ok, img_utility_ok)
        candidate_mask_all = torch.where(choose_bev, bev_candidate_ok, img_candidate_ok)
        rank_score_all = torch.where(choose_bev, bev_rank_score, img_rank_score)
        candidate_source_id = torch.where(
            choose_bev,
            fused_score.new_full(fused_score.shape, bev_source_id),
            fused_score.new_full(fused_score.shape, img_source_id))
        replace_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=device)
        selected_img_idx = torch.zeros((batch_size, num_queries), dtype=torch.long, device=device)
        selected_source_id = torch.full(
            (batch_size, num_queries), img_source_id, dtype=fused_score.dtype, device=device)
        selected_margin_score = torch.zeros((batch_size, num_queries), dtype=fused_score.dtype, device=device)
        selected_slot_utility_score = torch.zeros(
            (batch_size, num_queries), dtype=fused_score.dtype, device=device)
        marginal_checked = 0
        marginal_rejected = 0
        marginal_passed = 0
        blended_outs['supp_query_added'] = replace_mask.float()
        blended_outs['supp_query_selected_candidate_idx'] = selected_img_idx.unsqueeze(-1)
        blended_outs['supp_query_img_score'] = torch.gather(
            img_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_branch_score'] = torch.gather(
            candidate_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_bev_score'] = torch.gather(
            bev_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_fused_score'] = fused_score.unsqueeze(-1)
        blended_outs['supp_query_img_class'] = torch.gather(
            img_class, dim=1, index=selected_img_idx)
        blended_outs['supp_query_branch_class'] = torch.gather(
            candidate_class, dim=1, index=selected_img_idx)
        blended_outs['supp_query_utility_score'] = torch.gather(
            utility_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_source_branch'] = selected_source_id.unsqueeze(-1)
        blended_outs['supp_query_fused_slot_utility_score'] = fused_slot_utility_score.unsqueeze(-1)
        blended_outs['supp_query_marginal_utility_score'] = selected_margin_score.unsqueeze(-1)

        for b in range(batch_size):
            if not bool(scene_gate[b].item()):
                continue
            replace_slot_ok = fused_score[b] <= self.supp_fused_replace_score_ceiling
            replace_order = torch.argsort(fused_score[b], descending=False)
            replace_slots = [int(idx.item()) for idx in replace_order if bool(replace_slot_ok[idx].item())]
            if not replace_slots:
                continue
            max_add = min(int(effective_max_add[b].item()), len(replace_slots), num_queries)
            if max_add <= 0:
                continue
            candidate_mask = candidate_mask_all[b]
            candidate_order = torch.argsort(rank_score_all[b], descending=True)
            selected = []
            per_class_counts = {}
            fused_ref_threshold = self.supp_img_score_threshold
            if branch_aware_available:
                fused_ref_threshold = min(
                    self.supp_img_score_threshold, self.supp_bev_score_threshold)
            fused_ref_mask = fused_score[b] > fused_ref_threshold
            fused_ref_center = fused_center[-1, b]
            fused_ref_class = fused_class[b]
            candidate_center_b = candidate_center[-1, b]
            for idx_tensor in candidate_order:
                idx = int(idx_tensor.item())
                if len(selected) >= max_add:
                    break
                if not bool(candidate_mask[idx].item()):
                    continue
                slot_idx = replace_slots[len(selected)]
                if marginal_enabled:
                    required_margin = (
                        dual_marginal_margin
                        if bool(dual_damage_gate[b].item())
                        else marginal_margin
                    )
                    margin_value = (
                        candidate_utility_score[b, idx] -
                        fused_slot_utility_score[b, slot_idx]
                    )
                    marginal_checked += 1
                    if float(margin_value.item()) < required_margin:
                        marginal_rejected += 1
                        continue
                    marginal_passed += 1
                cls_id = int(candidate_class[b, idx].item())
                if self.supp_per_class_cap > 0 and per_class_counts.get(cls_id, 0) >= self.supp_per_class_cap:
                    continue
                if self.supp_nms_center_dist > 0:
                    same_fused = fused_ref_mask & (fused_ref_class == cls_id)
                    if bool(same_fused.any().item()):
                        dist = (fused_ref_center[same_fused] - candidate_center_b[idx]).norm(p=2, dim=-1)
                        if bool((dist < self.supp_nms_center_dist).any().item()):
                            continue
                    duplicate = False
                    for prev_idx in selected:
                        if int(candidate_class[b, prev_idx].item()) != cls_id:
                            continue
                        dist = (candidate_center_b[prev_idx] - candidate_center_b[idx]).norm(p=2)
                        if float(dist.item()) < self.supp_nms_center_dist:
                            duplicate = True
                            break
                    if duplicate:
                        continue
                selected.append(idx)
                per_class_counts[cls_id] = per_class_counts.get(cls_id, 0) + 1
            for slot, img_idx in zip(replace_slots, selected):
                replace_mask[b, slot] = True
                selected_img_idx[b, slot] = int(img_idx)
                selected_source_id[b, slot] = candidate_source_id[b, img_idx]
                selected_slot_utility_score[b, slot] = fused_slot_utility_score[b, slot]
                selected_margin_score[b, slot] = (
                    candidate_utility_score[b, img_idx] -
                    fused_slot_utility_score[b, slot]
                )

        candidate_ratio_all = float(candidate_mask_all.float().mean().item())
        img_candidate_ratio = float(img_candidate_ok.float().mean().item())
        bev_candidate_ratio = float(bev_candidate_ok.float().mean().item())
        utility_candidate_ratio = float(utility_ok.float().mean().item())
        replace_slot_ratio = float(
            (fused_score <= self.supp_fused_replace_score_ceiling).float().mean().item())
        selected_bev_ratio = float(
            ((selected_source_id == bev_source_id) & replace_mask).float().mean().item())
        selected_img_ratio = float(
            ((selected_source_id == img_source_id) & replace_mask).float().mean().item())
        source_bev_query_ratio = float(choose_bev.float().mean().item())

        def _write_qsem_gate_telemetry():
            blended_outs[f'{telemetry_prefix}_qsem_gate_enabled'] = float(
                1.0 if qsem_gate_enabled else 0.0)
            blended_outs[f'{telemetry_prefix}_qsem_gate_ratio'] = float(
                qsem_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_qsem_disagree_gate_ratio'] = float(
                qsem_disagree_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_qsem_margin_gate_ratio'] = float(
                qsem_margin_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_qsem_disagree_score'] = float(
                qsem_disagree_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_qsem_margin_score'] = float(
                qsem_margin_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_qsem_disagree_threshold'] = float(
                self.supp_qsem_disagree_threshold)
            blended_outs[f'{telemetry_prefix}_qsem_margin_threshold'] = float(
                self.supp_qsem_margin_threshold)
            blended_outs[f'{telemetry_prefix}_qsem_disagree_available'] = float(
                1.0 if qsem_disagree_available else 0.0)
            blended_outs[f'{telemetry_prefix}_qsem_margin_available'] = float(
                1.0 if qsem_margin_available else 0.0)

        def _write_contextheight_gate_telemetry():
            blended_outs[f'{telemetry_prefix}_contextheight_gate_enabled'] = float(
                1.0 if contextheight_gate_enabled else 0.0)
            blended_outs[f'{telemetry_prefix}_contextheight_gate_ratio'] = float(
                contextheight_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_context_gate_ratio'] = float(
                contextheight_context_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_height_gate_ratio'] = float(
                contextheight_height_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_v9_bev_gate_ratio'] = float(
                contextheight_v9_bev_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_context_score'] = float(
                contextheight_context_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_height_score'] = float(
                contextheight_height_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_v9_bev_score'] = float(
                contextheight_v9_bev_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_contextheight_context_threshold'] = float(
                self.supp_contextheight_context_threshold)
            blended_outs[f'{telemetry_prefix}_contextheight_height_threshold'] = float(
                self.supp_contextheight_height_threshold)
            blended_outs[f'{telemetry_prefix}_contextheight_v9_bev_threshold'] = float(
                self.supp_contextheight_v9_bev_threshold)
            blended_outs[f'{telemetry_prefix}_contextheight_context_available'] = float(
                1.0 if contextheight_context_available else 0.0)
            blended_outs[f'{telemetry_prefix}_contextheight_height_available'] = float(
                1.0 if contextheight_height_available else 0.0)
            blended_outs[f'{telemetry_prefix}_contextheight_v9_bev_available'] = float(
                1.0 if contextheight_v9_bev_available else 0.0)

        if not bool(replace_mask.any().item()):
            scene_gate_ratio = float(scene_gate.float().mean().item())
            candidate_ratio = candidate_ratio_all
            noadd_reason = 4.0
            if scene_gate_ratio <= 0.0:
                noadd_reason = (
                    15.0
                    if guard_enabled and guard_mode_code in (3, 4, 5, 6) and
                    bool(guard_abstain_mask.any().item())
                    else 10.0)
            elif replace_slot_ratio <= 0.0:
                noadd_reason = 11.0
            elif candidate_ratio <= 0.0:
                noadd_reason = 13.0 if utility_required and utility_candidate_ratio <= 0.0 else 12.0
            elif marginal_enabled and marginal_checked > 0 and marginal_passed <= 0:
                noadd_reason = 14.0
            blended_outs[f'{telemetry_prefix}_status_code'] = 4.0
            blended_outs[f'{telemetry_prefix}_noadd_reason_code'] = noadd_reason
            blended_outs[f'{telemetry_prefix}_scene_gate_ratio'] = float(scene_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_scene_gate_pre_guard_ratio'] = float(
                scene_gate_pre_guard.float().mean().item())
            blended_outs[f'{telemetry_prefix}_scene_gate_post_guard_ratio'] = float(
                scene_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_img_damage_guard_enabled'] = float(
                1.0 if guard_enabled else 0.0)
            blended_outs[f'{telemetry_prefix}_img_damage_guard_mode_code'] = float(guard_mode_code)
            blended_outs[f'{telemetry_prefix}_img_damage_gate_ratio'] = float(
                img_damage_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_dual_damage_gate_ratio'] = float(
                dual_damage_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_dual_gate_mode_code'] = float(dual_gate_mode_code)
            blended_outs[f'{telemetry_prefix}_scene_mapper_dual_gate_ratio'] = float(
                scene_mapper_dual_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_bev_damage_gate_ratio'] = float(
                bev_damage_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_bev_damage_upper_gate_ratio'] = float(
                bev_damage_upper_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_fused_damage_gate_ratio'] = float(
                fused_damage_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_bev_fused_dominance_gate_ratio'] = float(
                bev_fused_dominance_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_bev_fused_normalized_ratio_mean'] = float(
                bev_fused_normalized_ratio.float().mean().item())
            blended_outs[f'{telemetry_prefix}_dual_abstain_gate_ratio'] = float(
                dual_abstain_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_img_abstain_gate_ratio'] = float(
                img_abstain_gate.float().mean().item())
            blended_outs[f'{telemetry_prefix}_dual_abstain_score'] = float(
                dual_abstain_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_img_abstain_score'] = float(
                img_abstain_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_dual_abstain_threshold'] = float(
                self.supp_dual_abstain_threshold)
            blended_outs[f'{telemetry_prefix}_img_abstain_threshold'] = float(
                self.supp_img_abstain_threshold)
            blended_outs[f'{telemetry_prefix}_dual_abstain_available'] = float(
                1.0 if dual_abstain_available else 0.0)
            blended_outs[f'{telemetry_prefix}_img_abstain_available'] = float(
                1.0 if img_abstain_available else 0.0)
            blended_outs[f'{telemetry_prefix}_img_damage_threshold'] = float(
                img_damage_threshold_value)
            blended_outs[f'{telemetry_prefix}_img_damage_score'] = float(
                img_damage_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_bev_damage_threshold'] = float(
                bev_damage_threshold_value)
            blended_outs[f'{telemetry_prefix}_bev_damage_upper_threshold'] = float(
                bev_damage_upper_threshold_value)
            blended_outs[f'{telemetry_prefix}_img_damage_threshold_available'] = float(
                1.0 if img_damage_threshold_available else 0.0)
            blended_outs[f'{telemetry_prefix}_bev_damage_threshold_available'] = float(
                1.0 if bev_damage_threshold_available else 0.0)
            blended_outs[f'{telemetry_prefix}_bev_damage_upper_threshold_available'] = float(
                1.0 if bev_damage_upper_threshold_available else 0.0)
            blended_outs[f'{telemetry_prefix}_fused_damage_threshold'] = float(
                fused_damage_threshold_value)
            blended_outs[f'{telemetry_prefix}_fused_damage_threshold_available'] = float(
                1.0 if fused_damage_threshold_available else 0.0)
            blended_outs[f'{telemetry_prefix}_bev_fused_reference_available'] = float(
                1.0 if dominance_reference_available else 0.0)
            blended_outs[f'{telemetry_prefix}_fused_damage_threshold_scale'] = float(
                self.supp_fused_damage_threshold_scale)
            blended_outs[f'{telemetry_prefix}_img_damage_threshold_scale'] = float(
                self.supp_img_damage_threshold_scale)
            blended_outs[f'{telemetry_prefix}_bev_fused_dominance_ratio'] = float(
                self.supp_bev_fused_dominance_ratio)
            blended_outs[f'{telemetry_prefix}_bev_damage_upper_threshold_scale'] = float(
                self.supp_bev_damage_upper_threshold_scale)
            _write_qsem_gate_telemetry()
            _write_contextheight_gate_telemetry()
            blended_outs[f'{telemetry_prefix}_dual_budget_ratio'] = float(
                self.supp_dual_damage_budget_ratio)
            blended_outs[f'{telemetry_prefix}_dual_utility_threshold'] = float(
                self.supp_dual_damage_utility_threshold)
            blended_outs[f'{telemetry_prefix}_dual_scene_mapper_threshold'] = float(
                self.supp_dual_scene_mapper_threshold)
            blended_outs[f'{telemetry_prefix}_dual_bev_threshold_scale'] = float(
                self.supp_dual_bev_threshold_scale)
            blended_outs[f'{telemetry_prefix}_dual_bev_rank_boost'] = float(
                self.supp_dual_bev_rank_boost)
            blended_outs[f'{telemetry_prefix}_candidate_ratio'] = candidate_ratio
            blended_outs[f'{telemetry_prefix}_img_candidate_ratio'] = img_candidate_ratio
            blended_outs[f'{telemetry_prefix}_bev_candidate_ratio'] = bev_candidate_ratio
            blended_outs[f'{telemetry_prefix}_source_bev_query_ratio'] = source_bev_query_ratio
            blended_outs[f'{telemetry_prefix}_selected_img_ratio'] = selected_img_ratio
            blended_outs[f'{telemetry_prefix}_selected_bev_ratio'] = selected_bev_ratio
            blended_outs[f'{telemetry_prefix}_branch_aware_enabled'] = float(
                1.0 if branch_aware_available else 0.0)
            blended_outs[f'{telemetry_prefix}_branch_mapper_enabled'] = branch_mapper_enabled_ratio
            blended_outs[f'{telemetry_prefix}_img_utility_candidate_ratio'] = img_utility_candidate_ratio
            blended_outs[f'{telemetry_prefix}_bev_utility_candidate_ratio'] = bev_utility_candidate_ratio
            blended_outs[f'{telemetry_prefix}_img_utility_score_mean'] = img_utility_score_mean
            blended_outs[f'{telemetry_prefix}_bev_utility_score_mean'] = bev_utility_score_mean
            blended_outs[f'{telemetry_prefix}_utility_candidate_ratio'] = utility_candidate_ratio
            blended_outs[f'{telemetry_prefix}_utility_score_mean'] = float(utility_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_utility_threshold'] = float(self.supp_utility_threshold)
            blended_outs[f'{telemetry_prefix}_img_utility_threshold'] = float(
                self.supp_img_utility_threshold)
            blended_outs[f'{telemetry_prefix}_bev_utility_threshold'] = float(
                self.supp_bev_utility_threshold)
            blended_outs[f'{telemetry_prefix}_fused_slot_utility_score_mean'] = (
                fused_slot_utility_score_mean)
            blended_outs[f'{telemetry_prefix}_marginal_enabled'] = float(
                1.0 if marginal_enabled else 0.0)
            blended_outs[f'{telemetry_prefix}_marginal_margin'] = float(marginal_margin)
            blended_outs[f'{telemetry_prefix}_dual_marginal_margin'] = float(
                dual_marginal_margin)
            blended_outs[f'{telemetry_prefix}_marginal_checked_ratio'] = float(
                marginal_checked / max(batch_size * num_queries, 1))
            blended_outs[f'{telemetry_prefix}_marginal_reject_ratio'] = float(
                marginal_rejected / max(marginal_checked, 1))
            blended_outs[f'{telemetry_prefix}_marginal_pass_ratio'] = float(
                marginal_passed / max(marginal_checked, 1))
            blended_outs[f'{telemetry_prefix}_selected_margin_score_mean'] = 0.0
            blended_outs[f'{telemetry_prefix}_added_ratio'] = 0.0
            blended_outs[f'{telemetry_prefix}_replace_slot_ratio'] = replace_slot_ratio
            blended_outs[f'{telemetry_prefix}_bev_threshold'] = float(bev_threshold_value)
            blended_outs[f'{telemetry_prefix}_img_ceiling'] = float(img_ceiling_value)
            blended_outs[f'{telemetry_prefix}_margin_threshold'] = float(aux_threshold_value)
            blended_outs[f'{telemetry_prefix}_scene_bev_score'] = float(bev_scene_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_scene_img_score'] = float(img_scene_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_scene_fused_score'] = float(
                fused_scene_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_scene_aux_score'] = float(aux_scene_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_max_add_queries'] = float(self.supp_max_add_queries)
            blended_outs[f'{telemetry_prefix}_scene_mapper_score_mean'] = float(
                scene_mapper_score.float().mean().item())
            blended_outs[f'{telemetry_prefix}_scene_mapper_gate_ratio'] = float(scene_mapper_gate_ratio)
            blended_outs[f'{telemetry_prefix}_dynamic_budget_ratio'] = float(
                dynamic_budget_ratio.float().mean().item())
            blended_outs[f'{telemetry_prefix}_effective_max_add_queries'] = float(
                effective_max_add.float().mean().item())
            blended_outs[f'{telemetry_prefix}_rank_mode_code'] = float(
                {'img_score': 0, 'utility_product': 1, 'util_product': 1,
                 'utility_sum': 2, 'util_sum': 2, 'semantic_margin': 3,
                 'img_margin': 3}.get(self.supp_rank_mode.lower(), 0))
            blended_outs[f'{telemetry_prefix}_img_score_threshold'] = float(self.supp_img_score_threshold)
            blended_outs[f'{telemetry_prefix}_bev_score_threshold'] = float(self.supp_bev_score_threshold)
            blended_outs[f'{telemetry_prefix}_nms_center_dist'] = float(self.supp_nms_center_dist)
            blended_outs[f'{telemetry_prefix}_per_class_cap'] = float(self.supp_per_class_cap)
            return blended_outs

        output_mask = replace_mask.unsqueeze(0).unsqueeze(-1)
        gather_base = selected_img_idx.unsqueeze(0).unsqueeze(-1)
        candidate_source_is_bev = candidate_source_id.unsqueeze(0).unsqueeze(-1) == bev_source_id
        for key in ('cls_logits', 'cls_alpha', 'cls_beta', 'center', 'center_nig',
                    'height', 'height_nig', 'dim', 'dim_nig', 'rot', 'vel'):
            img_candidate_tensor = self._get_single_source_tensor(
                source_outs, key, 'img', modalities, num_queries_per_modality)
            if img_candidate_tensor is None or blended_outs.get(key) is None:
                continue
            candidate_tensor = img_candidate_tensor
            if branch_aware_available:
                bev_candidate_tensor = self._get_single_source_tensor(
                    source_outs, key, 'bev', modalities, num_queries_per_modality)
                if bev_candidate_tensor is not None:
                    candidate_tensor = torch.where(
                        candidate_source_is_bev.expand_as(img_candidate_tensor),
                        bev_candidate_tensor,
                        img_candidate_tensor)
            gather_idx = gather_base.expand(
                candidate_tensor.shape[0], -1, -1, candidate_tensor.shape[-1])
            selected_tensor = torch.gather(candidate_tensor, dim=2, index=gather_idx)
            blended_outs[key] = torch.where(output_mask, selected_tensor, blended_outs[key])

        blended_outs['supp_query_added'] = replace_mask.float()
        blended_outs['supp_query_selected_candidate_idx'] = selected_img_idx.unsqueeze(-1)
        blended_outs['supp_query_img_score'] = torch.gather(
            img_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_branch_score'] = torch.gather(
            candidate_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_bev_score'] = torch.gather(
            bev_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_fused_score'] = fused_score.unsqueeze(-1)
        blended_outs['supp_query_img_class'] = torch.gather(
            img_class, dim=1, index=selected_img_idx)
        blended_outs['supp_query_branch_class'] = torch.gather(
            candidate_class, dim=1, index=selected_img_idx)
        blended_outs['supp_query_utility_score'] = torch.gather(
            utility_score, dim=1, index=selected_img_idx).unsqueeze(-1)
        blended_outs['supp_query_source_branch'] = selected_source_id.unsqueeze(-1)
        blended_outs['supp_query_fused_slot_utility_score'] = selected_slot_utility_score.unsqueeze(-1)
        blended_outs['supp_query_marginal_utility_score'] = selected_margin_score.unsqueeze(-1)
        blended_outs[f'{telemetry_prefix}_scene_gate_ratio'] = float(scene_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_gate_pre_guard_ratio'] = float(
            scene_gate_pre_guard.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_gate_post_guard_ratio'] = float(
            scene_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_img_damage_guard_enabled'] = float(
            1.0 if guard_enabled else 0.0)
        blended_outs[f'{telemetry_prefix}_img_damage_guard_mode_code'] = float(guard_mode_code)
        blended_outs[f'{telemetry_prefix}_img_damage_gate_ratio'] = float(
            img_damage_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_dual_damage_gate_ratio'] = float(
            dual_damage_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_dual_gate_mode_code'] = float(dual_gate_mode_code)
        blended_outs[f'{telemetry_prefix}_scene_mapper_dual_gate_ratio'] = float(
            scene_mapper_dual_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_damage_gate_ratio'] = float(
            bev_damage_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_damage_upper_gate_ratio'] = float(
            bev_damage_upper_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_fused_damage_gate_ratio'] = float(
            fused_damage_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_fused_dominance_gate_ratio'] = float(
            bev_fused_dominance_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_fused_normalized_ratio_mean'] = float(
            bev_fused_normalized_ratio.float().mean().item())
        blended_outs[f'{telemetry_prefix}_dual_abstain_gate_ratio'] = float(
            dual_abstain_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_img_abstain_gate_ratio'] = float(
            img_abstain_gate.float().mean().item())
        blended_outs[f'{telemetry_prefix}_dual_abstain_score'] = float(
            dual_abstain_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_img_abstain_score'] = float(
            img_abstain_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_dual_abstain_threshold'] = float(
            self.supp_dual_abstain_threshold)
        blended_outs[f'{telemetry_prefix}_img_abstain_threshold'] = float(
            self.supp_img_abstain_threshold)
        blended_outs[f'{telemetry_prefix}_dual_abstain_available'] = float(
            1.0 if dual_abstain_available else 0.0)
        blended_outs[f'{telemetry_prefix}_img_abstain_available'] = float(
            1.0 if img_abstain_available else 0.0)
        blended_outs[f'{telemetry_prefix}_img_damage_threshold'] = float(img_damage_threshold_value)
        blended_outs[f'{telemetry_prefix}_img_damage_score'] = float(
            img_damage_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_damage_threshold'] = float(bev_damage_threshold_value)
        blended_outs[f'{telemetry_prefix}_bev_damage_upper_threshold'] = float(
            bev_damage_upper_threshold_value)
        blended_outs[f'{telemetry_prefix}_img_damage_threshold_available'] = float(
            1.0 if img_damage_threshold_available else 0.0)
        blended_outs[f'{telemetry_prefix}_bev_damage_threshold_available'] = float(
            1.0 if bev_damage_threshold_available else 0.0)
        blended_outs[f'{telemetry_prefix}_bev_damage_upper_threshold_available'] = float(
            1.0 if bev_damage_upper_threshold_available else 0.0)
        blended_outs[f'{telemetry_prefix}_fused_damage_threshold'] = float(
            fused_damage_threshold_value)
        blended_outs[f'{telemetry_prefix}_fused_damage_threshold_available'] = float(
            1.0 if fused_damage_threshold_available else 0.0)
        blended_outs[f'{telemetry_prefix}_bev_fused_reference_available'] = float(
            1.0 if dominance_reference_available else 0.0)
        blended_outs[f'{telemetry_prefix}_fused_damage_threshold_scale'] = float(
            self.supp_fused_damage_threshold_scale)
        blended_outs[f'{telemetry_prefix}_img_damage_threshold_scale'] = float(
            self.supp_img_damage_threshold_scale)
        blended_outs[f'{telemetry_prefix}_bev_fused_dominance_ratio'] = float(
            self.supp_bev_fused_dominance_ratio)
        blended_outs[f'{telemetry_prefix}_bev_damage_upper_threshold_scale'] = float(
            self.supp_bev_damage_upper_threshold_scale)
        _write_qsem_gate_telemetry()
        _write_contextheight_gate_telemetry()
        blended_outs[f'{telemetry_prefix}_dual_budget_ratio'] = float(
            self.supp_dual_damage_budget_ratio)
        blended_outs[f'{telemetry_prefix}_dual_utility_threshold'] = float(
            self.supp_dual_damage_utility_threshold)
        blended_outs[f'{telemetry_prefix}_dual_scene_mapper_threshold'] = float(
            self.supp_dual_scene_mapper_threshold)
        blended_outs[f'{telemetry_prefix}_dual_bev_threshold_scale'] = float(
            self.supp_dual_bev_threshold_scale)
        blended_outs[f'{telemetry_prefix}_dual_bev_rank_boost'] = float(
            self.supp_dual_bev_rank_boost)
        blended_outs[f'{telemetry_prefix}_status_code'] = 100.0
        blended_outs[f'{telemetry_prefix}_noadd_reason_code'] = 0.0
        blended_outs[f'{telemetry_prefix}_candidate_ratio'] = candidate_ratio_all
        blended_outs[f'{telemetry_prefix}_img_candidate_ratio'] = img_candidate_ratio
        blended_outs[f'{telemetry_prefix}_bev_candidate_ratio'] = bev_candidate_ratio
        blended_outs[f'{telemetry_prefix}_source_bev_query_ratio'] = source_bev_query_ratio
        blended_outs[f'{telemetry_prefix}_selected_img_ratio'] = selected_img_ratio
        blended_outs[f'{telemetry_prefix}_selected_bev_ratio'] = selected_bev_ratio
        blended_outs[f'{telemetry_prefix}_branch_aware_enabled'] = float(
            1.0 if branch_aware_available else 0.0)
        blended_outs[f'{telemetry_prefix}_branch_mapper_enabled'] = branch_mapper_enabled_ratio
        blended_outs[f'{telemetry_prefix}_img_utility_candidate_ratio'] = img_utility_candidate_ratio
        blended_outs[f'{telemetry_prefix}_bev_utility_candidate_ratio'] = bev_utility_candidate_ratio
        blended_outs[f'{telemetry_prefix}_img_utility_score_mean'] = img_utility_score_mean
        blended_outs[f'{telemetry_prefix}_bev_utility_score_mean'] = bev_utility_score_mean
        blended_outs[f'{telemetry_prefix}_utility_candidate_ratio'] = utility_candidate_ratio
        blended_outs[f'{telemetry_prefix}_utility_score_mean'] = float(
            utility_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_utility_threshold'] = float(self.supp_utility_threshold)
        blended_outs[f'{telemetry_prefix}_img_utility_threshold'] = float(
            self.supp_img_utility_threshold)
        blended_outs[f'{telemetry_prefix}_bev_utility_threshold'] = float(
            self.supp_bev_utility_threshold)
        added_count = int(replace_mask.sum().item())
        selected_margin_mean = (
            float(selected_margin_score[replace_mask].float().mean().item())
            if added_count > 0 else 0.0
        )
        blended_outs[f'{telemetry_prefix}_fused_slot_utility_score_mean'] = (
            fused_slot_utility_score_mean)
        blended_outs[f'{telemetry_prefix}_marginal_enabled'] = float(
            1.0 if marginal_enabled else 0.0)
        blended_outs[f'{telemetry_prefix}_marginal_margin'] = float(marginal_margin)
        blended_outs[f'{telemetry_prefix}_dual_marginal_margin'] = float(
            dual_marginal_margin)
        blended_outs[f'{telemetry_prefix}_marginal_checked_ratio'] = float(
            marginal_checked / max(batch_size * num_queries, 1))
        blended_outs[f'{telemetry_prefix}_marginal_reject_ratio'] = float(
            marginal_rejected / max(marginal_checked, 1))
        blended_outs[f'{telemetry_prefix}_marginal_pass_ratio'] = float(
            marginal_passed / max(marginal_checked, 1))
        blended_outs[f'{telemetry_prefix}_selected_margin_score_mean'] = selected_margin_mean
        blended_outs[f'{telemetry_prefix}_added_ratio'] = float(replace_mask.float().mean().item())
        blended_outs[f'{telemetry_prefix}_replace_slot_ratio'] = float(
            (fused_score <= self.supp_fused_replace_score_ceiling).float().mean().item())
        blended_outs[f'{telemetry_prefix}_bev_threshold'] = float(bev_threshold_value)
        blended_outs[f'{telemetry_prefix}_img_ceiling'] = float(img_ceiling_value)
        blended_outs[f'{telemetry_prefix}_margin_threshold'] = float(aux_threshold_value)
        blended_outs[f'{telemetry_prefix}_scene_bev_score'] = float(bev_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_img_score'] = float(img_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_fused_score'] = float(
            fused_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_aux_score'] = float(aux_scene_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_max_add_queries'] = float(self.supp_max_add_queries)
        blended_outs[f'{telemetry_prefix}_scene_mapper_score_mean'] = float(
            scene_mapper_score.float().mean().item())
        blended_outs[f'{telemetry_prefix}_scene_mapper_gate_ratio'] = float(scene_mapper_gate_ratio)
        blended_outs[f'{telemetry_prefix}_dynamic_budget_ratio'] = float(
            dynamic_budget_ratio.float().mean().item())
        blended_outs[f'{telemetry_prefix}_effective_max_add_queries'] = float(
            effective_max_add.float().mean().item())
        blended_outs[f'{telemetry_prefix}_rank_mode_code'] = float(
            {'img_score': 0, 'utility_product': 1, 'util_product': 1,
             'utility_sum': 2, 'util_sum': 2, 'semantic_margin': 3,
             'img_margin': 3}.get(self.supp_rank_mode.lower(), 0))
        blended_outs[f'{telemetry_prefix}_img_score_threshold'] = float(self.supp_img_score_threshold)
        blended_outs[f'{telemetry_prefix}_bev_score_threshold'] = float(self.supp_bev_score_threshold)
        blended_outs[f'{telemetry_prefix}_nms_center_dist'] = float(self.supp_nms_center_dist)
        blended_outs[f'{telemetry_prefix}_per_class_cap'] = float(self.supp_per_class_cap)
        return blended_outs

    def _maybe_apply_img_supplement_admission(
            self,
            use_supp_post_inference,
            blended_outs,
            source_outs,
            v7_info,
            modalities,
            num_queries_per_modality,
            raw_unc_by_source=None,
            source_cls_unc=None,
            v8_info=None):
        """Dispatch supplement admission without touching the matched path."""
        if not use_supp_post_inference:
            return blended_outs
        if v7_info is None:
            blended_outs['supp_status_code'] = 3.0
            blended_outs['supp_noadd_reason_code'] = 3.0
            return blended_outs
        return self._apply_img_supplement_admission(
            blended_outs,
            source_outs,
            v7_info,
            modalities,
            num_queries_per_modality,
            raw_unc_by_source=raw_unc_by_source,
            source_cls_unc=source_cls_unc,
            telemetry_prefix='supp',
            v8_info=v8_info)

    def _compute_v5_fused_anomaly(
            self,
            fused_cls_unc,
            fused_center_unc,
            fused_height_unc,
            fused_cls_logits,
            fused_center):
        stats = self._load_v5_reference_stats(require_stats=not self.training)
        if stats is None:
            return None

        bs, nq, _ = fused_center_unc.shape
        pred_class = fused_cls_logits.argmax(dim=-1)
        distance_xy = fused_center[..., :2].norm(dim=-1, keepdim=True)
        distance_bin = self._bucketize_v5_distance(distance_xy).squeeze(-1)

        cls_anom = fused_cls_unc.new_zeros(bs, nq, 1)
        center_anom = fused_center_unc.new_zeros(bs, nq, 1)
        height_anom = fused_height_unc.new_zeros(bs, nq, 1)
        combined_anom = fused_height_unc.new_zeros(bs, nq, 1)
        fallback_level = pred_class.new_full((bs, nq), 3)

        cls_w = self.v5_anomaly_component_weights['cls']
        center_w = self.v5_anomaly_component_weights['center']
        height_w = self.v5_anomaly_component_weights['height']

        for batch_idx in range(bs):
            for query_idx in range(nq):
                entry, fallback = self._lookup_v5_reference_entry(
                    stats,
                    pred_class[batch_idx, query_idx].item(),
                    distance_bin[batch_idx, query_idx].item())
                fallback_level[batch_idx, query_idx] = fallback
                if entry is None:
                    continue
                cls_score = self._compute_v5_single_anomaly(
                    fused_cls_unc[batch_idx, query_idx, 0].item(),
                    entry.get('cls_unc'),
                    self.v5_anomaly_score_mode)
                center_score = self._compute_v5_single_anomaly(
                    fused_center_unc[batch_idx, query_idx, 0].item(),
                    entry.get('center_unc'),
                    self.v5_anomaly_score_mode)
                height_score = self._compute_v5_single_anomaly(
                    fused_height_unc[batch_idx, query_idx, 0].item(),
                    entry.get('height_unc'),
                    self.v5_anomaly_score_mode)
                cls_anom[batch_idx, query_idx, 0] = cls_score
                center_anom[batch_idx, query_idx, 0] = center_score
                height_anom[batch_idx, query_idx, 0] = height_score
                combined_anom[batch_idx, query_idx, 0] = (
                    cls_w * cls_score + center_w * center_score + height_w * height_score
                )

        return {
            'pred_class': pred_class,
            'distance_bin': distance_bin,
            'fused_cls_anom': cls_anom,
            'fused_center_anom': center_anom,
            'fused_height_anom': height_anom,
            'fused_combined_anom': combined_anom,
            'fallback_level': fallback_level,
        }

    def _compute_v6_fused_anomaly(
            self,
            fused_cls_unc,
            fused_center_unc,
            fused_height_unc,
            fused_cls_logits,
            fused_center):
        stats = self._load_v6_reference_stats(require_stats=not self.training)
        if stats is None:
            return None

        bs, nq, _ = fused_center_unc.shape
        pred_class = fused_cls_logits.argmax(dim=-1)
        distance_xy = fused_center[..., :2].norm(dim=-1, keepdim=True)
        distance_bin = self._bucketize_v6_distance(distance_xy).squeeze(-1)

        cls_anom = fused_cls_unc.new_zeros(bs, nq, 1)
        center_anom = fused_center_unc.new_zeros(bs, nq, 1)
        height_anom = fused_height_unc.new_zeros(bs, nq, 1)
        combined_anom = fused_height_unc.new_zeros(bs, nq, 1)
        fallback_level = pred_class.new_full((bs, nq), 3)

        cls_w = self.v6_anomaly_component_weights['cls']
        center_w = self.v6_anomaly_component_weights['center']
        height_w = self.v6_anomaly_component_weights['height']

        for batch_idx in range(bs):
            for query_idx in range(nq):
                entry, fallback = self._lookup_v6_reference_entry(
                    stats,
                    pred_class[batch_idx, query_idx].item(),
                    distance_bin[batch_idx, query_idx].item())
                fallback_level[batch_idx, query_idx] = fallback
                if entry is None:
                    continue
                cls_score = self._compute_v5_single_anomaly(
                    fused_cls_unc[batch_idx, query_idx, 0].item(),
                    entry.get('cls_unc'),
                    self.v6_anomaly_score_mode)
                center_score = self._compute_v5_single_anomaly(
                    fused_center_unc[batch_idx, query_idx, 0].item(),
                    entry.get('center_unc'),
                    self.v6_anomaly_score_mode)
                height_score = self._compute_v5_single_anomaly(
                    fused_height_unc[batch_idx, query_idx, 0].item(),
                    entry.get('height_unc'),
                    self.v6_anomaly_score_mode)
                cls_anom[batch_idx, query_idx, 0] = cls_score
                center_anom[batch_idx, query_idx, 0] = center_score
                height_anom[batch_idx, query_idx, 0] = height_score
                combined_anom[batch_idx, query_idx, 0] = (
                    cls_w * cls_score + center_w * center_score + height_w * height_score
                )

        return {
            'pred_class': pred_class,
            'distance_bin': distance_bin,
            'fused_cls_anom': cls_anom,
            'fused_center_anom': center_anom,
            'fused_height_anom': height_anom,
            'fused_combined_anom': combined_anom,
            'fallback_level': fallback_level,
        }

    def forward(
            self,
            x,
            reference,
            outs,
            nig_params_dict,
            modalities,
            num_queries_per_modality,
            task_id,
            pc_range,
            attn_masks=None,
            img_metas=None):
        source_outs = outs

        x_last = x[-1].transpose(0, 1)
        x_split = list(x_last.split(num_queries_per_modality, dim=0))
        x_proj = []
        for idx, modality in enumerate(modalities):
            x_proj.append(self.modality_proj[modality](x_split[idx]))

        if self.use_value_scaling:
            for idx, modality in enumerate(modalities):
                if modality not in nig_params_dict or nig_params_dict[modality] is None:
                    continue
                unc = self.compute_uncertainty(nig_params_dict[modality])
                if unc is None:
                    continue
                reliability = 1.0 / (1.0 + unc)
                x_proj[idx] = x_proj[idx] * reliability.transpose(0, 1)

        fused_idx = modalities.index("fused")
        target = x_proj[fused_idx]
        memory = torch.cat(x_proj, dim=0)

        center = outs["center"][-1]
        box_pos_embed = pos2embed(center, self.embed_dims)
        box_pos_embed = self.box_pos_embedding(box_pos_embed).transpose(0, 1)
        box_pos_embed_split = list(box_pos_embed.split(num_queries_per_modality, dim=0))

        query_box_pos_embed = box_pos_embed_split[fused_idx]
        key_box_pos_embed = torch.cat(box_pos_embed_split, dim=0)

        center_split = list(center.split(num_queries_per_modality, dim=1))
        center_q = center_split[fused_idx]
        center_kv = torch.cat(center_split, dim=1)
        dist = (center_q.unsqueeze(2) - center_kv.unsqueeze(1)).norm(p=2, dim=-1)
        dist_mask = dist * self.dist_scaler + self.dist_bias

        if self.use_unc_bias:
            unc_per_modality = []
            for modality in modalities:
                unc = None
                if modality in nig_params_dict and nig_params_dict[modality] is not None:
                    unc = self.compute_uncertainty(nig_params_dict[modality])
                if unc is None:
                    nq_m = num_queries_per_modality[modalities.index(modality)]
                    unc = torch.zeros(center_q.shape[0], nq_m, 1, device=dist.device)
                unc_per_modality.append(unc)
            unc_kv = torch.cat(unc_per_modality, dim=1).squeeze(-1)
            unc_mask = self.unc_scaler * unc_kv + self.unc_bias_param
            dist_mask = dist_mask + unc_mask.unsqueeze(1)

        nq_fused = num_queries_per_modality[fused_idx]
        if attn_masks is None:
            attn_masks = torch.zeros((nq_fused, nq_fused), dtype=torch.bool, device=target.device)

        attn_masks = torch.zeros_like(attn_masks, dtype=torch.float).masked_fill(attn_masks, float("-inf"))
        attn_masks = attn_masks.repeat(1, len(x_proj))
        attn_masks = attn_masks + dist_mask
        attn_masks = attn_masks.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)

        outs_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_box_pos_embed,
            key_pos=key_box_pos_embed,
            attn_masks=[attn_masks])
        outs_dec = outs_dec.transpose(1, 2)

        if self.ensemble_mode in (
                'soft_selector',
                'primary_router',
                'hard_selective_switch',
                'gated_primary_router',
                'modality_damage_aware_router',
                'clean_reference_gate_router',
                'damage_gated_soft_router'):
            use_primary_router = self.ensemble_mode == 'primary_router'
            use_v8_router = self.ensemble_mode == 'damage_gated_soft_router'
            use_v5_inference = self.ensemble_mode == 'hard_selective_switch' and not self.training
            use_v6_inference = self.ensemble_mode == 'gated_primary_router' and not self.training
            use_v7_router_inference = self.ensemble_mode == 'modality_damage_aware_router' and not self.training
            use_crg_standalone_inference = self.ensemble_mode == 'clean_reference_gate_router' and not self.training
            use_crg_post_inference = (
                self.crg_enable and
                self.ensemble_mode == 'damage_gated_soft_router' and
                not self.training)
            use_scrg_post_inference = (
                self.scrg_enable and
                self.ensemble_mode == 'damage_gated_soft_router' and
                not self.training)
            use_supp_post_inference = (
                self.supp_enable and
                self.ensemble_mode == 'damage_gated_soft_router' and
                not self.training)
            use_v7_inference = (
                use_v7_router_inference or
                use_crg_standalone_inference or
                use_crg_post_inference or
                use_scrg_post_inference)
            use_soft_selector_behavior = (
                self.ensemble_mode == 'soft_selector' or
                self.ensemble_mode == 'gated_primary_router' or
                self.ensemble_mode == 'modality_damage_aware_router' or
                self.ensemble_mode == 'clean_reference_gate_router' or
                self.ensemble_mode == 'damage_gated_soft_router' or
                (self.ensemble_mode == 'hard_selective_switch' and self.training)
            )
            non_fused_modalities = [modality for modality in modalities if modality != 'fused']
            for key, tensor in source_outs.items():
                if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
                    continue
                split_tensors = list(tensor.split(num_queries_per_modality, dim=2))
                if use_soft_selector_behavior:
                    check_tensors = split_tensors
                    ref_nq = nq_fused
                else:
                    check_tensors = [
                        split_tensors[modalities.index(modality)]
                        for modality in non_fused_modalities
                    ]
                    ref_nq = check_tensors[0].shape[2]
                for split_tensor in check_tensors:
                    if split_tensor.shape[2] != ref_nq:
                        raise AssertionError(
                            f"Soft selector requires aligned query counts. "
                            f"Got {split_tensor.shape[2]} vs ref {ref_nq} for key {key}.")

            nb_dec, bs, nq, _ = outs_dec.shape
            context_logits = self.selector_context_proj(outs_dec)
            distance_prior_scores = []
            residual_scores = []
            raw_per_modality_unc = {}
            transformed_per_modality_unc = {}
            transformed_unc_by_source = {}
            raw_unc_by_source = {}
            distance_xy_map = {}
            distance_bin_stats_by_source = {}
            score_component_terms = {}
            damage_feature_terms = {}
            disagreement_summary = {}
            absolute_unc_summary = {}

            for source_name in self.pme_unc_sources:
                source_unc_map = {}
                transformed_source_unc_map = {}
                for modality in modalities:
                    modality_nig_inputs = nig_params_dict.get(modality)
                    raw_unc = self._compute_source_uncertainty(
                        modality_nig_inputs, source_name, (bs, nq), outs_dec.device)
                    source_unc_map[modality] = raw_unc
                    transformed_source_unc_map[modality] = self._transform_uncertainty(raw_unc)
                transformed_unc_by_source[source_name] = self._normalize_uncertainty_dict(
                    transformed_source_unc_map, modalities)
                raw_unc_by_source[source_name] = source_unc_map

            source_center = self._get_source_tensor_last(
                source_outs, 'center', modalities, num_queries_per_modality)
            source_height = self._get_source_tensor_last(
                source_outs, 'height', modalities, num_queries_per_modality)
            source_cls_logits = self._get_source_tensor_last(
                source_outs, 'cls_logits', modalities, num_queries_per_modality)
            source_cls_unc = self._get_source_cls_unc_last(
                source_outs, modalities, num_queries_per_modality)
            v7_img_router_logits = None
            v7_bev_router_logits = None
            if self.ensemble_mode == 'modality_damage_aware_router':
                query_context_last = outs_dec[-1]
                v7_img_router_logits = self.v7_img_fallback_router(query_context_last)
                v7_bev_router_logits = self.v7_bev_fallback_router(query_context_last)

            for modality in modalities:
                if source_center is None or modality not in source_center:
                    distance_xy_map[modality] = torch.zeros(bs, nq, 1, device=outs_dec.device)
                else:
                    center_xy = source_center[modality][..., :2]
                    distance_xy_map[modality] = center_xy.norm(dim=-1, keepdim=True)

            zero_feature = torch.zeros(bs, nq, 1, device=outs_dec.device)
            source_cls_percentile = {modality: zero_feature for modality in modalities}
            if source_cls_unc is not None:
                transformed_cls_unc = {
                    modality: self._transform_uncertainty(source_cls_unc[modality])
                    for modality in modalities
                }
                transformed_cls_unc = self._normalize_uncertainty_dict(
                    transformed_cls_unc, modalities)
                _cls_conditioned, cls_percentile_map, _cls_distance_stats = self._distance_conditioned_stats(
                    transformed_cls_unc, distance_xy_map, modalities)
                source_cls_percentile = cls_percentile_map

            conditioned_unc_by_source = {}
            percentile_unc_by_source = {}
            for source_name in self.pme_unc_sources:
                conditioned_map, percentile_map, distance_bin_stats = self._distance_conditioned_stats(
                    transformed_unc_by_source[source_name], distance_xy_map, modalities)
                conditioned_unc_by_source[source_name] = conditioned_map
                percentile_unc_by_source[source_name] = percentile_map
                distance_bin_stats_by_source[source_name] = distance_bin_stats

            relative_unc_summary = {}
            tail_ratio_summary = {}
            alpha_conditioned_terms = []
            alpha_percentile_terms = []
            alpha_disagreement_terms = []
            for modality in modalities:
                global_modality_idx = self.modality_names.index(modality)
                raw_unc_values = []
                relative_unc_values = []
                conditioned_unc_values = []
                percentile_values = []
                residual_features = []

                dist_xy = distance_xy_map[modality]
                fused_dist_xy = distance_xy_map[modalities[fused_idx]]
                dist_feature = torch.cat([
                    dist_xy,
                    torch.log1p(dist_xy.clamp(min=0.0)),
                    dist_xy - fused_dist_xy,
                    self._distance_bin_embed(dist_xy.squeeze(-1))
                ], dim=-1)
                dist_score = self.distance_prior_head(
                    dist_feature.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                )
                distance_prior_scores.append(dist_score)

                for source_name in self.pme_unc_sources:
                    source_unc = transformed_unc_by_source[source_name][modality]
                    fused_unc = transformed_unc_by_source[source_name][modalities[fused_idx]]
                    relative_unc = source_unc - fused_unc
                    conditioned_unc = conditioned_unc_by_source[source_name][modality]
                    percentile_unc = percentile_unc_by_source[source_name][modality]
                    raw_unc_values.append(raw_unc_by_source[source_name][modality])
                    relative_unc_values.append(relative_unc)
                    conditioned_unc_values.append(conditioned_unc)
                    percentile_values.append(percentile_unc)
                    residual_features.extend([relative_unc, conditioned_unc, percentile_unc])

                if source_center is not None:
                    center_disagreement = torch.abs(
                        source_center[modality] - source_center[modalities[fused_idx]]
                    ).mean(dim=-1, keepdim=True)
                else:
                    center_disagreement = torch.zeros(bs, nq, 1, device=outs_dec.device)
                if source_height is not None:
                    height_disagreement = torch.abs(
                        source_height[modality] - source_height[modalities[fused_idx]]
                    ).mean(dim=-1, keepdim=True)
                else:
                    height_disagreement = torch.zeros(bs, nq, 1, device=outs_dec.device)
                if source_cls_logits is not None:
                    fused_prob = torch.sigmoid(source_cls_logits[modalities[fused_idx]]).max(dim=-1, keepdim=True)[0]
                    modality_prob = torch.sigmoid(source_cls_logits[modality]).max(dim=-1, keepdim=True)[0]
                    cls_gap = torch.abs(modality_prob - fused_prob)
                else:
                    cls_gap = torch.zeros(bs, nq, 1, device=outs_dec.device)

                center_unc_raw = raw_unc_by_source.get('center', {}).get(modality, zero_feature)
                fused_center_unc_raw = raw_unc_by_source.get('center', {}).get(
                    modalities[fused_idx], zero_feature)
                height_unc_raw = raw_unc_by_source.get('height', {}).get(modality, zero_feature)
                fused_height_unc_raw = raw_unc_by_source.get('height', {}).get(
                    modalities[fused_idx], zero_feature)
                if source_cls_unc is not None and modality in source_cls_unc:
                    cls_unc_raw = source_cls_unc[modality]
                    fused_cls_unc_raw = source_cls_unc[modalities[fused_idx]]
                else:
                    cls_unc_raw = zero_feature
                    fused_cls_unc_raw = zero_feature
                center_rel_unc_raw = torch.relu(
                    torch.log1p(center_unc_raw.clamp(min=0.0)) -
                    torch.log1p(fused_center_unc_raw.clamp(min=0.0)))
                height_rel_unc_raw = torch.relu(
                    torch.log1p(height_unc_raw.clamp(min=0.0)) -
                    torch.log1p(fused_height_unc_raw.clamp(min=0.0)))
                cls_rel_unc_raw = torch.relu(
                    torch.log1p(cls_unc_raw.clamp(min=0.0)) -
                    torch.log1p(fused_cls_unc_raw.clamp(min=0.0)))
                center_percentile_unc = percentile_unc_by_source.get(
                    'center', {}).get(modality, zero_feature)
                height_percentile_unc = percentile_unc_by_source.get(
                    'height', {}).get(modality, zero_feature)
                cls_percentile_unc = source_cls_percentile.get(modality, zero_feature)
                a4_damage_features = torch.cat([
                    center_unc_raw,
                    height_unc_raw,
                    cls_unc_raw,
                    center_rel_unc_raw,
                    height_rel_unc_raw,
                    cls_rel_unc_raw,
                    center_percentile_unc.clamp(min=0.0, max=1.0),
                    height_percentile_unc.clamp(min=0.0, max=1.0),
                    cls_percentile_unc.clamp(min=0.0, max=1.0),
                    center_disagreement,
                    height_disagreement,
                    cls_gap,
                ], dim=-1)

                context_feature = context_logits[..., global_modality_idx:global_modality_idx + 1]
                residual_features.extend([center_disagreement, height_disagreement, cls_gap, context_feature[-1]])
                residual_input = torch.cat(residual_features, dim=-1)
                residual_score = self.residual_router_head(
                    residual_input.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                )
                residual_scores.append(residual_score)
                alpha_conditioned_terms.append(torch.cat(conditioned_unc_values, dim=-1).abs())
                alpha_percentile_terms.append(torch.cat(percentile_values, dim=-1))
                alpha_disagreement_terms.append(
                    torch.cat([center_disagreement, height_disagreement, cls_gap], dim=-1)
                )
                raw_per_modality_unc[modality] = torch.cat(raw_unc_values, dim=-1).mean(dim=-1, keepdim=True)
                transformed_per_modality_unc[modality] = torch.cat([
                    transformed_unc_by_source[source_name][modality]
                    for source_name in self.pme_unc_sources
                ], dim=-1).mean(dim=-1, keepdim=True)
                absolute_unc_summary[modality] = float(
                    transformed_per_modality_unc[modality].mean().item()
                )
                disagreement_summary[modality] = float(
                    torch.cat([center_disagreement, height_disagreement, cls_gap], dim=-1).mean().item()
                )
                if relative_unc_values:
                    relative_unc_summary[modality] = float(
                        torch.cat(relative_unc_values, dim=-1).mean().item()
                    )
                    tail_ratio_summary[modality] = float(
                        (torch.cat(percentile_values, dim=-1) > 0.9).float().mean().item()
                    )
                else:
                    relative_unc_summary[modality] = 0.0
                    tail_ratio_summary[modality] = 0.0
                score_component_terms[modality] = {
                    'distance': dist_score,
                    'context': context_feature,
                    'delta': residual_score,
                    'unc': transformed_per_modality_unc[modality],
                    'relative_unc': (
                        torch.cat(relative_unc_values, dim=-1).mean(dim=-1, keepdim=True)
                        if relative_unc_values else torch.zeros(bs, nq, 1, device=outs_dec.device)
                    ),
                    'disagreement': torch.cat(
                        [center_disagreement, height_disagreement, cls_gap], dim=-1
                    ).mean(dim=-1, keepdim=True),
                }
                damage_feature_terms[modality] = {
                    'unc': transformed_per_modality_unc[modality],
                    'relative_unc': score_component_terms[modality]['relative_unc'],
                    'conditioned_unc': (
                        torch.cat(conditioned_unc_values, dim=-1).mean(dim=-1, keepdim=True)
                        if conditioned_unc_values else torch.zeros(bs, nq, 1, device=outs_dec.device)
                    ),
                    'percentile_unc': (
                        torch.cat(percentile_values, dim=-1).mean(dim=-1, keepdim=True)
                        if percentile_values else torch.zeros(bs, nq, 1, device=outs_dec.device)
                    ),
                    'disagreement': torch.cat(
                        [center_disagreement, height_disagreement], dim=-1
                    ).mean(dim=-1, keepdim=True),
                    'cls_gap': cls_gap,
                    'a4_features': torch.nan_to_num(
                        a4_damage_features, nan=0.0, posinf=1e4, neginf=0.0),
                }

            distance_scores_all = torch.cat(distance_prior_scores, dim=-1)
            context_scores_all = context_logits
            base_scores_all = (
                self.selector_distance_score_weight * distance_scores_all +
                self.selector_context_score_weight * context_scores_all
            )
            delta_scores_all = torch.cat(residual_scores, dim=-1)
            alpha_input = torch.cat([
                torch.cat(alpha_percentile_terms, dim=-1).amax(dim=-1, keepdim=True),
                torch.cat(alpha_conditioned_terms, dim=-1).amax(dim=-1, keepdim=True),
                torch.cat(alpha_disagreement_terms, dim=-1).amax(dim=-1, keepdim=True)
            ], dim=-1)
            alpha_raw = torch.sigmoid(self.alpha_res_head(
                alpha_input.unsqueeze(0).expand(nb_dec, -1, -1, -1)
            ))
            alpha_res = self.residual_floor + (1.0 - self.residual_floor) * alpha_raw
            routing_modalities = (
                list(modalities)
                if use_soft_selector_behavior
                else non_fused_modalities
            )
            routing_indices = [modalities.index(modality) for modality in routing_modalities]
            prior = base_scores_all.new_zeros((1, 1, 1, len(routing_modalities)))
            if (not use_primary_router) and (not use_v5_inference) and 'fused' in routing_modalities:
                prior[..., routing_modalities.index('fused')] = self.selector_fused_score_bias

            unc_penalty_all = torch.cat([
                score_component_terms[modality]['unc']
                for modality in modalities
            ], dim=-1).unsqueeze(0).expand(nb_dec, -1, -1, -1)
            relative_unc_penalty_all = torch.cat([
                score_component_terms[modality]['relative_unc']
                for modality in modalities
            ], dim=-1).unsqueeze(0).expand(nb_dec, -1, -1, -1)
            disagreement_penalty_all = torch.cat([
                score_component_terms[modality]['disagreement']
                for modality in modalities
            ], dim=-1).unsqueeze(0).expand(nb_dec, -1, -1, -1)

            distance_scores = torch.cat([
                distance_scores_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)
            context_scores = torch.cat([
                context_scores_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)
            base_scores = torch.cat([
                base_scores_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)
            delta_scores = torch.cat([
                delta_scores_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)
            unc_penalty = torch.cat([
                unc_penalty_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)
            relative_unc_penalty = torch.cat([
                relative_unc_penalty_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)
            disagreement_penalty = torch.cat([
                disagreement_penalty_all[..., idx:idx + 1] for idx in routing_indices
            ], dim=-1)

            if self.selector_score_mode == 'quality_minus_uncertainty':
                selector_scores = (
                    base_scores
                    - self.selector_unc_score_weight * unc_penalty
                    - self.selector_rel_unc_score_weight * relative_unc_penalty
                    - self.selector_disagreement_score_weight * disagreement_penalty
                )
                if use_soft_selector_behavior:
                    selector_scores = selector_scores + alpha_res * delta_scores
                else:
                    selector_scores = selector_scores + delta_scores
            else:
                selector_scores = base_scores + alpha_res * delta_scores

            base_selector_probs = F.softmax(
                (base_scores + prior) / self.selector_softmax_temperature, dim=-1)
            selector_probs = F.softmax(
                (selector_scores + prior) / self.selector_softmax_temperature, dim=-1)

            v8_info = None
            if use_v8_router:
                def _v8_damage_features(modality):
                    terms = damage_feature_terms[modality]
                    features = torch.cat([
                        torch.relu(terms['unc']),
                        torch.relu(terms['relative_unc']),
                        torch.relu(terms['conditioned_unc']),
                        terms['percentile_unc'].clamp(min=0.0, max=1.0),
                        torch.relu(terms['disagreement']),
                        torch.relu(terms['cls_gap']),
                    ], dim=-1)
                    return torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=0.0)

                def _v8_damage_score(modality):
                    terms = damage_feature_terms[modality]
                    score = (
                        self.v8_unc_score_weight * torch.relu(terms['unc']) +
                        self.v8_rel_unc_score_weight * torch.relu(terms['relative_unc']) +
                        self.v8_conditioned_score_weight * torch.relu(terms['conditioned_unc']) +
                        self.v8_percentile_score_weight * terms['percentile_unc'].clamp(min=0.0, max=1.0) +
                        self.v8_disagreement_score_weight * torch.relu(terms['disagreement']) +
                        self.v8_cls_gap_score_weight * torch.relu(terms['cls_gap'])
                    )
                    return torch.nan_to_num(score, nan=0.0, posinf=1e4, neginf=0.0)

                use_v8_a2 = self.v8_router_variant == 'a2_reliability_soft'
                use_v8_a3 = self.v8_router_variant == 'a3_balanced_norm_reliability'
                use_v8_a4 = self.v8_router_variant == 'a4_clean_anchor_domain_reliability'
                use_v9 = self.v8_router_variant == 'v9_lite_domain_anchored'
                use_v8_reliability = use_v8_a2 or use_v8_a3 or use_v8_a4 or use_v9
                context_bev_damage = torch.zeros(bs, nq, 1, device=outs_dec.device)
                context_img_damage = torch.zeros(bs, nq, 1, device=outs_dec.device)
                a4_bev_damage_z = torch.zeros(bs, nq, 1, device=outs_dec.device)
                a4_img_damage_z = torch.zeros(bs, nq, 1, device=outs_dec.device)
                a4_bev_cls_unc_z = torch.zeros(bs, nq, 1, device=outs_dec.device)
                a4_img_cls_unc_z = torch.zeros(bs, nq, 1, device=outs_dec.device)
                a4_route_score_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                a4_route_score_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_a_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_a_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_a_bev_above_tau = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_a_img_above_tau = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_clean_like_mask = torch.zeros(bs, nq, 1, dtype=torch.bool, device=outs_dec.device)
                v9_anchor_distance_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_anchor_distance_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_consistency_bev = torch.ones(bs, nq, 1, device=outs_dec.device)
                v9_consistency_img = torch.ones(bs, nq, 1, device=outs_dec.device)
                v9_tail_prob_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_tail_prob_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_tail_excess_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_tail_excess_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_cvar_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_cvar_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_q95_shift_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_q95_shift_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_route_score_bev = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_route_score_img = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_mean_disagree = torch.zeros(bs, nq, 1, device=outs_dec.device)
                v9_velocity_l1 = torch.zeros(bs, nq, 1, device=outs_dec.device)
                if use_v8_a4 or use_v9:
                    bev_features = damage_feature_terms['bev']['a4_features']
                    img_features = damage_feature_terms['img']['a4_features']
                    if self.training:
                        clean_mask = self._v8_clean_sample_mask(img_metas, bs, outs_dec.device)
                        self._v8_update_a4_clean_anchor(bev_features, img_features, clean_mask)
                    bev_z = self._v8_a4_anchor_z(bev_features, 0)
                    img_z = self._v8_a4_anchor_z(img_features, 1)
                    bev_z_pos = torch.relu(bev_z)
                    img_z_pos = torch.relu(img_z)
                    a4_bev_damage_z = bev_z_pos.mean(dim=-1, keepdim=True)
                    a4_img_damage_z = img_z_pos.mean(dim=-1, keepdim=True)
                    a4_bev_cls_unc_z = bev_z[..., 2:3]
                    a4_img_cls_unc_z = img_z[..., 2:3]
                    if use_v9:
                        if self.v8_a3_damage_context_head is not None:
                            context_damage_logits = self.v8_a3_damage_context_head(outs_dec[-1])
                            zero_centered_softplus = (
                                F.softplus(context_damage_logits) -
                                context_damage_logits.new_tensor(0.6931471805599453)
                            )
                            context_bev_damage = zero_centered_softplus[..., 0:1] * self.v8_a3_context_damage_scale
                            context_img_damage = zero_centered_softplus[..., 1:2] * self.v8_a3_context_damage_scale
                        bev_scene_tail, bev_query_tail, bev_q95_shift, bev_tail_prob, bev_tail_excess, bev_cvar = (
                            self._v9_tail_features(bev_z)
                        )
                        img_scene_tail, img_query_tail, img_q95_shift, img_tail_prob, img_tail_excess, img_cvar = (
                            self._v9_tail_features(img_z)
                        )
                        v9_tail_prob_bev = bev_tail_prob.expand(-1, nq, -1)
                        v9_tail_prob_img = img_tail_prob.expand(-1, nq, -1)
                        v9_tail_excess_bev = bev_tail_excess.expand(-1, nq, -1)
                        v9_tail_excess_img = img_tail_excess.expand(-1, nq, -1)
                        v9_cvar_bev = bev_cvar.expand(-1, nq, -1)
                        v9_cvar_img = img_cvar.expand(-1, nq, -1)
                        v9_q95_shift_bev = bev_q95_shift.expand(-1, nq, -1)
                        v9_q95_shift_img = img_q95_shift.expand(-1, nq, -1)
                        bev_anomaly_input = torch.cat([bev_z, bev_scene_tail, bev_query_tail], dim=-1)
                        if source_cls_logits is not None:
                            fused_prob = torch.sigmoid(source_cls_logits[modalities[fused_idx]]).max(
                                dim=-1, keepdim=True)[0]
                            img_prob = torch.sigmoid(source_cls_logits.get(
                                'img', source_cls_logits[modalities[fused_idx]])).max(dim=-1, keepdim=True)[0]
                            cls_drop = fused_prob - img_prob
                        else:
                            cls_drop = torch.zeros(bs, nq, 1, device=outs_dec.device)
                        img_specific = torch.cat([cls_drop, img_z[..., 9:10]], dim=-1)
                        img_anomaly_input = torch.cat([img_z, img_scene_tail, img_query_tail, img_specific], dim=-1)
                        if self.v9_anomaly_head_detach_input:
                            bev_anomaly_input = bev_anomaly_input.detach()
                            img_anomaly_input = img_anomaly_input.detach()
                        v9_a_bev = self.v9_bev_anomaly_head(torch.nan_to_num(
                            bev_anomaly_input, nan=0.0, posinf=6.0, neginf=-2.0))
                        v9_a_img = self.v9_img_anomaly_head(torch.nan_to_num(
                            img_anomaly_input, nan=0.0, posinf=6.0, neginf=-2.0))
                        v9_anchor_distance_bev = torch.sqrt(
                            torch.relu(bev_z).pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12))
                        v9_anchor_distance_img = torch.sqrt(
                            torch.relu(img_z).pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12))
                        if self.training:
                            clean_mask = self._v8_clean_sample_mask(img_metas, bs, outs_dec.device)
                            self._v9_update_clean_thresholds(
                                v9_a_bev, v9_a_img,
                                v9_anchor_distance_bev, v9_anchor_distance_img,
                                bev_z, img_z, clean_mask)
                        tau_bev = self.v9_tau_bev.to(device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                        tau_img = self.v9_tau_img.to(device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                        thr_bev = self.v9_clean_distance_threshold_bev.to(
                            device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                        thr_img = self.v9_clean_distance_threshold_img.to(
                            device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                        v9_a_bev_above_tau = (v9_a_bev >= tau_bev).to(v9_a_bev.dtype)
                        v9_a_img_above_tau = (v9_a_img >= tau_img).to(v9_a_img.dtype)
                        v9_clean_like_mask = (
                            (v9_anchor_distance_bev < thr_bev) &
                            (v9_anchor_distance_img < thr_img) &
                            (v9_a_bev < tau_bev) &
                            (v9_a_img < tau_img)
                        )
                        bev_center_delta = damage_feature_terms['bev']['a4_features'][..., 9:10]
                        img_center_delta = damage_feature_terms['img']['a4_features'][..., 9:10]
                        v9_consistency_bev = torch.exp(
                            -torch.relu(bev_center_delta) / self.v9_disagreement_temperature)
                        v9_consistency_img = torch.exp(
                            -torch.relu(img_center_delta) / self.v9_disagreement_temperature)
                        v9_mean_disagree = 1.0 - 0.5 * (v9_consistency_bev + v9_consistency_img)
                        current_epoch = max(float(getattr(self, '_current_epoch', 0)), 0.0)
                        if self.training and self.v9_route_w_warmup_epochs > 0:
                            warm = min(current_epoch / max(float(self.v9_route_w_warmup_epochs), 1.0), 1.0)
                            w_other = 0.3 + warm * (self.v9_route_w_other_damage - 0.3)
                        else:
                            w_other = self.v9_route_w_other_damage
                        bev_idx = modalities.index('bev')
                        img_idx = modalities.index('img')
                        v9_a_bev_seq = v9_a_bev.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                        v9_a_img_seq = v9_a_img.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                        v9_consistency_bev_seq = v9_consistency_bev.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                        v9_consistency_img_seq = v9_consistency_img.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                        bev_route_score_seq = (
                            w_other * v9_a_img_seq -
                            self.v9_route_w_self_damage * v9_a_bev_seq +
                            self.v9_route_w_context * context_scores_all[..., bev_idx:bev_idx + 1] +
                            self.v9_route_w_distance * distance_scores_all[..., bev_idx:bev_idx + 1] +
                            self.v9_route_w_consistency * v9_consistency_bev_seq
                        )
                        img_route_score_seq = (
                            w_other * v9_a_bev_seq -
                            self.v9_route_w_self_damage * v9_a_img_seq +
                            self.v9_route_w_context * context_scores_all[..., img_idx:img_idx + 1] +
                            self.v9_route_w_distance * distance_scores_all[..., img_idx:img_idx + 1] +
                            self.v9_route_w_consistency * v9_consistency_img_seq
                        )
                        v9_bev_score_last = bev_route_score_seq[-1]
                        v9_img_score_last = img_route_score_seq[-1]
                        v9_route_score_bev = v9_bev_score_last
                        v9_route_score_img = v9_img_score_last
                        v8_route_seq = torch.cat([bev_route_score_seq, img_route_score_seq], dim=-1)
                        v8_non_fused_probs = F.softmax(v8_route_seq / self.v9_route_temperature, dim=-1)
                        bev_damage = v9_a_bev
                        img_damage = v9_a_img
                        bev_reliability = (1.0 - v9_a_bev).clamp(min=1e-6, max=1.0 - 1e-6)
                        img_reliability = (1.0 - v9_a_img).clamp(min=1e-6, max=1.0 - 1e-6)
                        a4_route_score_bev = v9_route_score_bev
                        a4_route_score_img = v9_route_score_img
                    else:
                        damage_weights = F.softplus(self.v8_a4_damage_feature_weights_raw)
                        if self.v8_a3_damage_context_head is not None:
                            context_damage_logits = self.v8_a3_damage_context_head(outs_dec[-1])
                            zero_centered_softplus = (
                                F.softplus(context_damage_logits) -
                                context_damage_logits.new_tensor(0.6931471805599453)
                            )
                            context_bev_damage = zero_centered_softplus[..., 0:1] * self.v8_a3_context_damage_scale
                            context_img_damage = zero_centered_softplus[..., 1:2] * self.v8_a3_context_damage_scale
                        bev_damage_logit = (
                            bev_z_pos * damage_weights[0].view(1, 1, -1)
                        ).sum(dim=-1, keepdim=True) + self.v8_a4_damage_feature_bias[0] + context_bev_damage
                        img_damage_logit = (
                            img_z_pos * damage_weights[1].view(1, 1, -1)
                        ).sum(dim=-1, keepdim=True) + self.v8_a4_damage_feature_bias[1] + context_img_damage
                        bev_damage = F.softplus(bev_damage_logit)
                        img_damage = F.softplus(img_damage_logit)
                elif use_v8_reliability:
                    bev_features = _v8_damage_features('bev')
                    img_features = _v8_damage_features('img')
                    damage_weights = F.softplus(self.v8_damage_feature_weights_raw)
                    bev_raw_damage = (
                        bev_features * damage_weights[0].view(1, 1, -1)
                    ).sum(dim=-1, keepdim=True)
                    img_raw_damage = (
                        img_features * damage_weights[1].view(1, 1, -1)
                    ).sum(dim=-1, keepdim=True)
                    if use_v8_a3:
                        def _robust_normalize_damage_features(features):
                            ref = torch.nan_to_num(
                                features.detach(), nan=0.0, posinf=1e4, neginf=0.0)
                            ref_float = ref.float()
                            q50 = torch.quantile(ref_float, 0.50, dim=1, keepdim=True).to(features.dtype)
                            q90 = torch.quantile(ref_float, 0.90, dim=1, keepdim=True).to(features.dtype)
                            scale = (q90 - q50).clamp(min=1e-4)
                            normalized = (features - q50) / scale
                            return torch.nan_to_num(
                                normalized, nan=0.0, posinf=3.0, neginf=0.0
                            ).clamp(min=0.0, max=3.0)

                        bev_norm_features = _robust_normalize_damage_features(bev_features)
                        img_norm_features = _robust_normalize_damage_features(img_features)
                        bev_norm_damage = (
                            bev_norm_features * damage_weights[0].view(1, 1, -1)
                        ).sum(dim=-1, keepdim=True)
                        img_norm_damage = (
                            img_norm_features * damage_weights[1].view(1, 1, -1)
                        ).sum(dim=-1, keepdim=True)
                        if self.v8_a3_damage_context_head is not None:
                            context_damage_logits = self.v8_a3_damage_context_head(outs_dec[-1])
                            zero_centered_softplus = (
                                F.softplus(context_damage_logits) -
                                context_damage_logits.new_tensor(0.6931471805599453)
                            )
                            context_damage = zero_centered_softplus * self.v8_a3_context_damage_scale
                            context_bev_damage = context_damage[..., 0:1]
                            context_img_damage = context_damage[..., 1:2]
                        bev_damage = (
                            bev_norm_damage +
                            self.v8_a3_raw_damage_weight * bev_raw_damage +
                            self.v8_damage_feature_bias[0] +
                            context_bev_damage
                        ).clamp(min=0.0)
                        img_damage = (
                            img_norm_damage +
                            self.v8_a3_raw_damage_weight * img_raw_damage +
                            self.v8_damage_feature_bias[1] +
                            context_img_damage
                        ).clamp(min=0.0)
                    else:
                        bev_damage = bev_raw_damage + self.v8_damage_feature_bias[0]
                        img_damage = img_raw_damage + self.v8_damage_feature_bias[1]
                    bev_damage = torch.nan_to_num(bev_damage, nan=0.0, posinf=1e4, neginf=0.0)
                    img_damage = torch.nan_to_num(img_damage, nan=0.0, posinf=1e4, neginf=0.0)
                else:
                    bev_damage = _v8_damage_score('bev')
                    img_damage = _v8_damage_score('img')

                dual_damage = torch.minimum(bev_damage, img_damage)
                if use_v9:
                    reliability_margin = torch.abs(bev_reliability - img_reliability)
                elif use_v8_a4:
                    reliability_prior = torch.logit(bev_damage.new_tensor(0.92))
                    bev_reliability = torch.sigmoid(
                        reliability_prior - bev_damage / self.v8_a4_reliability_temperature)
                    img_reliability = torch.sigmoid(
                        reliability_prior - img_damage / self.v8_a4_reliability_temperature)
                    reliability_margin = torch.abs(bev_reliability - img_reliability)
                else:
                    reliability_prior = torch.logit(
                        bev_damage.new_tensor(self.v8_reliability_init))
                    bev_reliability = torch.sigmoid(reliability_prior - bev_damage)
                    img_reliability = torch.sigmoid(reliability_prior - img_damage)
                    reliability_margin = torch.abs(bev_reliability - img_reliability)

                if use_v9:
                    pass
                elif use_v8_a4:
                    bev_idx = modalities.index('bev')
                    img_idx = modalities.index('img')
                    bev_rel_logit = torch.logit(
                        bev_reliability.clamp(min=1e-6, max=1.0 - 1e-6)
                    ).unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    img_rel_logit = torch.logit(
                        img_reliability.clamp(min=1e-6, max=1.0 - 1e-6)
                    ).unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    bev_center_disagreement_z = torch.relu(bev_z[..., 9:10]).unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    img_center_disagreement_z = torch.relu(img_z[..., 9:10]).unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    bev_route_score_seq = (
                        bev_rel_logit +
                        self.v8_a4_context_route_weight * context_scores_all[..., bev_idx:bev_idx + 1] +
                        self.v8_a4_distance_route_weight * distance_scores_all[..., bev_idx:bev_idx + 1] -
                        self.v8_a4_center_disagreement_route_weight * bev_center_disagreement_z
                    )
                    img_route_score_seq = (
                        img_rel_logit +
                        self.v8_a4_context_route_weight * context_scores_all[..., img_idx:img_idx + 1] +
                        self.v8_a4_distance_route_weight * distance_scores_all[..., img_idx:img_idx + 1] -
                        self.v8_a4_center_disagreement_route_weight * img_center_disagreement_z
                    )
                    a4_route_score_bev = bev_route_score_seq[-1]
                    a4_route_score_img = img_route_score_seq[-1]
                    v8_route_seq = torch.cat([bev_route_score_seq, img_route_score_seq], dim=-1)
                    v8_non_fused_probs = F.softmax(
                        v8_route_seq / self.v8_a4_route_temperature, dim=-1)
                elif use_v8_reliability:
                    v8_route_seq = torch.stack([bev_reliability, img_reliability], dim=-1).squeeze(-2)
                    v8_route_seq = v8_route_seq.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    v8_non_fused_probs = F.softmax(
                        v8_route_seq / self.v8_reliability_temperature, dim=-1)
                else:
                    v8_damage_seq = torch.stack([bev_damage, img_damage], dim=-1).squeeze(-2)
                    v8_damage_seq = v8_damage_seq.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    v8_non_fused_probs = F.softmax(
                        -v8_damage_seq / self.v8_damage_temperature, dim=-1)

                top2 = torch.topk(v8_non_fused_probs, k=min(2, v8_non_fused_probs.shape[-1]), dim=-1).values
                if top2.shape[-1] == 1:
                    non_fused_margin = torch.ones_like(v8_non_fused_probs[..., :1])
                else:
                    non_fused_margin = (top2[..., 0:1] - top2[..., 1:2])
                if use_v9:
                    gate_input_last = torch.cat([
                        v9_a_bev,
                        v9_a_img,
                        torch.minimum(v9_a_bev, v9_a_img),
                        torch.maximum(v9_a_bev, v9_a_img),
                        torch.abs(v9_a_bev - v9_a_img),
                        non_fused_margin[-1],
                        v9_mean_disagree,
                    ], dim=-1)
                elif use_v8_reliability:
                    gate_input_last = torch.cat([
                        bev_reliability,
                        img_reliability,
                        torch.minimum(bev_reliability, img_reliability),
                        torch.maximum(bev_reliability, img_reliability),
                        reliability_margin,
                        bev_reliability * img_reliability,
                    ], dim=-1)
                else:
                    gate_input_last = torch.cat([
                        bev_damage,
                        img_damage,
                        dual_damage,
                        torch.abs(bev_damage - img_damage),
                        torch.maximum(bev_damage, img_damage),
                        non_fused_margin[-1],
                    ], dim=-1)
                gate_input = gate_input_last.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                gate_head = self.v9_fused_gate_head if use_v9 else self.v8_fused_gate_head
                fused_keep_unit = torch.sigmoid(gate_head(gate_input))
                fused_keep_at_min = (fused_keep_unit <= 0.05).to(fused_keep_unit.dtype)
                fused_keep = (
                    self.v8_fused_keep_min +
                    (self.v8_fused_keep_max - self.v8_fused_keep_min) * fused_keep_unit
                )
                if use_v9:
                    floor = fused_keep.new_tensor(self.v9_clean_fused_keep_floor)
                    clean_floor = torch.maximum(fused_keep, floor.expand_as(fused_keep))
                    fused_keep = torch.where(
                        v9_clean_like_mask.unsqueeze(0).expand_as(fused_keep),
                        clean_floor,
                        fused_keep)
                final_weight_seq = torch.zeros(
                    nb_dec, bs, nq, len(self.modality_names),
                    device=outs_dec.device,
                    dtype=v8_non_fused_probs.dtype)
                final_weight_seq[..., fused_idx] = fused_keep[..., 0]
                non_fused_indices = [modalities.index(modality) for modality in non_fused_modalities]
                for local_idx, global_idx in enumerate(non_fused_indices):
                    final_weight_seq[..., global_idx] = (
                        (1.0 - fused_keep[..., 0]) * v8_non_fused_probs[..., local_idx]
                    )
                selector_probs = final_weight_seq
                if use_v9 and self.v9_emit_velocity_diagnostic:
                    softmix_vel = self._blend_source_tensor(
                        source_outs, 'vel', final_weight_seq, modalities,
                        num_queries_per_modality, use_selector=True)
                    fused_vel = self._get_single_source_tensor(
                        source_outs, 'vel', 'fused', modalities, num_queries_per_modality)
                    if softmix_vel is not None and fused_vel is not None:
                        v9_velocity_l1 = torch.abs(
                            softmix_vel[-1] - fused_vel[-1]
                        ).mean(dim=-1, keepdim=True)

                if use_v9:
                    tau_bev = self.v9_tau_bev.to(device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                    tau_img = self.v9_tau_img.to(device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                    clean_like = v9_clean_like_mask[..., 0]
                    bev_damaged = (
                        (v9_a_bev >= tau_bev) &
                        (v9_a_img < tau_img)
                    )[..., 0]
                    img_damaged = (
                        (v9_a_img >= tau_img) &
                        (v9_a_bev < tau_bev)
                    )[..., 0]
                    dual_damaged = (
                        (v9_a_bev >= tau_bev) &
                        (v9_a_img >= tau_img)
                    )[..., 0]
                elif use_v8_a4:
                    damage_delta = bev_damage - img_damage
                    clean_like = (
                        (bev_damage < self.v8_a4_clean_damage_threshold) &
                        (img_damage < self.v8_a4_clean_damage_threshold)
                    )[..., 0]
                    bev_damaged = (
                        (bev_damage >= self.v8_a4_single_damage_threshold) &
                        (img_damage < self.v8_a4_single_safe_threshold) &
                        (damage_delta >= self.v8_a4_damage_margin)
                    )[..., 0]
                    img_damaged = (
                        (img_damage >= self.v8_a4_single_damage_threshold) &
                        (bev_damage < self.v8_a4_single_safe_threshold) &
                        ((-damage_delta) >= self.v8_a4_damage_margin)
                    )[..., 0]
                    dual_damaged = (
                        (bev_damage >= self.v8_a4_dual_damage_threshold) &
                        (img_damage >= self.v8_a4_dual_damage_threshold)
                    )[..., 0]
                elif use_v8_reliability:
                    def _batch_quantile(values):
                        flat = values.squeeze(-1)
                        q = torch.quantile(flat, self.v8_unreliable_quantile, dim=1, keepdim=True)
                        return q.unsqueeze(-1)

                    bev_unreliable = 1.0 - bev_reliability
                    img_unreliable = 1.0 - img_reliability
                    unreliable_min = (
                        self.v8_a3_unreliable_min if use_v8_a3 else self.v8_unreliable_min
                    )
                    bev_thr = torch.maximum(
                        _batch_quantile(bev_unreliable),
                        bev_unreliable.new_tensor(unreliable_min))
                    img_thr = torch.maximum(
                        _batch_quantile(img_unreliable),
                        img_unreliable.new_tensor(unreliable_min))
                    bev_unreliable_high = bev_unreliable >= bev_thr
                    img_unreliable_high = img_unreliable >= img_thr
                    reliability_delta = img_reliability - bev_reliability
                    both_reliable = (
                        (bev_reliability >= self.v8_reliable_threshold) &
                        (img_reliability >= self.v8_reliable_threshold)
                    )
                    bev_damaged = (
                        bev_unreliable_high &
                        (reliability_delta >= self.v8_reliability_margin)
                    )[..., 0]
                    img_damaged = (
                        img_unreliable_high &
                        ((-reliability_delta) >= self.v8_reliability_margin)
                    )[..., 0]
                    dual_damaged = (
                        bev_unreliable_high &
                        img_unreliable_high &
                        (~bev_damaged.unsqueeze(-1)) &
                        (~img_damaged.unsqueeze(-1))
                    )[..., 0]
                    clean_like = both_reliable[..., 0]
                else:
                    bev_damaged = bev_damage[..., 0] > self.v8_high_damage_threshold
                    img_damaged = img_damage[..., 0] > self.v8_high_damage_threshold
                    dual_damaged = bev_damaged & img_damaged
                    clean_like = ~(bev_damaged | img_damaged)

                damage_state_idx = torch.zeros(bs, nq, dtype=torch.long, device=outs_dec.device)
                damage_state_idx = torch.where(
                    bev_damaged & (~img_damaged),
                    damage_state_idx.new_full(damage_state_idx.shape, 1),
                    damage_state_idx)
                damage_state_idx = torch.where(
                    img_damaged & (~bev_damaged),
                    damage_state_idx.new_full(damage_state_idx.shape, 2),
                    damage_state_idx)
                damage_state_idx = torch.where(
                    dual_damaged,
                    damage_state_idx.new_full(damage_state_idx.shape, 3),
                    damage_state_idx)
                if use_v8_reliability:
                    assigned_mask = clean_like | (damage_state_idx > 0)
                    damage_state_idx = torch.where(
                        ~assigned_mask,
                        damage_state_idx.new_full(damage_state_idx.shape, 4),
                        damage_state_idx)
                high_damage_mask = (damage_state_idx > 0) & (damage_state_idx < 4)
                non_fused_choice_idx = v8_non_fused_probs[-1].argmax(dim=-1)
                routing_index_tensor = torch.tensor(
                    non_fused_indices, device=outs_dec.device, dtype=torch.long)
                non_fused_choice_global_idx = routing_index_tensor[non_fused_choice_idx]
                final_weight_last = final_weight_seq[-1]
                cls_weight_seq = final_weight_seq
                height_weight_seq = final_weight_seq
                vel_weight_seq = None
                cls_fused_keep = fused_keep
                height_fused_keep = fused_keep
                vel_fused_keep = fused_keep
                center_cap_fused_keep = fused_keep
                v9_state_classification = damage_state_idx.new_full(damage_state_idx.shape, 4)
                v9_cap_applied = fused_keep.new_zeros(bs, nq, 1)
                v9_center_safe_skipped = fused_keep.new_zeros(bs, nq, 1)
                v9_state_clean_like = fused_keep.new_zeros(bs, nq, 1)
                v9_state_confirmed_img = fused_keep.new_zeros(bs, nq, 1)
                v9_state_confirmed_bev = fused_keep.new_zeros(bs, nq, 1)
                v9_state_confirmed_dual = fused_keep.new_zeros(bs, nq, 1)
                v9_state_neutral = fused_keep.new_ones(bs, nq, 1)
                v9_soft_weight_fused_cls = final_weight_last[..., fused_idx:fused_idx + 1]
                v9_soft_weight_bev_cls = final_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                v9_soft_weight_img_cls = final_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                v9_soft_weight_fused_height = v9_soft_weight_fused_cls
                v9_soft_weight_bev_height = v9_soft_weight_bev_cls
                v9_soft_weight_img_height = v9_soft_weight_img_cls
                v9_soft_weight_fused_vel = fused_keep.new_ones(bs, nq, 1)
                v9_soft_weight_bev_vel = fused_keep.new_zeros(bs, nq, 1)
                v9_soft_weight_img_vel = fused_keep.new_zeros(bs, nq, 1)
                v9_soft_weight_fused_center = final_weight_last[..., fused_idx:fused_idx + 1]
                v9_soft_weight_bev_center = final_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                v9_soft_weight_img_center = final_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                if use_v9:
                    tau_bev_state = self.v9_tau_bev.to(
                        device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                    tau_img_state = self.v9_tau_img.to(
                        device=outs_dec.device, dtype=outs_dec.dtype).view(1, 1, 1)
                    nf_bev_prob = fused_keep.new_zeros(bs, nq, 1)
                    nf_img_prob = fused_keep.new_zeros(bs, nq, 1)
                    if 'bev' in non_fused_modalities:
                        nf_bev_prob = v8_non_fused_probs[-1, ..., non_fused_modalities.index('bev'):non_fused_modalities.index('bev') + 1]
                    if 'img' in non_fused_modalities:
                        nf_img_prob = v8_non_fused_probs[-1, ..., non_fused_modalities.index('img'):non_fused_modalities.index('img') + 1]
                    clean_state = v9_clean_like_mask.bool()
                    confirmed_img = (
                        (v9_a_img >= tau_img_state) &
                        (v9_a_bev < tau_bev_state) &
                        (nf_bev_prob > 0.5)
                    )
                    confirmed_bev = (
                        (v9_a_bev >= tau_bev_state) &
                        (v9_a_img < tau_img_state) &
                        (nf_img_prob > 0.5)
                    )
                    confirmed_dual = (
                        (v9_a_bev >= tau_bev_state) &
                        (v9_a_img >= tau_img_state)
                    )
                    confirmed_img = confirmed_img & (~clean_state)
                    confirmed_bev = confirmed_bev & (~clean_state)
                    confirmed_dual = confirmed_dual & (~clean_state)
                    assigned_state = clean_state | confirmed_img | confirmed_bev | confirmed_dual
                    neutral_state = ~assigned_state
                    v9_state_classification = torch.where(
                        clean_state[..., 0],
                        v9_state_classification.new_zeros(v9_state_classification.shape),
                        v9_state_classification)
                    v9_state_classification = torch.where(
                        confirmed_img[..., 0],
                        v9_state_classification.new_full(v9_state_classification.shape, 1),
                        v9_state_classification)
                    v9_state_classification = torch.where(
                        confirmed_bev[..., 0],
                        v9_state_classification.new_full(v9_state_classification.shape, 2),
                        v9_state_classification)
                    v9_state_classification = torch.where(
                        confirmed_dual[..., 0],
                        v9_state_classification.new_full(v9_state_classification.shape, 3),
                        v9_state_classification)
                    v9_state_clean_like = clean_state.to(fused_keep.dtype)
                    v9_state_confirmed_img = confirmed_img.to(fused_keep.dtype)
                    v9_state_confirmed_bev = confirmed_bev.to(fused_keep.dtype)
                    v9_state_confirmed_dual = confirmed_dual.to(fused_keep.dtype)
                    v9_state_neutral = neutral_state.to(fused_keep.dtype)

                    def _weights_from_fused_keep(head_keep):
                        head_weights = torch.zeros_like(final_weight_seq)
                        head_weights[..., fused_idx] = head_keep[..., 0]
                        for local_idx, global_idx in enumerate(non_fused_indices):
                            head_weights[..., global_idx] = (
                                (1.0 - head_keep[..., 0]) *
                                v8_non_fused_probs[..., local_idx]
                            )
                        return head_weights

                    def _apply_v9_cap(head_keep, cap_value):
                        cap = head_keep.new_tensor(cap_value)
                        dual_cap = cap * head_keep.new_tensor(self.v9_cap_dual_scale)
                        clean_floor_value = head_keep.new_tensor(self.v9_clean_fused_keep_floor)
                        clean_keep = torch.maximum(
                            head_keep,
                            clean_floor_value.expand_as(head_keep))
                        single_capped = torch.minimum(head_keep, cap.expand_as(head_keep))
                        dual_capped = torch.minimum(head_keep, dual_cap.expand_as(head_keep))
                        clean_seq = clean_state.unsqueeze(0).expand_as(head_keep)
                        single_seq = (confirmed_img | confirmed_bev).unsqueeze(0).expand_as(head_keep)
                        dual_seq = confirmed_dual.unsqueeze(0).expand_as(head_keep)
                        capped_keep = torch.where(clean_seq, clean_keep, head_keep)
                        capped_keep = torch.where(single_seq, single_capped, capped_keep)
                        capped_keep = torch.where(dual_seq, dual_capped, capped_keep)
                        return capped_keep

                    if (
                            self.v9_enable_per_head_dispatch and
                            self.v9_cap_mode in ('safe', 'hard')):
                        cls_fused_keep = _apply_v9_cap(fused_keep, self.v9_cap_cls)
                        height_fused_keep = _apply_v9_cap(fused_keep, self.v9_cap_height)
                        cls_weight_seq = _weights_from_fused_keep(cls_fused_keep)
                        height_weight_seq = _weights_from_fused_keep(height_fused_keep)
                        if self.v9_cap_vel is not None:
                            vel_fused_keep = _apply_v9_cap(fused_keep, self.v9_cap_vel)
                            vel_weight_seq = _weights_from_fused_keep(vel_fused_keep)
                        if self.v9_cap_mode == 'hard' and self.v9_cap_center is not None:
                            center_cap_fused_keep = _apply_v9_cap(fused_keep, self.v9_cap_center)
                        v9_cap_applied = (
                            ((cls_fused_keep[-1] + 1e-6) < fused_keep[-1]) |
                            ((height_fused_keep[-1] + 1e-6) < fused_keep[-1]) |
                            ((vel_fused_keep[-1] + 1e-6) < fused_keep[-1]) |
                            ((center_cap_fused_keep[-1] + 1e-6) < fused_keep[-1])
                        ).to(fused_keep.dtype)
                        if self.v9_cap_mode == 'hard':
                            v9_center_safe_skipped = (
                                confirmed_img | confirmed_bev | confirmed_dual
                            ).to(fused_keep.dtype)
                        cls_weight_last = cls_weight_seq[-1]
                        height_weight_last = height_weight_seq[-1]
                        v9_soft_weight_fused_cls = cls_weight_last[..., fused_idx:fused_idx + 1]
                        v9_soft_weight_bev_cls = cls_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                        v9_soft_weight_img_cls = cls_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                        v9_soft_weight_fused_height = height_weight_last[..., fused_idx:fused_idx + 1]
                        v9_soft_weight_bev_height = height_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                        v9_soft_weight_img_height = height_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                        if vel_weight_seq is not None:
                            vel_weight_last = vel_weight_seq[-1]
                            v9_soft_weight_fused_vel = vel_weight_last[..., fused_idx:fused_idx + 1]
                            v9_soft_weight_bev_vel = vel_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                            v9_soft_weight_img_vel = vel_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                center_fused_keep = fused_keep
                center_route_allowed = torch.zeros(bs, nq, 1, dtype=torch.bool, device=outs_dec.device)
                center_weight_seq = final_weight_seq
                v10_info = None
                if use_v8_a3 or use_v8_a4 or use_v9:
                    single_damage_mask = (
                        (damage_state_idx == 1) | (damage_state_idx == 2)
                    ).unsqueeze(-1)
                    center_route_allowed = (
                        single_damage_mask &
                        (reliability_margin >= self.v8_a3_center_route_margin) &
                        (non_fused_margin[-1] >= self.v8_a3_center_non_fused_margin)
                    )
                    if use_v9 and self.v9_center_domain_consistent_only:
                        v9_domain_match = (
                            ((damage_state_idx == 1).unsqueeze(-1) & (v9_a_bev_above_tau > 0.5) & (v9_a_img_above_tau <= 0.5)) |
                            ((damage_state_idx == 2).unsqueeze(-1) & (v9_a_img_above_tau > 0.5) & (v9_a_bev_above_tau <= 0.5))
                        )
                        center_route_allowed = center_route_allowed & v9_domain_match
                    safe_floor = fused_keep.new_full(
                        (bs, nq, 1), self.v8_a3_center_safe_keep_floor)
                    single_floor = fused_keep.new_full(
                        (bs, nq, 1), self.v8_a3_center_single_damage_keep_floor)
                    center_floor = torch.where(center_route_allowed, single_floor, safe_floor)
                    center_fused_keep = torch.maximum(
                        fused_keep, center_floor.unsqueeze(0).expand_as(fused_keep))
                    if (
                            use_v9 and
                            self.v9_enable_per_head_dispatch and
                            self.v9_cap_mode == 'hard' and
                            self.v9_cap_center is not None):
                        hard_center_state = (
                            (v9_state_confirmed_img > 0.5) |
                            (v9_state_confirmed_bev > 0.5) |
                            (v9_state_confirmed_dual > 0.5)
                        ).unsqueeze(0).expand_as(center_fused_keep)
                        center_fused_keep = torch.where(
                            hard_center_state,
                            center_cap_fused_keep,
                            center_fused_keep)
                    center_weight_seq = torch.zeros_like(final_weight_seq)
                    center_weight_seq[..., fused_idx] = center_fused_keep[..., 0]
                    for local_idx, global_idx in enumerate(non_fused_indices):
                        center_weight_seq[..., global_idx] = (
                            (1.0 - center_fused_keep[..., 0]) *
                            v8_non_fused_probs[..., local_idx]
                        )
                if (
                        self.v10_enable and
                        self.v10_dd_head is not None and
                        self.v10_router_head_cls is not None and
                        self.v10_router_head_height is not None and
                        self.v10_router_head_center is not None and
                        self.v10_center_safe_gate is not None):
                    def _score_term(modality, name):
                        return score_component_terms.get(modality, {}).get(name)

                    def _damage_term(modality, name):
                        return damage_feature_terms.get(modality, {}).get(name)

                    def _dist_score(modality):
                        if modality not in modalities:
                            return None
                        idx = modalities.index(modality)
                        if idx >= len(distance_prior_scores):
                            return None
                        return distance_prior_scores[idx]

                    zero_q = fused_keep.new_zeros(bs, nq, 1)
                    bev_idx = modalities.index('bev') if 'bev' in modalities else None
                    img_idx = modalities.index('img') if 'img' in modalities else None
                    nf_bev = (
                        v8_non_fused_probs[-1, ..., non_fused_modalities.index('bev'):non_fused_modalities.index('bev') + 1]
                        if 'bev' in non_fused_modalities else zero_q)
                    nf_img = (
                        v8_non_fused_probs[-1, ..., non_fused_modalities.index('img'):non_fused_modalities.index('img') + 1]
                        if 'img' in non_fused_modalities else zero_q)
                    dd_feature_terms = [
                        bev_damage, img_damage, dual_damage,
                        bev_reliability, img_reliability, reliability_margin,
                        _damage_term('bev', 'a4_features'),
                        _damage_term('img', 'a4_features'),
                        _damage_term('bev', 'conditioned_unc'),
                        _damage_term('img', 'conditioned_unc'),
                        _damage_term('bev', 'percentile_unc'),
                        _damage_term('img', 'percentile_unc'),
                        _damage_term('bev', 'disagreement'),
                        _damage_term('img', 'disagreement'),
                        _damage_term('bev', 'cls_gap'),
                        _damage_term('img', 'cls_gap'),
                        a4_bev_damage_z, a4_img_damage_z,
                        v9_a_bev, v9_a_img,
                        v9_anchor_distance_bev, v9_anchor_distance_img,
                        v9_consistency_bev, v9_consistency_img,
                        v9_tail_prob_bev, v9_tail_prob_img,
                        v9_tail_excess_bev, v9_tail_excess_img,
                        v9_route_score_bev, v9_route_score_img,
                        nf_bev, nf_img,
                    ]
                    dd_features = self._v10_pack_features(
                        dd_feature_terms, self.v10_dd_input_dim,
                        bs, nq, outs_dec.device, outs_dec.dtype,
                        detach=self.v10_input_detach)
                    p_dmg = self.v10_dd_head(dd_features)

                    context_last = context_logits[-1] if isinstance(context_logits, torch.Tensor) else None
                    router_feature_terms = [
                        p_dmg,
                        bev_damage, img_damage, dual_damage,
                        bev_reliability, img_reliability, reliability_margin,
                        a4_route_score_bev, a4_route_score_img,
                        v9_a_bev, v9_a_img,
                        v9_route_score_bev, v9_route_score_img,
                        v9_mean_disagree,
                        v9_velocity_l1,
                        v9_tail_prob_bev, v9_tail_prob_img,
                        v9_tail_excess_bev, v9_tail_excess_img,
                        nf_bev, nf_img,
                        context_last,
                        _dist_score('fused'),
                        _dist_score('bev'),
                        _dist_score('img'),
                        _score_term('bev', 'disagreement'),
                        _score_term('img', 'disagreement'),
                    ]
                    router_features = self._v10_pack_features(
                        router_feature_terms, self.v10_router_input_dim,
                        bs, nq, outs_dec.device, outs_dec.dtype,
                        detach=self.v10_input_detach)
                    delta_cls = self.v10_router_head_cls(router_features)
                    delta_height = self.v10_router_head_height(router_features)
                    delta_center = self.v10_router_head_center(router_features)

                    csg_feature_terms = [
                        p_dmg[..., 0:1],
                        p_dmg[..., 1:2],
                        p_dmg[..., 2:3],
                        reliability_margin,
                        center_route_allowed.to(dtype=outs_dec.dtype),
                        v9_mean_disagree,
                        nf_bev,
                        nf_img,
                    ]
                    csg_features = self._v10_pack_features(
                        csg_feature_terms, self.v10_csg_input_dim,
                        bs, nq, outs_dec.device, outs_dec.dtype,
                        detach=self.v10_input_detach)
                    center_floor = 0.75 + 0.25 * torch.sigmoid(
                        self.v10_center_safe_gate(csg_features))

                    keep_unit = fused_keep_unit.clamp(1e-4, 1.0 - 1e-4)
                    v9_base_keep_logit = torch.logit(keep_unit)
                    base_non_fused_logits = torch.log(v8_non_fused_probs.clamp(min=1e-6))

                    def _expand_delta(delta):
                        return delta.unsqueeze(0).expand(nb_dec, -1, -1, -1)

                    def _weights_from_keep_and_probs(head_keep, non_fused_probs):
                        head_weights = torch.zeros_like(final_weight_seq)
                        head_weights[..., fused_idx] = head_keep[..., 0]
                        for local_idx, global_idx in enumerate(non_fused_indices):
                            head_weights[..., global_idx] = (
                                (1.0 - head_keep[..., 0]) *
                                non_fused_probs[..., local_idx]
                            )
                        return head_weights

                    def _v10_head_weights(delta, floor_value):
                        delta_seq = _expand_delta(delta)
                        floor = fused_keep.new_tensor(float(floor_value))
                        floor = torch.clamp(floor, min=0.0, max=self.v8_fused_keep_max - 1e-6)
                        keep = floor + (
                            self.v8_fused_keep_max - floor
                        ) * torch.sigmoid(v9_base_keep_logit + delta_seq[..., 0:1])
                        keep = keep.clamp(0.0, 1.0)
                        nf_probs = torch.softmax(
                            base_non_fused_logits + delta_seq[..., 1:3],
                            dim=-1)
                        return keep, nf_probs, _weights_from_keep_and_probs(keep, nf_probs)

                    cls_fused_keep, cls_nf_probs, cls_weight_seq = _v10_head_weights(
                        delta_cls, self.v10_cls_fused_keep_min)
                    height_fused_keep, height_nf_probs, height_weight_seq = _v10_head_weights(
                        delta_height, self.v10_cls_fused_keep_min)

                    delta_center_seq = _expand_delta(delta_center)
                    center_floor_seq = center_floor.unsqueeze(0).expand(nb_dec, -1, -1, -1)
                    v9_floor = fused_keep.new_tensor(float(self.v8_fused_keep_min)).expand_as(center_floor_seq)
                    center_min = torch.maximum(v9_floor, center_floor_seq)
                    center_span = (self.v8_fused_keep_max - center_min).clamp(min=1e-6)
                    center_raw = center_min + center_span * torch.sigmoid(
                        v9_base_keep_logit + delta_center_seq[..., 0:1])
                    center_fused_keep = torch.maximum(
                        center_raw, center_floor_seq).clamp(0.0, 1.0)
                    center_nf_probs = torch.softmax(
                        base_non_fused_logits + delta_center_seq[..., 1:3],
                        dim=-1)
                    center_weight_seq = _weights_from_keep_and_probs(
                        center_fused_keep, center_nf_probs)

                    cls_last = cls_weight_seq[-1]
                    height_last = height_weight_seq[-1]
                    center_last = center_weight_seq[-1]
                    v9_soft_weight_fused_cls = cls_last[..., fused_idx:fused_idx + 1]
                    v9_soft_weight_bev_cls = cls_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                    v9_soft_weight_img_cls = cls_last[..., modalities.index('img'):modalities.index('img') + 1]
                    v9_soft_weight_fused_height = height_last[..., fused_idx:fused_idx + 1]
                    v9_soft_weight_bev_height = height_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                    v9_soft_weight_img_height = height_last[..., modalities.index('img'):modalities.index('img') + 1]
                    v9_soft_weight_fused_center = center_last[..., fused_idx:fused_idx + 1]
                    v9_soft_weight_bev_center = center_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                    v9_soft_weight_img_center = center_last[..., modalities.index('img'):modalities.index('img') + 1]
                    v10_info = {
                        'v10_p_dmg_bev': p_dmg[..., 0:1],
                        'v10_p_dmg_img': p_dmg[..., 1:2],
                        'v10_p_dmg_dual': p_dmg[..., 2:3],
                        'v10_w_fused_cls': v9_soft_weight_fused_cls,
                        'v10_w_bev_cls': v9_soft_weight_bev_cls,
                        'v10_w_img_cls': v9_soft_weight_img_cls,
                        'v10_w_fused_height': v9_soft_weight_fused_height,
                        'v10_w_bev_height': v9_soft_weight_bev_height,
                        'v10_w_img_height': v9_soft_weight_img_height,
                        'v10_w_fused_center': v9_soft_weight_fused_center,
                        'v10_w_bev_center': v9_soft_weight_bev_center,
                        'v10_w_img_center': v9_soft_weight_img_center,
                        'v10_center_floor': center_floor,
                        'v10_cls_fused_keep_min_current': fused_keep.new_tensor(
                            float(self.v10_cls_fused_keep_min)),
                    }
                force_modality_active = self.force_modality is not None
                forced_branch = self.force_modality
                force_weight_seq = None
                if force_modality_active:
                    if forced_branch not in modalities:
                        raise ValueError(
                            f"force_modality={forced_branch} is unavailable in modalities={modalities}")
                    force_idx = modalities.index(forced_branch)
                    force_weight_seq = torch.zeros_like(final_weight_seq)
                    force_weight_seq[..., force_idx] = 1.0
                    final_weight_seq = force_weight_seq
                    selector_probs = force_weight_seq
                    cls_weight_seq = force_weight_seq
                    height_weight_seq = force_weight_seq
                    center_weight_seq = force_weight_seq
                    vel_weight_seq = force_weight_seq
                    cls_fused_keep = force_weight_seq[..., fused_idx:fused_idx + 1]
                    height_fused_keep = cls_fused_keep
                    center_fused_keep = cls_fused_keep
                    vel_fused_keep = cls_fused_keep
                    center_cap_fused_keep = cls_fused_keep
                    forced_last = force_weight_seq[-1]
                    v9_soft_weight_fused_cls = forced_last[..., fused_idx:fused_idx + 1]
                    v9_soft_weight_bev_cls = forced_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                    v9_soft_weight_img_cls = forced_last[..., modalities.index('img'):modalities.index('img') + 1]
                    v9_soft_weight_fused_height = v9_soft_weight_fused_cls
                    v9_soft_weight_bev_height = v9_soft_weight_bev_cls
                    v9_soft_weight_img_height = v9_soft_weight_img_cls
                    v9_soft_weight_fused_center = v9_soft_weight_fused_cls
                    v9_soft_weight_bev_center = v9_soft_weight_bev_cls
                    v9_soft_weight_img_center = v9_soft_weight_img_cls
                    v9_soft_weight_fused_vel = v9_soft_weight_fused_cls
                    v9_soft_weight_bev_vel = v9_soft_weight_bev_cls
                    v9_soft_weight_img_vel = v9_soft_weight_img_cls
                    if v10_info is not None:
                        v10_info.update({
                            'v10_w_fused_cls': v9_soft_weight_fused_cls,
                            'v10_w_bev_cls': v9_soft_weight_bev_cls,
                            'v10_w_img_cls': v9_soft_weight_img_cls,
                            'v10_w_fused_height': v9_soft_weight_fused_height,
                            'v10_w_bev_height': v9_soft_weight_bev_height,
                            'v10_w_img_height': v9_soft_weight_img_height,
                            'v10_w_fused_center': v9_soft_weight_fused_center,
                            'v10_w_bev_center': v9_soft_weight_bev_center,
                            'v10_w_img_center': v9_soft_weight_img_center,
                        })
                center_weight_last = center_weight_seq[-1]
                final_weight_last = final_weight_seq[-1]
                v9_soft_weight_fused_center = center_weight_last[..., fused_idx:fused_idx + 1]
                v9_soft_weight_bev_center = center_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                v9_soft_weight_img_center = center_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                v9_center_safe_applied = (
                    ((center_fused_keep[-1] + 1e-6) > fused_keep[-1]) &
                    (v9_center_safe_skipped <= 0.5)
                ).to(fused_keep.dtype)
                center_source_idx = center_weight_last.argmax(dim=-1)
                gate_reason_idx = torch.where(
                    fused_keep[-1, ..., 0] >= 0.5,
                    damage_state_idx.new_zeros(damage_state_idx.shape),
                    damage_state_idx)
                v8_info = {
                    'router_variant': self.v8_router_variant,
                    'bev_damage': bev_damage,
                    'img_damage': img_damage,
                    'dual_damage': dual_damage,
                    'bev_reliability': bev_reliability,
                    'img_reliability': img_reliability,
                    'reliability_margin': reliability_margin,
                    'fused_keep': fused_keep,
                    'center_fused_keep': center_fused_keep,
                    'damage_gate': 1.0 - fused_keep,
                    'non_fused_probs': v8_non_fused_probs,
                    'damage_state_idx': damage_state_idx,
                    'reliability_state_idx': damage_state_idx,
                    'high_damage_mask': high_damage_mask,
                    'center_route_allowed': center_route_allowed,
                    'non_fused_choice_idx': non_fused_choice_global_idx,
                    'center_source_idx': center_source_idx,
                    'gate_reason_idx': gate_reason_idx,
                    'final_weight_last': final_weight_last,
                    'cls_weight_seq': cls_weight_seq,
                    'center_weight_seq': center_weight_seq,
                    'height_weight_seq': height_weight_seq,
                    'vel_weight_seq': vel_weight_seq,
                    'force_weight_seq': force_weight_seq,
                    'force_modality_active': force_modality_active,
                    'forced_branch': forced_branch,
                    'context_bev_damage': context_bev_damage,
                    'context_img_damage': context_img_damage,
                    'a4_anchor_ready': bev_damage.new_tensor(
                        1.0 if float(self.v8_a4_anchor_count.item()) > 0.0 else 0.0),
                    'a4_bev_damage_z': a4_bev_damage_z,
                    'a4_img_damage_z': a4_img_damage_z,
                    'a4_bev_cls_unc_z': a4_bev_cls_unc_z,
                    'a4_img_cls_unc_z': a4_img_cls_unc_z,
                    'a4_route_score_bev': a4_route_score_bev,
                    'a4_route_score_img': a4_route_score_img,
                    'a4_clean_anchor_update_count': bev_damage.new_tensor(
                        float(self.v8_a4_anchor_count.item())),
                    'v9_a_bev': v9_a_bev,
                    'v9_a_img': v9_a_img,
                    'v9_a_bev_above_tau': v9_a_bev_above_tau,
                    'v9_a_img_above_tau': v9_a_img_above_tau,
                    'v9_clean_like_mask': v9_clean_like_mask.float(),
                    'v9_anchor_distance_bev': v9_anchor_distance_bev,
                    'v9_anchor_distance_img': v9_anchor_distance_img,
                    'v9_consistency_bev': v9_consistency_bev,
                    'v9_consistency_img': v9_consistency_img,
                    'v9_tail_prob_bev': v9_tail_prob_bev,
                    'v9_tail_prob_img': v9_tail_prob_img,
                    'v9_tail_excess_bev': v9_tail_excess_bev,
                    'v9_tail_excess_img': v9_tail_excess_img,
                    'v9_cvar_bev': v9_cvar_bev,
                    'v9_cvar_img': v9_cvar_img,
                    'v9_q95_shift_bev': v9_q95_shift_bev,
                    'v9_q95_shift_img': v9_q95_shift_img,
                    'v9_route_score_bev': v9_route_score_bev,
                    'v9_route_score_img': v9_route_score_img,
                    'v9_mean_disagree': v9_mean_disagree,
                    'v9_velocity_l1': v9_velocity_l1,
                    'v9_fused_keep_unit': fused_keep_unit[-1],
                    'v9_fused_keep_at_min': fused_keep_at_min[-1],
                    'v9_state_classification': v9_state_classification,
                    'v9_state_clean_like': v9_state_clean_like,
                    'v9_state_confirmed_img': v9_state_confirmed_img,
                    'v9_state_confirmed_bev': v9_state_confirmed_bev,
                    'v9_state_confirmed_dual': v9_state_confirmed_dual,
                    'v9_state_neutral': v9_state_neutral,
                    'v9_cap_applied': v9_cap_applied,
                    'v9_center_safe_applied': v9_center_safe_applied,
                    'v9_center_safe_skipped': v9_center_safe_skipped,
                    'v9_fused_keep_pre_cap_cls': fused_keep[-1],
                    'v9_fused_keep_pre_cap_center': fused_keep[-1],
                    'v9_fused_keep_pre_cap_height': fused_keep[-1],
                    'v9_fused_keep_pre_cap_vel': fused_keep[-1],
                    'v9_fused_keep_post_cap_cls': cls_fused_keep[-1],
                    'v9_fused_keep_post_cap_center': center_cap_fused_keep[-1],
                    'v9_fused_keep_post_cap_height': height_fused_keep[-1],
                    'v9_fused_keep_post_cap_vel': vel_fused_keep[-1],
                    'v9_fused_keep_final_cls': cls_fused_keep[-1],
                    'v9_fused_keep_final_center': center_fused_keep[-1],
                    'v9_fused_keep_final_height': height_fused_keep[-1],
                    'v9_fused_keep_final_vel': vel_fused_keep[-1],
                    'v9_soft_weight_fused_cls': v9_soft_weight_fused_cls,
                    'v9_soft_weight_bev_cls': v9_soft_weight_bev_cls,
                    'v9_soft_weight_img_cls': v9_soft_weight_img_cls,
                    'v9_soft_weight_fused_center': v9_soft_weight_fused_center,
                    'v9_soft_weight_bev_center': v9_soft_weight_bev_center,
                    'v9_soft_weight_img_center': v9_soft_weight_img_center,
                    'v9_soft_weight_fused_height': v9_soft_weight_fused_height,
                    'v9_soft_weight_bev_height': v9_soft_weight_bev_height,
                    'v9_soft_weight_img_height': v9_soft_weight_img_height,
                    'v9_soft_weight_fused_vel': v9_soft_weight_fused_vel,
                    'v9_soft_weight_bev_vel': v9_soft_weight_bev_vel,
                    'v9_soft_weight_img_vel': v9_soft_weight_img_vel,
                    'v9_current_keep_min': bev_damage.new_tensor(float(self.v8_fused_keep_min)),
                    'v9_current_lambda_gate': bev_damage.new_tensor(
                        float(getattr(self, 'v9_current_lambda_gate', float('nan')))),
                    'v9_current_lambda_route': bev_damage.new_tensor(
                        float(getattr(self, 'v9_current_lambda_route', float('nan')))),
                    'v9_tau_bev_current': bev_damage.new_tensor(float(self.v9_tau_bev.item())),
                    'v9_tau_img_current': bev_damage.new_tensor(float(self.v9_tau_img.item())),
                    'v9_tau_update_count': bev_damage.new_tensor(float(self.v9_tau_update_count.item())),
                    'v9_clean_distance_threshold_bev': bev_damage.new_tensor(
                        float(self.v9_clean_distance_threshold_bev.item())),
                    'v9_clean_distance_threshold_img': bev_damage.new_tensor(
                        float(self.v9_clean_distance_threshold_img.item())),
                    'v9_clean_distance_update_count': bev_damage.new_tensor(
                        float(self.v9_clean_distance_update_count.item())),
                    'v9_clean_q95_z_bev': bev_damage.new_tensor(float(self.v9_clean_q95_z_bev.item())),
                    'v9_clean_q95_z_img': bev_damage.new_tensor(float(self.v9_clean_q95_z_img.item())),
                    'v9_clean_anchor_frozen': bev_damage.new_tensor(
                        1.0 if (
                            use_v9 and self.v9_freeze_clean_anchor_epochs > 0 and
                            getattr(self, '_current_epoch', 0) < self.v9_freeze_clean_anchor_epochs
                        ) else 0.0),
                }
                if v10_info is not None:
                    v8_info.update(v10_info)

            v5_info = None
            v6_info = None
            v7_info = None
            if self.ensemble_mode == 'hard_selective_switch':
                fused_center_last = source_center.get('fused') if source_center is not None else None
                fused_height_unc_last = raw_unc_by_source.get('height', {}).get('fused')
                fused_center_unc_last = raw_unc_by_source.get('center', {}).get('fused')
                fused_cls_logits_last = source_cls_logits.get('fused') if source_cls_logits is not None else None
                fused_cls_unc_last = source_cls_unc.get('fused') if source_cls_unc is not None else None
                if (
                    fused_center_last is not None and
                    fused_height_unc_last is not None and
                    fused_center_unc_last is not None and
                    fused_cls_logits_last is not None and
                    fused_cls_unc_last is not None
                ):
                    v5_info = self._compute_v5_fused_anomaly(
                        fused_cls_unc_last,
                        fused_center_unc_last,
                        fused_height_unc_last,
                        fused_cls_logits_last,
                        fused_center_last,
                    )
            elif (
                self.ensemble_mode in ('modality_damage_aware_router', 'clean_reference_gate_router') or
                use_crg_post_inference or
                use_scrg_post_inference or
                use_supp_post_inference
            ):
                fused_center_last = source_center.get('fused') if source_center is not None else None
                bev_center_last = source_center.get('bev') if source_center is not None else None
                img_center_last = source_center.get('img') if source_center is not None else None
                fused_center_unc_last = raw_unc_by_source.get('center', {}).get('fused')
                bev_center_unc_last = raw_unc_by_source.get('center', {}).get('bev')
                img_center_unc_last = raw_unc_by_source.get('center', {}).get('img')
                if (
                    fused_center_last is not None and
                    bev_center_last is not None and
                    img_center_last is not None and
                    fused_center_unc_last is not None and
                    bev_center_unc_last is not None and
                    img_center_unc_last is not None
                ):
                    if source_cls_logits is not None:
                        blended_cls_logits = self._blend_source_tensor(
                            source_outs,
                            'cls_logits',
                            selector_probs,
                            modalities,
                            num_queries_per_modality,
                            use_selector=('cls_logits' in self.selector_apply_heads),
                        )
                        pred_class = blended_cls_logits[-1].argmax(dim=-1)
                    else:
                        pred_class = torch.zeros(
                            bs, nq, dtype=torch.long, device=outs_dec.device)
                    v7_info = self._compute_v7_center_anomaly(
                        pred_class,
                        fused_center_last,
                        bev_center_last,
                        img_center_last,
                        fused_center_unc_last,
                        bev_center_unc_last,
                        img_center_unc_last,
                    )
            elif self.ensemble_mode == 'gated_primary_router':
                fused_center_last = source_center.get('fused') if source_center is not None else None
                fused_height_unc_last = raw_unc_by_source.get('height', {}).get('fused')
                fused_center_unc_last = raw_unc_by_source.get('center', {}).get('fused')
                fused_cls_logits_last = source_cls_logits.get('fused') if source_cls_logits is not None else None
                fused_cls_unc_last = source_cls_unc.get('fused') if source_cls_unc is not None else None
                if (
                    fused_center_last is not None and
                    fused_height_unc_last is not None and
                    fused_center_unc_last is not None and
                    fused_cls_logits_last is not None and
                    fused_cls_unc_last is not None
                ):
                    v6_info = self._compute_v6_fused_anomaly(
                        fused_cls_unc_last,
                        fused_center_unc_last,
                        fused_height_unc_last,
                        fused_cls_logits_last,
                        fused_center_last,
                    )

            blended_outs = {}
            if use_supp_post_inference:
                blended_outs['supp_enabled_runtime'] = 1.0
                blended_outs['supp_status_code'] = 0.0
                blended_outs['supp_noadd_reason_code'] = 0.0
            fused_residual_gate = alpha_res
            if use_v5_inference:
                routing_index_tensor = torch.tensor(routing_indices, device=outs_dec.device, dtype=torch.long)
                candidate_local_idx = base_selector_probs[-1].argmax(dim=-1)
                candidate_global_idx = routing_index_tensor[candidate_local_idx]
                switch_mask = (
                    v5_info['fused_combined_anom'][..., 0] > self.v5_anomaly_threshold
                    if v5_info is not None else
                    torch.zeros(bs, nq, dtype=torch.bool, device=outs_dec.device)
                )
                selected_global_idx = torch.full_like(candidate_global_idx, fused_idx)
                selected_global_idx = torch.where(switch_mask, candidate_global_idx, selected_global_idx)

                if 'cls' in self.v5_switch_heads:
                    cls_alpha = self._hard_select_source_tensor(
                        source_outs, 'cls_alpha', selected_global_idx, modalities, num_queries_per_modality)
                    cls_beta = self._hard_select_source_tensor(
                        source_outs, 'cls_beta', selected_global_idx, modalities, num_queries_per_modality)
                    if cls_alpha is not None and cls_beta is not None:
                        blended_outs['cls_alpha'] = cls_alpha
                        blended_outs['cls_beta'] = cls_beta
                        blended_outs['cls_logits'] = torch.log(cls_alpha / cls_beta + 1e-8)
                    else:
                        blended_outs['cls_logits'] = self._hard_select_source_tensor(
                            source_outs, 'cls_logits', selected_global_idx, modalities, num_queries_per_modality)
                else:
                    cls_alpha = self._get_single_source_tensor(
                        source_outs, 'cls_alpha', 'fused', modalities, num_queries_per_modality)
                    cls_beta = self._get_single_source_tensor(
                        source_outs, 'cls_beta', 'fused', modalities, num_queries_per_modality)
                    if cls_alpha is not None and cls_beta is not None:
                        blended_outs['cls_alpha'] = cls_alpha
                        blended_outs['cls_beta'] = cls_beta
                        blended_outs['cls_logits'] = torch.log(cls_alpha / cls_beta + 1e-8)
                    else:
                        blended_outs['cls_logits'] = self._get_single_source_tensor(
                            source_outs, 'cls_logits', 'fused', modalities, num_queries_per_modality)

                if 'center' in self.v5_switch_heads:
                    blended_outs['center'] = self._hard_select_source_tensor(
                        source_outs, 'center', selected_global_idx, modalities, num_queries_per_modality)
                    blended_outs['center_nig'] = self._hard_select_source_tensor(
                        source_outs, 'center_nig', selected_global_idx, modalities, num_queries_per_modality)
                else:
                    blended_outs['center'] = self._get_single_source_tensor(
                        source_outs, 'center', 'fused', modalities, num_queries_per_modality)
                    blended_outs['center_nig'] = self._get_single_source_tensor(
                        source_outs, 'center_nig', 'fused', modalities, num_queries_per_modality)

                if 'height' in self.v5_switch_heads:
                    blended_outs['height'] = self._hard_select_source_tensor(
                        source_outs, 'height', selected_global_idx, modalities, num_queries_per_modality)
                    blended_outs['height_nig'] = self._hard_select_source_tensor(
                        source_outs, 'height_nig', selected_global_idx, modalities, num_queries_per_modality)
                else:
                    blended_outs['height'] = self._get_single_source_tensor(
                        source_outs, 'height', 'fused', modalities, num_queries_per_modality)
                    blended_outs['height_nig'] = self._get_single_source_tensor(
                        source_outs, 'height_nig', 'fused', modalities, num_queries_per_modality)

                for key in ('dim', 'dim_nig', 'rot', 'vel'):
                    frozen_tensor = self._get_single_source_tensor(
                        source_outs, key, 'fused', modalities, num_queries_per_modality)
                    if frozen_tensor is not None:
                        blended_outs[key] = frozen_tensor

                blended_outs['selector_teacher_cls_logits'] = self._hard_select_source_tensor(
                    source_outs, 'cls_logits', candidate_global_idx, modalities, num_queries_per_modality)
                blended_outs['selector_teacher_center'] = self._hard_select_source_tensor(
                    source_outs, 'center', candidate_global_idx, modalities, num_queries_per_modality)
                blended_outs['selector_teacher_height'] = self._hard_select_source_tensor(
                    source_outs, 'height', candidate_global_idx, modalities, num_queries_per_modality)
                blended_outs['selector_fused_weight'] = (~switch_mask).float().unsqueeze(0).unsqueeze(-1)
                blended_outs['selector_modalities'] = list(modalities)
                blended_outs['selector_probs_last'] = base_selector_probs[-1]
                blended_outs['v5_query_switch_triggered'] = switch_mask.float()
                blended_outs['v5_query_selected_modality_idx'] = selected_global_idx
                blended_outs['v5_query_candidate_modality_idx'] = candidate_global_idx
                if v5_info is not None:
                    blended_outs['v5_query_fused_cls_anom'] = v5_info['fused_cls_anom']
                    blended_outs['v5_query_fused_center_anom'] = v5_info['fused_center_anom']
                    blended_outs['v5_query_fused_height_anom'] = v5_info['fused_height_anom']
                    blended_outs['v5_query_fused_anomaly_score'] = v5_info['fused_combined_anom']
                    blended_outs['v5_query_reference_distance_bin'] = v5_info['distance_bin']
                    blended_outs['v5_query_reference_fallback_level'] = v5_info['fallback_level']
                    blended_outs['v5_query_pred_class'] = v5_info['pred_class']
                blended_outs['v5_fused_cls_unc_last'] = source_cls_unc['fused'] if source_cls_unc is not None else None
                blended_outs['v5_fused_center_unc_last'] = raw_unc_by_source.get('center', {}).get('fused')
                blended_outs['v5_fused_height_unc_last'] = raw_unc_by_source.get('height', {}).get('fused')
            else:
                cls_selector_probs = selector_probs
                height_selector_probs = selector_probs
                if use_v8_router and v8_info is not None:
                    cls_selector_probs = v8_info.get('cls_weight_seq', selector_probs)
                    height_selector_probs = v8_info.get('height_weight_seq', selector_probs)
                cls_blend_probs = (
                    cls_selector_probs
                    if self.v10_use_per_head_cls_dispatch else selector_probs)
                if 'cls_logits' in self.selector_apply_heads:
                    if use_primary_router:
                        cls_alpha = self._blend_selected_source_tensor(
                            source_outs, 'cls_alpha', cls_blend_probs, routing_modalities,
                            modalities, num_queries_per_modality, use_selector=True)
                        cls_beta = self._blend_selected_source_tensor(
                            source_outs, 'cls_beta', cls_blend_probs, routing_modalities,
                            modalities, num_queries_per_modality, use_selector=True)
                        fused_cls_alpha = self._get_single_source_tensor(
                            source_outs, 'cls_alpha', 'fused', modalities, num_queries_per_modality)
                        fused_cls_beta = self._get_single_source_tensor(
                            source_outs, 'cls_beta', 'fused', modalities, num_queries_per_modality)
                        if cls_alpha is not None and fused_cls_alpha is not None:
                            cls_alpha = cls_alpha + fused_residual_gate * (fused_cls_alpha - cls_alpha)
                        if cls_beta is not None and fused_cls_beta is not None:
                            cls_beta = cls_beta + fused_residual_gate * (fused_cls_beta - cls_beta)
                    else:
                        cls_alpha = self._blend_source_tensor(
                            source_outs, 'cls_alpha', cls_blend_probs, modalities,
                            num_queries_per_modality, use_selector=True)
                        cls_beta = self._blend_source_tensor(
                            source_outs, 'cls_beta', cls_blend_probs, modalities,
                            num_queries_per_modality, use_selector=True)
                    if cls_alpha is not None and cls_beta is not None:
                        blended_outs['cls_alpha'] = cls_alpha
                        blended_outs['cls_beta'] = cls_beta
                        blended_outs['cls_logits'] = torch.log(cls_alpha / cls_beta + 1e-8)
                    else:
                        if use_primary_router:
                            primary_cls = self._blend_selected_source_tensor(
                                source_outs, 'cls_logits', cls_blend_probs, routing_modalities,
                                modalities, num_queries_per_modality, use_selector=True)
                            fused_cls = self._get_single_source_tensor(
                                source_outs, 'cls_logits', 'fused', modalities, num_queries_per_modality)
                            blended_outs['cls_logits'] = primary_cls + fused_residual_gate * (
                                fused_cls - primary_cls
                            )
                        else:
                            blended_outs['cls_logits'] = self._blend_source_tensor(
                                source_outs, 'cls_logits', cls_blend_probs, modalities,
                                num_queries_per_modality, use_selector=True)
                else:
                    cls_alpha = self._blend_source_tensor(
                        source_outs, 'cls_alpha', selector_probs, modalities,
                        num_queries_per_modality, use_selector=False)
                    cls_beta = self._blend_source_tensor(
                        source_outs, 'cls_beta', selector_probs, modalities,
                        num_queries_per_modality, use_selector=False)
                    if cls_alpha is not None and cls_beta is not None:
                        blended_outs['cls_alpha'] = cls_alpha
                        blended_outs['cls_beta'] = cls_beta
                        blended_outs['cls_logits'] = torch.log(cls_alpha / cls_beta + 1e-8)
                    else:
                        blended_outs['cls_logits'] = self._blend_source_tensor(
                            source_outs, 'cls_logits', selector_probs, modalities,
                            num_queries_per_modality, use_selector=False)

                use_center_selector = 'center' in self.selector_apply_heads
                use_height_selector = 'height' in self.selector_apply_heads
                center_selector_probs = selector_probs
                if use_v8_router and v8_info is not None:
                    center_selector_probs = v8_info.get('center_weight_seq', selector_probs)
                if use_primary_router:
                    primary_center = self._blend_selected_source_tensor(
                        source_outs, 'center', selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=use_center_selector)
                    primary_center_nig = self._blend_selected_source_tensor(
                        source_outs, 'center_nig', selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=use_center_selector)
                    primary_height = self._blend_selected_source_tensor(
                        source_outs, 'height', height_selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=use_height_selector)
                    primary_height_nig = self._blend_selected_source_tensor(
                        source_outs, 'height_nig', height_selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=use_height_selector)
                    fused_center = self._get_single_source_tensor(
                        source_outs, 'center', 'fused', modalities, num_queries_per_modality)
                    fused_center_nig = self._get_single_source_tensor(
                        source_outs, 'center_nig', 'fused', modalities, num_queries_per_modality)
                    fused_height = self._get_single_source_tensor(
                        source_outs, 'height', 'fused', modalities, num_queries_per_modality)
                    fused_height_nig = self._get_single_source_tensor(
                        source_outs, 'height_nig', 'fused', modalities, num_queries_per_modality)
                    blended_outs['center'] = primary_center + fused_residual_gate * (
                        fused_center - primary_center
                    )
                    blended_outs['center_nig'] = primary_center_nig + fused_residual_gate * (
                        fused_center_nig - primary_center_nig
                    )
                    blended_outs['height'] = primary_height + fused_residual_gate * (
                        fused_height - primary_height
                    )
                    blended_outs['height_nig'] = primary_height_nig + fused_residual_gate * (
                        fused_height_nig - primary_height_nig
                    )
                else:
                    blended_outs['center'] = self._blend_source_tensor(
                        source_outs, 'center', center_selector_probs, modalities,
                        num_queries_per_modality, use_selector=use_center_selector)
                    blended_outs['center_nig'] = self._blend_source_tensor(
                        source_outs, 'center_nig', center_selector_probs, modalities,
                        num_queries_per_modality, use_selector=use_center_selector)
                    blended_outs['height'] = self._blend_source_tensor(
                        source_outs, 'height', height_selector_probs, modalities,
                        num_queries_per_modality, use_selector=use_height_selector)
                    blended_outs['height_nig'] = self._blend_source_tensor(
                        source_outs, 'height_nig', height_selector_probs, modalities,
                        num_queries_per_modality, use_selector=use_height_selector)

                vel_selector_probs = None
                force_selector_probs = None
                force_modality_active = False
                if use_v8_router and v8_info is not None:
                    vel_selector_probs = v8_info.get('vel_weight_seq')
                    force_selector_probs = v8_info.get('force_weight_seq')
                    force_modality_active = bool(v8_info.get('force_modality_active', False))
                for key in ('dim', 'dim_nig', 'rot', 'vel'):
                    head_selector_probs = selector_probs
                    use_head_selector = False
                    if force_modality_active and force_selector_probs is not None:
                        head_selector_probs = force_selector_probs
                        use_head_selector = True
                    elif key == 'vel' and vel_selector_probs is not None:
                        head_selector_probs = vel_selector_probs
                        use_head_selector = True
                    frozen_tensor = self._blend_source_tensor(
                        source_outs, key, head_selector_probs, modalities,
                        num_queries_per_modality, use_selector=use_head_selector)
                    if frozen_tensor is not None:
                        blended_outs[key] = frozen_tensor

                if use_primary_router:
                    blended_outs['selector_teacher_cls_logits'] = self._blend_selected_source_tensor(
                        source_outs, 'cls_logits', selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=True)
                    blended_outs['selector_teacher_center'] = self._blend_selected_source_tensor(
                        source_outs, 'center', selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=True)
                    blended_outs['selector_teacher_height'] = self._blend_selected_source_tensor(
                        source_outs, 'height', selector_probs, routing_modalities,
                        modalities, num_queries_per_modality, use_selector=True)
                    blended_outs['selector_fused_weight'] = fused_residual_gate
                else:
                    blended_outs['selector_teacher_cls_logits'] = self._blend_source_tensor(
                        source_outs, 'cls_logits', selector_probs, modalities,
                        num_queries_per_modality, use_selector=False)
                    blended_outs['selector_teacher_center'] = self._blend_source_tensor(
                        source_outs, 'center', selector_probs, modalities,
                        num_queries_per_modality, use_selector=False)
                    blended_outs['selector_teacher_height'] = self._blend_source_tensor(
                        source_outs, 'height', selector_probs, modalities,
                        num_queries_per_modality, use_selector=False)
                    blended_outs['selector_fused_weight'] = selector_probs[..., fused_idx:fused_idx + 1]
                blended_outs['selector_modalities'] = list(routing_modalities)
                blended_outs['selector_probs_last'] = selector_probs[-1]

                if use_v6_inference and v6_info is not None:
                    if self.v6_candidate_rule != 'non_fused_selector':
                        raise NotImplementedError(
                            f"Unsupported v6_candidate_rule: {self.v6_candidate_rule}")
                    non_fused_indices = torch.tensor(
                        [modalities.index(modality) for modality in non_fused_modalities],
                        device=outs_dec.device,
                        dtype=torch.long,
                    )
                    non_fused_scores_last = selector_scores[-1][..., non_fused_indices]
                    non_fused_probs = F.softmax(
                        non_fused_scores_last / self.selector_softmax_temperature, dim=-1)
                    candidate_local_idx = non_fused_probs.argmax(dim=-1)
                    candidate_global_idx = non_fused_indices[candidate_local_idx]
                    candidate_prob = non_fused_probs.gather(
                        dim=-1, index=candidate_local_idx.unsqueeze(-1)).squeeze(-1)
                    top2 = torch.topk(non_fused_probs, k=min(2, non_fused_probs.shape[-1]), dim=-1).values
                    if top2.shape[-1] == 1:
                        candidate_margin = torch.ones_like(candidate_prob)
                    else:
                        candidate_margin = top2[..., 0] - top2[..., 1]

                    switch_mask = (
                        (v6_info['fused_combined_anom'][..., 0] > self.v6_anomaly_threshold) &
                        (candidate_prob >= self.v6_candidate_min_prob) &
                        (candidate_margin >= self.v6_candidate_min_margin)
                    )
                    gate_pass_anomaly = (v6_info['fused_combined_anom'][..., 0] > self.v6_anomaly_threshold)
                    gate_pass_candidate_prob = (candidate_prob >= self.v6_candidate_min_prob)
                    gate_pass_candidate_margin = (candidate_margin >= self.v6_candidate_min_margin)

                    soft_blend_idx = torch.full_like(candidate_global_idx, -1)
                    center_source_idx = soft_blend_idx.clone()
                    height_source_idx = soft_blend_idx.clone()
                    cls_source_idx = soft_blend_idx.clone()

                    if 'center' in self.v6_switch_heads:
                        candidate_center = self._hard_select_source_tensor(
                            source_outs, 'center', candidate_global_idx, modalities, num_queries_per_modality)
                        candidate_center_nig = self._hard_select_source_tensor(
                            source_outs, 'center_nig', candidate_global_idx, modalities, num_queries_per_modality)
                        if candidate_center is not None:
                            center_mask = switch_mask.unsqueeze(0).unsqueeze(-1)
                            blended_outs['center'] = torch.where(center_mask, candidate_center, blended_outs['center'])
                            if candidate_center_nig is not None and blended_outs.get('center_nig') is not None:
                                blended_outs['center_nig'] = torch.where(
                                    center_mask, candidate_center_nig, blended_outs['center_nig'])
                            center_source_idx = torch.where(switch_mask, candidate_global_idx, center_source_idx)

                    if 'height' in self.v6_switch_heads:
                        candidate_height = self._hard_select_source_tensor(
                            source_outs, 'height', candidate_global_idx, modalities, num_queries_per_modality)
                        candidate_height_nig = self._hard_select_source_tensor(
                            source_outs, 'height_nig', candidate_global_idx, modalities, num_queries_per_modality)
                        if candidate_height is not None:
                            height_mask = switch_mask.unsqueeze(0).unsqueeze(-1)
                            blended_outs['height'] = torch.where(height_mask, candidate_height, blended_outs['height'])
                            if candidate_height_nig is not None and blended_outs.get('height_nig') is not None:
                                blended_outs['height_nig'] = torch.where(
                                    height_mask, candidate_height_nig, blended_outs['height_nig'])
                            height_source_idx = torch.where(switch_mask, candidate_global_idx, height_source_idx)

                    blended_outs['v6_query_switch_triggered'] = switch_mask.float()
                    blended_outs['v6_query_candidate_modality_idx'] = candidate_global_idx
                    blended_outs['v6_query_baseline_selected_modality_idx'] = selector_probs[-1].argmax(dim=-1)
                    blended_outs['v6_query_center_source_idx'] = center_source_idx
                    blended_outs['v6_query_height_source_idx'] = height_source_idx
                    blended_outs['v6_query_cls_source_idx'] = cls_source_idx
                    blended_outs['v6_query_gate_anomaly_score'] = v6_info['fused_combined_anom']
                    blended_outs['v6_query_gate_passed_confidence'] = candidate_prob.unsqueeze(-1)
                    blended_outs['v6_query_gate_passed_margin'] = candidate_margin.unsqueeze(-1)
                    blended_outs['v6_query_gate_pass_anomaly'] = gate_pass_anomaly.float()
                    blended_outs['v6_query_gate_pass_candidate_prob'] = gate_pass_candidate_prob.float()
                    blended_outs['v6_query_gate_pass_candidate_margin'] = gate_pass_candidate_margin.float()
                    blended_outs['v6_query_gate_pass_final'] = switch_mask.float()
                    blended_outs['v6_query_non_fused_selector_probs'] = non_fused_probs
                    blended_outs['v6_query_reference_distance_bin'] = v6_info['distance_bin']
                    blended_outs['v6_query_reference_fallback_level'] = v6_info['fallback_level']
                    blended_outs['v6_query_pred_class'] = v6_info['pred_class']

                if self.ensemble_mode == 'modality_damage_aware_router':
                    blended_outs['v7_img_router_logits_last'] = v7_img_router_logits
                    blended_outs['v7_bev_router_logits_last'] = v7_bev_router_logits

                if use_crg_standalone_inference and v7_info is not None:
                    blended_outs = self._apply_crg_img_fallback(
                        blended_outs,
                        source_outs,
                        v7_info,
                        modalities,
                        num_queries_per_modality,
                        telemetry_prefix='crg')

                if use_v7_router_inference and v7_info is not None:
                    img_router_probs = F.softmax(v7_img_router_logits, dim=-1)
                    bev_router_probs = F.softmax(v7_bev_router_logits, dim=-1)
                    img_fallback_prob = img_router_probs[..., 1]
                    bev_fallback_prob = bev_router_probs[..., 1]
                    img_fallback_margin = img_router_probs[..., 1] - img_router_probs[..., 0]
                    bev_fallback_margin = bev_router_probs[..., 1] - bev_router_probs[..., 0]

                    bev_damaged = v7_info['bev_center_anom'][..., 0] > self.v7_bev_damage_threshold
                    img_damaged = v7_info['img_center_anom'][..., 0] > self.v7_img_damage_threshold
                    bev_healthy = v7_info['bev_center_anom'][..., 0] < self.v7_bev_healthy_ceiling
                    img_healthy = v7_info['img_center_anom'][..., 0] < self.v7_img_healthy_ceiling

                    lidar_failure_state = bev_damaged & img_healthy
                    image_failure_state = img_damaged & bev_healthy
                    dual_damage_state = bev_damaged & img_damaged
                    neutral_state = ~(lidar_failure_state | image_failure_state | dual_damage_state)

                    img_router_pass = (
                        (img_fallback_prob >= self.v7_img_fallback_prob_threshold) &
                        (img_fallback_margin >= self.v7_img_fallback_margin_threshold)
                    )
                    bev_router_pass = (
                        (bev_fallback_prob >= self.v7_bev_fallback_prob_threshold) &
                        (bev_fallback_margin >= self.v7_bev_fallback_margin_threshold)
                    )
                    img_switch_mask = lidar_failure_state & img_router_pass
                    bev_switch_mask = image_failure_state & bev_router_pass
                    switch_mask = img_switch_mask | bev_switch_mask

                    soft_blend_idx = torch.full_like(v7_info['pred_class'], -1)
                    center_source_idx = soft_blend_idx.clone()
                    center_source_idx = torch.where(
                        img_switch_mask,
                        center_source_idx.new_full(center_source_idx.shape, modalities.index('img')),
                        center_source_idx)
                    center_source_idx = torch.where(
                        bev_switch_mask,
                        center_source_idx.new_full(center_source_idx.shape, modalities.index('bev')),
                        center_source_idx)

                    selected_global_idx = torch.where(
                        img_switch_mask,
                        center_source_idx.new_full(center_source_idx.shape, modalities.index('img')),
                        center_source_idx.new_full(center_source_idx.shape, modalities.index('bev')))
                    candidate_center = self._hard_select_source_tensor(
                        source_outs, 'center', selected_global_idx, modalities, num_queries_per_modality)
                    candidate_center_nig = self._hard_select_source_tensor(
                        source_outs, 'center_nig', selected_global_idx, modalities, num_queries_per_modality)
                    center_mask = switch_mask.unsqueeze(0).unsqueeze(-1)
                    if candidate_center is not None:
                        blended_outs['center'] = torch.where(center_mask, candidate_center, blended_outs['center'])
                    if candidate_center_nig is not None and blended_outs.get('center_nig') is not None:
                        blended_outs['center_nig'] = torch.where(
                            center_mask, candidate_center_nig, blended_outs['center_nig'])

                    damage_state_idx = torch.zeros_like(v7_info['pred_class'])
                    damage_state_idx = torch.where(
                        lidar_failure_state, damage_state_idx.new_full(damage_state_idx.shape, 1), damage_state_idx)
                    damage_state_idx = torch.where(
                        image_failure_state, damage_state_idx.new_full(damage_state_idx.shape, 2), damage_state_idx)
                    damage_state_idx = torch.where(
                        dual_damage_state, damage_state_idx.new_full(damage_state_idx.shape, 3), damage_state_idx)

                    reference_distance_bin = v7_info['fused_distance_bin'].clone()
                    reference_distance_bin = torch.where(
                        lidar_failure_state, v7_info['bev_distance_bin'], reference_distance_bin)
                    reference_distance_bin = torch.where(
                        image_failure_state, v7_info['img_distance_bin'], reference_distance_bin)

                    blended_outs['v7_query_damage_state_idx'] = damage_state_idx
                    blended_outs['v7_query_switch_triggered'] = switch_mask.float()
                    blended_outs['v7_query_center_source_idx'] = center_source_idx
                    blended_outs['v7_query_bev_center_anom'] = v7_info['bev_center_anom']
                    blended_outs['v7_query_img_center_anom'] = v7_info['img_center_anom']
                    blended_outs['v7_query_fused_center_anom'] = v7_info['fused_center_anom']
                    blended_outs['v7_query_img_fallback_prob'] = img_fallback_prob.unsqueeze(-1)
                    blended_outs['v7_query_img_fallback_margin'] = img_fallback_margin.unsqueeze(-1)
                    blended_outs['v7_query_bev_fallback_prob'] = bev_fallback_prob.unsqueeze(-1)
                    blended_outs['v7_query_bev_fallback_margin'] = bev_fallback_margin.unsqueeze(-1)
                    blended_outs['v7_query_reference_distance_bin'] = reference_distance_bin
                    blended_outs['v7_query_pred_class'] = v7_info['pred_class']
                    blended_outs['v7_query_bev_reference_distance_bin'] = v7_info['bev_distance_bin']
                    blended_outs['v7_query_img_reference_distance_bin'] = v7_info['img_distance_bin']
                    blended_outs['v7_query_fused_reference_distance_bin'] = v7_info['fused_distance_bin']
                    blended_outs['v7_query_bev_reference_fallback_level'] = v7_info['bev_fallback_level']
                    blended_outs['v7_query_img_reference_fallback_level'] = v7_info['img_fallback_level']
                    blended_outs['v7_query_fused_reference_fallback_level'] = v7_info['fused_fallback_level']

                if use_v8_router and v8_info is not None:
                    non_fused_probs = v8_info['non_fused_probs']
                    blended_outs['selector_teacher_cls_logits'] = self._blend_selected_source_tensor(
                        source_outs, 'cls_logits', non_fused_probs, non_fused_modalities,
                        modalities, num_queries_per_modality, use_selector=True)
                    blended_outs['selector_teacher_center'] = self._blend_selected_source_tensor(
                        source_outs, 'center', non_fused_probs, non_fused_modalities,
                        modalities, num_queries_per_modality, use_selector=True)
                    blended_outs['selector_teacher_height'] = self._blend_selected_source_tensor(
                        source_outs, 'height', non_fused_probs, non_fused_modalities,
                        modalities, num_queries_per_modality, use_selector=True)
                    blended_outs['selector_fused_weight'] = v8_info['fused_keep']
                    blended_outs['selector_modalities'] = list(non_fused_modalities)
                    blended_outs['selector_probs_last'] = non_fused_probs[-1]
                    blended_outs['v8_router_variant'] = v8_info['router_variant']
                    blended_outs['v8_fused_keep'] = v8_info['fused_keep'][-1]
                    blended_outs['v8_center_fused_keep'] = v8_info['center_fused_keep'][-1]
                    blended_outs['v8_damage_gate'] = v8_info['damage_gate'][-1]
                    blended_outs['v8_bev_damage'] = v8_info['bev_damage']
                    blended_outs['v8_img_damage'] = v8_info['img_damage']
                    blended_outs['v8_dual_damage'] = v8_info['dual_damage']
                    blended_outs['v8_context_bev_damage'] = v8_info['context_bev_damage']
                    blended_outs['v8_context_img_damage'] = v8_info['context_img_damage']
                    blended_outs['v8_a4_anchor_ready'] = v8_info['a4_anchor_ready']
                    blended_outs['v8_a4_bev_damage_z'] = v8_info['a4_bev_damage_z']
                    blended_outs['v8_a4_img_damage_z'] = v8_info['a4_img_damage_z']
                    blended_outs['v8_a4_cls_unc_z_bev'] = v8_info['a4_bev_cls_unc_z']
                    blended_outs['v8_a4_cls_unc_z_img'] = v8_info['a4_img_cls_unc_z']
                    blended_outs['v8_a4_route_score_bev'] = v8_info['a4_route_score_bev']
                    blended_outs['v8_a4_route_score_img'] = v8_info['a4_route_score_img']
                    blended_outs['v8_a4_clean_anchor_update_count'] = v8_info['a4_clean_anchor_update_count']
                    for v9_key in (
                            'v9_a_bev', 'v9_a_img',
                            'v9_a_bev_above_tau', 'v9_a_img_above_tau',
                            'v9_clean_like_mask',
                            'v9_anchor_distance_bev', 'v9_anchor_distance_img',
                            'v9_consistency_bev', 'v9_consistency_img',
                            'v9_tail_prob_bev', 'v9_tail_prob_img',
                            'v9_tail_excess_bev', 'v9_tail_excess_img',
                            'v9_cvar_bev', 'v9_cvar_img',
                            'v9_q95_shift_bev', 'v9_q95_shift_img',
                            'v9_route_score_bev', 'v9_route_score_img',
                            'v9_mean_disagree', 'v9_velocity_l1',
                            'v9_fused_keep_unit', 'v9_fused_keep_at_min',
                            'v9_state_clean_like', 'v9_state_confirmed_img',
                            'v9_state_confirmed_bev', 'v9_state_confirmed_dual',
                            'v9_state_neutral',
                            'v9_cap_applied',
                            'v9_center_safe_applied', 'v9_center_safe_skipped',
                            'v9_fused_keep_pre_cap_cls',
                            'v9_fused_keep_pre_cap_center',
                            'v9_fused_keep_pre_cap_height',
                            'v9_fused_keep_pre_cap_vel',
                            'v9_fused_keep_post_cap_cls',
                            'v9_fused_keep_post_cap_center',
                            'v9_fused_keep_post_cap_height',
                            'v9_fused_keep_post_cap_vel',
                            'v9_fused_keep_final_cls',
                            'v9_fused_keep_final_center',
                            'v9_fused_keep_final_height',
                            'v9_fused_keep_final_vel',
                            'v9_soft_weight_fused_cls',
                            'v9_soft_weight_bev_cls',
                            'v9_soft_weight_img_cls',
                            'v9_soft_weight_fused_center',
                            'v9_soft_weight_bev_center',
                            'v9_soft_weight_img_center',
                            'v9_soft_weight_fused_height',
                            'v9_soft_weight_bev_height',
                            'v9_soft_weight_img_height',
                            'v9_soft_weight_fused_vel',
                            'v9_soft_weight_bev_vel',
                            'v9_soft_weight_img_vel'):
                        blended_outs[v9_key] = v8_info[v9_key]
                    blended_outs['v9_state_classification'] = v8_info['v9_state_classification']
                    for v9_key in (
                            'v9_tau_bev_current', 'v9_tau_img_current',
                            'v9_tau_update_count',
                            'v9_clean_distance_threshold_bev',
                            'v9_clean_distance_threshold_img',
                            'v9_clean_distance_update_count',
                            'v9_clean_q95_z_bev',
                            'v9_clean_q95_z_img',
                            'v9_clean_anchor_frozen',
                            'v9_current_keep_min',
                            'v9_current_lambda_gate',
                            'v9_current_lambda_route'):
                        blended_outs[v9_key] = v8_info[v9_key]
                    for v10_key in (
                            'v10_p_dmg_bev', 'v10_p_dmg_img', 'v10_p_dmg_dual',
                            'v10_w_fused_cls', 'v10_w_bev_cls', 'v10_w_img_cls',
                            'v10_w_fused_center', 'v10_w_bev_center', 'v10_w_img_center',
                            'v10_w_fused_height', 'v10_w_bev_height', 'v10_w_img_height',
                            'v10_center_floor', 'v10_cls_fused_keep_min_current'):
                        if v10_key in v8_info:
                            blended_outs[v10_key] = v8_info[v10_key]
                    blended_outs['force_modality_active'] = bool(
                        v8_info.get('force_modality_active', False))
                    blended_outs['forced_branch'] = v8_info.get('forced_branch')
                    blended_outs['v8_bev_reliability'] = v8_info['bev_reliability']
                    blended_outs['v8_img_reliability'] = v8_info['img_reliability']
                    blended_outs['v8_reliability_margin'] = v8_info['reliability_margin']
                    blended_outs['v8_non_fused_probs'] = non_fused_probs[-1]
                    blended_outs['v8_non_fused_choice'] = v8_info['non_fused_choice_idx']
                    blended_outs['v8_center_source'] = v8_info['center_source_idx']
                    blended_outs['v8_gate_reason'] = v8_info['gate_reason_idx']
                    blended_outs['v8_damage_state'] = v8_info['damage_state_idx']
                    blended_outs['v8_reliability_state'] = v8_info['reliability_state_idx']
                    blended_outs['v8_high_damage_query'] = v8_info['high_damage_mask'].float()
                    blended_outs['v8_center_route_allowed'] = v8_info['center_route_allowed'].float()
                    final_weight_last = v8_info['final_weight_last']
                    blended_outs['v8_soft_weight_fused'] = final_weight_last[..., fused_idx:fused_idx + 1]
                    blended_outs['v8_soft_weight_bev'] = final_weight_last[..., modalities.index('bev'):modalities.index('bev') + 1]
                    blended_outs['v8_soft_weight_img'] = final_weight_last[..., modalities.index('img'):modalities.index('img') + 1]
                    fused_cls_logits = self._get_single_source_tensor(
                        source_outs, 'cls_logits', 'fused', modalities, num_queries_per_modality)
                    fused_center = self._get_single_source_tensor(
                        source_outs, 'center', 'fused', modalities, num_queries_per_modality)
                    fused_height = self._get_single_source_tensor(
                        source_outs, 'height', 'fused', modalities, num_queries_per_modality)
                    if fused_cls_logits is not None:
                        blended_outs['v8_output_delta_cls'] = torch.abs(
                            blended_outs['cls_logits'][-1] - fused_cls_logits[-1]
                        ).mean(dim=-1, keepdim=True)
                    if fused_center is not None:
                        blended_outs['v8_output_delta_center'] = torch.abs(
                            blended_outs['center'][-1] - fused_center[-1]
                        ).mean(dim=-1, keepdim=True)
                    if fused_height is not None:
                        blended_outs['v8_output_delta_height'] = torch.abs(
                            blended_outs['height'][-1] - fused_height[-1]
                        ).mean(dim=-1, keepdim=True)

                    if v10_info is not None:
                        blended_outs.update(self._v10_build_edlcue_info(
                            blended_outs,
                            source_cls_logits,
                            source_cls_unc,
                            v10_info,
                            dict(
                                reliability_margin=reliability_margin,
                                v9_mean_disagree=v9_mean_disagree,
                                v9_tail_prob_bev=v9_tail_prob_bev,
                                v9_tail_prob_img=v9_tail_prob_img,
                                v9_tail_excess_bev=v9_tail_excess_bev,
                                v9_tail_excess_img=v9_tail_excess_img)))

            if use_supp_post_inference and self.supp_scene_mapper is not None:
                pre_supp_selector_probs = selector_probs[-1]
                blended_outs['selector_entropy'] = float(
                    -(pre_supp_selector_probs * pre_supp_selector_probs.clamp(min=1e-8).log())
                    .sum(dim=-1).mean().item()
                )
                if use_v5_inference:
                    fused_weight_value = blended_outs['selector_fused_weight'].mean().item()
                elif use_primary_router:
                    fused_weight_value = fused_residual_gate[-1].mean().item()
                else:
                    fused_weight_value = pre_supp_selector_probs[..., fused_idx].mean().item()
                blended_outs['selector_fused_weight_mean'] = float(fused_weight_value)

            blended_outs = self._publish_raw_modality_uncertainty(
                blended_outs, raw_per_modality_unc)
            blended_outs['selector_distance_bin_abnormality_stats'] = {
                source_name: distance_bin_stats_by_source[source_name]
                for source_name in distance_bin_stats_by_source
            }
            blended_outs['distance_bin_abnormality_stats'] = (
                blended_outs['selector_distance_bin_abnormality_stats'])
            if (
                    bool(getattr(self, 'ev_query_semantic_log_enable', False)) or
                    bool(getattr(self, 'supp_qsem_gate_enable', False))):
                query_sem_img_cls = self._get_single_source_tensor(
                    source_outs, 'cls_logits', 'img', modalities, num_queries_per_modality)
                query_sem_bev_cls = (
                    self._get_single_source_tensor(
                        source_outs, 'cls_logits', 'bev', modalities, num_queries_per_modality)
                    if 'bev' in modalities else None)
                self._publish_query_semantic_telemetry(
                    blended_outs,
                    blended_outs.get('cls_logits'),
                    query_sem_img_cls,
                    bev_cls=query_sem_bev_cls,
                    task_id=task_id)

            if use_crg_post_inference and v7_info is not None:
                blended_outs = self._apply_crg_img_fallback(
                    blended_outs,
                    source_outs,
                    v7_info,
                    modalities,
                    num_queries_per_modality,
                    telemetry_prefix='crg_post')

            if use_scrg_post_inference and v7_info is not None:
                blended_outs = self._apply_scrg_img_fallback(
                    blended_outs,
                    source_outs,
                    v7_info,
                    modalities,
                    num_queries_per_modality,
                    telemetry_prefix='scrg_post')

            blended_outs = self._maybe_apply_img_supplement_admission(
                use_supp_post_inference,
                blended_outs,
                source_outs,
                v7_info,
                modalities,
                num_queries_per_modality,
                raw_unc_by_source=raw_unc_by_source,
                source_cls_unc=source_cls_unc,
                v8_info=v8_info)

            if source_cls_logits is not None:
                blended_outs['selector_source_cls_logits'] = source_cls_logits
            if source_cls_unc is not None:
                blended_outs['selector_source_cls_unc'] = source_cls_unc
            if source_center is not None:
                blended_outs['selector_source_center'] = source_center
            if source_height is not None:
                blended_outs['selector_source_height'] = source_height

            self._maybe_export_ev_branch_candidates(
                blended_outs,
                source_outs,
                modalities,
                num_queries_per_modality,
                pc_range,
                raw_unc_by_source=raw_unc_by_source,
                source_cls_unc=source_cls_unc,
                task_id=task_id,
                v8_info=v8_info)

            last_probs = selector_probs[-1]
            last_base_probs = base_selector_probs[-1]
            base_top1_idx = last_base_probs.argmax(dim=-1)
            selector_weight_stats = {modality: 0.0 for modality in self.modality_names}
            selector_top1_ratio = {modality: 0.0 for modality in self.modality_names}
            distance_prior_top1_ratio = {modality: 0.0 for modality in self.modality_names}
            if use_v5_inference:
                switch_mask = blended_outs['v5_query_switch_triggered'] > 0.5
                selected_global_idx = blended_outs['v5_query_selected_modality_idx']
                candidate_global_idx = blended_outs['v5_query_candidate_modality_idx']
                for global_idx, modality in enumerate(modalities):
                    selector_weight_stats[modality] = float((selected_global_idx == global_idx).float().mean().item())
                    selector_top1_ratio[modality] = selector_weight_stats[modality]
                for idx, modality in enumerate(routing_modalities):
                    global_idx = routing_indices[idx]
                    distance_prior_top1_ratio[modality] = float((base_top1_idx == idx).float().mean().item())
                    selector_weight_stats[modality] = float((selected_global_idx == global_idx).float().mean().item())
                selector_weight_stats['fused'] = float((selected_global_idx == fused_idx).float().mean().item())
                selector_top1_ratio = dict(selector_weight_stats)
            else:
                mean_weights = last_probs.mean(dim=(0, 1))
                top1_idx = last_probs.argmax(dim=-1)
                for idx, modality in enumerate(routing_modalities):
                    selector_weight_stats[modality] = float(mean_weights[idx].item())
                    selector_top1_ratio[modality] = float((top1_idx == idx).float().mean().item())
                    distance_prior_top1_ratio[modality] = float((base_top1_idx == idx).float().mean().item())
            blended_outs['selector_weight_stats'] = selector_weight_stats
            blended_outs['selector_top1_ratio'] = selector_top1_ratio
            blended_outs['selector_distance_prior_top1_ratio'] = distance_prior_top1_ratio
            blended_outs['selector_quality_top1_ratio'] = distance_prior_top1_ratio
            blended_outs['selector_entropy'] = float(
                -(last_probs * last_probs.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
            )
            blended_outs['selector_fused_weight_mean'] = float(
                (
                    blended_outs['selector_fused_weight'].mean().item()
                    if use_v5_inference else
                    (fused_residual_gate[-1].mean().item() if use_primary_router else mean_weights[fused_idx].item())
                )
            )
            blended_outs['selector_alpha_res_last'] = alpha_res[-1]
            blended_outs['selector_alpha_res_mean'] = float(alpha_res[-1].mean().item())
            blended_outs['selector_distance_prior_probs_last'] = last_base_probs
            if use_v8_router and v8_info is not None:
                non_fused_indices = [modalities.index(modality) for modality in non_fused_modalities]
                base_non_fused = last_base_probs[..., non_fused_indices]
                base_non_fused = base_non_fused / base_non_fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                blended_outs['selector_distance_prior_probs_last'] = base_non_fused
            blended_outs['selector_residual_override_ratio'] = float(
	                (
	                    (selector_top1_ratio.get('fused', 0.0) < 1.0 and blended_outs['v5_query_switch_triggered'].mean().item())
                    if use_v5_inference else
                    (
                        blended_outs['v6_query_switch_triggered'].mean().item()
                        if use_v6_inference else
                        (top1_idx != base_top1_idx).float().mean().item()
                    )
                )
            )
            blended_outs['selector_per_modality_unc'] = {
                modality: float(raw_per_modality_unc[modality].mean().item())
                for modality in modalities
            }
            blended_outs['selector_relative_unc_summary'] = relative_unc_summary
            blended_outs['selector_absolute_unc_summary'] = absolute_unc_summary
            blended_outs['selector_disagreement_summary'] = disagreement_summary
            blended_outs['selector_uncertainty_tail_query_ratio'] = tail_ratio_summary
            blended_outs['selector_distance_bin_abnormality_stats'] = {
                source_name: distance_bin_stats_by_source[source_name]
                for source_name in distance_bin_stats_by_source
            }
            if 'center' in raw_unc_by_source:
                center_unc_export = raw_unc_by_source['center']
                for modality in modalities:
                    blended_outs[f'v7_ref_{modality}_center_unc'] = center_unc_export.get(modality)
                    if source_center is not None and modality in source_center:
                        blended_outs[f'v7_ref_{modality}_distance'] = source_center[modality][..., :2].norm(
                            dim=-1, keepdim=True)
                blended_outs['v7_ref_pred_class'] = blended_outs['cls_logits'][-1].argmax(dim=-1)
            if self.selector_log_score_components:
                selector_score_component_stats = {}
                for idx, modality in enumerate(routing_modalities):
                    selector_score_component_stats[modality] = {
                        'distance': float(distance_scores[-1, :, :, idx].mean().item()),
                        'context': float(context_scores[-1, :, :, idx].mean().item()),
                        'quality': float(base_scores[-1, :, :, idx].mean().item()),
                        'uncertainty_penalty': float(
                            (self.selector_unc_score_weight * unc_penalty[-1, :, :, idx]).mean().item()
                        ),
                        'relative_uncertainty_penalty': float(
                            (self.selector_rel_unc_score_weight * relative_unc_penalty[-1, :, :, idx]).mean().item()
                        ),
                        'disagreement_penalty': float(
                            (self.selector_disagreement_score_weight * disagreement_penalty[-1, :, :, idx]).mean().item()
                        ),
                        'residual_delta': float((alpha_res[-1, :, :, 0] * delta_scores[-1, :, :, idx]).mean().item()),
                        'final_score': float(selector_scores[-1, :, :, idx].mean().item()),
                    }
                blended_outs['selector_score_component_stats'] = selector_score_component_stats

            if use_v5_inference and v5_info is not None:
                candidate_global_idx = blended_outs['v5_query_candidate_modality_idx']
                selected_global_idx = blended_outs['v5_query_selected_modality_idx']
                switch_mask = blended_outs['v5_query_switch_triggered'] > 0.5
                fused_cls_logits = self._get_single_source_tensor(
                    source_outs, 'cls_logits', 'fused', modalities, num_queries_per_modality)
                fused_center = self._get_single_source_tensor(
                    source_outs, 'center', 'fused', modalities, num_queries_per_modality)
                fused_height = self._get_single_source_tensor(
                    source_outs, 'height', 'fused', modalities, num_queries_per_modality)
                switched_count = float(switch_mask.float().sum().item())
                cls_shift = 0.0
                center_shift = 0.0
                height_shift = 0.0
                if switched_count > 0:
                    cls_shift = float(
                        (torch.abs(blended_outs['cls_logits'][-1] - fused_cls_logits[-1]).mean(dim=-1)[switch_mask]).mean().item()
                    )
                    center_shift = float(
                        (torch.abs(blended_outs['center'][-1] - fused_center[-1]).mean(dim=-1)[switch_mask]).mean().item()
                    )
                    height_shift = float(
                        (torch.abs(blended_outs['height'][-1] - fused_height[-1]).mean(dim=-1)[switch_mask]).mean().item()
                    )
                blended_outs['v5_switch_ratio'] = float(switch_mask.float().mean().item())
                blended_outs['v5_keep_fused_ratio'] = float((~switch_mask).float().mean().item())
                blended_outs['v5_switch_to_bev_ratio'] = float(
                    (switch_mask & (selected_global_idx == modalities.index('bev'))).float().mean().item()
                )
                blended_outs['v5_switch_to_img_ratio'] = float(
                    (switch_mask & (selected_global_idx == modalities.index('img'))).float().mean().item()
                )
                blended_outs['v5_fused_cls_anom_mean'] = float(v5_info['fused_cls_anom'].mean().item())
                blended_outs['v5_fused_center_anom_mean'] = float(v5_info['fused_center_anom'].mean().item())
                blended_outs['v5_fused_height_anom_mean'] = float(v5_info['fused_height_anom'].mean().item())
                blended_outs['v5_fused_combined_anom_mean'] = float(v5_info['fused_combined_anom'].mean().item())
                blended_outs['v5_trigger_ratio_over_threshold'] = float(switch_mask.float().mean().item())
                blended_outs['v5_bin_coverage_ratio'] = float((v5_info['fallback_level'] == 0).float().mean().item())
                blended_outs['v5_fallback_class_ratio'] = float((v5_info['fallback_level'] == 1).float().mean().item())
                blended_outs['v5_fallback_global_ratio'] = float((v5_info['fallback_level'] == 2).float().mean().item())
                blended_outs['v5_distance_prior_candidate_ratio'] = {
                    modality: float((candidate_global_idx == modalities.index(modality)).float().mean().item())
                    for modality in routing_modalities
                }
                blended_outs['v5_distance_prior_switched_ratio'] = {
                    modality: float(
                        (
                            (switch_mask & (selected_global_idx == modalities.index(modality))).float().sum() /
                            switch_mask.float().sum().clamp(min=1.0)
                        ).item()
                    )
                    for modality in routing_modalities
                }
                blended_outs['v5_switch_agreement_with_prior'] = float(
                    torch.where(
                        switch_mask,
                        (selected_global_idx == candidate_global_idx).float(),
                        torch.ones_like(selected_global_idx, dtype=torch.float)
                    ).mean().item()
                )
                blended_outs['v5_head_shift'] = {
                    'cls': cls_shift,
                    'center': center_shift,
                    'height': height_shift,
                }
            if use_v6_inference and v6_info is not None:
                switch_mask = blended_outs['v6_query_switch_triggered'] > 0.5
                candidate_global_idx = blended_outs['v6_query_candidate_modality_idx']
                center_source_idx = blended_outs['v6_query_center_source_idx']
                height_source_idx = blended_outs['v6_query_height_source_idx']
                blended_outs['v6_switch_ratio'] = float(switch_mask.float().mean().item())
                blended_outs['v6_gate_anomaly_mean'] = float(v6_info['fused_combined_anom'].mean().item())
                blended_outs['v6_gate_pass_anomaly_ratio'] = float(
                    blended_outs['v6_query_gate_pass_anomaly'].mean().item()
                )
                blended_outs['v6_gate_pass_candidate_prob_ratio'] = float(
                    blended_outs['v6_query_gate_pass_candidate_prob'].mean().item()
                )
                blended_outs['v6_gate_pass_candidate_margin_ratio'] = float(
                    blended_outs['v6_query_gate_pass_candidate_margin'].mean().item()
                )
                blended_outs['v6_gate_pass_final_ratio'] = float(
                    blended_outs['v6_query_gate_pass_final'].mean().item()
                )
                blended_outs['v6_candidate_modality_ratio'] = {
                    modality: float((candidate_global_idx == modalities.index(modality)).float().mean().item())
                    for modality in non_fused_modalities
                }
                blended_outs['v6_center_source_ratio'] = {
                    'soft_blend': float((center_source_idx < 0).float().mean().item()),
                    'bev': float((center_source_idx == modalities.index('bev')).float().mean().item()),
                    'img': float((center_source_idx == modalities.index('img')).float().mean().item()),
                }
                blended_outs['v6_height_source_ratio'] = {
                    'soft_blend': float((height_source_idx < 0).float().mean().item()),
                    'bev': float((height_source_idx == modalities.index('bev')).float().mean().item()),
                    'img': float((height_source_idx == modalities.index('img')).float().mean().item()),
                }
                blended_outs['v6_reference_fallback_ratio'] = {
                    'class_distance': float((v6_info['fallback_level'] == 0).float().mean().item()),
                    'class_global': float((v6_info['fallback_level'] == 1).float().mean().item()),
                    'global': float((v6_info['fallback_level'] == 2).float().mean().item()),
                    'missing': float((v6_info['fallback_level'] == 3).float().mean().item()),
                }
            if (
                use_v7_inference and
                v7_info is not None and
                'v7_query_switch_triggered' in blended_outs
            ):
                switch_mask = blended_outs['v7_query_switch_triggered'] > 0.5
                damage_state_idx = blended_outs['v7_query_damage_state_idx']
                center_source_idx = blended_outs['v7_query_center_source_idx']
                blended_outs['v7_switch_ratio'] = float(switch_mask.float().mean().item())
                blended_outs['v7_damage_state_ratio'] = {
                    'neutral': float((damage_state_idx == 0).float().mean().item()),
                    'lidar_failure': float((damage_state_idx == 1).float().mean().item()),
                    'image_failure': float((damage_state_idx == 2).float().mean().item()),
                    'dual_damage': float((damage_state_idx == 3).float().mean().item()),
                }
                blended_outs['v7_center_source_ratio'] = {
                    'soft_blend': float((center_source_idx < 0).float().mean().item()),
                    'bev_fallback': float((center_source_idx == modalities.index('bev')).float().mean().item()),
                    'img_fallback': float((center_source_idx == modalities.index('img')).float().mean().item()),
                }
                blended_outs['v7_bev_center_anom_mean'] = float(v7_info['bev_center_anom'].mean().item())
                blended_outs['v7_img_center_anom_mean'] = float(v7_info['img_center_anom'].mean().item())
                blended_outs['v7_fused_center_anom_mean'] = float(v7_info['fused_center_anom'].mean().item())

            if use_v8_router and v8_info is not None:
                damage_state_idx = blended_outs['v8_damage_state']
                reliability_state_idx = blended_outs['v8_reliability_state']
                center_source_idx = blended_outs['v8_center_source']
                non_fused_probs = blended_outs['v8_non_fused_probs']
                blended_outs['v8_fused_keep_mean'] = float(blended_outs['v8_fused_keep'].mean().item())
                blended_outs['v8_center_fused_keep_mean'] = float(
                    blended_outs['v8_center_fused_keep'].mean().item())
                blended_outs['v8_damage_gate_mean'] = float(blended_outs['v8_damage_gate'].mean().item())
                blended_outs['v8_high_damage_query_ratio'] = float(
                    blended_outs['v8_high_damage_query'].mean().item())
                blended_outs['v8_center_route_allowed_ratio'] = float(
                    blended_outs['v8_center_route_allowed'].mean().item())
                blended_outs['v8_bev_damage_mean'] = float(blended_outs['v8_bev_damage'].mean().item())
                blended_outs['v8_img_damage_mean'] = float(blended_outs['v8_img_damage'].mean().item())
                blended_outs['v8_dual_damage_mean'] = float(blended_outs['v8_dual_damage'].mean().item())
                blended_outs['v8_context_bev_damage_mean'] = float(
                    blended_outs['v8_context_bev_damage'].mean().item())
                blended_outs['v8_context_img_damage_mean'] = float(
                    blended_outs['v8_context_img_damage'].mean().item())
                blended_outs['v8_a4_anchor_ready'] = float(
                    blended_outs['v8_a4_anchor_ready'].detach().item())
                blended_outs['v8_a4_bev_damage_z_mean'] = float(
                    blended_outs['v8_a4_bev_damage_z'].mean().item())
                blended_outs['v8_a4_img_damage_z_mean'] = float(
                    blended_outs['v8_a4_img_damage_z'].mean().item())
                blended_outs['v8_a4_cls_unc_z_bev_mean'] = float(
                    blended_outs['v8_a4_cls_unc_z_bev'].mean().item())
                blended_outs['v8_a4_cls_unc_z_img_mean'] = float(
                    blended_outs['v8_a4_cls_unc_z_img'].mean().item())
                blended_outs['v8_a4_route_score_bev_mean'] = float(
                    blended_outs['v8_a4_route_score_bev'].mean().item())
                blended_outs['v8_a4_route_score_img_mean'] = float(
                    blended_outs['v8_a4_route_score_img'].mean().item())
                blended_outs['v8_a4_clean_anchor_update_count'] = float(
                    blended_outs['v8_a4_clean_anchor_update_count'].detach().item())
                if blended_outs.get('v8_router_variant') == 'v9_lite_domain_anchored':
                    for key in (
                            'v9_a_bev', 'v9_a_img',
                            'v9_a_bev_above_tau', 'v9_a_img_above_tau',
                            'v9_clean_like_mask',
                            'v9_anchor_distance_bev', 'v9_anchor_distance_img',
                            'v9_consistency_bev', 'v9_consistency_img',
                            'v9_tail_prob_bev', 'v9_tail_prob_img',
                            'v9_tail_excess_bev', 'v9_tail_excess_img',
                            'v9_cvar_bev', 'v9_cvar_img',
                            'v9_q95_shift_bev', 'v9_q95_shift_img',
                            'v9_route_score_bev', 'v9_route_score_img',
                            'v9_mean_disagree', 'v9_velocity_l1',
                            'v9_fused_keep_unit', 'v9_fused_keep_at_min',
                            'v9_state_clean_like', 'v9_state_confirmed_img',
                            'v9_state_confirmed_bev', 'v9_state_confirmed_dual',
                            'v9_state_neutral',
                            'v9_cap_applied',
                            'v9_center_safe_applied', 'v9_center_safe_skipped',
                            'v9_fused_keep_pre_cap_cls',
                            'v9_fused_keep_pre_cap_center',
                            'v9_fused_keep_pre_cap_height',
                            'v9_fused_keep_pre_cap_vel',
                            'v9_fused_keep_post_cap_cls',
                            'v9_fused_keep_post_cap_center',
                            'v9_fused_keep_post_cap_height',
                            'v9_fused_keep_post_cap_vel',
                            'v9_fused_keep_final_cls',
                            'v9_fused_keep_final_center',
                            'v9_fused_keep_final_height',
                            'v9_fused_keep_final_vel',
                            'v9_soft_weight_fused_cls',
                            'v9_soft_weight_bev_cls',
                            'v9_soft_weight_img_cls',
                            'v9_soft_weight_fused_center',
                            'v9_soft_weight_bev_center',
                            'v9_soft_weight_img_center',
                            'v9_soft_weight_fused_height',
                            'v9_soft_weight_bev_height',
                            'v9_soft_weight_img_height',
                            'v9_soft_weight_fused_vel',
                            'v9_soft_weight_bev_vel',
                            'v9_soft_weight_img_vel'):
                        blended_outs[f'{key}_mean'] = float(blended_outs[key].mean().item())
                    if 'v9_fused_keep_unit' in blended_outs:
                        keep_unit_flat = blended_outs['v9_fused_keep_unit'].reshape(-1).float()
                        if keep_unit_flat.numel() > 0:
                            blended_outs['v9_fused_keep_unit_q05'] = float(
                                torch.quantile(keep_unit_flat, 0.05).item())
                            blended_outs['v9_fused_keep_unit_q50'] = float(
                                torch.quantile(keep_unit_flat, 0.50).item())
                            blended_outs['v9_fused_keep_unit_q95'] = float(
                                torch.quantile(keep_unit_flat, 0.95).item())
                    for key in (
                            'v9_tau_bev_current', 'v9_tau_img_current',
                            'v9_tau_update_count',
                            'v9_clean_distance_threshold_bev',
                            'v9_clean_distance_threshold_img',
                            'v9_clean_distance_update_count',
                            'v9_clean_q95_z_bev',
                            'v9_clean_q95_z_img',
                            'v9_clean_anchor_frozen',
                            'v9_current_keep_min',
                            'v9_current_lambda_gate',
                            'v9_current_lambda_route'):
                        value = blended_outs[key]
                        if isinstance(value, torch.Tensor):
                            blended_outs[key] = float(value.detach().item())
                blended_outs['v8_bev_reliability_mean'] = float(blended_outs['v8_bev_reliability'].mean().item())
                blended_outs['v8_img_reliability_mean'] = float(blended_outs['v8_img_reliability'].mean().item())
                blended_outs['v8_reliability_margin_mean'] = float(
                    blended_outs['v8_reliability_margin'].mean().item())
                blended_outs['v8_non_fused_weight_ratio'] = {
                    'bev': float(non_fused_probs[..., 0].mean().item()),
                    'img': float(non_fused_probs[..., 1].mean().item()),
                }
                blended_outs['v8_soft_weight_ratio'] = {
                    'fused': float(blended_outs['v8_soft_weight_fused'].mean().item()),
                    'bev': float(blended_outs['v8_soft_weight_bev'].mean().item()),
                    'img': float(blended_outs['v8_soft_weight_img'].mean().item()),
                }
                blended_outs['v8_damage_state_ratio'] = {
                    'clean_like': float((damage_state_idx == 0).float().mean().item()),
                    'bev_damage': float((damage_state_idx == 1).float().mean().item()),
                    'img_damage': float((damage_state_idx == 2).float().mean().item()),
                    'dual_damage': float((damage_state_idx == 3).float().mean().item()),
                    'ambiguous': float((damage_state_idx == 4).float().mean().item()),
                }
                blended_outs['v8_reliability_state_ratio'] = {
                    'both_reliable': float((reliability_state_idx == 0).float().mean().item()),
                    'bev_unreliable': float((reliability_state_idx == 1).float().mean().item()),
                    'img_unreliable': float((reliability_state_idx == 2).float().mean().item()),
                    'dual_unreliable': float((reliability_state_idx == 3).float().mean().item()),
                    'ambiguous': float((reliability_state_idx == 4).float().mean().item()),
                }
                blended_outs['v8_selected_source_ratio'] = {
                    'fused': float((center_source_idx == modalities.index('fused')).float().mean().item()),
                    'bev': float((center_source_idx == modalities.index('bev')).float().mean().item()),
                    'img': float((center_source_idx == modalities.index('img')).float().mean().item()),
                }
                keep_values = blended_outs['v8_fused_keep'][..., 0]
                keep_by_state = {}
                for state_name, state_id in [
                        ('both_reliable', 0),
                        ('bev_unreliable', 1),
                        ('img_unreliable', 2),
                        ('dual_unreliable', 3),
                        ('ambiguous', 4)]:
                    state_mask = reliability_state_idx == state_id
                    keep_by_state[state_name] = (
                        float(keep_values[state_mask].mean().item())
                        if state_mask.any() else float('nan')
                    )
                blended_outs['v8_fused_keep_by_state'] = keep_by_state
                blended_outs['v8_non_fused_entropy_mean'] = float(
                    -(non_fused_probs * non_fused_probs.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
                )
                for delta_key in ('cls', 'center', 'height'):
                    full_key = f'v8_output_delta_{delta_key}'
                    if full_key in blended_outs:
                        blended_outs[f'{full_key}_mean'] = float(blended_outs[full_key].mean().item())
                if 'v8_output_delta_center' in blended_outs:
                    blended_outs['v9_center_l1'] = blended_outs['v8_output_delta_center']
                    blended_outs['v9_center_l1_mean'] = float(
                        blended_outs['v9_center_l1'].mean().item())
                    center_l1_flat = blended_outs['v9_center_l1'].reshape(-1).float()
                    if center_l1_flat.numel() > 0:
                        blended_outs['v9_center_l1_p95'] = float(
                            torch.quantile(center_l1_flat, 0.95).item())

            if not self.training:
                modality_weights = selector_weight_stats
                per_mod_unc = blended_outs['selector_per_modality_unc']
                blended_outs['per_modality_unc'] = per_mod_unc
                blended_outs['modality_weights'] = modality_weights
                blended_outs['selected_modality'] = max(modality_weights, key=modality_weights.get)
                blended_outs['fused_weight'] = float(modality_weights.get('fused', 0.0))
                blended_outs['distance_prior_top1_ratio'] = distance_prior_top1_ratio
                blended_outs['quality_top1_ratio'] = distance_prior_top1_ratio
                blended_outs['residual_override_ratio'] = blended_outs['selector_residual_override_ratio']
                blended_outs['alpha_res_mean'] = blended_outs['selector_alpha_res_mean']
                blended_outs['uncertainty_tail_query_ratio'] = tail_ratio_summary
                blended_outs['distance_bin_abnormality_stats'] = blended_outs['selector_distance_bin_abnormality_stats']
                blended_outs['absolute_unc_summary'] = absolute_unc_summary
                blended_outs['disagreement_summary'] = disagreement_summary
                if self.selector_log_score_components:
                    blended_outs['score_component_stats'] = blended_outs['selector_score_component_stats']
                if use_v5_inference and v5_info is not None:
                    blended_outs['v5_switch_triggered'] = blended_outs['v5_query_switch_triggered']
                    blended_outs['v5_selected_modality_idx'] = blended_outs['v5_query_selected_modality_idx']
                    blended_outs['v5_candidate_modality_idx'] = blended_outs['v5_query_candidate_modality_idx']
                    blended_outs['v5_fused_anomaly_score'] = blended_outs['v5_query_fused_anomaly_score']
                    blended_outs['v5_fused_anomaly_components'] = {
                        'cls': blended_outs['v5_query_fused_cls_anom'],
                        'center': blended_outs['v5_query_fused_center_anom'],
                        'height': blended_outs['v5_query_fused_height_anom'],
                    }
                    blended_outs['v5_reference_distance_bin'] = blended_outs['v5_query_reference_distance_bin']
                    blended_outs['v5_reference_fallback_level'] = blended_outs['v5_query_reference_fallback_level']
                    blended_outs['v5_reference_pred_class'] = blended_outs['v5_query_pred_class']
                    blended_outs['v5_fused_cls_unc'] = blended_outs['v5_fused_cls_unc_last']
                    blended_outs['v5_fused_center_unc'] = blended_outs['v5_fused_center_unc_last']
                    blended_outs['v5_fused_height_unc'] = blended_outs['v5_fused_height_unc_last']
                if use_v6_inference and v6_info is not None:
                    blended_outs['v6_switch_triggered'] = blended_outs['v6_query_switch_triggered']
                    blended_outs['v6_candidate_modality_idx'] = blended_outs['v6_query_candidate_modality_idx']
                    blended_outs['v6_gate_anomaly_score'] = blended_outs['v6_query_gate_anomaly_score']
                    blended_outs['v6_gate_passed_confidence'] = blended_outs['v6_query_gate_passed_confidence']
                    blended_outs['v6_gate_passed_margin'] = blended_outs['v6_query_gate_passed_margin']
                    blended_outs['v6_gate_pass_anomaly'] = blended_outs['v6_query_gate_pass_anomaly']
                    blended_outs['v6_gate_pass_candidate_prob'] = blended_outs['v6_query_gate_pass_candidate_prob']
                    blended_outs['v6_gate_pass_candidate_margin'] = blended_outs['v6_query_gate_pass_candidate_margin']
                    blended_outs['v6_gate_pass_final'] = blended_outs['v6_query_gate_pass_final']
                    blended_outs['v6_reference_distance_bin'] = blended_outs['v6_query_reference_distance_bin']
                    blended_outs['v6_reference_fallback_level'] = blended_outs['v6_query_reference_fallback_level']
                    blended_outs['v6_reference_pred_class'] = blended_outs['v6_query_pred_class']
                    blended_outs['v6_baseline_selected_modality_idx'] = blended_outs['v6_query_baseline_selected_modality_idx']
                    blended_outs['v6_center_source_idx'] = blended_outs['v6_query_center_source_idx']
                    blended_outs['v6_height_source_idx'] = blended_outs['v6_query_height_source_idx']
                    blended_outs['v6_cls_source_idx'] = blended_outs['v6_query_cls_source_idx']
                    blended_outs['v6_non_fused_selector_probs'] = blended_outs['v6_query_non_fused_selector_probs']
                if (
                    use_v7_inference and
                    v7_info is not None and
                    'v7_query_switch_triggered' in blended_outs
                ):
                    blended_outs['v7_switch_triggered'] = blended_outs['v7_query_switch_triggered']
                    blended_outs['v7_damage_state_idx'] = blended_outs['v7_query_damage_state_idx']
                    blended_outs['v7_center_source_idx'] = blended_outs['v7_query_center_source_idx']
                    blended_outs['v7_bev_center_anom'] = blended_outs['v7_query_bev_center_anom']
                    blended_outs['v7_img_center_anom'] = blended_outs['v7_query_img_center_anom']
                    blended_outs['v7_fused_center_anom'] = blended_outs['v7_query_fused_center_anom']
                    blended_outs['v7_img_fallback_prob'] = blended_outs['v7_query_img_fallback_prob']
                    blended_outs['v7_img_fallback_margin'] = blended_outs['v7_query_img_fallback_margin']
                    blended_outs['v7_bev_fallback_prob'] = blended_outs['v7_query_bev_fallback_prob']
                    blended_outs['v7_bev_fallback_margin'] = blended_outs['v7_query_bev_fallback_margin']
                    blended_outs['v7_reference_distance_bin'] = blended_outs['v7_query_reference_distance_bin']
                    blended_outs['v7_pred_class'] = blended_outs['v7_query_pred_class']
                if use_v8_router and v8_info is not None:
                    blended_outs['v8_fused_keep'] = blended_outs['v8_fused_keep']
                    blended_outs['v8_center_fused_keep'] = blended_outs['v8_center_fused_keep']
                    blended_outs['v8_damage_gate'] = blended_outs['v8_damage_gate']
                    blended_outs['v8_bev_damage'] = blended_outs['v8_bev_damage']
                    blended_outs['v8_img_damage'] = blended_outs['v8_img_damage']
                    blended_outs['v8_dual_damage'] = blended_outs['v8_dual_damage']
                    blended_outs['v8_context_bev_damage'] = blended_outs['v8_context_bev_damage']
                    blended_outs['v8_context_img_damage'] = blended_outs['v8_context_img_damage']
                    blended_outs['v8_a4_anchor_ready'] = blended_outs['v8_a4_anchor_ready']
                    blended_outs['v8_a4_bev_damage_z'] = blended_outs['v8_a4_bev_damage_z']
                    blended_outs['v8_a4_img_damage_z'] = blended_outs['v8_a4_img_damage_z']
                    blended_outs['v8_a4_cls_unc_z_bev'] = blended_outs['v8_a4_cls_unc_z_bev']
                    blended_outs['v8_a4_cls_unc_z_img'] = blended_outs['v8_a4_cls_unc_z_img']
                    blended_outs['v8_a4_route_score_bev'] = blended_outs['v8_a4_route_score_bev']
                    blended_outs['v8_a4_route_score_img'] = blended_outs['v8_a4_route_score_img']
                    blended_outs['v8_a4_clean_anchor_update_count'] = blended_outs['v8_a4_clean_anchor_update_count']
                    for v9_key in (
                            'v9_a_bev', 'v9_a_img',
                            'v9_a_bev_above_tau', 'v9_a_img_above_tau',
                            'v9_clean_like_mask',
                            'v9_anchor_distance_bev', 'v9_anchor_distance_img',
                            'v9_consistency_bev', 'v9_consistency_img',
                            'v9_tail_prob_bev', 'v9_tail_prob_img',
                            'v9_tail_excess_bev', 'v9_tail_excess_img',
                            'v9_cvar_bev', 'v9_cvar_img',
                            'v9_q95_shift_bev', 'v9_q95_shift_img',
                            'v9_route_score_bev', 'v9_route_score_img',
                            'v9_mean_disagree', 'v9_velocity_l1'):
                        if v9_key in blended_outs:
                            blended_outs[v9_key] = blended_outs[v9_key]
                    blended_outs['v8_non_fused_probs'] = blended_outs['v8_non_fused_probs']
                    blended_outs['v8_non_fused_choice'] = blended_outs['v8_non_fused_choice']
                    blended_outs['v8_center_source'] = blended_outs['v8_center_source']
                    blended_outs['v8_gate_reason'] = blended_outs['v8_gate_reason']
                    blended_outs['v8_damage_state'] = blended_outs['v8_damage_state']
                    blended_outs['v8_high_damage_query'] = blended_outs['v8_high_damage_query']
                    blended_outs['v8_center_route_allowed'] = blended_outs['v8_center_route_allowed']
                if 'supp_query_added' in blended_outs:
                    blended_outs['supp_added'] = blended_outs['supp_query_added']
                    blended_outs['supp_img_score'] = blended_outs['supp_query_img_score']
                    blended_outs['supp_branch_score'] = blended_outs.get(
                        'supp_query_branch_score', blended_outs['supp_query_img_score'])
                    if 'supp_query_bev_score' in blended_outs:
                        blended_outs['supp_bev_score'] = blended_outs['supp_query_bev_score']
                    blended_outs['supp_fused_score'] = blended_outs['supp_query_fused_score']
                    blended_outs['supp_img_class'] = blended_outs['supp_query_img_class']
                    if 'supp_query_branch_class' in blended_outs:
                        blended_outs['supp_branch_class'] = blended_outs['supp_query_branch_class']
                    blended_outs['supp_utility_score'] = blended_outs['supp_query_utility_score']
                    if 'supp_query_fused_slot_utility_score' in blended_outs:
                        blended_outs['supp_fused_slot_utility_score'] = (
                            blended_outs['supp_query_fused_slot_utility_score'])
                    if 'supp_query_marginal_utility_score' in blended_outs:
                        blended_outs['supp_marginal_utility_score'] = (
                            blended_outs['supp_query_marginal_utility_score'])
                    if 'supp_query_source_branch' in blended_outs:
                        blended_outs['supp_source_branch'] = blended_outs['supp_query_source_branch']

            return blended_outs

        outs = self.task_heads[task_id](outs_dec)

        reference_fused = reference.split(num_queries_per_modality, dim=1)[fused_idx]

        center_raw = outs['center']
        nb_dec, bs, nq, _ = center_raw.shape
        center_nig = center_raw.view(nb_dec, bs, nq, 2, 4)
        gamma_center_norm = (center_nig[..., 0] + reference_fused[None, :, :, :2]).sigmoid()
        gamma_center_abs = gamma_center_norm.new_zeros(gamma_center_norm.shape)
        gamma_center_abs[..., 0:1] = gamma_center_norm[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        gamma_center_abs[..., 1:2] = gamma_center_norm[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        outs['center_nig'] = torch.cat(
            [gamma_center_norm.unsqueeze(-1), center_nig[..., 1:]], dim=-1).flatten(-2, -1)
        outs['center'] = gamma_center_abs

        height_raw = outs['height']
        height_nig = height_raw.view(nb_dec, bs, nq, 1, 4)
        gamma_height_norm = (height_nig[..., 0] + reference_fused[None, :, :, 2:3]).sigmoid()
        gamma_height_abs = gamma_height_norm.new_zeros(gamma_height_norm.shape)
        gamma_height_abs[..., 0:1] = gamma_height_norm[..., 0:1] * (pc_range[5] - pc_range[2]) + pc_range[2]
        outs['height_nig'] = torch.cat(
            [gamma_height_norm.unsqueeze(-1), height_nig[..., 1:]], dim=-1).flatten(-2, -1)
        outs['height'] = gamma_height_abs

        if self.use_dim_edl and 'dim' in outs:
            dim_raw = outs['dim']
            nb_dec, bs, nq, _ = dim_raw.shape
            dim_nig = dim_raw.view(nb_dec, bs, nq, 3, 4)
            outs['dim_nig'] = dim_raw
            outs['dim'] = dim_nig[..., 0]

        cls_raw = outs['cls_logits']
        nb_dec, bs, nq, c_cls = cls_raw.shape
        num_cls = c_cls // 2
        cls_reshaped = cls_raw.view(nb_dec, bs, nq, num_cls, 2)
        alpha = F.softplus(cls_reshaped[..., 0]) + 1.0
        beta = F.softplus(cls_reshaped[..., 1]) + 1.0
        outs['cls_alpha'] = alpha
        outs['cls_beta'] = beta
        outs['cls_logits'] = torch.log(alpha / beta + 1e-8)

        if not self.training:
            per_mod_unc = {}
            for modality in modalities:
                if modality in nig_params_dict and nig_params_dict[modality] is not None:
                    unc = self.compute_uncertainty(nig_params_dict[modality])
                    per_mod_unc[modality] = float(unc.mean().item()) if unc is not None else 0.0
                else:
                    per_mod_unc[modality] = 0.0
            attn_approx = F.softmax(-dist_mask, dim=-1)
            attn_split = attn_approx.split(num_queries_per_modality, dim=-1)
            modality_weights = {}
            for idx, modality in enumerate(modalities):
                modality_weights[modality] = float(attn_split[idx].sum(dim=-1).mean().item())
            outs['per_modality_unc'] = per_mod_unc
            outs['modality_weights'] = modality_weights
            if modality_weights:
                outs['selected_modality'] = max(modality_weights, key=modality_weights.get)

        return outs
