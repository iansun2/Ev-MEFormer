# ------------------------------------------------------------------------
# EDL Uncertainty Logger
# Collects per-sample uncertainty and modality weight info during inference
# and saves to JSON for post-hoc analysis and visualization.
# ------------------------------------------------------------------------
import gzip
import json
import os
import numpy as np
import torch

V5_MODALITY_MAP = {0: "fused", 1: "bev", 2: "img"}
V5_FALLBACK_MAP = {
    0: "class_distance",
    1: "class_global",
    2: "global",
    3: "missing",
}
V6_SOURCE_MAP = {
    -1: "soft_blend",
    0: "fused",
    1: "bev",
    2: "img",
}
V7_DAMAGE_STATE_MAP = {
    0: "neutral",
    1: "lidar_failure",
    2: "image_failure",
    3: "dual_damage",
}
V7_CENTER_SOURCE_MAP = {
    -1: "soft_blend",
    1: "bev_fallback",
    2: "img_fallback",
}
V8_SOURCE_MAP = {
    0: "fused",
    1: "bev",
    2: "img",
}
V8_DAMAGE_STATE_MAP = {
    0: "clean_like",
    1: "bev_damage",
    2: "img_damage",
    3: "dual_damage",
    4: "ambiguous",
}
V8_GATE_REASON_MAP = {
    0: "keep_fused",
    1: "bev_damage",
    2: "img_damage",
    3: "dual_damage",
    4: "ambiguous",
}
V8_RELIABILITY_STATE_MAP = {
    0: "both_reliable",
    1: "bev_unreliable",
    2: "img_unreliable",
    3: "dual_unreliable",
    4: "ambiguous",
}


class EDLUncertaintyLogger:
    """Collect and save per-sample EDL uncertainty to JSON.
    
    Supports two scenarios:
    - Stage 1 (MOAD-only): center_unc, height_unc, cls_unc, combined_unc
    - Stage 2 (MOAD+PME): above + per_modality_unc, modality_weights,
      selected_modality, fused_weight, selector_entropy
    
    Args:
        output_dir (str): Directory to save the JSON log.
        class_names (list[str]): Class names for label mapping.
    """
    
    def __init__(self, output_dir, class_names, prefix="edl", rank=None,
                 world_size=None, compression="none", keep_full_json=True,
                 stream_full=False, prediction_full_stream=False,
                 prediction_light=True, flush_interval=0,
                 sample_context=None, extra_meta=None,
                 branch_candidates=False, branch_candidate_topk=300,
                 branch_candidate_score_thr=0.0,
                 branch_candidate_branches=None,
                 query_semantic=False, query_semantic_topk=0,
                 query_semantic_score_thr=0.0):
        self.output_dir = output_dir
        self.class_names = class_names
        self.prefix = prefix
        self.rank = rank
        self.world_size = world_size
        self.compression_requested = compression
        self.compression = self._normalize_compression(compression)
        self.keep_full_json = bool(keep_full_json)
        self.stream_full = bool(stream_full)
        self.prediction_full_stream = bool(prediction_full_stream)
        self.prediction_light = bool(prediction_light)
        self.flush_interval = int(flush_interval or 0)
        self.sample_context = dict(sample_context or {})
        self.extra_meta = dict(extra_meta or {})
        self.branch_candidates = bool(branch_candidates)
        self.branch_candidate_topk = int(branch_candidate_topk or 0)
        self.branch_candidate_score_thr = float(branch_candidate_score_thr or 0.0)
        self.query_semantic = bool(query_semantic)
        self.query_semantic_topk = int(query_semantic_topk or 0)
        self.query_semantic_score_thr = float(query_semantic_score_thr or 0.0)
        if isinstance(branch_candidate_branches, str):
            branch_candidate_branches = [
                item.strip() for item in branch_candidate_branches.split(',')
                if item.strip()
            ]
        self.branch_candidate_branches = list(branch_candidate_branches or ["fused", "bev", "img"])
        self.logs = []
        os.makedirs(output_dir, exist_ok=True)
        rank_suffix = "" if rank is None else f".rank{int(rank):03d}"
        stream_ext = ".jsonl" + self._compression_extension(self.compression)
        if prefix == "edl" and rank is None and self.compression == "none":
            self.scene_summary_filename = "edl_scene_summary.jsonl"
            self.prediction_light_filename = (
                "edl_prediction_light.jsonl" if self.prediction_light else None)
            self.branch_candidates_filename = (
                "edl_branch_candidates.jsonl" if self.branch_candidates else None)
            self.query_semantic_filename = (
                "edl_query_semantic.jsonl" if self.query_semantic else None)
            self.sidecar_meta_filename = "edl_sidecar_meta.json"
            self.sample_full_filename = None
            self.prediction_full_filename = None
        else:
            self.scene_summary_filename = f"{prefix}_scene_summary{rank_suffix}{stream_ext}"
            self.prediction_light_filename = (
                f"{prefix}_prediction_light{rank_suffix}{stream_ext}"
                if self.prediction_light else None)
            self.branch_candidates_filename = (
                f"{prefix}_branch_candidates{rank_suffix}{stream_ext}"
                if self.branch_candidates else None)
            self.query_semantic_filename = (
                f"{prefix}_query_semantic{rank_suffix}{stream_ext}"
                if self.query_semantic else None)
            self.sidecar_meta_filename = f"{prefix}_meta{rank_suffix}.json"
            self.sample_full_filename = (
                f"{prefix}_sample_full{rank_suffix}{stream_ext}"
                if self.stream_full else None)
            self.prediction_full_filename = (
                f"{prefix}_prediction_full{rank_suffix}{stream_ext}"
                if self.stream_full and self.prediction_full_stream else None)
        self._scene_summary_path = os.path.join(output_dir, self.scene_summary_filename)
        self._prediction_light_path = (
            os.path.join(output_dir, self.prediction_light_filename)
            if self.prediction_light_filename else None)
        self._branch_candidates_path = (
            os.path.join(output_dir, self.branch_candidates_filename)
            if self.branch_candidates_filename else None)
        self._query_semantic_path = (
            os.path.join(output_dir, self.query_semantic_filename)
            if self.query_semantic_filename else None)
        self._sidecar_meta_path = os.path.join(output_dir, self.sidecar_meta_filename)
        self._sample_full_path = (
            os.path.join(output_dir, self.sample_full_filename)
            if self.sample_full_filename else None)
        self._prediction_full_path = (
            os.path.join(output_dir, self.prediction_full_filename)
            if self.prediction_full_filename else None)
        self._scene_summary_fp = self._open_text_writer(self._scene_summary_path)
        self._prediction_light_fp = (
            self._open_text_writer(self._prediction_light_path)
            if self._prediction_light_path else None)
        self._branch_candidates_fp = (
            self._open_text_writer(self._branch_candidates_path)
            if self._branch_candidates_path else None)
        self._query_semantic_fp = (
            self._open_text_writer(self._query_semantic_path)
            if self._query_semantic_path else None)
        self._sample_full_fp = (
            self._open_text_writer(self._sample_full_path)
            if self.stream_full and self._sample_full_path else None)
        self._prediction_full_fp = (
            self._open_text_writer(self._prediction_full_path)
            if self._prediction_full_path else None)
        self._sidecar_counts = {
            "num_samples": 0,
            "num_predictions": 0,
            "num_branch_candidates": 0,
            "num_query_semantic": 0,
        }
        self._v6_schema_samples = 0
        self._v6_missing_field_counts = {}
        self._v7_schema_samples = 0
        self._v7_missing_field_counts = {}
        self._v8_schema_samples = 0
        self._v8_missing_field_counts = {}
        self._v6_expected_scene_keys = [
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
        ]
        self._v6_expected_query_keys = [
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
        ]
        self._v7_expected_scene_keys = [
            'v7_switch_ratio',
            'v7_damage_state_ratio',
            'v7_center_source_ratio',
            'v7_bev_center_anom_mean',
            'v7_img_center_anom_mean',
            'v7_fused_center_anom_mean',
        ]
        self._v7_expected_query_keys = [
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
        ]
        self._v8_expected_scene_keys = [
            'v8_fused_keep_mean',
            'v8_center_fused_keep_mean',
            'v8_damage_gate_mean',
            'v8_non_fused_weight_ratio',
            'v8_damage_state_ratio',
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
            'v8_soft_weight_ratio',
            'v8_reliability_state_ratio',
            'v8_fused_keep_by_state',
            'v8_non_fused_entropy_mean',
            'v8_output_delta_cls_mean',
            'v8_output_delta_center_mean',
            'v8_output_delta_height_mean',
            'v9_a_bev_mean',
            'v9_a_img_mean',
            'v9_a_bev_above_tau_mean',
            'v9_a_img_above_tau_mean',
            'v9_clean_like_mask_mean',
            'v9_anchor_distance_bev_mean',
            'v9_anchor_distance_img_mean',
            'v9_tail_prob_bev_mean',
            'v9_tail_prob_img_mean',
            'v9_tail_excess_bev_mean',
            'v9_tail_excess_img_mean',
            'v9_cvar_bev_mean',
            'v9_cvar_img_mean',
            'v9_q95_shift_bev_mean',
            'v9_q95_shift_img_mean',
            'v9_route_score_bev_mean',
            'v9_route_score_img_mean',
            'v9_velocity_l1_mean',
            'v9_state_clean_like_mean',
            'v9_state_confirmed_img_mean',
            'v9_state_confirmed_bev_mean',
            'v9_state_confirmed_dual_mean',
            'v9_state_neutral_mean',
            'v9_cap_applied_mean',
            'v9_center_safe_applied_mean',
            'v9_center_safe_skipped_mean',
            'v9_fused_keep_pre_cap_cls_mean',
            'v9_fused_keep_pre_cap_center_mean',
            'v9_fused_keep_pre_cap_height_mean',
            'v9_fused_keep_pre_cap_vel_mean',
            'v9_fused_keep_post_cap_cls_mean',
            'v9_fused_keep_post_cap_center_mean',
            'v9_fused_keep_post_cap_height_mean',
            'v9_fused_keep_post_cap_vel_mean',
            'v9_fused_keep_final_cls_mean',
            'v9_fused_keep_final_center_mean',
            'v9_fused_keep_final_height_mean',
            'v9_fused_keep_final_vel_mean',
            'v9_soft_weight_fused_cls_mean',
            'v9_soft_weight_bev_cls_mean',
            'v9_soft_weight_img_cls_mean',
            'v9_soft_weight_fused_center_mean',
            'v9_soft_weight_bev_center_mean',
            'v9_soft_weight_img_center_mean',
            'v9_soft_weight_fused_height_mean',
            'v9_soft_weight_bev_height_mean',
            'v9_soft_weight_img_height_mean',
            'v9_soft_weight_fused_vel_mean',
            'v9_soft_weight_bev_vel_mean',
            'v9_soft_weight_img_vel_mean',
            'v9_center_l1_mean',
            'v9_center_l1_p95',
            'v9_tau_bev_current',
            'v9_tau_img_current',
            'v9_clean_distance_threshold_bev',
            'v9_clean_distance_threshold_img',
            'v9_clean_q95_z_bev',
            'v9_clean_q95_z_img',
            'v9_clean_anchor_frozen',
            'v9_motionblur_img_bce_mean',
            'v9_motionblur_img_bce_count',
            'v9_beams_bev_bce_mean',
            'v9_beams_bev_bce_count',
            'v9_points_bev_bce_mean',
            'v9_points_bev_bce_count',
            'v9_dark_img_bce_mean',
            'v9_dark_img_bce_count',
            'v9_brightness_img_bce_mean',
            'v9_brightness_img_bce_count',
            'v9_missingcamera_img_bce_mean',
            'v9_missingcamera_img_bce_count',
            'v9_fog_bev_bce_mean',
            'v9_fog_bev_bce_count',
            'v9_fog_img_bce_mean',
            'v9_fog_img_bce_count',
        ]
        self._v8_expected_query_keys = [
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
            'v8_non_fused_probs',
            'v8_non_fused_choice',
            'v8_center_source',
            'v8_gate_reason',
            'v8_damage_state',
            'v8_bev_reliability',
            'v8_img_reliability',
            'v8_reliability_margin',
            'v8_reliability_state',
            'v8_soft_weight_fused',
            'v8_soft_weight_bev',
            'v8_soft_weight_img',
            'v8_output_delta_cls',
            'v8_output_delta_center',
            'v8_output_delta_height',
            'v8_high_damage_query',
            'v8_center_route_allowed',
            'v9_a_bev',
            'v9_a_img',
            'v9_a_bev_above_tau',
            'v9_a_img_above_tau',
            'v9_clean_like_mask',
            'v9_anchor_distance_bev',
            'v9_anchor_distance_img',
            'v9_consistency_bev',
            'v9_consistency_img',
            'v9_tail_prob_bev',
            'v9_tail_prob_img',
            'v9_tail_excess_bev',
            'v9_tail_excess_img',
            'v9_cvar_bev',
            'v9_cvar_img',
            'v9_q95_shift_bev',
            'v9_q95_shift_img',
            'v9_route_score_bev',
            'v9_route_score_img',
            'v9_velocity_l1',
            'v9_state_classification',
            'v9_state_clean_like',
            'v9_state_confirmed_img',
            'v9_state_confirmed_bev',
            'v9_state_confirmed_dual',
            'v9_state_neutral',
            'v9_cap_applied',
            'v9_center_safe_applied',
            'v9_center_safe_skipped',
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
            'v9_soft_weight_img_vel',
            'v9_center_l1',
        ]
        self._v10_expected_scene_keys = [
            'v10_cls_fused_keep_min_current',
            'v10_n_pairs_local',
            'v10_n_pairs_global',
            'v10_quality_gap_p50',
            'v10_quality_gap_p95',
            'v10_oracle_entropy_mean',
            'v10_matched_count_per_corruption',
            'loss_v10_route_kl',
            'loss_v10_route_margin',
            'loss_v10_dd_anomaly',
            'loss_v10_dd_consistency',
            'loss_v10_csg',
            'force_modality_active',
            'forced_branch',
        ]
        self._v10_expected_query_keys = [
            'v10_w_fused_cls',
            'v10_w_bev_cls',
            'v10_w_img_cls',
            'v10_w_fused_center',
            'v10_w_bev_center',
            'v10_w_img_center',
            'v10_w_fused_height',
            'v10_w_bev_height',
            'v10_w_img_height',
            'v10_p_dmg_bev',
            'v10_p_dmg_img',
            'v10_p_dmg_dual',
            'v10_center_floor',
        ]
        self._v8_expected_scene_keys.extend(self._v10_expected_scene_keys)
        self._v8_expected_query_keys.extend(self._v10_expected_query_keys)
        self._sidecar_fields = {
            "scene_summary": [
                "sample_token",
                "scene_token",
                "ev_split",
                "ev_model_label",
                "corruption_type",
                "corruption_severity",
                "num_predictions",
                "scene_avg_unc",
                "scene_avg_weights",
                "scene_fused_weight",
                "scene_selector_entropy",
                "scene_quality_top1_ratio",
                "scene_distance_prior_top1_ratio",
                "scene_residual_override_ratio",
                "scene_alpha_res_mean",
                "scene_second_opinion_ratio",
                "scene_second_opinion_teacher_match_ratio",
                "scene_uncertainty_tail_query_ratio",
                "scene_absolute_unc_summary",
                "scene_disagreement_summary",
                "scene_score_component_stats",
                "scene_distance_bin_abnormality_stats",
                "scene_v5_switch_ratio",
                "scene_v5_keep_fused_ratio",
                "scene_v5_switch_to_bev_ratio",
                "scene_v5_switch_to_img_ratio",
                "scene_v5_fused_cls_anom_mean",
                "scene_v5_fused_center_anom_mean",
                "scene_v5_fused_height_anom_mean",
                "scene_v5_fused_combined_anom_mean",
                "scene_v5_trigger_ratio_over_threshold",
                "scene_v5_bin_coverage_ratio",
                "scene_v5_fallback_class_ratio",
                "scene_v5_fallback_global_ratio",
                "scene_v5_distance_prior_candidate_ratio",
                "scene_v5_distance_prior_switched_ratio",
                "scene_v5_switch_agreement_with_prior",
                "scene_v5_head_shift",
                "scene_v6_switch_ratio",
                "scene_v6_gate_anomaly_mean",
                "scene_v6_gate_pass_anomaly_ratio",
                "scene_v6_gate_pass_candidate_prob_ratio",
                "scene_v6_gate_pass_candidate_margin_ratio",
                "scene_v6_gate_pass_final_ratio",
                "scene_v6_candidate_modality_ratio",
                "scene_v6_center_source_ratio",
                "scene_v6_height_source_ratio",
                "scene_v6_reference_fallback_ratio",
                "scene_v7_switch_ratio",
                "scene_v7_damage_state_ratio",
                "scene_v7_center_source_ratio",
                "scene_v7_bev_center_anom_mean",
                "scene_v7_img_center_anom_mean",
                "scene_v7_fused_center_anom_mean",
                "scene_v8_fused_keep_mean",
                "scene_v8_center_fused_keep_mean",
                "scene_v8_damage_gate_mean",
                "scene_v8_non_fused_weight_ratio",
                "scene_v8_damage_state_ratio",
                "scene_v8_selected_source_ratio",
                "scene_v8_high_damage_query_ratio",
                "scene_v8_center_route_allowed_ratio",
                "scene_v8_bev_damage_mean",
                "scene_v8_img_damage_mean",
                "scene_v8_dual_damage_mean",
                "scene_v8_context_bev_damage_mean",
                "scene_v8_context_img_damage_mean",
                "scene_v8_a4_anchor_ready",
                "scene_v8_a4_bev_damage_z_mean",
                "scene_v8_a4_img_damage_z_mean",
                "scene_v8_a4_cls_unc_z_bev_mean",
                "scene_v8_a4_cls_unc_z_img_mean",
                "scene_v8_a4_route_score_bev_mean",
                "scene_v8_a4_route_score_img_mean",
                "scene_v8_a4_clean_anchor_update_count",
                "scene_v8_bev_reliability_mean",
                "scene_v8_img_reliability_mean",
                "scene_v8_reliability_margin_mean",
                "scene_v8_soft_weight_ratio",
                "scene_v8_reliability_state_ratio",
                "scene_v8_fused_keep_by_state",
                "scene_v8_non_fused_entropy_mean",
                "scene_v8_output_delta_cls_mean",
                "scene_v8_output_delta_center_mean",
                "scene_v8_output_delta_height_mean",
                "scene_v9_a_bev_mean",
                "scene_v9_a_img_mean",
                "scene_v9_a_bev_above_tau_mean",
                "scene_v9_a_img_above_tau_mean",
                "scene_v9_clean_like_mask_mean",
                "scene_v9_anchor_distance_bev_mean",
                "scene_v9_anchor_distance_img_mean",
                "scene_v9_tail_prob_bev_mean",
                "scene_v9_tail_prob_img_mean",
                "scene_v9_tail_excess_bev_mean",
                "scene_v9_tail_excess_img_mean",
                "scene_v9_cvar_bev_mean",
                "scene_v9_cvar_img_mean",
                "scene_v9_q95_shift_bev_mean",
                "scene_v9_q95_shift_img_mean",
                "scene_v9_route_score_bev_mean",
                "scene_v9_route_score_img_mean",
                "scene_v9_velocity_l1_mean",
                "scene_v9_state_clean_like_mean",
                "scene_v9_state_confirmed_img_mean",
                "scene_v9_state_confirmed_bev_mean",
                "scene_v9_state_confirmed_dual_mean",
                "scene_v9_state_neutral_mean",
                "scene_v9_cap_applied_mean",
                "scene_v9_center_safe_applied_mean",
                "scene_v9_center_safe_skipped_mean",
                "scene_v9_fused_keep_pre_cap_cls_mean",
                "scene_v9_fused_keep_pre_cap_center_mean",
                "scene_v9_fused_keep_pre_cap_height_mean",
                "scene_v9_fused_keep_pre_cap_vel_mean",
                "scene_v9_fused_keep_post_cap_cls_mean",
                "scene_v9_fused_keep_post_cap_center_mean",
                "scene_v9_fused_keep_post_cap_height_mean",
                "scene_v9_fused_keep_post_cap_vel_mean",
                "scene_v9_fused_keep_final_cls_mean",
                "scene_v9_fused_keep_final_center_mean",
                "scene_v9_fused_keep_final_height_mean",
                "scene_v9_fused_keep_final_vel_mean",
                "scene_v9_soft_weight_fused_cls_mean",
                "scene_v9_soft_weight_bev_cls_mean",
                "scene_v9_soft_weight_img_cls_mean",
                "scene_v9_soft_weight_fused_center_mean",
                "scene_v9_soft_weight_bev_center_mean",
                "scene_v9_soft_weight_img_center_mean",
                "scene_v9_soft_weight_fused_height_mean",
                "scene_v9_soft_weight_bev_height_mean",
                "scene_v9_soft_weight_img_height_mean",
                "scene_v9_soft_weight_fused_vel_mean",
                "scene_v9_soft_weight_bev_vel_mean",
                "scene_v9_soft_weight_img_vel_mean",
                "scene_v9_center_l1_mean",
                "scene_v9_center_l1_p95",
                "scene_v9_fused_keep_unit_mean",
                "scene_v9_fused_keep_unit_q05",
                "scene_v9_fused_keep_unit_q50",
                "scene_v9_fused_keep_unit_q95",
                "scene_v9_fused_keep_at_min_ratio",
                "scene_v9_current_keep_min",
                "scene_v9_current_lambda_gate",
                "scene_v9_current_lambda_route",
                "scene_v9_tau_bev_current",
                "scene_v9_tau_img_current",
                "scene_v9_clean_distance_threshold_bev",
                "scene_v9_clean_distance_threshold_img",
                "scene_v9_clean_q95_z_bev",
                "scene_v9_clean_q95_z_img",
                "scene_v9_clean_anchor_frozen",
                "scene_v9_motionblur_img_bce_mean",
                "scene_v9_motionblur_img_bce_count",
                "scene_v9_beams_bev_bce_mean",
                "scene_v9_beams_bev_bce_count",
                "scene_v9_points_bev_bce_mean",
                "scene_v9_points_bev_bce_count",
                "scene_v9_dark_img_bce_mean",
                "scene_v9_dark_img_bce_count",
                "scene_v9_brightness_img_bce_mean",
                "scene_v9_brightness_img_bce_count",
                "scene_v9_missingcamera_img_bce_mean",
                "scene_v9_missingcamera_img_bce_count",
                "scene_v9_fog_bev_bce_mean",
                "scene_v9_fog_bev_bce_count",
                "scene_v9_fog_img_bce_mean",
                "scene_v9_fog_img_bce_count",
                "scene_qsem_img_score_mean",
                "scene_qsem_img_score_q05",
                "scene_qsem_img_score_q50",
                "scene_qsem_img_score_p95",
                "scene_qsem_img_margin_mean",
                "scene_qsem_img_margin_q05",
                "scene_qsem_img_margin_q50",
                "scene_qsem_img_margin_p95",
                "scene_qsem_img_entropy_mean",
                "scene_qsem_img_entropy_q50",
                "scene_qsem_img_entropy_p95",
                "scene_qsem_fused_score_mean",
                "scene_qsem_fused_margin_mean",
                "scene_qsem_fused_entropy_mean",
                "scene_qsem_bev_score_mean",
                "scene_qsem_bev_margin_mean",
                "scene_qsem_bev_entropy_mean",
                "scene_qsem_img_low_score020_ratio",
                "scene_qsem_img_low_score030_ratio",
                "scene_qsem_img_low_margin005_ratio",
                "scene_qsem_img_low_margin010_ratio",
                "scene_qsem_img_high_entropy080_ratio",
                "scene_qsem_img_fused_class_disagree_ratio",
                "scene_qsem_img_fused_highconf_disagree_ratio",
                "scene_qsem_img_bev_class_disagree_ratio",
                "scene_qsem_img_bev_highconf_disagree_ratio",
                "scene_qsem_fused_bev_class_disagree_ratio",
                "scene_qsem_fused_bev_highconf_disagree_ratio",
                "scene_qsem_img_score_minus_fused_mean",
                "scene_qsem_img_margin_minus_fused_mean",
                "scene_qsem_img_score_minus_bev_mean",
                "scene_qsem_img_margin_minus_bev_mean",
                "scene_qsem_top100_query_count",
                "scene_qsem_top100_max_score_mean",
                "scene_qsem_top100_img_score_mean",
                "scene_qsem_top100_img_margin_mean",
                "scene_qsem_top100_img_entropy_mean",
                "scene_qsem_top100_img_low_score030_ratio",
                "scene_qsem_top100_img_low_margin010_ratio",
                "scene_qsem_top100_img_high_entropy080_ratio",
                "scene_qsem_top100_img_fused_class_disagree_ratio",
                "scene_qsem_top100_img_fused_highconf_disagree_ratio",
                "scene_qsem_top100_img_bev_class_disagree_ratio",
                "scene_qsem_top100_img_bev_highconf_disagree_ratio",
                "scene_qsem_top300_query_count",
                "scene_qsem_top300_max_score_mean",
                "scene_qsem_top300_img_score_mean",
                "scene_qsem_top300_img_margin_mean",
                "scene_qsem_top300_img_entropy_mean",
                "scene_qsem_top300_img_low_score030_ratio",
                "scene_qsem_top300_img_low_margin010_ratio",
                "scene_qsem_top300_img_high_entropy080_ratio",
                "scene_qsem_top300_img_fused_class_disagree_ratio",
                "scene_qsem_top300_img_fused_highconf_disagree_ratio",
                "scene_qsem_top300_img_bev_class_disagree_ratio",
                "scene_qsem_top300_img_bev_highconf_disagree_ratio",
                "scene_supp_scene_gate_ratio",
                "scene_supp_candidate_ratio",
                "scene_supp_img_candidate_ratio",
                "scene_supp_bev_candidate_ratio",
                "scene_supp_source_bev_query_ratio",
                "scene_supp_selected_img_ratio",
                "scene_supp_selected_bev_ratio",
                "scene_supp_branch_aware_enabled",
                "scene_supp_branch_mapper_enabled",
                "scene_supp_img_utility_candidate_ratio",
                "scene_supp_bev_utility_candidate_ratio",
                "scene_supp_img_utility_score_mean",
                "scene_supp_bev_utility_score_mean",
                "scene_supp_utility_candidate_ratio",
                "scene_supp_utility_score_mean",
                "scene_supp_utility_threshold",
                "scene_supp_img_utility_threshold",
                "scene_supp_bev_utility_threshold",
                "scene_supp_fused_slot_utility_score_mean",
                "scene_supp_marginal_enabled",
                "scene_supp_marginal_margin",
                "scene_supp_dual_marginal_margin",
                "scene_supp_marginal_checked_ratio",
                "scene_supp_marginal_reject_ratio",
                "scene_supp_marginal_pass_ratio",
                "scene_supp_selected_margin_score_mean",
                "scene_supp_added_ratio",
                "scene_supp_replace_slot_ratio",
                "scene_supp_scene_gate_pre_guard_ratio",
                "scene_supp_scene_gate_post_guard_ratio",
                "scene_supp_img_damage_guard_enabled",
                "scene_supp_img_damage_guard_mode_code",
                "scene_supp_img_damage_gate_ratio",
                "scene_supp_dual_damage_gate_ratio",
                "scene_supp_dual_gate_mode_code",
                "scene_supp_scene_mapper_dual_gate_ratio",
                "scene_supp_bev_damage_gate_ratio",
                "scene_supp_bev_damage_upper_gate_ratio",
                "scene_supp_fused_damage_gate_ratio",
                "scene_supp_bev_fused_dominance_gate_ratio",
                "scene_supp_bev_fused_normalized_ratio_mean",
                "scene_supp_dual_abstain_gate_ratio",
                "scene_supp_img_abstain_gate_ratio",
                "scene_supp_dual_abstain_score",
                "scene_supp_img_abstain_score",
                "scene_supp_dual_abstain_threshold",
                "scene_supp_img_abstain_threshold",
                "scene_supp_dual_abstain_available",
                "scene_supp_img_abstain_available",
                "scene_supp_qsem_gate_enabled",
                "scene_supp_qsem_gate_ratio",
                "scene_supp_qsem_disagree_gate_ratio",
                "scene_supp_qsem_margin_gate_ratio",
                "scene_supp_qsem_disagree_score",
                "scene_supp_qsem_margin_score",
                "scene_supp_qsem_disagree_threshold",
                "scene_supp_qsem_margin_threshold",
                "scene_supp_qsem_disagree_available",
                "scene_supp_qsem_margin_available",
                "scene_supp_contextheight_gate_enabled",
                "scene_supp_contextheight_gate_ratio",
                "scene_supp_contextheight_context_gate_ratio",
                "scene_supp_contextheight_height_gate_ratio",
                "scene_supp_contextheight_v9_bev_gate_ratio",
                "scene_supp_contextheight_context_score",
                "scene_supp_contextheight_height_score",
                "scene_supp_contextheight_v9_bev_score",
                "scene_supp_contextheight_context_threshold",
                "scene_supp_contextheight_height_threshold",
                "scene_supp_contextheight_v9_bev_threshold",
                "scene_supp_contextheight_context_available",
                "scene_supp_contextheight_height_available",
                "scene_supp_contextheight_v9_bev_available",
                "scene_supp_img_damage_threshold",
                "scene_supp_img_damage_score",
                "scene_supp_bev_damage_threshold",
                "scene_supp_bev_damage_upper_threshold",
                "scene_supp_fused_damage_threshold",
                "scene_supp_img_damage_threshold_available",
                "scene_supp_bev_damage_threshold_available",
                "scene_supp_bev_damage_upper_threshold_available",
                "scene_supp_fused_damage_threshold_available",
                "scene_supp_bev_fused_reference_available",
                "scene_supp_img_damage_threshold_scale",
                "scene_supp_fused_damage_threshold_scale",
                "scene_supp_bev_fused_dominance_ratio",
                "scene_supp_bev_damage_upper_threshold_scale",
                "scene_supp_dual_budget_ratio",
                "scene_supp_dual_utility_threshold",
                "scene_supp_dual_scene_mapper_threshold",
                "scene_supp_dual_bev_threshold_scale",
                "scene_supp_dual_bev_rank_boost",
                "scene_supp_bev_threshold",
                "scene_supp_img_ceiling",
                "scene_supp_margin_threshold",
                "scene_supp_scene_bev_score",
                "scene_supp_scene_img_score",
                "scene_supp_scene_fused_score",
                "scene_supp_scene_aux_score",
                "scene_supp_max_add_queries",
                "scene_supp_scene_mapper_score_mean",
                "scene_supp_scene_mapper_gate_ratio",
                "scene_supp_scene_mapper_configured",
                "scene_supp_scene_mapper_feature_expected_count",
                "scene_supp_scene_mapper_feature_available_count",
                "scene_supp_scene_mapper_feature_finite_count",
                "scene_supp_scene_mapper_feature_missing_count",
                "scene_supp_scene_mapper_contract_ready",
                "scene_supp_scene_mapper_q95_available",
                "scene_supp_scene_mapper_q95_runtime_mean",
                "scene_supp_dynamic_budget_ratio",
                "scene_supp_effective_max_add_queries",
                "scene_supp_rank_mode_code",
                "scene_supp_img_score_threshold",
                "scene_supp_bev_score_threshold",
                "scene_supp_nms_center_dist",
                "scene_supp_per_class_cap",
                "scene_supp_enabled_runtime",
                "scene_supp_status_code",
                "scene_supp_noadd_reason_code",
                "scene_v10_cls_fused_keep_min_current",
                "scene_v10_n_pairs_local",
                "scene_v10_n_pairs_global",
                "scene_v10_quality_gap_p50",
                "scene_v10_quality_gap_p95",
                "scene_v10_oracle_entropy_mean",
                "scene_v10_matched_count_per_corruption",
                "scene_loss_v10_route_kl",
                "scene_loss_v10_route_margin",
                "scene_loss_v10_dd_anomaly",
                "scene_loss_v10_dd_consistency",
                "scene_loss_v10_csg",
                "scene_force_modality_active",
                "scene_forced_branch",
                "scene_branch_candidate_enabled",
                "scene_branch_candidate_count",
                "scene_branch_candidate_valid_count",
                "scene_branch_candidate_valid_ratio",
                "scene_branch_candidate_limit",
                "scene_branch_candidate_limit_per_branch",
                "scene_branch_candidate_total_available",
                "scene_branch_candidate_score_threshold",
            ],
            "prediction_light": [
                "sample_token",
                "scene_token",
                "corruption_type",
                "corruption_severity",
                "box_idx",
                "label",
                "label_name",
                "score",
                "bbox_center",
                "distance_to_ego",
                "selected_modality",
                "combined_unc",
                "center_unc",
                "height_unc",
                "cls_unc",
                "v5_fused_cls_unc",
                "v5_fused_center_unc",
                "v5_fused_height_unc",
                "fused_anomaly_score",
                "fused_anomaly_components",
                "v5_switch_triggered",
                "v5_selected_modality",
                "v5_candidate_modality",
                "v5_reference_bin",
                "v5_reference_fallback_level",
                "v5_reference_fallback_name",
                "v5_reference_pred_class",
                "v6_switch_triggered",
                "v6_candidate_modality",
                "v6_gate_anomaly_score",
                "v6_gate_passed_confidence",
                "v6_gate_passed_margin",
                "v6_gate_pass_anomaly",
                "v6_gate_pass_candidate_prob",
                "v6_gate_pass_candidate_margin",
                "v6_gate_pass_final",
                "v6_reference_bin",
                "v6_reference_fallback_level",
                "v6_reference_fallback_name",
                "v6_reference_pred_class",
                "v6_baseline_selected_modality",
                "v6_center_source",
                "v6_height_source",
                "v6_cls_source",
                "v6_non_fused_selector_probs",
                "v7_switch_triggered",
                "v7_damage_state",
                "v7_center_source",
                "v7_bev_center_anom",
                "v7_img_center_anom",
                "v7_fused_center_anom",
                "v7_img_fallback_prob",
                "v7_img_fallback_margin",
                "v7_bev_fallback_prob",
                "v7_bev_fallback_margin",
                "v7_reference_bin",
                "v7_pred_class",
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
                "v8_non_fused_probs",
                "v8_non_fused_choice",
                "v8_center_source",
                "v8_gate_reason",
                "v8_damage_state",
                "v8_bev_reliability",
                "v8_img_reliability",
                "v8_reliability_margin",
                "v8_reliability_state",
                "v8_soft_weight_fused",
                "v8_soft_weight_bev",
                "v8_soft_weight_img",
                "v8_output_delta_cls",
                "v8_output_delta_center",
                "v8_output_delta_height",
                "v8_high_damage_query",
                "v8_center_route_allowed",
                "v8_corruption_type",
                "v8_corruption_severity",
                "v9_a_bev",
                "v9_a_img",
                "v9_a_bev_above_tau",
                "v9_a_img_above_tau",
                "v9_clean_like_mask",
                "v9_anchor_distance_bev",
                "v9_anchor_distance_img",
                "v9_consistency_bev",
                "v9_consistency_img",
                "v9_tail_prob_bev",
                "v9_tail_prob_img",
                "v9_tail_excess_bev",
                "v9_tail_excess_img",
                "v9_cvar_bev",
                "v9_cvar_img",
                "v9_q95_shift_bev",
                "v9_q95_shift_img",
                "v9_route_score_bev",
                "v9_route_score_img",
                "v9_velocity_l1",
                "v9_state_classification",
                "v9_state_clean_like",
                "v9_state_confirmed_img",
                "v9_state_confirmed_bev",
                "v9_state_confirmed_dual",
                "v9_state_neutral",
                "v9_cap_applied",
                "v9_center_safe_applied",
                "v9_center_safe_skipped",
                "v9_fused_keep_pre_cap_cls",
                "v9_fused_keep_pre_cap_center",
                "v9_fused_keep_pre_cap_height",
                "v9_fused_keep_pre_cap_vel",
                "v9_fused_keep_post_cap_cls",
                "v9_fused_keep_post_cap_center",
                "v9_fused_keep_post_cap_height",
                "v9_fused_keep_post_cap_vel",
                "v9_fused_keep_final_cls",
                "v9_fused_keep_final_center",
                "v9_fused_keep_final_height",
                "v9_fused_keep_final_vel",
                "v9_soft_weight_fused_cls",
                "v9_soft_weight_bev_cls",
                "v9_soft_weight_img_cls",
                "v9_soft_weight_fused_center",
                "v9_soft_weight_bev_center",
                "v9_soft_weight_img_center",
                "v9_soft_weight_fused_height",
                "v9_soft_weight_bev_height",
                "v9_soft_weight_img_height",
                "v9_soft_weight_fused_vel",
                "v9_soft_weight_bev_vel",
                "v9_soft_weight_img_vel",
                "v9_center_l1",
                "v9_fused_keep_unit",
                "v9_fused_keep_at_min",
                "v10_w_fused_cls",
                "v10_w_bev_cls",
                "v10_w_img_cls",
                "v10_w_fused_center",
                "v10_w_bev_center",
                "v10_w_img_center",
                "v10_w_fused_height",
                "v10_w_bev_height",
                "v10_w_img_height",
                "v10_p_dmg_bev",
                "v10_p_dmg_img",
                "v10_p_dmg_dual",
                "v10_center_floor",
                "supp_added",
                "supp_img_score",
                "supp_bev_score",
                "supp_branch_score",
                "supp_fused_score",
                "supp_utility_score",
                "supp_fused_slot_utility_score",
                "supp_marginal_utility_score",
                "supp_source_branch",
                "supp_img_class",
                "supp_branch_class",
                "final_decode_query_idx",
                "final_decode_task_idx",
                "final_decode_bbox_idx",
                "final_decode_topk_rank",
                "final_source_alignment_valid",
                "final_source_is_supplement",
                "final_source_branch",
                "final_supp_img_score",
                "final_supp_bev_score",
                "final_supp_branch_score",
                "final_supp_fused_score",
                "final_supp_utility_score",
                "final_supp_fused_slot_utility_score",
                "final_supp_marginal_utility_score",
                "final_supp_candidate_query_idx",
                "final_supp_img_class",
                "final_supp_branch_class",
            ],
            "branch_candidates": [
                "sample_token",
                "scene_token",
                "ev_split",
                "ev_model_label",
                "corruption_type",
                "corruption_severity",
                "candidate_idx",
                "candidate_rank",
                "logging_rank",
                "branch_rank",
                "candidate_uid",
                "task_id",
                "task_class_offset",
                "query_idx",
                "branch",
                "branch_idx",
                "label",
                "local_label",
                "global_label",
                "label_name",
                "score",
                "bbox_center",
                "center_norm",
                "height_norm",
                "dim",
                "rot",
                "vel",
                "distance_to_ego",
                "valid",
                "debug_raw_cls_unc",
                "debug_raw_center_unc",
                "debug_raw_height_unc",
                "debug_raw_combined_unc",
                "cls_weight",
                "center_weight",
                "height_weight",
                "non_fused_prob",
                "fused_keep",
                "damage_gate",
                "bev_damage",
                "img_damage",
                "dual_damage",
                "damage_state",
                "center_source",
                "admitted_by_current_policy",
                "action_label",
                "action_space",
                "selected_for_cls",
                "selected_for_center",
                "selected_for_height",
            ],
            "query_semantic": [
                "sample_token",
                "scene_token",
                "ev_split",
                "ev_model_label",
                "corruption_type",
                "corruption_severity",
                "row_idx",
                "rank",
                "task_id",
                "task_class_offset",
                "query_idx",
                "score_rank",
                "max_score",
                "fused_score",
                "fused_margin",
                "fused_entropy",
                "fused_local_class",
                "fused_global_class",
                "bev_score",
                "bev_margin",
                "bev_entropy",
                "bev_local_class",
                "bev_global_class",
                "img_score",
                "img_margin",
                "img_entropy",
                "img_local_class",
                "img_global_class",
                "img_fused_class_disagree",
                "img_fused_highconf_disagree",
                "img_bev_class_disagree",
                "img_bev_highconf_disagree",
                "fused_bev_class_disagree",
                "fused_bev_highconf_disagree",
                "img_low_score020",
                "img_low_score030",
                "img_low_margin005",
                "img_low_margin010",
                "img_high_entropy080",
            ],
        }
    
    def _to_float(self, val):
        """Convert tensor/numpy to Python float."""
        if isinstance(val, torch.Tensor):
            return float(val.cpu().item()) if val.numel() == 1 else float(val.cpu().mean().item())
        elif isinstance(val, np.ndarray):
            return float(val.mean())
        return float(val)

    @staticmethod
    def _normalize_compression(compression):
        value = str(compression or "none").lower()
        if value in ("", "none", "false", "0"):
            return "none"
        if value in ("gzip", "gz"):
            return "gzip"
        if value in ("zstd", "zst"):
            try:
                import zstandard  # noqa: F401
                return "zstd"
            except Exception:
                print("[EDL Logger][WARN] zstandard is unavailable; falling back to gzip.")
                return "gzip"
        raise ValueError(f"Unsupported log compression: {compression}")

    @staticmethod
    def _compression_extension(compression):
        if compression == "gzip":
            return ".gz"
        if compression == "zstd":
            return ".zst"
        return ""

    def _open_text_writer(self, path):
        if self.compression == "gzip":
            return gzip.open(path, "wt", encoding="utf-8")
        if self.compression == "zstd":
            import io
            import zstandard as zstd
            raw_fp = open(path, "wb")
            writer = zstd.ZstdCompressor(level=3).stream_writer(raw_fp, closefd=True)
            return io.TextIOWrapper(writer, encoding="utf-8")
        return open(path, "w", encoding="utf-8")

    def _to_jsonable(self, val):
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return float(val.detach().cpu().item())
            return val.detach().cpu().tolist()
        if isinstance(val, np.ndarray):
            return val.tolist() if val.ndim > 0 else float(val.item())
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, dict):
            return {str(k): self._to_jsonable(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [self._to_jsonable(v) for v in val]
        if isinstance(val, (str, bool, int, float)) or val is None:
            return val
        return str(val)

    def _write_jsonl(self, fp, payload):
        fp.write(json.dumps(self._to_jsonable(payload), ensure_ascii=False) + "\n")

    def _flush_streams_if_needed(self):
        if self.flush_interval <= 0:
            return
        if self._sidecar_counts["num_samples"] % self.flush_interval != 0:
            return
        for fp in (
                self._scene_summary_fp,
                self._prediction_light_fp,
                self._branch_candidates_fp,
                self._query_semantic_fp,
                self._sample_full_fp,
                self._prediction_full_fp):
            if fp is not None:
                fp.flush()

    def _note_v6_missing_field(self, field_name):
        self._v6_missing_field_counts[field_name] = self._v6_missing_field_counts.get(field_name, 0) + 1

    def _note_v7_missing_field(self, field_name):
        self._v7_missing_field_counts[field_name] = self._v7_missing_field_counts.get(field_name, 0) + 1

    def _note_v8_missing_field(self, field_name):
        self._v8_missing_field_counts[field_name] = self._v8_missing_field_counts.get(field_name, 0) + 1

    def _build_scene_summary_record(self, sample_log):
        record = {"sample_token": sample_log["sample_token"]}
        for key in self._sidecar_fields["scene_summary"]:
            if key == "sample_token":
                continue
            if key in sample_log:
                record[key] = sample_log[key]
        return record

    def _build_prediction_light_record(self, sample_log, pred):
        record = {"sample_token": sample_log["sample_token"]}
        for sample_key in [
                "scene_token", "ev_split", "ev_model_label",
                "corruption_type", "corruption_severity"]:
            if sample_key in sample_log:
                record[sample_key] = sample_log[sample_key]
        for key in self._sidecar_fields["prediction_light"]:
            if key in (
                    "sample_token", "scene_token", "ev_split", "ev_model_label",
                    "corruption_type", "corruption_severity"):
                continue
            if key in pred:
                record[key] = pred[key]
        return record

    def _build_prediction_full_record(self, sample_log, pred):
        record = {}
        for sample_key in [
                "sample_token", "scene_token", "corruption_type",
                "corruption_severity", "ev_split", "ev_rank", "ev_world_size"]:
            if sample_key in sample_log:
                record[sample_key] = sample_log[sample_key]
        record.update(pred)
        return record

    def _candidate_value(self, edl_info, key, idx, default=None):
        value = edl_info.get(f"branch_candidate_{key}")
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return self._to_jsonable(value)
            if idx >= value.shape[0]:
                return default
            item = value[idx]
            if isinstance(item, torch.Tensor) and item.numel() == 1:
                return float(item.detach().cpu().item())
            return self._to_jsonable(item)
        if isinstance(value, (list, tuple)):
            if idx >= len(value):
                return default
            return self._to_jsonable(value[idx])
        return self._to_jsonable(value)

    def _candidate_count(self, edl_info):
        score = edl_info.get('branch_candidate_score')
        if isinstance(score, torch.Tensor) and score.dim() >= 1:
            count = int(score.shape[0])
        elif isinstance(score, (list, tuple)):
            count = len(score)
        else:
            return 0
        return max(0, count)

    def _is_candidate_valid(self, edl_info, idx):
        valid = self._candidate_value(edl_info, 'valid', idx, default=1.0)
        score = self._candidate_value(edl_info, 'score', idx, default=None)
        try:
            valid_ok = float(valid) > 0.5
        except Exception:
            valid_ok = bool(valid)
        try:
            score_ok = score is not None and float(score) >= self.branch_candidate_score_thr
        except Exception:
            score_ok = False
        return valid_ok and score_ok

    def _build_branch_candidate_record(self, sample_log, edl_info, candidate_idx):
        branch_idx = self._candidate_value(edl_info, 'branch_idx', candidate_idx)
        try:
            branch_idx_int = int(branch_idx)
        except Exception:
            branch_idx_int = -1
        label = self._candidate_value(edl_info, 'label', candidate_idx)
        try:
            label_int = int(label)
        except Exception:
            label_int = -1
        global_label = self._candidate_value(edl_info, 'global_label', candidate_idx, default=label_int)
        try:
            global_label_int = int(global_label)
        except Exception:
            global_label_int = label_int
        center = self._candidate_value(edl_info, 'center_abs', candidate_idx)
        record = {
            "sample_token": sample_log["sample_token"],
            "candidate_idx": int(candidate_idx),
            "candidate_rank": int(self._candidate_value(
                edl_info, 'rank', candidate_idx, default=candidate_idx)),
            "logging_rank": int(self._candidate_value(
                edl_info, 'logging_rank', candidate_idx, default=candidate_idx)),
            "branch_rank": int(self._candidate_value(
                edl_info, 'branch_rank', candidate_idx, default=candidate_idx)),
            "candidate_uid": int(self._candidate_value(
                edl_info, 'uid', candidate_idx, default=-1)),
            "task_id": int(self._candidate_value(
                edl_info, 'task_id', candidate_idx, default=-1)),
            "task_class_offset": int(self._candidate_value(
                edl_info, 'task_class_offset', candidate_idx, default=0)),
            "query_idx": int(self._candidate_value(
                edl_info, 'query_idx', candidate_idx, default=-1)),
            "branch_idx": branch_idx_int,
            "branch": V8_SOURCE_MAP.get(branch_idx_int, "unknown"),
            "label": global_label_int,
            "local_label": int(self._candidate_value(
                edl_info, 'local_label', candidate_idx, default=label_int)),
            "global_label": global_label_int,
            "label_name": (
                self.class_names[global_label_int]
                if 0 <= global_label_int < len(self.class_names) else "unknown"
            ),
            "score": self._candidate_value(edl_info, 'score', candidate_idx),
            "bbox_center": center,
            "center_norm": self._candidate_value(edl_info, 'center_norm', candidate_idx),
            "height_norm": self._candidate_value(edl_info, 'height_norm', candidate_idx),
            "dim": self._candidate_value(edl_info, 'dim', candidate_idx),
            "rot": self._candidate_value(edl_info, 'rot', candidate_idx),
            "vel": self._candidate_value(edl_info, 'vel', candidate_idx),
            "valid": bool(self._is_candidate_valid(edl_info, candidate_idx)),
        }
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            record["distance_to_ego"] = float((float(center[0]) ** 2 + float(center[1]) ** 2) ** 0.5)
        for sample_key in [
                "scene_token", "ev_split", "ev_model_label",
                "corruption_type", "corruption_severity"]:
            if sample_key in sample_log:
                record[sample_key] = sample_log[sample_key]
        for key in ["cls_unc", "center_unc", "height_unc", "combined_unc"]:
            value = self._candidate_value(edl_info, key, candidate_idx)
            if value is not None:
                record[f"debug_raw_{key}"] = value
        for key in [
                "cls_weight", "center_weight", "height_weight",
                "non_fused_prob", "fused_keep", "damage_gate",
                "bev_damage", "img_damage", "dual_damage"]:
            value = self._candidate_value(edl_info, key, candidate_idx)
            if value is not None:
                record[key] = value
        for key in [
                "admitted_by_current_policy", "action_label", "action_space",
                "selected_for_cls", "selected_for_center", "selected_for_height"]:
            value = self._candidate_value(edl_info, key, candidate_idx)
            if value is not None:
                try:
                    record[key] = int(float(value))
                except Exception:
                    record[key] = value
        damage_state = self._candidate_value(edl_info, 'damage_state', candidate_idx)
        if damage_state is not None:
            try:
                damage_state_int = int(damage_state)
            except Exception:
                damage_state_int = -1
            record["damage_state"] = V8_DAMAGE_STATE_MAP.get(damage_state_int, "unknown")
        center_source = self._candidate_value(edl_info, 'center_source', candidate_idx)
        if center_source is not None:
            try:
                center_source_int = int(center_source)
            except Exception:
                center_source_int = -1
            record["center_source"] = V8_SOURCE_MAP.get(center_source_int, "unknown")
        return record

    def _iter_branch_candidate_records(self, sample_log, edl_info):
        if not self.branch_candidates or not isinstance(edl_info, dict):
            return
        for idx in range(self._candidate_count(edl_info)):
            if not self._is_candidate_valid(edl_info, idx):
                continue
            yield self._build_branch_candidate_record(sample_log, edl_info, idx)

    def _qsem_value(self, edl_info, key, idx, default=None):
        value = edl_info.get(f"qsem_{key}")
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return self._to_jsonable(value)
            if idx >= value.shape[0]:
                return default
            item = value[idx]
            if isinstance(item, torch.Tensor) and item.numel() == 1:
                value = float(item.detach().cpu().item())
                if not np.isfinite(value):
                    return default
                return value
            return self._to_jsonable(item)
        if isinstance(value, (list, tuple)):
            if idx >= len(value):
                return default
            return self._to_jsonable(value[idx])
        return self._to_jsonable(value)

    def _qsem_count(self, edl_info):
        score = edl_info.get('qsem_img_score')
        if isinstance(score, torch.Tensor) and score.dim() >= 1:
            return max(0, int(score.shape[0]))
        if isinstance(score, (list, tuple)):
            return len(score)
        return 0

    def _qsem_score_for_rank(self, edl_info, idx):
        values = []
        for key in ('img_score', 'bev_score', 'fused_score'):
            value = self._qsem_value(edl_info, key, idx, default=None)
            try:
                values.append(float(value))
            except Exception:
                pass
        return max(values) if values else float('-inf')

    def _build_query_semantic_record(self, sample_log, edl_info, idx, rank):
        def number(key, default=None):
            return self._qsem_value(edl_info, key, idx, default=default)

        def integer(key, default=-1):
            value = number(key, default=default)
            try:
                return int(float(value))
            except Exception:
                return int(default)

        record = {
            "sample_token": sample_log["sample_token"],
            "row_idx": int(idx),
            "rank": int(rank),
            "task_id": integer("task_id"),
            "task_class_offset": integer("task_class_offset", default=0),
            "query_idx": integer("query_idx"),
            "max_score": self._qsem_score_for_rank(edl_info, idx),
        }
        for sample_key in [
                "scene_token", "ev_split", "ev_model_label",
                "corruption_type", "corruption_severity"]:
            if sample_key in sample_log:
                record[sample_key] = sample_log[sample_key]
        for branch in ("fused", "bev", "img"):
            for key in ("score", "margin", "entropy"):
                value = number(f"{branch}_{key}")
                if value is not None:
                    record[f"{branch}_{key}"] = value
            for key in ("local_class", "global_class"):
                value = integer(f"{branch}_{key}")
                if value >= 0:
                    record[f"{branch}_{key}"] = value
        for name in (
                "img_fused_class_disagree_flag",
                "img_fused_highconf_disagree_flag",
                "img_bev_class_disagree_flag",
                "img_bev_highconf_disagree_flag",
                "fused_bev_class_disagree_flag",
                "fused_bev_highconf_disagree_flag",
                "img_low_score020_flag",
                "img_low_score030_flag",
                "img_low_margin005_flag",
                "img_low_margin010_flag",
                "img_high_entropy080_flag"):
            value = number(name, default=None)
            if value is not None:
                record[name[:-5]] = bool(float(value) > 0.5)
        return record

    def _iter_query_semantic_records(self, sample_log, edl_info):
        if not self.query_semantic or not isinstance(edl_info, dict):
            return
        count = self._qsem_count(edl_info)
        if count <= 0:
            return
        ranked = [
            (idx, self._qsem_score_for_rank(edl_info, idx))
            for idx in range(count)
        ]
        score_thr = float(self.query_semantic_score_thr)
        ranked = [
            (idx, score) for idx, score in ranked
            if np.isfinite(score) and score >= score_thr
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        topk = int(self.query_semantic_topk)
        if topk > 0:
            ranked = ranked[:topk]
        for rank, (idx, _) in enumerate(ranked):
            record = self._build_query_semantic_record(
                sample_log, edl_info, idx, rank)
            record["score_rank"] = int(rank)
            yield record
    
    def log_sample(self, sample_token, bboxes_tensor, scores, labels, edl_info, data_info=None):
        """Log predictions for a single sample.
        
        Args:
            sample_token (str): Unique sample identifier.
            bboxes_tensor (Tensor): [N, >=3] box parameters (x, y, z, ...).
            scores (Tensor): [N] prediction scores.
            labels (Tensor): [N] predicted class indices.
            edl_info (dict): Structured uncertainty dict from get_bboxes.
        """
        sample_log = {
            "sample_token": str(sample_token),
            "num_predictions": int(len(scores)),
            "predictions": []
        }
        if self.rank is not None:
            sample_log["ev_rank"] = int(self.rank)
        if self.world_size is not None:
            sample_log["ev_world_size"] = int(self.world_size)
        for key, value in self.sample_context.items():
            if value is not None:
                sample_log[key] = value

        if isinstance(data_info, dict):
            for src_key, dst_key in [
                ('scene_token', 'scene_token'),
                ('corruption_type', 'corruption_type'),
                ('corruption_severity', 'corruption_severity'),
            ]:
                if src_key in data_info:
                    sample_log[dst_key] = data_info[src_key]

        branch_candidate_count = (
            self._candidate_count(edl_info)
            if self.branch_candidates and isinstance(edl_info, dict) else 0
        )
        branch_candidate_valid_count = 0
        if branch_candidate_count > 0:
            for candidate_idx in range(branch_candidate_count):
                if self._is_candidate_valid(edl_info, candidate_idx):
                    branch_candidate_valid_count += 1
        if self.branch_candidates:
            sample_log["scene_branch_candidate_enabled"] = 1.0
            sample_log["scene_branch_candidate_count"] = int(branch_candidate_count)
            sample_log["scene_branch_candidate_valid_count"] = int(branch_candidate_valid_count)
            sample_log["scene_branch_candidate_valid_ratio"] = (
                float(branch_candidate_valid_count) / float(branch_candidate_count)
                if branch_candidate_count > 0 else 0.0
            )
            for src_key, dst_key in [
                    ("branch_candidate_limit", "scene_branch_candidate_limit"),
                    ("branch_candidate_limit_per_branch", "scene_branch_candidate_limit_per_branch"),
                    ("branch_candidate_total_available", "scene_branch_candidate_total_available"),
                    ("branch_candidate_score_threshold", "scene_branch_candidate_score_threshold")]:
                if src_key in edl_info:
                    sample_log[dst_key] = self._to_float(edl_info[src_key])

        if 'per_modality_unc' in edl_info:
            sample_log["scene_avg_unc"] = edl_info['per_modality_unc']
        if 'modality_weights' in edl_info:
            sample_log["scene_avg_weights"] = edl_info['modality_weights']
        if 'fused_weight' in edl_info:
            sample_log["scene_fused_weight"] = edl_info['fused_weight']
        if 'selector_entropy' in edl_info:
            sample_log["scene_selector_entropy"] = edl_info['selector_entropy']
        if 'quality_top1_ratio' in edl_info:
            sample_log["scene_quality_top1_ratio"] = edl_info['quality_top1_ratio']
        if 'distance_prior_top1_ratio' in edl_info:
            sample_log["scene_distance_prior_top1_ratio"] = edl_info['distance_prior_top1_ratio']
        if 'residual_override_ratio' in edl_info:
            sample_log["scene_residual_override_ratio"] = edl_info['residual_override_ratio']
        if 'alpha_res_mean' in edl_info:
            sample_log["scene_alpha_res_mean"] = edl_info['alpha_res_mean']
        if 'second_opinion_ratio' in edl_info:
            sample_log["scene_second_opinion_ratio"] = edl_info['second_opinion_ratio']
        if 'second_opinion_teacher_match_ratio' in edl_info:
            sample_log["scene_second_opinion_teacher_match_ratio"] = edl_info['second_opinion_teacher_match_ratio']
        if 'uncertainty_tail_query_ratio' in edl_info:
            sample_log["scene_uncertainty_tail_query_ratio"] = edl_info['uncertainty_tail_query_ratio']
        if 'absolute_unc_summary' in edl_info:
            sample_log["scene_absolute_unc_summary"] = edl_info['absolute_unc_summary']
        if 'disagreement_summary' in edl_info:
            sample_log["scene_disagreement_summary"] = edl_info['disagreement_summary']
        if 'score_component_stats' in edl_info:
            sample_log["scene_score_component_stats"] = edl_info['score_component_stats']
        if 'distance_bin_abnormality_stats' in edl_info:
            sample_log["scene_distance_bin_abnormality_stats"] = edl_info['distance_bin_abnormality_stats']
        has_v6_schema = any(
            key in edl_info
            for key in (self._v6_expected_scene_keys + self._v6_expected_query_keys)
        )
        if has_v6_schema:
            self._v6_schema_samples += 1
            for key in self._v6_expected_scene_keys:
                if key not in edl_info:
                    self._note_v6_missing_field(key)
            for key in self._v6_expected_query_keys:
                if key not in edl_info:
                    self._note_v6_missing_field(key)
        has_v7_schema = any(
            key in edl_info
            for key in (self._v7_expected_scene_keys + self._v7_expected_query_keys)
        )
        if has_v7_schema:
            self._v7_schema_samples += 1
            for key in self._v7_expected_scene_keys:
                if key not in edl_info:
                    self._note_v7_missing_field(key)
            for key in self._v7_expected_query_keys:
                if key not in edl_info:
                    self._note_v7_missing_field(key)
        has_v8_schema = any(
            key in edl_info
            for key in (self._v8_expected_scene_keys + self._v8_expected_query_keys)
        )
        if has_v8_schema:
            self._v8_schema_samples += 1
            for key in self._v8_expected_scene_keys:
                if key not in edl_info:
                    self._note_v8_missing_field(key)
            for key in self._v8_expected_query_keys:
                if key not in edl_info:
                    self._note_v8_missing_field(key)
        for src_key, dst_key in [
            ('v5_switch_ratio', 'scene_v5_switch_ratio'),
            ('v5_keep_fused_ratio', 'scene_v5_keep_fused_ratio'),
            ('v5_switch_to_bev_ratio', 'scene_v5_switch_to_bev_ratio'),
            ('v5_switch_to_img_ratio', 'scene_v5_switch_to_img_ratio'),
            ('v5_fused_cls_anom_mean', 'scene_v5_fused_cls_anom_mean'),
            ('v5_fused_center_anom_mean', 'scene_v5_fused_center_anom_mean'),
            ('v5_fused_height_anom_mean', 'scene_v5_fused_height_anom_mean'),
            ('v5_fused_combined_anom_mean', 'scene_v5_fused_combined_anom_mean'),
            ('v5_trigger_ratio_over_threshold', 'scene_v5_trigger_ratio_over_threshold'),
            ('v5_bin_coverage_ratio', 'scene_v5_bin_coverage_ratio'),
            ('v5_fallback_class_ratio', 'scene_v5_fallback_class_ratio'),
            ('v5_fallback_global_ratio', 'scene_v5_fallback_global_ratio'),
            ('v5_switch_agreement_with_prior', 'scene_v5_switch_agreement_with_prior'),
        ]:
            if src_key in edl_info:
                sample_log[dst_key] = self._to_float(edl_info[src_key])
        if 'v5_distance_prior_candidate_ratio' in edl_info:
            sample_log["scene_v5_distance_prior_candidate_ratio"] = edl_info['v5_distance_prior_candidate_ratio']
        if 'v5_distance_prior_switched_ratio' in edl_info:
            sample_log["scene_v5_distance_prior_switched_ratio"] = edl_info['v5_distance_prior_switched_ratio']
        if 'v5_head_shift' in edl_info:
            sample_log["scene_v5_head_shift"] = edl_info['v5_head_shift']
        for src_key, dst_key in [
            ('v6_switch_ratio', 'scene_v6_switch_ratio'),
            ('v6_gate_anomaly_mean', 'scene_v6_gate_anomaly_mean'),
            ('v6_gate_pass_anomaly_ratio', 'scene_v6_gate_pass_anomaly_ratio'),
            ('v6_gate_pass_candidate_prob_ratio', 'scene_v6_gate_pass_candidate_prob_ratio'),
            ('v6_gate_pass_candidate_margin_ratio', 'scene_v6_gate_pass_candidate_margin_ratio'),
            ('v6_gate_pass_final_ratio', 'scene_v6_gate_pass_final_ratio'),
        ]:
            if src_key in edl_info:
                sample_log[dst_key] = self._to_float(edl_info[src_key])
        for src_key, dst_key in [
            ('v6_candidate_modality_ratio', 'scene_v6_candidate_modality_ratio'),
            ('v6_center_source_ratio', 'scene_v6_center_source_ratio'),
            ('v6_height_source_ratio', 'scene_v6_height_source_ratio'),
            ('v6_reference_fallback_ratio', 'scene_v6_reference_fallback_ratio'),
        ]:
            if src_key in edl_info:
                sample_log[dst_key] = edl_info[src_key]
        for src_key, dst_key in [
            ('v7_switch_ratio', 'scene_v7_switch_ratio'),
            ('v7_damage_state_ratio', 'scene_v7_damage_state_ratio'),
            ('v7_center_source_ratio', 'scene_v7_center_source_ratio'),
            ('v7_bev_center_anom_mean', 'scene_v7_bev_center_anom_mean'),
            ('v7_img_center_anom_mean', 'scene_v7_img_center_anom_mean'),
            ('v7_fused_center_anom_mean', 'scene_v7_fused_center_anom_mean'),
        ]:
            if src_key in edl_info:
                sample_log[dst_key] = (
                    self._to_float(edl_info[src_key])
                    if not isinstance(edl_info[src_key], dict) else edl_info[src_key]
                )
        for src_key, dst_key in [
            ('v8_fused_keep_mean', 'scene_v8_fused_keep_mean'),
            ('v8_center_fused_keep_mean', 'scene_v8_center_fused_keep_mean'),
            ('v8_damage_gate_mean', 'scene_v8_damage_gate_mean'),
            ('v8_non_fused_weight_ratio', 'scene_v8_non_fused_weight_ratio'),
            ('v8_damage_state_ratio', 'scene_v8_damage_state_ratio'),
            ('v8_selected_source_ratio', 'scene_v8_selected_source_ratio'),
            ('v8_high_damage_query_ratio', 'scene_v8_high_damage_query_ratio'),
            ('v8_center_route_allowed_ratio', 'scene_v8_center_route_allowed_ratio'),
            ('v8_bev_damage_mean', 'scene_v8_bev_damage_mean'),
            ('v8_img_damage_mean', 'scene_v8_img_damage_mean'),
            ('v8_dual_damage_mean', 'scene_v8_dual_damage_mean'),
            ('v8_context_bev_damage_mean', 'scene_v8_context_bev_damage_mean'),
            ('v8_context_img_damage_mean', 'scene_v8_context_img_damage_mean'),
            ('v8_a4_anchor_ready', 'scene_v8_a4_anchor_ready'),
            ('v8_a4_bev_damage_z_mean', 'scene_v8_a4_bev_damage_z_mean'),
            ('v8_a4_img_damage_z_mean', 'scene_v8_a4_img_damage_z_mean'),
            ('v8_a4_cls_unc_z_bev_mean', 'scene_v8_a4_cls_unc_z_bev_mean'),
            ('v8_a4_cls_unc_z_img_mean', 'scene_v8_a4_cls_unc_z_img_mean'),
            ('v8_a4_route_score_bev_mean', 'scene_v8_a4_route_score_bev_mean'),
            ('v8_a4_route_score_img_mean', 'scene_v8_a4_route_score_img_mean'),
            ('v8_a4_clean_anchor_update_count', 'scene_v8_a4_clean_anchor_update_count'),
            ('v8_bev_reliability_mean', 'scene_v8_bev_reliability_mean'),
            ('v8_img_reliability_mean', 'scene_v8_img_reliability_mean'),
            ('v8_reliability_margin_mean', 'scene_v8_reliability_margin_mean'),
            ('v8_soft_weight_ratio', 'scene_v8_soft_weight_ratio'),
            ('v8_reliability_state_ratio', 'scene_v8_reliability_state_ratio'),
            ('v8_fused_keep_by_state', 'scene_v8_fused_keep_by_state'),
            ('v8_non_fused_entropy_mean', 'scene_v8_non_fused_entropy_mean'),
            ('v8_output_delta_cls_mean', 'scene_v8_output_delta_cls_mean'),
            ('v8_output_delta_center_mean', 'scene_v8_output_delta_center_mean'),
            ('v8_output_delta_height_mean', 'scene_v8_output_delta_height_mean'),
            ('v9_a_bev_mean', 'scene_v9_a_bev_mean'),
            ('v9_a_img_mean', 'scene_v9_a_img_mean'),
            ('v9_a_bev_above_tau_mean', 'scene_v9_a_bev_above_tau_mean'),
            ('v9_a_img_above_tau_mean', 'scene_v9_a_img_above_tau_mean'),
            ('v9_clean_like_mask_mean', 'scene_v9_clean_like_mask_mean'),
            ('v9_anchor_distance_bev_mean', 'scene_v9_anchor_distance_bev_mean'),
            ('v9_anchor_distance_img_mean', 'scene_v9_anchor_distance_img_mean'),
            ('v9_tail_prob_bev_mean', 'scene_v9_tail_prob_bev_mean'),
            ('v9_tail_prob_img_mean', 'scene_v9_tail_prob_img_mean'),
            ('v9_tail_excess_bev_mean', 'scene_v9_tail_excess_bev_mean'),
            ('v9_tail_excess_img_mean', 'scene_v9_tail_excess_img_mean'),
            ('v9_cvar_bev_mean', 'scene_v9_cvar_bev_mean'),
            ('v9_cvar_img_mean', 'scene_v9_cvar_img_mean'),
            ('v9_q95_shift_bev_mean', 'scene_v9_q95_shift_bev_mean'),
            ('v9_q95_shift_img_mean', 'scene_v9_q95_shift_img_mean'),
            ('v9_route_score_bev_mean', 'scene_v9_route_score_bev_mean'),
            ('v9_route_score_img_mean', 'scene_v9_route_score_img_mean'),
            ('v9_velocity_l1_mean', 'scene_v9_velocity_l1_mean'),
            ('v9_state_clean_like_mean', 'scene_v9_state_clean_like_mean'),
            ('v9_state_confirmed_img_mean', 'scene_v9_state_confirmed_img_mean'),
            ('v9_state_confirmed_bev_mean', 'scene_v9_state_confirmed_bev_mean'),
            ('v9_state_confirmed_dual_mean', 'scene_v9_state_confirmed_dual_mean'),
            ('v9_state_neutral_mean', 'scene_v9_state_neutral_mean'),
            ('v9_cap_applied_mean', 'scene_v9_cap_applied_mean'),
            ('v9_center_safe_applied_mean', 'scene_v9_center_safe_applied_mean'),
            ('v9_center_safe_skipped_mean', 'scene_v9_center_safe_skipped_mean'),
            ('v9_fused_keep_pre_cap_cls_mean', 'scene_v9_fused_keep_pre_cap_cls_mean'),
            ('v9_fused_keep_pre_cap_center_mean', 'scene_v9_fused_keep_pre_cap_center_mean'),
            ('v9_fused_keep_pre_cap_height_mean', 'scene_v9_fused_keep_pre_cap_height_mean'),
            ('v9_fused_keep_pre_cap_vel_mean', 'scene_v9_fused_keep_pre_cap_vel_mean'),
            ('v9_fused_keep_post_cap_cls_mean', 'scene_v9_fused_keep_post_cap_cls_mean'),
            ('v9_fused_keep_post_cap_center_mean', 'scene_v9_fused_keep_post_cap_center_mean'),
            ('v9_fused_keep_post_cap_height_mean', 'scene_v9_fused_keep_post_cap_height_mean'),
            ('v9_fused_keep_post_cap_vel_mean', 'scene_v9_fused_keep_post_cap_vel_mean'),
            ('v9_fused_keep_final_cls_mean', 'scene_v9_fused_keep_final_cls_mean'),
            ('v9_fused_keep_final_center_mean', 'scene_v9_fused_keep_final_center_mean'),
            ('v9_fused_keep_final_height_mean', 'scene_v9_fused_keep_final_height_mean'),
            ('v9_fused_keep_final_vel_mean', 'scene_v9_fused_keep_final_vel_mean'),
            ('v9_soft_weight_fused_cls_mean', 'scene_v9_soft_weight_fused_cls_mean'),
            ('v9_soft_weight_bev_cls_mean', 'scene_v9_soft_weight_bev_cls_mean'),
            ('v9_soft_weight_img_cls_mean', 'scene_v9_soft_weight_img_cls_mean'),
            ('v9_soft_weight_fused_center_mean', 'scene_v9_soft_weight_fused_center_mean'),
            ('v9_soft_weight_bev_center_mean', 'scene_v9_soft_weight_bev_center_mean'),
            ('v9_soft_weight_img_center_mean', 'scene_v9_soft_weight_img_center_mean'),
            ('v9_soft_weight_fused_height_mean', 'scene_v9_soft_weight_fused_height_mean'),
            ('v9_soft_weight_bev_height_mean', 'scene_v9_soft_weight_bev_height_mean'),
            ('v9_soft_weight_img_height_mean', 'scene_v9_soft_weight_img_height_mean'),
            ('v9_soft_weight_fused_vel_mean', 'scene_v9_soft_weight_fused_vel_mean'),
            ('v9_soft_weight_bev_vel_mean', 'scene_v9_soft_weight_bev_vel_mean'),
            ('v9_soft_weight_img_vel_mean', 'scene_v9_soft_weight_img_vel_mean'),
            ('v9_center_l1_mean', 'scene_v9_center_l1_mean'),
            ('v9_center_l1_p95', 'scene_v9_center_l1_p95'),
            ('v9_fused_keep_unit_mean', 'scene_v9_fused_keep_unit_mean'),
            ('v9_fused_keep_unit_q05', 'scene_v9_fused_keep_unit_q05'),
            ('v9_fused_keep_unit_q50', 'scene_v9_fused_keep_unit_q50'),
            ('v9_fused_keep_unit_q95', 'scene_v9_fused_keep_unit_q95'),
            ('v9_fused_keep_at_min_mean', 'scene_v9_fused_keep_at_min_ratio'),
            ('v9_current_keep_min', 'scene_v9_current_keep_min'),
            ('v9_current_lambda_gate', 'scene_v9_current_lambda_gate'),
            ('v9_current_lambda_route', 'scene_v9_current_lambda_route'),
            ('v9_tau_bev_current', 'scene_v9_tau_bev_current'),
            ('v9_tau_img_current', 'scene_v9_tau_img_current'),
            ('v9_clean_distance_threshold_bev', 'scene_v9_clean_distance_threshold_bev'),
            ('v9_clean_distance_threshold_img', 'scene_v9_clean_distance_threshold_img'),
            ('v9_clean_q95_z_bev', 'scene_v9_clean_q95_z_bev'),
            ('v9_clean_q95_z_img', 'scene_v9_clean_q95_z_img'),
            ('v9_clean_anchor_frozen', 'scene_v9_clean_anchor_frozen'),
            ('v9_motionblur_img_bce_mean', 'scene_v9_motionblur_img_bce_mean'),
            ('v9_motionblur_img_bce_count', 'scene_v9_motionblur_img_bce_count'),
            ('v9_beams_bev_bce_mean', 'scene_v9_beams_bev_bce_mean'),
            ('v9_beams_bev_bce_count', 'scene_v9_beams_bev_bce_count'),
            ('v9_points_bev_bce_mean', 'scene_v9_points_bev_bce_mean'),
            ('v9_points_bev_bce_count', 'scene_v9_points_bev_bce_count'),
            ('v9_dark_img_bce_mean', 'scene_v9_dark_img_bce_mean'),
            ('v9_dark_img_bce_count', 'scene_v9_dark_img_bce_count'),
            ('v9_brightness_img_bce_mean', 'scene_v9_brightness_img_bce_mean'),
            ('v9_brightness_img_bce_count', 'scene_v9_brightness_img_bce_count'),
            ('v9_missingcamera_img_bce_mean', 'scene_v9_missingcamera_img_bce_mean'),
            ('v9_missingcamera_img_bce_count', 'scene_v9_missingcamera_img_bce_count'),
            ('v9_fog_bev_bce_mean', 'scene_v9_fog_bev_bce_mean'),
            ('v9_fog_bev_bce_count', 'scene_v9_fog_bev_bce_count'),
            ('v9_fog_img_bce_mean', 'scene_v9_fog_img_bce_mean'),
            ('v9_fog_img_bce_count', 'scene_v9_fog_img_bce_count'),
            ('qsem_img_score_mean', 'scene_qsem_img_score_mean'),
            ('qsem_img_score_q05', 'scene_qsem_img_score_q05'),
            ('qsem_img_score_q50', 'scene_qsem_img_score_q50'),
            ('qsem_img_score_p95', 'scene_qsem_img_score_p95'),
            ('qsem_img_margin_mean', 'scene_qsem_img_margin_mean'),
            ('qsem_img_margin_q05', 'scene_qsem_img_margin_q05'),
            ('qsem_img_margin_q50', 'scene_qsem_img_margin_q50'),
            ('qsem_img_margin_p95', 'scene_qsem_img_margin_p95'),
            ('qsem_img_entropy_mean', 'scene_qsem_img_entropy_mean'),
            ('qsem_img_entropy_q50', 'scene_qsem_img_entropy_q50'),
            ('qsem_img_entropy_p95', 'scene_qsem_img_entropy_p95'),
            ('qsem_fused_score_mean', 'scene_qsem_fused_score_mean'),
            ('qsem_fused_margin_mean', 'scene_qsem_fused_margin_mean'),
            ('qsem_fused_entropy_mean', 'scene_qsem_fused_entropy_mean'),
            ('qsem_bev_score_mean', 'scene_qsem_bev_score_mean'),
            ('qsem_bev_margin_mean', 'scene_qsem_bev_margin_mean'),
            ('qsem_bev_entropy_mean', 'scene_qsem_bev_entropy_mean'),
            ('qsem_img_low_score020_ratio', 'scene_qsem_img_low_score020_ratio'),
            ('qsem_img_low_score030_ratio', 'scene_qsem_img_low_score030_ratio'),
            ('qsem_img_low_margin005_ratio', 'scene_qsem_img_low_margin005_ratio'),
            ('qsem_img_low_margin010_ratio', 'scene_qsem_img_low_margin010_ratio'),
            ('qsem_img_high_entropy080_ratio', 'scene_qsem_img_high_entropy080_ratio'),
            ('qsem_img_fused_class_disagree_ratio',
             'scene_qsem_img_fused_class_disagree_ratio'),
            ('qsem_img_fused_highconf_disagree_ratio',
             'scene_qsem_img_fused_highconf_disagree_ratio'),
            ('qsem_img_bev_class_disagree_ratio',
             'scene_qsem_img_bev_class_disagree_ratio'),
            ('qsem_img_bev_highconf_disagree_ratio',
             'scene_qsem_img_bev_highconf_disagree_ratio'),
            ('qsem_fused_bev_class_disagree_ratio',
             'scene_qsem_fused_bev_class_disagree_ratio'),
            ('qsem_fused_bev_highconf_disagree_ratio',
             'scene_qsem_fused_bev_highconf_disagree_ratio'),
            ('qsem_img_score_minus_fused_mean',
             'scene_qsem_img_score_minus_fused_mean'),
            ('qsem_img_margin_minus_fused_mean',
             'scene_qsem_img_margin_minus_fused_mean'),
            ('qsem_img_score_minus_bev_mean',
             'scene_qsem_img_score_minus_bev_mean'),
            ('qsem_img_margin_minus_bev_mean',
             'scene_qsem_img_margin_minus_bev_mean'),
            ('qsem_top100_query_count', 'scene_qsem_top100_query_count'),
            ('qsem_top100_max_score_mean',
             'scene_qsem_top100_max_score_mean'),
            ('qsem_top100_img_score_mean',
             'scene_qsem_top100_img_score_mean'),
            ('qsem_top100_img_margin_mean',
             'scene_qsem_top100_img_margin_mean'),
            ('qsem_top100_img_entropy_mean',
             'scene_qsem_top100_img_entropy_mean'),
            ('qsem_top100_img_low_score030_ratio',
             'scene_qsem_top100_img_low_score030_ratio'),
            ('qsem_top100_img_low_margin010_ratio',
             'scene_qsem_top100_img_low_margin010_ratio'),
            ('qsem_top100_img_high_entropy080_ratio',
             'scene_qsem_top100_img_high_entropy080_ratio'),
            ('qsem_top100_img_fused_class_disagree_ratio',
             'scene_qsem_top100_img_fused_class_disagree_ratio'),
            ('qsem_top100_img_fused_highconf_disagree_ratio',
             'scene_qsem_top100_img_fused_highconf_disagree_ratio'),
            ('qsem_top100_img_bev_class_disagree_ratio',
             'scene_qsem_top100_img_bev_class_disagree_ratio'),
            ('qsem_top100_img_bev_highconf_disagree_ratio',
             'scene_qsem_top100_img_bev_highconf_disagree_ratio'),
            ('qsem_top300_query_count', 'scene_qsem_top300_query_count'),
            ('qsem_top300_max_score_mean',
             'scene_qsem_top300_max_score_mean'),
            ('qsem_top300_img_score_mean',
             'scene_qsem_top300_img_score_mean'),
            ('qsem_top300_img_margin_mean',
             'scene_qsem_top300_img_margin_mean'),
            ('qsem_top300_img_entropy_mean',
             'scene_qsem_top300_img_entropy_mean'),
            ('qsem_top300_img_low_score030_ratio',
             'scene_qsem_top300_img_low_score030_ratio'),
            ('qsem_top300_img_low_margin010_ratio',
             'scene_qsem_top300_img_low_margin010_ratio'),
            ('qsem_top300_img_high_entropy080_ratio',
             'scene_qsem_top300_img_high_entropy080_ratio'),
            ('qsem_top300_img_fused_class_disagree_ratio',
             'scene_qsem_top300_img_fused_class_disagree_ratio'),
            ('qsem_top300_img_fused_highconf_disagree_ratio',
             'scene_qsem_top300_img_fused_highconf_disagree_ratio'),
            ('qsem_top300_img_bev_class_disagree_ratio',
             'scene_qsem_top300_img_bev_class_disagree_ratio'),
            ('qsem_top300_img_bev_highconf_disagree_ratio',
             'scene_qsem_top300_img_bev_highconf_disagree_ratio'),
            ('supp_scene_gate_ratio', 'scene_supp_scene_gate_ratio'),
            ('supp_candidate_ratio', 'scene_supp_candidate_ratio'),
            ('supp_img_candidate_ratio', 'scene_supp_img_candidate_ratio'),
            ('supp_bev_candidate_ratio', 'scene_supp_bev_candidate_ratio'),
            ('supp_source_bev_query_ratio', 'scene_supp_source_bev_query_ratio'),
            ('supp_selected_img_ratio', 'scene_supp_selected_img_ratio'),
            ('supp_selected_bev_ratio', 'scene_supp_selected_bev_ratio'),
            ('supp_branch_aware_enabled', 'scene_supp_branch_aware_enabled'),
            ('supp_branch_mapper_enabled', 'scene_supp_branch_mapper_enabled'),
            ('supp_img_utility_candidate_ratio', 'scene_supp_img_utility_candidate_ratio'),
            ('supp_bev_utility_candidate_ratio', 'scene_supp_bev_utility_candidate_ratio'),
            ('supp_img_utility_score_mean', 'scene_supp_img_utility_score_mean'),
            ('supp_bev_utility_score_mean', 'scene_supp_bev_utility_score_mean'),
            ('supp_utility_candidate_ratio', 'scene_supp_utility_candidate_ratio'),
            ('supp_utility_score_mean', 'scene_supp_utility_score_mean'),
            ('supp_utility_threshold', 'scene_supp_utility_threshold'),
            ('supp_img_utility_threshold', 'scene_supp_img_utility_threshold'),
            ('supp_bev_utility_threshold', 'scene_supp_bev_utility_threshold'),
            ('supp_fused_slot_utility_score_mean', 'scene_supp_fused_slot_utility_score_mean'),
            ('supp_marginal_enabled', 'scene_supp_marginal_enabled'),
            ('supp_marginal_margin', 'scene_supp_marginal_margin'),
            ('supp_dual_marginal_margin', 'scene_supp_dual_marginal_margin'),
            ('supp_marginal_checked_ratio', 'scene_supp_marginal_checked_ratio'),
            ('supp_marginal_reject_ratio', 'scene_supp_marginal_reject_ratio'),
            ('supp_marginal_pass_ratio', 'scene_supp_marginal_pass_ratio'),
            ('supp_selected_margin_score_mean', 'scene_supp_selected_margin_score_mean'),
            ('supp_added_ratio', 'scene_supp_added_ratio'),
            ('supp_replace_slot_ratio', 'scene_supp_replace_slot_ratio'),
            ('supp_scene_gate_pre_guard_ratio', 'scene_supp_scene_gate_pre_guard_ratio'),
            ('supp_scene_gate_post_guard_ratio', 'scene_supp_scene_gate_post_guard_ratio'),
            ('supp_img_damage_guard_enabled', 'scene_supp_img_damage_guard_enabled'),
            ('supp_img_damage_guard_mode_code', 'scene_supp_img_damage_guard_mode_code'),
            ('supp_img_damage_gate_ratio', 'scene_supp_img_damage_gate_ratio'),
            ('supp_dual_damage_gate_ratio', 'scene_supp_dual_damage_gate_ratio'),
            ('supp_dual_gate_mode_code', 'scene_supp_dual_gate_mode_code'),
            ('supp_scene_mapper_dual_gate_ratio', 'scene_supp_scene_mapper_dual_gate_ratio'),
            ('supp_bev_damage_gate_ratio', 'scene_supp_bev_damage_gate_ratio'),
            ('supp_bev_damage_upper_gate_ratio',
             'scene_supp_bev_damage_upper_gate_ratio'),
            ('supp_fused_damage_gate_ratio', 'scene_supp_fused_damage_gate_ratio'),
            ('supp_bev_fused_dominance_gate_ratio',
             'scene_supp_bev_fused_dominance_gate_ratio'),
            ('supp_bev_fused_normalized_ratio_mean',
             'scene_supp_bev_fused_normalized_ratio_mean'),
            ('supp_dual_abstain_gate_ratio', 'scene_supp_dual_abstain_gate_ratio'),
            ('supp_img_abstain_gate_ratio', 'scene_supp_img_abstain_gate_ratio'),
            ('supp_dual_abstain_score', 'scene_supp_dual_abstain_score'),
            ('supp_img_abstain_score', 'scene_supp_img_abstain_score'),
            ('supp_dual_abstain_threshold', 'scene_supp_dual_abstain_threshold'),
            ('supp_img_abstain_threshold', 'scene_supp_img_abstain_threshold'),
            ('supp_dual_abstain_available', 'scene_supp_dual_abstain_available'),
            ('supp_img_abstain_available', 'scene_supp_img_abstain_available'),
            ('supp_qsem_gate_enabled', 'scene_supp_qsem_gate_enabled'),
            ('supp_qsem_gate_ratio', 'scene_supp_qsem_gate_ratio'),
            ('supp_qsem_disagree_gate_ratio',
             'scene_supp_qsem_disagree_gate_ratio'),
            ('supp_qsem_margin_gate_ratio', 'scene_supp_qsem_margin_gate_ratio'),
            ('supp_qsem_disagree_score', 'scene_supp_qsem_disagree_score'),
            ('supp_qsem_margin_score', 'scene_supp_qsem_margin_score'),
            ('supp_qsem_disagree_threshold',
             'scene_supp_qsem_disagree_threshold'),
            ('supp_qsem_margin_threshold', 'scene_supp_qsem_margin_threshold'),
            ('supp_qsem_disagree_available',
             'scene_supp_qsem_disagree_available'),
            ('supp_qsem_margin_available', 'scene_supp_qsem_margin_available'),
            ('supp_contextheight_gate_enabled',
             'scene_supp_contextheight_gate_enabled'),
            ('supp_contextheight_gate_ratio',
             'scene_supp_contextheight_gate_ratio'),
            ('supp_contextheight_context_gate_ratio',
             'scene_supp_contextheight_context_gate_ratio'),
            ('supp_contextheight_height_gate_ratio',
             'scene_supp_contextheight_height_gate_ratio'),
            ('supp_contextheight_v9_bev_gate_ratio',
             'scene_supp_contextheight_v9_bev_gate_ratio'),
            ('supp_contextheight_context_score',
             'scene_supp_contextheight_context_score'),
            ('supp_contextheight_height_score',
             'scene_supp_contextheight_height_score'),
            ('supp_contextheight_v9_bev_score',
             'scene_supp_contextheight_v9_bev_score'),
            ('supp_contextheight_context_threshold',
             'scene_supp_contextheight_context_threshold'),
            ('supp_contextheight_height_threshold',
             'scene_supp_contextheight_height_threshold'),
            ('supp_contextheight_v9_bev_threshold',
             'scene_supp_contextheight_v9_bev_threshold'),
            ('supp_contextheight_context_available',
             'scene_supp_contextheight_context_available'),
            ('supp_contextheight_height_available',
             'scene_supp_contextheight_height_available'),
            ('supp_contextheight_v9_bev_available',
             'scene_supp_contextheight_v9_bev_available'),
            ('supp_img_damage_threshold', 'scene_supp_img_damage_threshold'),
            ('supp_img_damage_score', 'scene_supp_img_damage_score'),
            ('supp_bev_damage_threshold', 'scene_supp_bev_damage_threshold'),
            ('supp_bev_damage_upper_threshold',
             'scene_supp_bev_damage_upper_threshold'),
            ('supp_fused_damage_threshold', 'scene_supp_fused_damage_threshold'),
            ('supp_img_damage_threshold_available', 'scene_supp_img_damage_threshold_available'),
            ('supp_bev_damage_threshold_available', 'scene_supp_bev_damage_threshold_available'),
            ('supp_bev_damage_upper_threshold_available',
             'scene_supp_bev_damage_upper_threshold_available'),
            ('supp_fused_damage_threshold_available',
             'scene_supp_fused_damage_threshold_available'),
            ('supp_bev_fused_reference_available',
             'scene_supp_bev_fused_reference_available'),
            ('supp_img_damage_threshold_scale',
             'scene_supp_img_damage_threshold_scale'),
            ('supp_fused_damage_threshold_scale',
             'scene_supp_fused_damage_threshold_scale'),
            ('supp_bev_fused_dominance_ratio',
             'scene_supp_bev_fused_dominance_ratio'),
            ('supp_bev_damage_upper_threshold_scale',
             'scene_supp_bev_damage_upper_threshold_scale'),
            ('supp_dual_budget_ratio', 'scene_supp_dual_budget_ratio'),
            ('supp_dual_utility_threshold', 'scene_supp_dual_utility_threshold'),
            ('supp_dual_scene_mapper_threshold', 'scene_supp_dual_scene_mapper_threshold'),
            ('supp_dual_bev_threshold_scale', 'scene_supp_dual_bev_threshold_scale'),
            ('supp_dual_bev_rank_boost', 'scene_supp_dual_bev_rank_boost'),
            ('supp_bev_threshold', 'scene_supp_bev_threshold'),
            ('supp_img_ceiling', 'scene_supp_img_ceiling'),
            ('supp_margin_threshold', 'scene_supp_margin_threshold'),
            ('supp_scene_bev_score', 'scene_supp_scene_bev_score'),
            ('supp_scene_img_score', 'scene_supp_scene_img_score'),
            ('supp_scene_fused_score', 'scene_supp_scene_fused_score'),
            ('supp_scene_aux_score', 'scene_supp_scene_aux_score'),
            ('supp_max_add_queries', 'scene_supp_max_add_queries'),
            ('supp_scene_mapper_score_mean', 'scene_supp_scene_mapper_score_mean'),
            ('supp_scene_mapper_gate_ratio', 'scene_supp_scene_mapper_gate_ratio'),
            ('supp_scene_mapper_configured',
             'scene_supp_scene_mapper_configured'),
            ('supp_scene_mapper_feature_expected_count',
             'scene_supp_scene_mapper_feature_expected_count'),
            ('supp_scene_mapper_feature_available_count',
             'scene_supp_scene_mapper_feature_available_count'),
            ('supp_scene_mapper_feature_finite_count',
             'scene_supp_scene_mapper_feature_finite_count'),
            ('supp_scene_mapper_feature_missing_count',
             'scene_supp_scene_mapper_feature_missing_count'),
            ('supp_scene_mapper_contract_ready',
             'scene_supp_scene_mapper_contract_ready'),
            ('supp_scene_mapper_q95_available',
             'scene_supp_scene_mapper_q95_available'),
            ('supp_scene_mapper_q95_runtime_mean',
             'scene_supp_scene_mapper_q95_runtime_mean'),
            ('supp_dynamic_budget_ratio', 'scene_supp_dynamic_budget_ratio'),
            ('supp_effective_max_add_queries', 'scene_supp_effective_max_add_queries'),
            ('supp_rank_mode_code', 'scene_supp_rank_mode_code'),
            ('supp_img_score_threshold', 'scene_supp_img_score_threshold'),
            ('supp_bev_score_threshold', 'scene_supp_bev_score_threshold'),
            ('supp_nms_center_dist', 'scene_supp_nms_center_dist'),
            ('supp_per_class_cap', 'scene_supp_per_class_cap'),
            ('supp_enabled_runtime', 'scene_supp_enabled_runtime'),
            ('supp_status_code', 'scene_supp_status_code'),
            ('supp_noadd_reason_code', 'scene_supp_noadd_reason_code'),
        ]:
            if src_key in edl_info:
                sample_log[dst_key] = (
                    self._to_float(edl_info[src_key])
                    if not isinstance(edl_info[src_key], dict) else edl_info[src_key]
                )
        for src_key in self._v10_expected_scene_keys:
            if src_key in edl_info:
                sample_log[f"scene_{src_key}"] = self._to_jsonable(edl_info[src_key])
        
        num_preds = len(scores)
        for i in range(num_preds):
            if not (
                self.keep_full_json or
                self._sample_full_fp is not None or
                self.prediction_light or
                self._prediction_full_fp is not None
            ):
                break
            pred = {
                "box_idx": i,
                "label": int(labels[i]),
                "label_name": self.class_names[int(labels[i])] if int(labels[i]) < len(self.class_names) else "unknown",
                "score": float(scores[i].cpu()),
            }
            
            # Box center and distance
            if bboxes_tensor is not None and bboxes_tensor.shape[0] > i:
                center = bboxes_tensor[i, :3].cpu().numpy()
                pred["bbox_center"] = center.tolist()
                pred["distance_to_ego"] = float(np.linalg.norm(center[:2]))
            
            # --- Per-box uncertainty (Stage 1 & 2) ---
            for unc_key in ['center_unc', 'height_unc', 'cls_unc', 'combined_unc']:
                if unc_key in edl_info and edl_info[unc_key] is not None:
                    unc_tensor = edl_info[unc_key]
                    if isinstance(unc_tensor, torch.Tensor) and i < unc_tensor.shape[0]:
                        pred[unc_key] = float(unc_tensor[i].cpu().squeeze())
            
            # --- PME-specific info (Stage 2 only) ---
            if 'per_modality_unc' in edl_info:
                pred["per_modality_unc"] = edl_info['per_modality_unc']
            
            if 'modality_weights' in edl_info:
                pred["modality_weights"] = edl_info['modality_weights']
                pred["ensemble_info"] = {
                    "attn_weights": edl_info['modality_weights']
                }
            if 'selected_modality' in edl_info:
                pred["selected_modality"] = edl_info['selected_modality']
            if 'fused_weight' in edl_info:
                pred["fused_weight"] = edl_info['fused_weight']
            if 'selector_entropy' in edl_info:
                pred["selector_entropy"] = edl_info['selector_entropy']
            if 'quality_top1_ratio' in edl_info:
                pred["quality_top1_ratio"] = edl_info['quality_top1_ratio']
            if 'distance_prior_top1_ratio' in edl_info:
                pred["distance_prior_top1_ratio"] = edl_info['distance_prior_top1_ratio']
            if 'residual_override_ratio' in edl_info:
                pred["residual_override_ratio"] = edl_info['residual_override_ratio']
            if 'alpha_res_mean' in edl_info:
                pred["alpha_res_mean"] = edl_info['alpha_res_mean']
            if 'second_opinion_ratio' in edl_info:
                pred["second_opinion_ratio"] = edl_info['second_opinion_ratio']
            if 'second_opinion_teacher_match_ratio' in edl_info:
                pred["second_opinion_teacher_match_ratio"] = edl_info['second_opinion_teacher_match_ratio']
            if 'uncertainty_tail_query_ratio' in edl_info:
                pred["uncertainty_tail_query_ratio"] = edl_info['uncertainty_tail_query_ratio']
            if 'absolute_unc_summary' in edl_info:
                pred["absolute_unc_summary"] = edl_info['absolute_unc_summary']
            if 'disagreement_summary' in edl_info:
                pred["disagreement_summary"] = edl_info['disagreement_summary']
            if 'score_component_stats' in edl_info:
                pred["score_component_stats"] = edl_info['score_component_stats']
            if 'distance_bin_abnormality_stats' in edl_info:
                pred["distance_bin_abnormality_stats"] = edl_info['distance_bin_abnormality_stats']

            if 'v5_fused_anomaly_score' in edl_info and isinstance(edl_info['v5_fused_anomaly_score'], torch.Tensor):
                unc_tensor = edl_info['v5_fused_anomaly_score']
                if i < unc_tensor.shape[0]:
                    pred["fused_anomaly_score"] = float(unc_tensor[i].cpu().squeeze())
            component_map = {}
            for src_key, dst_key in [
                ('v5_fused_cls_anom', 'cls'),
                ('v5_fused_center_anom', 'center'),
                ('v5_fused_height_anom', 'height'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor) and i < edl_info[src_key].shape[0]:
                    component_map[dst_key] = float(edl_info[src_key][i].cpu().squeeze())
            if component_map:
                pred["fused_anomaly_components"] = component_map
            for src_key, dst_key in [
                ('v5_fused_cls_unc', 'v5_fused_cls_unc'),
                ('v5_fused_center_unc', 'v5_fused_center_unc'),
                ('v5_fused_height_unc', 'v5_fused_height_unc'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor) and i < edl_info[src_key].shape[0]:
                    pred[dst_key] = float(edl_info[src_key][i].cpu().squeeze())
            if 'v5_switch_triggered' in edl_info and isinstance(edl_info['v5_switch_triggered'], torch.Tensor):
                tensor = edl_info['v5_switch_triggered']
                if i < tensor.shape[0]:
                    pred["v5_switch_triggered"] = bool(float(tensor[i].cpu().squeeze()) > 0.5)
            if 'v5_selected_modality_idx' in edl_info and isinstance(edl_info['v5_selected_modality_idx'], torch.Tensor):
                tensor = edl_info['v5_selected_modality_idx']
                if i < tensor.shape[0]:
                    modality_idx = int(tensor[i].cpu().item())
                    pred["v5_selected_modality"] = V5_MODALITY_MAP.get(modality_idx, "unknown")
                    pred["selected_modality"] = pred["v5_selected_modality"]
            if 'v5_candidate_modality_idx' in edl_info and isinstance(edl_info['v5_candidate_modality_idx'], torch.Tensor):
                tensor = edl_info['v5_candidate_modality_idx']
                if i < tensor.shape[0]:
                    modality_idx = int(tensor[i].cpu().item())
                    pred["v5_candidate_modality"] = V5_MODALITY_MAP.get(modality_idx, "unknown")
            if 'v5_reference_distance_bin' in edl_info and isinstance(edl_info['v5_reference_distance_bin'], torch.Tensor):
                tensor = edl_info['v5_reference_distance_bin']
                if i < tensor.shape[0]:
                    pred["v5_reference_bin"] = int(tensor[i].cpu().item())
            if 'v5_reference_fallback_level' in edl_info and isinstance(edl_info['v5_reference_fallback_level'], torch.Tensor):
                tensor = edl_info['v5_reference_fallback_level']
                if i < tensor.shape[0]:
                    level = int(tensor[i].cpu().item())
                    pred["v5_reference_fallback_level"] = level
                    pred["v5_reference_fallback_name"] = V5_FALLBACK_MAP.get(level, "unknown")
            if 'v5_reference_pred_class' in edl_info and isinstance(edl_info['v5_reference_pred_class'], torch.Tensor):
                tensor = edl_info['v5_reference_pred_class']
                if i < tensor.shape[0]:
                    pred["v5_reference_pred_class"] = int(tensor[i].cpu().item())
            if 'v6_switch_triggered' in edl_info and isinstance(edl_info['v6_switch_triggered'], torch.Tensor):
                tensor = edl_info['v6_switch_triggered']
                if i < tensor.shape[0]:
                    pred["v6_switch_triggered"] = bool(float(tensor[i].cpu().squeeze()) > 0.5)
            if 'v6_candidate_modality_idx' in edl_info and isinstance(edl_info['v6_candidate_modality_idx'], torch.Tensor):
                tensor = edl_info['v6_candidate_modality_idx']
                if i < tensor.shape[0]:
                    modality_idx = int(tensor[i].cpu().item())
                    pred["v6_candidate_modality"] = V5_MODALITY_MAP.get(modality_idx, "unknown")
            if 'v6_gate_anomaly_score' in edl_info and isinstance(edl_info['v6_gate_anomaly_score'], torch.Tensor):
                tensor = edl_info['v6_gate_anomaly_score']
                if i < tensor.shape[0]:
                    pred["v6_gate_anomaly_score"] = float(tensor[i].cpu().squeeze())
            if 'v6_gate_passed_confidence' in edl_info and isinstance(edl_info['v6_gate_passed_confidence'], torch.Tensor):
                tensor = edl_info['v6_gate_passed_confidence']
                if i < tensor.shape[0]:
                    pred["v6_gate_passed_confidence"] = float(tensor[i].cpu().squeeze())
            if 'v6_gate_passed_margin' in edl_info and isinstance(edl_info['v6_gate_passed_margin'], torch.Tensor):
                tensor = edl_info['v6_gate_passed_margin']
                if i < tensor.shape[0]:
                    pred["v6_gate_passed_margin"] = float(tensor[i].cpu().squeeze())
            for src_key, dst_key in [
                ('v6_gate_pass_anomaly', 'v6_gate_pass_anomaly'),
                ('v6_gate_pass_candidate_prob', 'v6_gate_pass_candidate_prob'),
                ('v6_gate_pass_candidate_margin', 'v6_gate_pass_candidate_margin'),
                ('v6_gate_pass_final', 'v6_gate_pass_final'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        pred[dst_key] = bool(float(tensor[i].cpu().squeeze()) > 0.5)
            if 'v6_reference_distance_bin' in edl_info and isinstance(edl_info['v6_reference_distance_bin'], torch.Tensor):
                tensor = edl_info['v6_reference_distance_bin']
                if i < tensor.shape[0]:
                    pred["v6_reference_bin"] = int(tensor[i].cpu().item())
            if 'v6_reference_fallback_level' in edl_info and isinstance(edl_info['v6_reference_fallback_level'], torch.Tensor):
                tensor = edl_info['v6_reference_fallback_level']
                if i < tensor.shape[0]:
                    level = int(tensor[i].cpu().item())
                    pred["v6_reference_fallback_level"] = level
                    pred["v6_reference_fallback_name"] = V5_FALLBACK_MAP.get(level, "unknown")
            if 'v6_reference_pred_class' in edl_info and isinstance(edl_info['v6_reference_pred_class'], torch.Tensor):
                tensor = edl_info['v6_reference_pred_class']
                if i < tensor.shape[0]:
                    pred["v6_reference_pred_class"] = int(tensor[i].cpu().item())
            if 'v6_baseline_selected_modality_idx' in edl_info and isinstance(edl_info['v6_baseline_selected_modality_idx'], torch.Tensor):
                tensor = edl_info['v6_baseline_selected_modality_idx']
                if i < tensor.shape[0]:
                    modality_idx = int(tensor[i].cpu().item())
                    pred["v6_baseline_selected_modality"] = V5_MODALITY_MAP.get(modality_idx, "unknown")
            for src_key, dst_key in [
                ('v6_center_source_idx', 'v6_center_source'),
                ('v6_height_source_idx', 'v6_height_source'),
                ('v6_cls_source_idx', 'v6_cls_source'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        source_idx = int(tensor[i].cpu().item())
                        pred[dst_key] = V6_SOURCE_MAP.get(source_idx, "unknown")
            if 'v6_non_fused_selector_probs' in edl_info and isinstance(edl_info['v6_non_fused_selector_probs'], torch.Tensor):
                tensor = edl_info['v6_non_fused_selector_probs']
                if i < tensor.shape[0] and tensor.shape[-1] >= 2:
                    probs = tensor[i].cpu().tolist()
                    pred["v6_non_fused_selector_probs"] = {
                        "bev": float(probs[0]),
                        "img": float(probs[1]),
                    }
            if 'v7_switch_triggered' in edl_info and isinstance(edl_info['v7_switch_triggered'], torch.Tensor):
                tensor = edl_info['v7_switch_triggered']
                if i < tensor.shape[0]:
                    pred["v7_switch_triggered"] = bool(float(tensor[i].cpu().squeeze()) > 0.5)
            if 'v7_damage_state_idx' in edl_info and isinstance(edl_info['v7_damage_state_idx'], torch.Tensor):
                tensor = edl_info['v7_damage_state_idx']
                if i < tensor.shape[0]:
                    pred["v7_damage_state"] = V7_DAMAGE_STATE_MAP.get(int(tensor[i].cpu().item()), "unknown")
            if 'v7_center_source_idx' in edl_info and isinstance(edl_info['v7_center_source_idx'], torch.Tensor):
                tensor = edl_info['v7_center_source_idx']
                if i < tensor.shape[0]:
                    pred["v7_center_source"] = V7_CENTER_SOURCE_MAP.get(int(tensor[i].cpu().item()), "unknown")
            for src_key, dst_key in [
                ('v7_bev_center_anom', 'v7_bev_center_anom'),
                ('v7_img_center_anom', 'v7_img_center_anom'),
                ('v7_fused_center_anom', 'v7_fused_center_anom'),
                ('v7_img_fallback_prob', 'v7_img_fallback_prob'),
                ('v7_img_fallback_margin', 'v7_img_fallback_margin'),
                ('v7_bev_fallback_prob', 'v7_bev_fallback_prob'),
                ('v7_bev_fallback_margin', 'v7_bev_fallback_margin'),
                ('v7_ref_fused_center_unc', 'v7_ref_fused_center_unc'),
                ('v7_ref_bev_center_unc', 'v7_ref_bev_center_unc'),
                ('v7_ref_img_center_unc', 'v7_ref_img_center_unc'),
                ('v7_ref_fused_distance', 'v7_ref_fused_distance'),
                ('v7_ref_bev_distance', 'v7_ref_bev_distance'),
                ('v7_ref_img_distance', 'v7_ref_img_distance'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        pred[dst_key] = float(tensor[i].cpu().squeeze())
            if 'v7_reference_distance_bin' in edl_info and isinstance(edl_info['v7_reference_distance_bin'], torch.Tensor):
                tensor = edl_info['v7_reference_distance_bin']
                if i < tensor.shape[0]:
                    pred["v7_reference_bin"] = int(tensor[i].cpu().item())
            for src_key, dst_key in [
                ('v7_pred_class', 'v7_pred_class'),
                ('v7_ref_pred_class', 'v7_ref_pred_class'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        pred[dst_key] = int(tensor[i].cpu().item())
            for src_key, dst_key in [
                ('v8_fused_keep', 'v8_fused_keep'),
                ('v8_center_fused_keep', 'v8_center_fused_keep'),
                ('v8_damage_gate', 'v8_damage_gate'),
                ('v8_bev_damage', 'v8_bev_damage'),
                ('v8_img_damage', 'v8_img_damage'),
                ('v8_dual_damage', 'v8_dual_damage'),
                ('v8_context_bev_damage', 'v8_context_bev_damage'),
                ('v8_context_img_damage', 'v8_context_img_damage'),
                ('v8_a4_bev_damage_z', 'v8_a4_bev_damage_z'),
                ('v8_a4_img_damage_z', 'v8_a4_img_damage_z'),
                ('v8_a4_cls_unc_z_bev', 'v8_a4_cls_unc_z_bev'),
                ('v8_a4_cls_unc_z_img', 'v8_a4_cls_unc_z_img'),
                ('v8_a4_route_score_bev', 'v8_a4_route_score_bev'),
                ('v8_a4_route_score_img', 'v8_a4_route_score_img'),
                ('v8_gscrg_raw_unc_fused', 'v8_gscrg_raw_unc_fused'),
                ('v8_gscrg_raw_unc_bev', 'v8_gscrg_raw_unc_bev'),
                ('v8_gscrg_raw_unc_img', 'v8_gscrg_raw_unc_img'),
                ('v8_bev_reliability', 'v8_bev_reliability'),
                ('v8_img_reliability', 'v8_img_reliability'),
                ('v8_reliability_margin', 'v8_reliability_margin'),
                ('v8_soft_weight_fused', 'v8_soft_weight_fused'),
                ('v8_soft_weight_bev', 'v8_soft_weight_bev'),
                ('v8_soft_weight_img', 'v8_soft_weight_img'),
                ('v8_output_delta_cls', 'v8_output_delta_cls'),
                ('v8_output_delta_center', 'v8_output_delta_center'),
                ('v8_output_delta_height', 'v8_output_delta_height'),
                ('v8_center_route_allowed', 'v8_center_route_allowed'),
                ('v9_a_bev', 'v9_a_bev'),
                ('v9_a_img', 'v9_a_img'),
                ('v9_a_bev_above_tau', 'v9_a_bev_above_tau'),
                ('v9_a_img_above_tau', 'v9_a_img_above_tau'),
                ('v9_clean_like_mask', 'v9_clean_like_mask'),
                ('v9_anchor_distance_bev', 'v9_anchor_distance_bev'),
                ('v9_anchor_distance_img', 'v9_anchor_distance_img'),
                ('v9_consistency_bev', 'v9_consistency_bev'),
                ('v9_consistency_img', 'v9_consistency_img'),
                ('v9_tail_prob_bev', 'v9_tail_prob_bev'),
                ('v9_tail_prob_img', 'v9_tail_prob_img'),
                ('v9_tail_excess_bev', 'v9_tail_excess_bev'),
                ('v9_tail_excess_img', 'v9_tail_excess_img'),
                ('v9_cvar_bev', 'v9_cvar_bev'),
                ('v9_cvar_img', 'v9_cvar_img'),
                ('v9_q95_shift_bev', 'v9_q95_shift_bev'),
                ('v9_q95_shift_img', 'v9_q95_shift_img'),
                ('v9_route_score_bev', 'v9_route_score_bev'),
                ('v9_route_score_img', 'v9_route_score_img'),
                ('v9_velocity_l1', 'v9_velocity_l1'),
                ('v9_state_clean_like', 'v9_state_clean_like'),
                ('v9_state_confirmed_img', 'v9_state_confirmed_img'),
                ('v9_state_confirmed_bev', 'v9_state_confirmed_bev'),
                ('v9_state_confirmed_dual', 'v9_state_confirmed_dual'),
                ('v9_state_neutral', 'v9_state_neutral'),
                ('v9_cap_applied', 'v9_cap_applied'),
                ('v9_center_safe_applied', 'v9_center_safe_applied'),
                ('v9_center_safe_skipped', 'v9_center_safe_skipped'),
                ('v9_fused_keep_pre_cap_cls', 'v9_fused_keep_pre_cap_cls'),
                ('v9_fused_keep_pre_cap_center', 'v9_fused_keep_pre_cap_center'),
                ('v9_fused_keep_pre_cap_height', 'v9_fused_keep_pre_cap_height'),
                ('v9_fused_keep_pre_cap_vel', 'v9_fused_keep_pre_cap_vel'),
                ('v9_fused_keep_post_cap_cls', 'v9_fused_keep_post_cap_cls'),
                ('v9_fused_keep_post_cap_center', 'v9_fused_keep_post_cap_center'),
                ('v9_fused_keep_post_cap_height', 'v9_fused_keep_post_cap_height'),
                ('v9_fused_keep_post_cap_vel', 'v9_fused_keep_post_cap_vel'),
                ('v9_fused_keep_final_cls', 'v9_fused_keep_final_cls'),
                ('v9_fused_keep_final_center', 'v9_fused_keep_final_center'),
                ('v9_fused_keep_final_height', 'v9_fused_keep_final_height'),
                ('v9_fused_keep_final_vel', 'v9_fused_keep_final_vel'),
                ('v9_soft_weight_fused_cls', 'v9_soft_weight_fused_cls'),
                ('v9_soft_weight_bev_cls', 'v9_soft_weight_bev_cls'),
                ('v9_soft_weight_img_cls', 'v9_soft_weight_img_cls'),
                ('v9_soft_weight_fused_center', 'v9_soft_weight_fused_center'),
                ('v9_soft_weight_bev_center', 'v9_soft_weight_bev_center'),
                ('v9_soft_weight_img_center', 'v9_soft_weight_img_center'),
                ('v9_soft_weight_fused_height', 'v9_soft_weight_fused_height'),
                ('v9_soft_weight_bev_height', 'v9_soft_weight_bev_height'),
                ('v9_soft_weight_img_height', 'v9_soft_weight_img_height'),
                ('v9_soft_weight_fused_vel', 'v9_soft_weight_fused_vel'),
                ('v9_soft_weight_bev_vel', 'v9_soft_weight_bev_vel'),
                ('v9_soft_weight_img_vel', 'v9_soft_weight_img_vel'),
                ('v9_center_l1', 'v9_center_l1'),
                ('v9_fused_keep_unit', 'v9_fused_keep_unit'),
                ('v9_fused_keep_at_min', 'v9_fused_keep_at_min'),
                ('supp_added', 'supp_added'),
                ('supp_img_score', 'supp_img_score'),
                ('supp_bev_score', 'supp_bev_score'),
                ('supp_branch_score', 'supp_branch_score'),
                ('supp_fused_score', 'supp_fused_score'),
                ('supp_utility_score', 'supp_utility_score'),
                ('supp_fused_slot_utility_score', 'supp_fused_slot_utility_score'),
                ('supp_marginal_utility_score', 'supp_marginal_utility_score'),
                ('supp_source_branch', 'supp_source_branch'),
                ('final_decode_query_idx', 'final_decode_query_idx'),
                ('final_decode_task_idx', 'final_decode_task_idx'),
                ('final_decode_bbox_idx', 'final_decode_bbox_idx'),
                ('final_decode_topk_rank', 'final_decode_topk_rank'),
                ('final_source_alignment_valid', 'final_source_alignment_valid'),
                ('final_source_is_supplement', 'final_source_is_supplement'),
                ('final_source_branch', 'final_source_branch'),
                ('final_supp_img_score', 'final_supp_img_score'),
                ('final_supp_bev_score', 'final_supp_bev_score'),
                ('final_supp_branch_score', 'final_supp_branch_score'),
                ('final_supp_fused_score', 'final_supp_fused_score'),
                ('final_supp_utility_score', 'final_supp_utility_score'),
                ('final_supp_fused_slot_utility_score', 'final_supp_fused_slot_utility_score'),
                ('final_supp_marginal_utility_score', 'final_supp_marginal_utility_score'),
                ('final_supp_candidate_query_idx', 'final_supp_candidate_query_idx'),
                ('final_supp_img_class', 'final_supp_img_class'),
                ('final_supp_branch_class', 'final_supp_branch_class'),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        pred[dst_key] = float(tensor[i].cpu().squeeze())
            if 'supp_img_class' in edl_info and isinstance(edl_info['supp_img_class'], torch.Tensor):
                tensor = edl_info['supp_img_class']
                if i < tensor.shape[0]:
                    pred['supp_img_class'] = int(tensor[i].cpu().squeeze())
            if 'supp_branch_class' in edl_info and isinstance(edl_info['supp_branch_class'], torch.Tensor):
                tensor = edl_info['supp_branch_class']
                if i < tensor.shape[0]:
                    pred['supp_branch_class'] = int(tensor[i].cpu().squeeze())
            for int_key in (
                    'final_decode_query_idx',
                    'final_decode_task_idx',
                    'final_decode_bbox_idx',
                    'final_decode_topk_rank',
                    'final_source_branch',
                    'final_supp_candidate_query_idx',
                    'final_supp_img_class',
                    'final_supp_branch_class'):
                if int_key in pred and pred[int_key] is not None:
                    pred[int_key] = int(pred[int_key])
            for src_key in self._v10_expected_query_keys:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        pred[src_key] = float(tensor[i].cpu().squeeze())
            if 'v9_state_classification' in edl_info and isinstance(edl_info['v9_state_classification'], torch.Tensor):
                tensor = edl_info['v9_state_classification']
                if i < tensor.shape[0]:
                    pred['v9_state_classification'] = int(tensor[i].cpu().squeeze())
            if 'v8_non_fused_probs' in edl_info and isinstance(edl_info['v8_non_fused_probs'], torch.Tensor):
                tensor = edl_info['v8_non_fused_probs']
                if i < tensor.shape[0] and tensor.shape[-1] >= 2:
                    probs = tensor[i].cpu().tolist()
                    pred["v8_non_fused_probs"] = {
                        "bev": float(probs[0]),
                        "img": float(probs[1]),
                    }
            for src_key, dst_key, mapping in [
                ('v8_non_fused_choice', 'v8_non_fused_choice', V8_SOURCE_MAP),
                ('v8_center_source', 'v8_center_source', V8_SOURCE_MAP),
                ('v8_gate_reason', 'v8_gate_reason', V8_GATE_REASON_MAP),
                ('v8_damage_state', 'v8_damage_state', V8_DAMAGE_STATE_MAP),
                ('v8_reliability_state', 'v8_reliability_state', V8_RELIABILITY_STATE_MAP),
            ]:
                if src_key in edl_info and isinstance(edl_info[src_key], torch.Tensor):
                    tensor = edl_info[src_key]
                    if i < tensor.shape[0]:
                        pred[dst_key] = mapping.get(int(tensor[i].cpu().item()), "unknown")
            if 'v8_high_damage_query' in edl_info and isinstance(edl_info['v8_high_damage_query'], torch.Tensor):
                tensor = edl_info['v8_high_damage_query']
                if i < tensor.shape[0]:
                    pred["v8_high_damage_query"] = bool(float(tensor[i].cpu().squeeze()) > 0.5)
            if "corruption_type" in sample_log:
                pred["v8_corruption_type"] = sample_log["corruption_type"]
            if "corruption_severity" in sample_log:
                pred["v8_corruption_severity"] = sample_log["corruption_severity"]
            sample_log["predictions"].append(pred)
        
        if self.keep_full_json:
            self.logs.append(sample_log)
        if self._sample_full_fp is not None:
            self._write_jsonl(self._sample_full_fp, sample_log)
        self._write_jsonl(self._scene_summary_fp, self._build_scene_summary_record(sample_log))
        branch_candidate_rows = 0
        if self.branch_candidates and self._branch_candidates_fp is not None:
            for candidate_record in self._iter_branch_candidate_records(sample_log, edl_info):
                self._write_jsonl(self._branch_candidates_fp, candidate_record)
                branch_candidate_rows += 1
        query_semantic_rows = 0
        if self.query_semantic and self._query_semantic_fp is not None:
            for semantic_record in self._iter_query_semantic_records(sample_log, edl_info):
                self._write_jsonl(self._query_semantic_fp, semantic_record)
                query_semantic_rows += 1
        if self.prediction_light or self._prediction_full_fp is not None:
            for pred in sample_log["predictions"]:
                if self.prediction_light:
                    self._write_jsonl(
                        self._prediction_light_fp,
                        self._build_prediction_light_record(sample_log, pred))
                if self._prediction_full_fp is not None:
                    self._write_jsonl(
                        self._prediction_full_fp,
                        self._build_prediction_full_record(sample_log, pred))
        self._sidecar_counts["num_samples"] += 1
        self._sidecar_counts["num_predictions"] += int(num_preds)
        self._sidecar_counts["num_branch_candidates"] += int(branch_candidate_rows)
        self._sidecar_counts["num_query_semantic"] += int(query_semantic_rows)
        self._flush_streams_if_needed()

    def save(self, filename="edl_uncertainty_log.json"):
        """Save collected logs to JSON file."""
        path = os.path.join(self.output_dir, filename)
        generated_from = filename
        if self.keep_full_json:
            with open(path, 'w') as f:
                json.dump(self._to_jsonable(self.logs), f, indent=2, ensure_ascii=False)
        else:
            path = None
            generated_from = self.sample_full_filename or self.scene_summary_filename
        self._scene_summary_fp.close()
        if self._prediction_light_fp is not None:
            self._prediction_light_fp.close()
        if self._branch_candidates_fp is not None:
            self._branch_candidates_fp.close()
        if self._query_semantic_fp is not None:
            self._query_semantic_fp.close()
        if self._sample_full_fp is not None:
            self._sample_full_fp.close()
        if self._prediction_full_fp is not None:
            self._prediction_full_fp.close()
        with open(self._sidecar_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "generated_from": generated_from,
                "prefix": self.prefix,
                "rank": self.rank,
                "world_size": self.world_size,
                "compression_requested": self.compression_requested,
                "compression": self.compression,
                "keep_full_json": self.keep_full_json,
                "stream_full": self.stream_full,
                "prediction_full_stream": self.prediction_full_stream,
                "prediction_light": self.prediction_light,
                "branch_candidates": self.branch_candidates,
                "query_semantic": self.query_semantic,
                "branch_candidate_topk": self.branch_candidate_topk,
                "branch_candidate_topk_per_branch": self.branch_candidate_topk,
                "branch_candidate_score_thr": self.branch_candidate_score_thr,
                "branch_candidate_branches": self.branch_candidate_branches,
                "query_semantic_topk": self.query_semantic_topk,
                "query_semantic_score_thr": self.query_semantic_score_thr,
                "num_samples": self._sidecar_counts["num_samples"],
                "num_predictions": self._sidecar_counts["num_predictions"],
                "num_branch_candidates": self._sidecar_counts["num_branch_candidates"],
                "num_query_semantic": self._sidecar_counts["num_query_semantic"],
                "scene_summary_path": self.scene_summary_filename,
                "prediction_light_path": self.prediction_light_filename,
                "branch_candidates_path": self.branch_candidates_filename,
                "query_semantic_path": self.query_semantic_filename,
                "sample_full_path": self.sample_full_filename,
                "prediction_full_path": self.prediction_full_filename,
                "fields": self._sidecar_fields,
                "sample_context": self.sample_context,
                "extra_meta": self.extra_meta,
                "diagnostics": {
                    "v6_schema_samples": self._v6_schema_samples,
                    "v6_missing_field_counts": self._v6_missing_field_counts,
                    "v7_schema_samples": self._v7_schema_samples,
                    "v7_missing_field_counts": self._v7_missing_field_counts,
                    "v8_schema_samples": self._v8_schema_samples,
                    "v8_missing_field_counts": self._v8_missing_field_counts,
                },
            }, f, indent=2, ensure_ascii=False)
        if path:
            print(f"\n[EDL Logger] Saved uncertainty log: {path}")
        else:
            print("\n[EDL Logger] Full JSON array disabled; stream shards are authoritative.")
            if self._sample_full_path:
                print(f"[EDL Logger] Saved full sample stream: {self._sample_full_path}")
            if self._prediction_full_path:
                print(f"[EDL Logger] Saved full prediction stream: {self._prediction_full_path}")
        print(f"[EDL Logger] Saved sidecar scene summary: {self._scene_summary_path}")
        if self._prediction_light_path:
            print(f"[EDL Logger] Saved sidecar prediction light: {self._prediction_light_path}")
        if self._branch_candidates_path:
            print(f"[EDL Logger] Saved branch candidates: {self._branch_candidates_path}")
        if self._query_semantic_path:
            print(f"[EDL Logger] Saved query semantic sidecar: {self._query_semantic_path}")
        print(f"[EDL Logger] Saved sidecar meta: {self._sidecar_meta_path}")
        print(f"[EDL Logger] Total samples: {self._sidecar_counts['num_samples']}, "
              f"Total predictions: {self._sidecar_counts['num_predictions']}, "
              f"Total branch candidates: {self._sidecar_counts['num_branch_candidates']}, "
              f"Total query semantic rows: {self._sidecar_counts['num_query_semantic']}")
        if self._v6_schema_samples > 0 and self._v6_missing_field_counts:
            print(
                "[EDL Logger][WARN] Incomplete v6 diagnostics detected. "
                f"samples_with_v6={self._v6_schema_samples}, "
                f"missing={self._v6_missing_field_counts}"
            )
        if self._v7_schema_samples > 0 and self._v7_missing_field_counts:
            print(
                "[EDL Logger][WARN] Incomplete v7 diagnostics detected. "
                f"samples_with_v7={self._v7_schema_samples}, "
                f"missing={self._v7_missing_field_counts}"
            )
        if self._v8_schema_samples > 0 and self._v8_missing_field_counts:
            print(
                "[EDL Logger][WARN] Incomplete v8 diagnostics detected. "
                f"samples_with_v8={self._v8_schema_samples}, "
                f"missing={self._v8_missing_field_counts}"
            )
        return path or self._sample_full_path or self._scene_summary_path
    
    def get_summary_stats(self):
        """Compute summary statistics from collected logs."""
        if not self.logs:
            return {}
        
        all_center_unc = []
        all_height_unc = []
        all_cls_unc = []
        all_combined_unc = []
        per_class_unc = {}
        
        for sample in self.logs:
            for pred in sample['predictions']:
                if 'combined_unc' in pred:
                    all_combined_unc.append(pred['combined_unc'])
                if 'center_unc' in pred:
                    all_center_unc.append(pred['center_unc'])
                if 'height_unc' in pred:
                    all_height_unc.append(pred['height_unc'])
                if 'cls_unc' in pred:
                    all_cls_unc.append(pred['cls_unc'])
                
                label_name = pred.get('label_name', 'unknown')
                if label_name not in per_class_unc:
                    per_class_unc[label_name] = []
                if 'combined_unc' in pred:
                    per_class_unc[label_name].append(pred['combined_unc'])
        
        stats = {}
        for name, vals in [('center_unc', all_center_unc), 
                           ('height_unc', all_height_unc),
                           ('cls_unc', all_cls_unc),
                           ('combined_unc', all_combined_unc)]:
            if vals:
                arr = np.array(vals)
                stats[name] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'median': float(np.median(arr)),
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                }
        
        # Per-class summary
        stats['per_class'] = {}
        for cls_name, vals in per_class_unc.items():
            if vals:
                arr = np.array(vals)
                stats['per_class'][cls_name] = {
                    'count': len(vals),
                    'mean_unc': float(arr.mean()),
                    'std_unc': float(arr.std()),
                }
        
        return stats
    
    def print_summary(self):
        """Print a formatted summary of uncertainty statistics."""
        stats = self.get_summary_stats()
        if not stats:
            print("[EDL Logger] No data collected.")
            return
        
        print("\n" + "=" * 60)
        print("EDL Uncertainty Summary")
        print("=" * 60)
        
        for key in ['center_unc', 'height_unc', 'cls_unc', 'combined_unc']:
            if key in stats:
                s = stats[key]
                print(f"  {key:15s}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                      f"median={s['median']:.4f}, range=[{s['min']:.4f}, {s['max']:.4f}]")
        
        if 'per_class' in stats:
            print("\nPer-Class Combined Uncertainty:")
            for cls_name in sorted(stats['per_class'].keys()):
                s = stats['per_class'][cls_name]
                print(f"  {cls_name:25s}: n={s['count']:5d}, "
                      f"mean={s['mean_unc']:.4f}, std={s['std_unc']:.4f}")
        
        # PME info (check first sample)
        if self.logs and self.logs[0]['predictions']:
            first_pred = self.logs[0]['predictions'][0]
            if 'per_modality_unc' in first_pred:
                # Compute average modality uncertainty across all samples
                mod_uncs = {}
                mod_weights = {}
                count = 0
                for sample in self.logs:
                    for pred in sample['predictions']:
                        if 'per_modality_unc' in pred:
                            for m, v in pred['per_modality_unc'].items():
                                mod_uncs.setdefault(m, []).append(v)
                        if 'modality_weights' in pred:
                            for m, v in pred['modality_weights'].items():
                                mod_weights.setdefault(m, []).append(v)
                            count += 1
                
                print("\nPME Per-Modality Uncertainty (mean across all predictions):")
                for m in sorted(mod_uncs.keys()):
                    arr = np.array(mod_uncs[m])
                    print(f"  {m:10s}: unc={arr.mean():.4f}")
                
                if mod_weights:
                    print("\nPME Approximate Attention Weights (mean):")
                    for m in sorted(mod_weights.keys()):
                        arr = np.array(mod_weights[m])
                        print(f"  {m:10s}: weight={arr.mean():.4f}")
        
        print("=" * 60)
