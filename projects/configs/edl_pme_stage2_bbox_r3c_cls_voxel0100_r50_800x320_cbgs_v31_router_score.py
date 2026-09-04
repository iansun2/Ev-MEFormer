_base_ = ['./edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs.py']

model = dict(
    pts_bbox_head=dict(
        lambda_fused_prior=0.0,
        ensemble=dict(
            selector_score_mode='quality_minus_uncertainty',
            selector_unc_score_weight=1.0,
            selector_rel_unc_score_weight=0.5,
            selector_disagreement_score_weight=0.5,
            selector_context_score_weight=1.0,
            selector_distance_score_weight=1.0,
            selector_fused_score_bias=0.0,
            selector_log_score_components=True,
        ),
    ),
)

work_dir = './work_dirs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v31_router_score'
