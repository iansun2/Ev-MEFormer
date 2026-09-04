"""VoV V10 selector-6 template aligned with the finalized R50 method.

This template is used only for train-only quality calibration. The complete
chain generates an immutable child config containing VoV/checkpoint-specific
oracle thresholds before launching the six-epoch selector training.
"""

_base_ = [
    './edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_umr.py'
]

total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=6)

load_from = (
    './work_dirs/vov_umr_chain/vov-umr-run1/umr/'
    '20260809-161531/epoch_4.pth'
)
resume_from = None
work_dir = './work_dirs/vov_v10_router_selector_placeholder'

v10_trainable_prefixes = [
    'pts_bbox_head.ensemble.v10_dd_head',
    'pts_bbox_head.ensemble.v10_router_head_cls',
    'pts_bbox_head.ensemble.v10_router_head_height',
    'pts_bbox_head.ensemble.v10_router_head_center',
    'pts_bbox_head.ensemble.v10_center_safe_gate',
]

custom_hooks = [
    dict(
        type='EDLEpochHook',
        v9_keep_min_schedule=[
            dict(epoch=0, value=0.95),
            dict(epoch=1, value=0.90),
            dict(epoch=2, value=0.85),
            dict(epoch=3, value=0.85),
        ],
        v9_gate_lambda_warmup=dict(
            enabled=True, start=0.0, end=0.0, warmup_epochs=1),
        v9_route_lambda_warmup=dict(
            enabled=True, start=0.0, end=0.0, warmup_epochs=1),
        selector_cls_lambda_warmup=dict(
            enabled=False, start=0.0, end=0.0, warmup_epochs=1),
        selector_box_lambda_warmup=dict(
            enabled=False, start=0.0, end=0.0, warmup_epochs=1),
        v10_cls_fused_keep_min_schedule=[
            dict(epoch=0, value=0.85),
            dict(epoch=1, value=0.70),
            dict(epoch=2, value=0.60),
            dict(epoch=3, value=0.60),
            dict(epoch=4, value=0.50),
            dict(epoch=5, value=0.50),
        ],
        v10_cls_fused_keep_min_schedule_conservative=[
            dict(epoch=0, value=0.85),
            dict(epoch=1, value=0.75),
            dict(epoch=2, value=0.70),
            dict(epoch=3, value=0.70),
            dict(epoch=4, value=0.65),
            dict(epoch=5, value=0.65),
        ],
        v10_route_kl_lambda_warmup=dict(
            enabled=True, start=0.05, end=0.20, warmup_epochs=1),
    ),
    dict(
        type='FreezeWeight',
        finetune_weight=v10_trainable_prefixes,
        freeze_bn=True,
    ),
]

model = dict(
    pts_bbox_head=dict(
        lambda_consistency_cls=0.0,
        lambda_consistency_box=0.0,
        lambda_v9_anomaly_supervision=0.01,
        lambda_v9_gate_domain=0.0,
        lambda_v9_route_consistency=0.0,
        v9_anomaly_head_lambda_warmup=False,
        v10_loss_enabled=True,
        lambda_v10_route_kl=0.05,
        lambda_v10_route_margin=0.10,
        lambda_v10_dd_anomaly=1.00,
        lambda_v10_dd_consistency=0.05,
        lambda_v10_csg=0.10,
        v10_route_margin_value=0.10,
        v10_csg_clean_target=0.90,
        v10_oracle_thresholds=dict(
            tau_margin=0.05,
            margin=0.10,
            tau_dmg=0.15,
            T_oracle=0.5,
        ),
        v10_matching_mode='union',
        ensemble=dict(
            v10_enable=True,
            v10_input_detach=True,
            v10_cls_fused_keep_min=0.85,
            v10_use_conservative_schedule=False,
            v10_use_per_head_cls_dispatch=False,
            v10_oracle_thresholds=dict(
                tau_margin=0.05,
                margin=0.10,
                tau_dmg=0.15,
                T_oracle=0.5,
            ),
        ),
    ),
)

optimizer = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys={
            'pts_bbox_head.ensemble.v10_dd_head': dict(
                lr_mult=1.0, decay_mult=1.0),
            'pts_bbox_head.ensemble.v10_router_head_cls': dict(
                lr_mult=1.0, decay_mult=1.0),
            'pts_bbox_head.ensemble.v10_router_head_height': dict(
                lr_mult=1.0, decay_mult=1.0),
            'pts_bbox_head.ensemble.v10_router_head_center': dict(
                lr_mult=1.0, decay_mult=1.0),
            'pts_bbox_head.ensemble.v10_center_safe_gate': dict(
                lr_mult=1.0, decay_mult=1.0),
            'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'img_neck': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_voxel_encoder': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_middle_encoder': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_neck': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0, decay_mult=0.0),
        },
    ),
)
