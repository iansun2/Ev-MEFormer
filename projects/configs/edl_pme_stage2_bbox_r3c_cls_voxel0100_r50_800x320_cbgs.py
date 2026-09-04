_base_ = ['./edl_pme_bbox_r3c_cls_voxel0100_r50_800x320_cbgs.py']

# Stage 2:
# - keep the validated minimal integration (edl-pme bbox + R3c cls)
# - enable PME ensemble for uncertainty-aware modality selection
# - prepare inference-time EDL JSON logging via tools/test.py --save-edl-log

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,  # raise if the data loader is the bottleneck
    )

model = dict(
    pts_bbox_head=dict(
        lambda_consistency_cls=0.5,
        lambda_consistency_box=0.5,
        lambda_fused_prior=0.02,
        lambda_selector_target=0.5,
        lambda_selector_target_base=0.1,
        lambda_selector_target_residual=0.5,
        lambda_residual_usage=0.05,
        lambda_selector_second_opinion=0.2,
        lambda_selector_entropy_hard=0.02,
        lambda_alpha_res_separation=0.05,
        selector_teacher_temperature=0.5,
        selector_entropy_target_hard=0.35,
        residual_usage_target=0.2,
        alpha_res_separation_margin=0.15,
        use_ensemble=True,
        modalities=dict(
            train=["fused", "bev", "img"],
            test=["fused", "bev", "img"],
        ),
        ensemble=dict(
            type='EDLPMETransformer',
            ensemble_mode='soft_selector',
            use_unc_bias=True,
            use_value_scaling=True,
            fused_prior_bias=0.15,
            selector_softmax_temperature=1.0,
            selector_apply_heads=['cls_logits', 'center', 'height'],
            selector_frozen_heads=['dim', 'rot', 'vel'],
            selector_unc_transform='log1p',
            selector_unc_norm='per_query_modality_norm',
            selector_use_relative_unc=True,
            residual_floor=0.1,
            residual_usage_target=0.2,
            distance_bin_edges=[20.0, 40.0],
            pme_unc_sources=['center', 'height'],
            decoder=dict(
                type="PETRTransformerDecoder",
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
                ),
            ),
        ),
    ),
)

total_epochs = 6
load_from = './work_dirs/edl_pme_bbox_r3c_cls_voxel0100_r50_800x320_cbgs/edl-moad/latest.pth'

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'img_neck': dict(lr_mult=0.0, decay_mult=0.0),
        }),
    weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.001,
    warmup="linear",
    warmup_iters=500)
momentum_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)

optimizer_config = dict(
    _delete_=True,
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(
        pts_voxel_encoder=False,
        pts_middle_encoder=False,
        pts_bbox_head=False,
    ))

custom_hooks = [
    dict(
        type="FreezeWeight",
        finetune_weight=["pts_bbox_head.ensemble"]
    )
]

evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
find_unused_parameters = True

work_dir = './work_dirs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs'
