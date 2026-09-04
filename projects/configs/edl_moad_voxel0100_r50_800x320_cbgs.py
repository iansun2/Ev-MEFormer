plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# --- 1. R50-specific settings (resolution and voxel) ---
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.1, 0.1, 0.2]  # R50 uses the coarser voxel
out_size_factor = 8
evaluation = dict(interval=4)
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

# R50 image augmentation settings (lower resolution, 800x320)
ida_aug_conf = {
    "resize_lim": (0.47, 0.625),
    "final_dim": (320, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

# --- 2. Data pipeline (keeps the R50 settings) ---
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='UnifiedObjectSample',
        sample_2d=True,
        mixup_rate=0.5,
        db_sampler=dict(
            type='UnifiedDataBaseSampler',
            data_root=data_root,
            info_path=data_root + 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ))),
    dict(
        type='GlobalRotScaleTransAll',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='CustomRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

# --- 3. Data Config ---
# samples_per_gpu is fixed at 2; see cumulative_iters for the effective batch size.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,  # raise if the data loader is the bottleneck
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

# --- 4. Model Architecture (R50 Backbone + EDL Head) ---
model = dict(
    type='MEFormerDetector',
    use_grid_mask=True,
    # [R50 backbone, unchanged]
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    # [R50 voxel settings, unchanged (grid size 1024, voxel 0.1)]
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024], # R50 grid size
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # [EDL head]
    pts_bbox_head=dict(
        type='EDLHead', # Changed from MEFormerHead
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        pc_range=point_cloud_range,
        modalities=dict(
            train=["fused", "bev", "img"],
            test=["fused"]
        ),
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]),
        ],
        bbox_coder=dict(
            type='EDLBBoxCoder', # Changed
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,  # note: this is the R50 value, 0.1
            num_classes=10),
        separate_head=dict(
            type='EDLNIGHead', init_bias=-2.19, final_kernel=1), # Changed
        transformer=dict(
            type='MOADTransformer',
            # The upstream R50 config leaves use_cam_embed off, but the MEFormer
            # paper generally enables it. It is enabled here to match the EDL VoV
            # config so both routes share the same head behaviour.
            use_cam_embed=True, 
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
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
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            ),
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        # [EDL loss settings] - EDL bbox + finalized EDL cls (R3c-style)
        loss_edl_weights=dict(
            l1_weight=1.0,            # L1 weight for the EDL tasks (center/height/dim)
            l1_weight_standard=1.0,   # separate L1 weight for the plain regression tasks (rot/vel); raised to resist NIG interference
            nig_nll_weight=0.3,       # NIG NLL loss weight (the code already applies a uniform 0.25 loss weight)
            nig_reg_weight=0.0001,    # NIG regularizer weight (the paper suggests lambda = 1e-4)
            warmup_iters=2000,        # 2000 iterations warmup
            # Controls the gradient reaching gamma through the NIG loss:
            # - True:  L1 supervises gamma, NIG supervises only v, alpha, beta
            #          (separation of concerns)
            # - False: NIG fully supervises gamma, v, alpha, beta (original E-DETR)
            detach_gamma=False,
            # [EDL CLS] Finalized R3c-style classification parameters
            cls_edl_weight=0.5,       # EDL NLL weight (Digamma-based)
            cls_kl_weight=0.1,        # strong paper-style KL weight
            cls_warmup_iters=7420,    # paper-style KL warmup horizon
            use_l1_quality_cls=True,
            cls_quality_tau=0.5,
            cls_neg_gamma=2.0,
            cls_neg_balance=1.0,
            use_dynamic_kl=True,
            # [MT-ENet] Lipschitz L1 Loss (faithful to Proposition 2)
            use_lipschitz_l1=True,    # enable Lipschitz L1 (the U_l threshold is derived dynamically from digamma)
            # ll1_zero_below_threshold:
            #   False = fall back to plain L1 (when chi^2 <= U_l, loss = |y - gamma|,
            #           so gamma always receives supervision)
            #   True  = original MT-ENet (when chi^2 <= U_l, loss = 0 and the NLL term
            #           takes over completely)
            ll1_zero_below_threshold=False,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1024, 1024, 40], # R50 Grid Size
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1024, 1024, 40], # R50 Grid Size
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))

# --- 5. Optimizer and schedule (EDL fine-tuning strategy) ---
optimizer = dict(
    type='AdamW',
    lr=0.00014,
    paramwise_cfg=dict(
        custom_keys={
            # # Fully frozen layers
            # 'img_backbone': dict(lr_mult=0.01, decay_mult=0.0),
            # 'pts_voxel_encoder': dict(lr_mult=0.0, decay_mult=0.0),
            # 'pts_middle_encoder': dict(lr_mult=0.0, decay_mult=0.0),
            
            # # Fine-tuned layers (0.1x lr)
            # 'img_neck': dict(lr_mult=0.0, decay_mult=0.1),
            # 'pts_backbone': dict(lr_mult=0.0, decay_mult=0.1),
            # 'pts_neck': dict(lr_mult=0.0, decay_mult=0.1),
            # 'shared_conv': dict(lr_mult=0.0, decay_mult=0.0),
            
            # # --- Fine-tuning region (LR x 0.1) ---
            # # The transformer performs feature fusion; a small LR lets it adapt to the EDL gradient flow
            # 'transformer': dict(lr_mult=0.0, decay_mult=0.0),
            
            # Normally trained layers (1.0x lr) - transformer and head
            # No entry needed; these use the base lr
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    cumulative_iters=2,  # 4 GPU * 2 samples * 2 accum = 16 Effective BS
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(pts_voxel_encoder=False, pts_middle_encoder=False, pts_bbox_head=False))

# lr_config = dict(
#     policy='CosineAnnealing', 
#     min_lr_ratio=0.01,       
#     warmup='linear',          
#     warmup_iters=1000,        
#     warmup_ratio=0.001,       
#     by_epoch=False
# )

lr_config = dict(
    policy='cyclic',
    target_ratio=(3, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)

total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None

# IMPORTANT
# Load the .pth produced by the plain R50 config here.
# If stage 1 has not been run yet, run moad_voxel0100_r50_800x320_cbgs.py first
# and point this at the resulting latest.pth.
# load_from = 'ckpts/moad_voxel0100_r50_800x320_cbgs.pth' 
# To start straight from ImageNet pretraining instead (not recommended - EDL
# converges poorly from there), use the line below:
load_from = 'ckpts/nuim_r50.pth'

resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)

custom_hooks = [
    dict(
        type="DropGTSamplingHook",
        priority="LOWEST",
        pipeline_name="UnifiedObjectSample",
        drop_epoch=15
    ),
    # EDLEpochHook has been removed in favour of iteration-based warmup,
    # which the loss functions update themselves.
]
