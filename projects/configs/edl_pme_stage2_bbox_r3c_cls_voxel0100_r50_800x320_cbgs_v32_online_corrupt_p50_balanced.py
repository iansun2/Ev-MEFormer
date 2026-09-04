_base_ = ['./edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v31_router_score.py']

# v3.2 online corruption: higher train corruption exposure, balanced taxonomy.

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_root = 'data/nuscenes/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)
ida_aug_conf = {
    "resize_lim": (0.47, 0.625),
    "final_dim": (320, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0, 1, 2, 3, 4]),
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
            points_loader=dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]))),
    dict(
        type='GlobalRotScaleTransAll',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(type='CustomRandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='OnlineSyntheticMultiCorrupt',
        corrupt_prob=0.50,
        severity_choices=[1, 2, 3],
        corruption_weights=dict(
            beamsreducing=0.18,
            pointsreducing=0.16,
            spatialmisalignment=0.13,
            temporalmisalignment=0.09,
            fog=0.14,
            snow=0.08,
            brightness=0.06,
            dark=0.07,
            motionblur=0.06,
            missingcamera=0.03,
        ),
        record_meta=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape',
            'scale_factor', 'flip', 'pcd_horizontal_flip',
            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
            'img_norm_cfg', 'pcd_trans', 'sample_idx',
            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
            'transformation_3d_flow', 'rot_degree',
            'gt_bboxes_3d', 'gt_labels_3d',
            'synthetic_corruption', 'corruption_type',
            'corruption_severity'))
]

data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))

work_dir = './work_dirs/edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v32_online_corrupt_p50_balanced'
