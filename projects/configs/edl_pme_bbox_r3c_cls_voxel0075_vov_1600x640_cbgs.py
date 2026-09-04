_base_ = ['./edl_moad_voxel0075_vov_1600x640_cbgs.py']

# Final Stage-1 EUD / R3c configuration using the original MEFormer VoV recipe:
# - VoVNet V-99-eSE image backbone with 1600x640 image input
# - voxel size 0.075 and 1440x1440 BEV grid
# - finalized R3c quality-aware classification and dynamic KL
# - no PME inference, no EvMVX, no ER, and no dimension NIG

model = dict(
    pts_bbox_head=dict(
        use_l1_quality_cls=True,
        cls_quality_tau=0.5,
        cls_neg_gamma=2.0,
        cls_neg_balance=1.0,
        use_dynamic_kl=True,
        # Replace the stale loss block in the historical VoV EUD base instead
        # of recursively merging with it (notably its unused warmup_epochs).
        loss_edl_weights=dict(
            _delete_=True,
            l1_weight=1.0,
            l1_weight_standard=1.0,
            nig_nll_weight=0.3,
            nig_reg_weight=0.0001,
            warmup_iters=2000,
            detach_gamma=False,
            cls_edl_weight=0.5,
            cls_kl_weight=0.1,
            cls_warmup_iters=7420,
            # Retain the finalized R3c values in the experiment config for
            # provenance; the active head-level switches are defined above.
            use_l1_quality_cls=True,
            cls_quality_tau=0.5,
            cls_neg_gamma=2.0,
            cls_neg_balance=1.0,
            use_dynamic_kl=True,
            use_lipschitz_l1=True,
            ll1_zero_below_threshold=False,
        ),
    )
)

# Match the base learning-rate change used by the original MEFormer VoV
# recipe while preserving the finalized R3c optimizer parameter groups.
optimizer = dict(lr=0.0001)

# The historical EUD VoV config has a separate experimental scheduler.
# Replace it with the scheduler used by the finalized R3c EUD run.
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(3, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4,
)

total_epochs = 20
checkpoint_config = dict(interval=4)

# Backbone-only initialization is the VoV counterpart of the nuImages R50
# initialization used by the finalized R3c EUD experiment.
load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None

work_dir = (
    './work_dirs/'
    'edl_pme_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs'
)
