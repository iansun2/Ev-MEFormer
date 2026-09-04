_base_ = ['./edl_moad_voxel0100_r50_800x320_cbgs.py']

# Minimal clean integration:
# - bbox/decode/assign semantics from feature/edl-pme
# - R3c cls weighting / dynamic KL from feature/cls-only-probe
# - no PME inference, no EvMVX, no ER, no dim NIG

model = dict(
    pts_bbox_head=dict(
        use_l1_quality_cls=True,
        cls_quality_tau=0.5,
        cls_neg_gamma=2.0,
        cls_neg_balance=1.0,
        use_dynamic_kl=True,
        loss_edl_weights=dict(
            cls_edl_weight=0.5,
            cls_kl_weight=0.1,
            cls_warmup_iters=7420,
        ),
    )
)

checkpoint_config = dict(interval=4)
work_dir = './work_dirs/edl_pme_bbox_r3c_cls_voxel0100_r50_800x320_cbgs'
