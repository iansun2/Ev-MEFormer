"""Reproducible R50 V10 selector used by the finalized UMR chain.

The historical ``v10_router_selector.py`` loads its oracle thresholds and
matching fallback from an ignored ``work_dirs/v10_quality_calibration.json``.
The values below are the ones recorded in the fully resolved config of the
finalized 2026-05-22 run. Keeping them in a tracked leaf config makes a fresh
checkout resolve to the same training and runtime contract.
"""

_base_ = [
    './edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_v10_router_selector_placeholder.py'
]

_frozen_v10_oracle_thresholds = dict(
    tau_margin=1.588853454589843,
    margin=0.10,
    tau_dmg=0.18098025321960445,
    T_oracle=1.7976929346720378,
)

model = dict(
    pts_bbox_head=dict(
        v10_oracle_thresholds=_frozen_v10_oracle_thresholds,
        v10_matching_mode='single_best',
        ensemble=dict(
            v10_oracle_thresholds=_frozen_v10_oracle_thresholds,
        ),
    ),
)

work_dir = './work_dirs/v10_router_selector_frozen'
