# Repository layout

Where everything lives, which configs are actually invoked, and how the historical experiment
names in filenames map to the functional stage names. See [model-architecture.md](model-architecture.md)
for what the model itself does, and the [README](../README.md) for setup, training and evaluation.

---

## Directory tree

> **Always run commands from the repository root.** Configs declare
> `plugin_dir = 'projects/mmdet3d_plugin/'` and the tool scripts resolve
> `projects/configs/...` relative to the current working directory. Running from
> `projects/` — or bind-mounting `projects/` instead of the repository root — breaks
> plugin loading and every config path.

```text
Ev-MEFormer/
├── dockerfile/
│   ├── Dockerfile                     # CUDA 11.8 devel + torch 1.13.1+cu117 + MMLab + flash-attn
│   └── requirements.txt               # pinned Python dependencies
├── scripts/
│   ├── build.sh                       # docker build -t ev-meformer
│   └── start.sh                       # docker run, mounts the repository root at /Ev-MEFormer
├── projects/
│   ├── configs/                       # 16 mmcv configs (see the config map below)
│   └── mmdet3d_plugin/                # the mmdet3d plugin registered by plugin_dir
│       ├── core/bbox/
│       │   ├── assigners/hungarian_assigner_3d.py
│       │   ├── coders/edl_bbox_coder.py            # decodes NIG params -> boxes + uncertainty
│       │   ├── coders/multi_task_bbox_coder.py
│       │   └── match_costs/match_cost.py
│       ├── datasets/
│       │   ├── custom_nuscenes_dataset.py
│       │   └── pipelines/
│       │       ├── transform_3d.py                 # image/BEV augmentation
│       │       ├── dbsampler.py                    # UnifiedDataBaseSampler (GT paste)
│       │       └── synthetic_corruptions.py        # OnlineSyntheticMultiCorrupt
│       ├── mmcv_custom/
│       │   ├── ops/voxel/spconv_voxelize.py        # SPConvVoxelization
│       │   └── runner/hooks/                       # FreezeWeight, DropGTSampling, EDLEpochHook,
│       │                                           #   CustomFp16OptimizerHook (grad accumulation)
│       └── models/
│           ├── backbones/vovnet.py                 # VoVNet V-99-eSE
│           ├── necks/cp_fpn.py                     # CPFPN
│           ├── detectors/meformer.py               # MEFormerDetector
│           ├── dense_heads/
│           │   ├── meformer_head.py                # baseline MEFormer head
│           │   ├── edl_head.py                     # EDLHead — EUD losses + UMR orchestration
│           │   ├── edl_nig_head.py                 # EDLNIGHead — per-task NIG/evidence output
│           │   └── separate_task_head.py
│           └── utils/
│               ├── moad_transformer.py             # MOADTransformer — per-modality decoding
│               ├── edl_pme_transformer.py          # EDLPMETransformer — routers and gating
│               ├── petr_transformer.py, attention.py, pme_transformer.py
│               ├── v10_quality.py                  # per-head conservative routing quality model
│               ├── edl_uncertainty_logger.py       # uncertainty sidecar logging
│               └── grid_mask.py
├── tools/                             # entry points, chain scripts, calibration and provenance
│   ├── train.py, test.py, dist_train.sh, dist_test.sh
│   ├── create_data.py, create_data.sh, data_converter/
│   ├── train_r50_umr_complete.sh      # R50: full UMR chain in one command
│   ├── train_vov_umr_complete.sh      # VoV: full UMR chain in one command
│   ├── prepare_r50_umr_sgsup_b.sh     # R50 bounded-candidate calibration -> runtime pack
│   ├── prepare_vov_umr_sgsup_b.sh     # VoV bounded-candidate calibration -> runtime pack
│   ├── umr_module_names.sh            # canonical module-name registry
│   ├── umr_collector_contract.py      # collector-code fingerprint used for provenance
│   ├── check_sgsup_admission_contract.py  # verifies the 18-feature admission contract
│   ├── stamp_umr_model_provenance.py  # stamps/validates runtime-pack provenance
│   ├── *_preflight.py                 # per-stage checkpoint/config contract checks
│   ├── visualize.py, runtime_visualize.sh, runtime_supplement_visualize.sh
│   ├── ev_meformer_corruption_viz.py, render_multicorrupt_examples.py, visualize_edl_logs.py
│   └── test_*.py                      # CPU unit tests for routing/provenance contracts
├── tests/                             # CPU unit tests for config parity and preflight coverage
├── multicorrupt_evaluation.sh         # full MultiCorrupt sweep (10 corruptions × 3 severities)
├── run_targeted_corruption_probe.sh   # single-corruption probe (used by generated runtime packs)
├── run_r50_three_lines.sh             # R50 three-line validation driver (EUD / UMR / supplement)
├── docs/
│   ├── model-architecture.md          # model, losses and the UMR training stages
│   ├── repository-layout.md           # this file
│   └── ev-meformer-architecture.png   # architecture figure used by the README
├── README.md
└── LICENSE
```

## Config map

`projects/configs/` holds 16 files: **8 you actually invoke**, and 8 that exist only because the
invoked ones inherit from them through `_base_`. All names below are prefixed
`edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_` (R50) or
`edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_` (VoV) unless shown in full.

**The 8 you invoke**, one per stage per backbone:

| Stage | R50 (800×320, voxel 0.1) | VoV (1600×640, voxel 0.075) |
|---|---|---|
| EUD training, 20 epochs | `edl_pme_bbox_r3c_cls_voxel0100_r50_800x320_cbgs.py` | `edl_pme_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs.py` |
| Branch reliability learning, 6 epochs | `..._v8_a4_clean_anchor_domain_p50_full_e6.py` | `..._v8_a4_clean_anchor_domain_p50_full_e6.py` |
| Damage-aware routing, 4 epochs | `..._v9_rev20_a4_routing_only.py` | `..._umr.py` |
| Conservative head routing, 6 epochs | `..._v10_router_selector_frozen.py` | `..._v10_router_selector_placeholder.py` |

The EUD config is also what you evaluate for line 1. The conservative-head-routing config is what
lines 2 and 3 evaluate — on R50 `_frozen` carries the calibrated `v10_oracle_thresholds` and
`v10_matching_mode` inline, so a fresh checkout resolves to the same runtime contract with no
external JSON. On VoV the `_placeholder` is a template: the chain script calibrates against that
run's damage-aware-routing checkpoint and generates an immutable child config before training.

**The 8 inheritance links.** These are never passed to `tools/train.py` or `tools/test.py`; each adds one layer
of settings that the next one builds on. Deleting any of them breaks the config above it:

| Config | What this layer contributes |
|---|---|
| `edl_moad_voxel0100_r50_800x320_cbgs.py` / `..._voxel0075_vov_...py` | the full model, data pipeline, optimizer and EDL loss weights — everything else is a delta on this |
| `edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs.py` / `..._vov_...py` | stage-2 batch/schedule settings for router training on a frozen detector |
| `..._v31_router_score.py` (R50) | turns on router scoring and the branch-consistency losses |
| `..._v32_online_corrupt_p50_balanced.py` (R50) | `OnlineSyntheticMultiCorrupt` at `corrupt_prob=0.5` with the balanced corruption mix |
| `..._v9_lite_p50_full_e6.py` (R50) | the anomaly-head and `tau` machinery the routing stage trains |
| `..._v10_router_selector_placeholder.py` (R50) | the per-head selector wiring, 6-epoch schedule and default thresholds |

Full chains, left to right from base to leaf:

```text
R50  edl_moad -> edl_pme_bbox_r3c_cls -> stage2 -> v31_router_score
     -> v32_online_corrupt_p50_balanced -> v8_a4_clean_anchor_domain_p50_full_e6
     -> v9_lite_p50_full_e6 -> v9_rev20_a4_routing_only
     -> v10_router_selector_placeholder -> v10_router_selector_frozen

VoV  edl_moad -> edl_pme_bbox_r3c_cls -> stage2
     -> v8_a4_clean_anchor_domain_p50_full_e6 -> umr
     -> v10_router_selector_placeholder
```

The R50 chain is longer because the routing stage was developed there first; the VoV route reuses
the finalized settings and so needs fewer intermediate layers. The `v8`/`v9`/`v10`/`v31`/`v32`
fragments in the filenames are the historical experiment identifiers those layers were introduced
under — the [module naming](#module-naming) table maps them to the functional stage names.

`umr_sgsup_b_r50_parity_contract.json` pins the R50/VoV parity contract checked by the preflight
and provenance tools.

`tools/` is scoped the same way: it holds the training and evaluation entry points, the UMR chain
and calibration scripts, the preflight/provenance checks, the visualization utilities and the unit
tests. Development-only tooling — signal-analysis and replay probes, parameter sweeps and their
monitors, superseded calibration wrappers — is not included here, and neither are the exploratory
config ablations (`v4`–`v9` router variants, curriculum and tau-calibration sweeps, distillation,
the plain MEFormer and MOAD baselines). All of it lives in the upstream development repository.

## Module naming

Thesis-level functional names are canonical; the historical experiment aliases are only
accepted so that existing checkpoint-resume commands keep working
(`tools/umr_module_names.sh` holds the registry).

| Functional name | Legacy alias | Epochs / nature |
|---|---|---|
| `branch_reliability_learning` | V8-A4 | 6 epochs |
| `damage_aware_routing` | Rev20-A | 4 epochs |
| `conservative_head_routing` | V10 selector | 6 epochs |
| `bounded_candidate_supplement` | SGSUP-B | train-only calibration, no weight updates |
