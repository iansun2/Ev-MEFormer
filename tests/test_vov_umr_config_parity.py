import copy
import unittest

from mmcv import Config


R50_A4_CONFIG = (
    'projects/configs/'
    'edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_'
    'v8_a4_clean_anchor_domain_p50_full_e6.py'
)
VOV_A4_CONFIG = (
    'projects/configs/'
    'edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_'
    'v8_a4_clean_anchor_domain_p50_full_e6.py'
)
R50_UMR_CONFIG = (
    'projects/configs/'
    'edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_'
    'v9_rev20_a4_routing_only.py'
)
VOV_UMR_CONFIG = (
    'projects/configs/'
    'edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_umr.py'
)
R50_V10_CONFIG = (
    'projects/configs/'
    'edl_pme_stage2_bbox_r3c_cls_voxel0100_r50_800x320_cbgs_'
    'v10_router_selector_placeholder.py'
)
VOV_V10_CONFIG = (
    'projects/configs/'
    'edl_pme_stage2_bbox_r3c_cls_voxel0075_vov_1600x640_cbgs_'
    'v10_router_selector_placeholder.py'
)


def _pipeline_step(cfg, step_type):
    return next(
        step for step in cfg.data.train.dataset.pipeline
        if step.get('type') == step_type)


def _resize_step(cfg, split):
    dataset = cfg.data[split]
    if dataset.get('type') == 'CBGSDataset':
        dataset = dataset.dataset
    pipeline = dataset.pipeline
    direct = [
        step for step in pipeline
        if step.get('type') == 'ResizeCropFlipImage'
    ]
    if direct:
        return direct[0]
    multiscale = next(
        step for step in pipeline
        if step.get('type') == 'MultiScaleFlipAug3D')
    return next(
        step for step in multiscale.transforms
        if step.get('type') == 'ResizeCropFlipImage')


def _normalized_head_pair(r50_cfg, vov_cfg):
    r50_head = copy.deepcopy(r50_cfg.model.pts_bbox_head)
    vov_head = copy.deepcopy(vov_cfg.model.pts_bbox_head)
    # The decoder semantics must match; the bbox coder alone follows the
    # backbone-native 0.075 VoV geometry instead of the R50 0.10 geometry.
    vov_head.bbox_coder.voxel_size = copy.deepcopy(
        r50_head.bbox_coder.voxel_size)
    return r50_head, vov_head


class VoVUMRConfigParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r50_a4 = Config.fromfile(R50_A4_CONFIG)
        cls.vov_a4 = Config.fromfile(VOV_A4_CONFIG)
        cls.r50_umr = Config.fromfile(R50_UMR_CONFIG)
        cls.vov_umr = Config.fromfile(VOV_UMR_CONFIG)
        cls.r50_v10 = Config.fromfile(R50_V10_CONFIG)
        cls.vov_v10 = Config.fromfile(VOV_V10_CONFIG)

    def test_a4_head_matches_r50_except_voxel_size(self):
        r50_head, vov_head = _normalized_head_pair(self.r50_a4, self.vov_a4)
        self.assertEqual(r50_head, vov_head)
        self.assertEqual(self.r50_a4.total_epochs, 6)
        self.assertEqual(self.vov_a4.total_epochs, 6)
        self.assertEqual(
            _pipeline_step(self.r50_a4, 'OnlineSyntheticMultiCorrupt'),
            _pipeline_step(self.vov_a4, 'OnlineSyntheticMultiCorrupt'))

    def test_a4_training_contract_and_freeze_hardening(self):
        self.assertEqual(
            self.r50_a4.custom_hooks,
            [{'type': 'FreezeWeight',
              'finetune_weight': ['pts_bbox_head.ensemble']}])
        self.assertEqual(
            self.vov_a4.custom_hooks,
            [{'type': 'FreezeWeight',
              'finetune_weight': ['pts_bbox_head.ensemble'],
              'freeze_bn': True}])
        self.assertEqual(self.vov_a4.data.samples_per_gpu, 2)
        self.assertEqual(self.vov_a4.optimizer_config.cumulative_iters, 2)
        self.assertEqual(self.vov_a4.lr_config.warmup_iters, 1000)
        self.assertEqual(
            self.vov_a4.data.samples_per_gpu *
            self.vov_a4.optimizer_config.cumulative_iters,
            self.r50_a4.data.samples_per_gpu)
        self.assertEqual(
            self.vov_a4.lr_config.warmup_iters,
            self.r50_a4.lr_config.warmup_iters *
            self.vov_a4.optimizer_config.cumulative_iters)

        r50_optimizer_config = copy.deepcopy(self.r50_a4.optimizer_config)
        vov_optimizer_config = copy.deepcopy(self.vov_a4.optimizer_config)
        vov_optimizer_config.pop('cumulative_iters')
        self.assertEqual(r50_optimizer_config, vov_optimizer_config)
        r50_lr = copy.deepcopy(self.r50_a4.lr_config)
        vov_lr = copy.deepcopy(self.vov_a4.lr_config)
        vov_lr.warmup_iters = r50_lr.warmup_iters
        self.assertEqual(r50_lr, vov_lr)

        frozen_prefixes = (
            'img_backbone', 'img_neck', 'pts_voxel_encoder',
            'pts_middle_encoder', 'pts_backbone', 'pts_neck',
            'pts_bbox_head')
        expected_custom_keys = {
            prefix: {'lr_mult': 0.0, 'decay_mult': 0.0}
            for prefix in frozen_prefixes
        }
        expected_custom_keys['pts_bbox_head.ensemble'] = {
            'lr_mult': 1.0, 'decay_mult': 1.0}
        custom_keys = self.vov_a4.optimizer.paramwise_cfg.custom_keys
        self.assertEqual(custom_keys, expected_custom_keys)

        # Once the intentional freeze hardening is normalized away, the full
        # optimizer contract (including type, global lr, and weight decay)
        # must remain identical to the final R50 A4 configuration.
        r50_optimizer = copy.deepcopy(self.r50_a4.optimizer)
        vov_optimizer = copy.deepcopy(self.vov_a4.optimizer)
        vov_optimizer.paramwise_cfg.custom_keys = copy.deepcopy(
            r50_optimizer.paramwise_cfg.custom_keys)
        self.assertEqual(r50_optimizer, vov_optimizer)

    def test_umr_head_matches_r50_except_voxel_size(self):
        r50_head, vov_head = _normalized_head_pair(self.r50_umr, self.vov_umr)
        self.assertEqual(r50_head, vov_head)

    def test_umr_routing_training_contract_matches_r50(self):
        self.assertEqual(self.r50_umr.custom_hooks, self.vov_umr.custom_hooks)
        self.assertEqual(self.r50_umr.optimizer, self.vov_umr.optimizer)
        self.assertEqual(
            _pipeline_step(self.r50_umr, 'OnlineSyntheticMultiCorrupt'),
            _pipeline_step(self.vov_umr, 'OnlineSyntheticMultiCorrupt'))
        self.assertEqual(self.r50_umr.total_epochs, self.vov_umr.total_epochs)
        self.assertEqual(self.vov_umr.total_epochs, 4)

        r50_lr = copy.deepcopy(self.r50_umr.lr_config)
        vov_lr = copy.deepcopy(self.vov_umr.lr_config)
        vov_lr.warmup_iters = r50_lr.warmup_iters
        self.assertEqual(r50_lr, vov_lr)

        self.assertEqual(self.r50_umr.data.samples_per_gpu, 4)
        self.assertEqual(self.r50_umr.optimizer_config.get('cumulative_iters', 1), 1)
        self.assertEqual(self.vov_umr.data.samples_per_gpu, 2)
        self.assertEqual(self.vov_umr.optimizer_config.cumulative_iters, 2)
        r50_optimizer_config = copy.deepcopy(self.r50_umr.optimizer_config)
        vov_optimizer_config = copy.deepcopy(self.vov_umr.optimizer_config)
        vov_optimizer_config.pop('cumulative_iters')
        self.assertEqual(r50_optimizer_config, vov_optimizer_config)
        self.assertEqual(
            self.r50_umr.data.samples_per_gpu *
            self.r50_umr.optimizer_config.get('cumulative_iters', 1),
            self.vov_umr.data.samples_per_gpu *
            self.vov_umr.optimizer_config.cumulative_iters)
        self.assertEqual(self.vov_umr.lr_config.warmup_iters, 1000)
        self.assertEqual(
            self.vov_umr.lr_config.warmup_iters,
            self.r50_umr.lr_config.warmup_iters *
            self.vov_umr.optimizer_config.cumulative_iters)

    def test_vov_geometry_is_native(self):
        for cfg in (self.vov_a4, self.vov_umr):
            self.assertEqual(cfg.model.img_backbone.type, 'VoVNet')
            self.assertEqual(cfg.model.img_backbone.spec_name, 'V-99-eSE')
            self.assertEqual(
                tuple(cfg.model.img_backbone.out_features),
                ('stage4', 'stage5'))
            self.assertEqual(list(cfg.model.img_neck.in_channels), [768, 1024])
            self.assertEqual(
                list(cfg.model.pts_voxel_layer.voxel_size),
                [0.075, 0.075, 0.2])
            self.assertEqual(
                list(cfg.model.pts_middle_encoder.sparse_shape),
                [41, 1440, 1440])
            self.assertEqual(
                list(cfg.model.pts_bbox_head.bbox_coder.voxel_size),
                [0.075, 0.075, 0.2])
            self.assertEqual(
                list(cfg.model.train_cfg.pts.voxel_size),
                [0.075, 0.075, 0.2])
            self.assertEqual(
                list(cfg.model.test_cfg.pts.voxel_size),
                [0.075, 0.075, 0.2])
            self.assertEqual(
                list(cfg.model.train_cfg.pts.grid_size), [1440, 1440, 40])
            self.assertEqual(
                list(cfg.model.test_cfg.pts.grid_size), [1440, 1440, 40])
            for split in ('train', 'val', 'test'):
                resize = _resize_step(cfg, split)
                self.assertEqual(
                    tuple(resize.data_aug_conf.final_dim), (640, 1600))
                self.assertEqual(
                    tuple(resize.data_aug_conf.resize_lim), (0.94, 1.25))

    def test_v10_selector_training_contract_matches_r50(self):
        self.assertEqual(self.vov_v10.total_epochs, 6)
        self.assertEqual(self.vov_v10.runner.max_epochs, 6)
        self.assertEqual(
            self.vov_v10.v10_trainable_prefixes,
            self.r50_v10.v10_trainable_prefixes)
        self.assertEqual(self.vov_v10.custom_hooks, self.r50_v10.custom_hooks)
        self.assertEqual(self.vov_v10.optimizer, self.r50_v10.optimizer)
        self.assertEqual(
            _pipeline_step(self.vov_v10, 'OnlineSyntheticMultiCorrupt'),
            _pipeline_step(self.r50_v10, 'OnlineSyntheticMultiCorrupt'))

        contract_keys = (
            'lambda_consistency_cls', 'lambda_consistency_box',
            'lambda_v9_anomaly_supervision', 'lambda_v9_gate_domain',
            'lambda_v9_route_consistency', 'v9_anomaly_head_lambda_warmup',
            'v10_loss_enabled', 'lambda_v10_route_kl',
            'lambda_v10_route_margin', 'lambda_v10_dd_anomaly',
            'lambda_v10_dd_consistency', 'lambda_v10_csg',
            'v10_route_margin_value', 'v10_csg_clean_target',
        )
        for key in contract_keys:
            self.assertEqual(
                self.vov_v10.model.pts_bbox_head[key],
                self.r50_v10.model.pts_bbox_head[key], key)
        ensemble_keys = (
            'v10_enable', 'v10_input_detach', 'v10_cls_fused_keep_min',
            'v10_use_conservative_schedule',
            'v10_use_per_head_cls_dispatch',
        )
        for key in ensemble_keys:
            self.assertEqual(
                self.vov_v10.model.pts_bbox_head.ensemble[key],
                self.r50_v10.model.pts_bbox_head.ensemble[key], key)

        self.assertEqual(self.vov_v10.data.samples_per_gpu, 2)
        self.assertEqual(self.vov_v10.optimizer_config.cumulative_iters, 2)
        self.assertEqual(self.r50_v10.data.samples_per_gpu, 4)
        self.assertEqual(
            self.vov_v10.data.samples_per_gpu *
            self.vov_v10.optimizer_config.cumulative_iters,
            self.r50_v10.data.samples_per_gpu)
        r50_optimizer_config = copy.deepcopy(self.r50_v10.optimizer_config)
        vov_optimizer_config = copy.deepcopy(self.vov_v10.optimizer_config)
        vov_optimizer_config.pop('cumulative_iters')
        self.assertEqual(vov_optimizer_config, r50_optimizer_config)
        r50_lr = copy.deepcopy(self.r50_v10.lr_config)
        vov_lr = copy.deepcopy(self.vov_v10.lr_config)
        vov_lr.warmup_iters = r50_lr.warmup_iters
        self.assertEqual(vov_lr, r50_lr)
        self.assertEqual(
            self.vov_v10.model.img_backbone.type, 'VoVNet')
        self.assertEqual(
            self.vov_v10.model.pts_bbox_head.v10_matching_mode, 'union')


if __name__ == '__main__':
    unittest.main()
