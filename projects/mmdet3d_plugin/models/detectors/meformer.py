# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models import build_model
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import SPConvVoxelization
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class MEFormerDetector(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 rev19_distill=None,
                 **kwargs):
        self.rev19_distill_cfg = rev19_distill or {}
        self.rev19_distill_enabled = bool(
            self.rev19_distill_cfg.get('enabled', False))
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(MEFormerDetector, self).__init__(**kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
        object.__setattr__(self, 'rev19_teacher_a4', None)
        object.__setattr__(self, 'rev19_teacher_v3', None)
        if self.rev19_distill_enabled:
            self._init_rev19_teachers()

    def _init_rev19_teacher(self, teacher_cfg):
        cfg_path = teacher_cfg.get('config')
        ckpt_path = teacher_cfg.get('checkpoint')
        if not cfg_path or not ckpt_path:
            raise ValueError('Rev19 teacher config and checkpoint are required')
        cfg = Config.fromfile(cfg_path)
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _init_rev19_teachers(self):
        teachers = self.rev19_distill_cfg.get('teachers', {})
        if 'a4' in teachers:
            object.__setattr__(
                self, 'rev19_teacher_a4',
                self._init_rev19_teacher(teachers['a4']))
        if 'v3' in teachers:
            object.__setattr__(
                self, 'rev19_teacher_v3',
                self._init_rev19_teacher(teachers['v3']))

    def train(self, mode=True):
        super(MEFormerDetector, self).train(mode)
        for teacher in (getattr(self, 'rev19_teacher_a4', None),
                        getattr(self, 'rev19_teacher_v3', None)):
            if teacher is not None:
                teacher.eval()
        return self

    @staticmethod
    def _rev19_meta_corruption_type(meta):
        if not isinstance(meta, dict):
            return 'clean'
        corruption_type = str(meta.get('corruption_type', 'clean') or 'clean').lower()
        if corruption_type in ('', 'none'):
            corruption_type = 'clean'
        synthetic = bool(meta.get('synthetic_corruption', False))
        if not synthetic and corruption_type != 'clean':
            return corruption_type
        return corruption_type

    def _rev19_distill_weights(self, img_metas, device, dtype):
        clean_weight = float(self.rev19_distill_cfg.get('clean_weight', 1.0))
        image_weights = self.rev19_distill_cfg.get('image_weights', {})
        weights_a4 = torch.zeros(len(img_metas), device=device, dtype=dtype)
        weights_v3 = torch.zeros(len(img_metas), device=device, dtype=dtype)
        for idx, meta in enumerate(img_metas):
            corruption_type = self._rev19_meta_corruption_type(meta)
            if corruption_type == 'clean':
                weights_a4[idx] = clean_weight
            elif corruption_type in image_weights:
                weights_v3[idx] = float(image_weights[corruption_type])
        return weights_a4, weights_v3

    @staticmethod
    def _rev19_teacher_points_fp32(points):
        if points is None:
            return None
        return [
            point.float() if (
                isinstance(point, torch.Tensor) and
                point.is_floating_point() and
                point.dtype != torch.float32
            ) else point
            for point in points
        ]

    @torch.no_grad()
    def _rev19_extract_teacher_queries(self, teacher, points, img, img_metas):
        device = None
        if points and points[0] is not None:
            device = points[0].device
        elif img is not None:
            device = img.device
        if device is not None:
            try:
                teacher_device = next(teacher.parameters()).device
            except StopIteration:
                teacher_device = device
            if teacher_device != device:
                teacher.to(device)
            teacher.eval()
        teacher_points = self._rev19_teacher_points_fp32(points)
        img_feats, pts_feats = teacher.extract_feat(
            teacher_points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        _outs, distill_queries = teacher.pts_bbox_head(
            pts_feats, img_feats, img_metas, return_distill_queries=True)
        if not distill_queries:
            return None
        return distill_queries[0].detach()

    def init_weights(self):
        """Initialize model weights."""
        super(MEFormerDetector, self).init_weights()

    def _detach_to_cpu(self, value):
        """Detach tensors before distributed result collection.

        Training-time evaluation collects per-rank result pickles on rank 0.
        Keeping EDL diagnostics on CUDA makes rank 0 deserialize every shard
        back to GPU memory, which can OOM after a full validation epoch.
        """
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._detach_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._detach_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._detach_to_cpu(v) for v in value)
        return value

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        rev19_distill = None
        if self.rev19_distill_enabled and img_metas is not None:
            device = None
            dtype = torch.float32
            if pts_feats:
                device = pts_feats[0].device
                dtype = pts_feats[0].dtype
            elif img_feats:
                device = img_feats[0].device
                dtype = img_feats[0].dtype
            if device is not None:
                weights_a4, weights_v3 = self._rev19_distill_weights(
                    img_metas, device, dtype)
                rev19_distill = {
                    'weights_a4': weights_a4,
                    'weights_v3': weights_v3,
                }
                if (
                        bool((weights_a4 > 0).any()) and
                        self.rev19_teacher_a4 is not None):
                    rev19_distill['teacher_queries_a4'] = (
                        self._rev19_extract_teacher_queries(
                            self.rev19_teacher_a4, points, img, img_metas))
                if (
                        bool((weights_v3 > 0).any()) and
                        self.rev19_teacher_v3 is not None):
                    rev19_distill['teacher_queries_v3'] = (
                        self._rev19_extract_teacher_queries(
                            self.rev19_teacher_v3, points, img, img_metas))
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(
                pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d,
                img_metas, gt_bboxes_ignore, rev19_distill=rev19_distill
            )
            losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          rev19_distill=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        if self.rev19_distill_enabled:
            outs, _distill_queries = self.pts_bbox_head(
                pts_feats, img_feats, img_metas, return_distill_queries=True)
        else:
            outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        loss_kwargs = dict(img_metas=img_metas)
        if rev19_distill:
            loss_kwargs.update(
                rev19_teacher_queries_a4=rev19_distill.get('teacher_queries_a4'),
                rev19_teacher_queries_v3=rev19_distill.get('teacher_queries_v3'),
                rev19_distill_weights_a4=rev19_distill.get('weights_a4'),
                rev19_distill_weights_v3=rev19_distill.get('weights_v3'))
        losses = self.pts_bbox_head.loss(*loss_inputs, **loss_kwargs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)

    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = []
        for result in bbox_list:
            bboxes, scores, labels = result[0], result[1], result[2]
            res = bbox3d2result(bboxes, scores, labels)
            # [EDL] Propagate edl_info if available
            if len(result) > 3:
                res['edl_info'] = self._detach_to_cpu(result[3])
            bbox_results.append(res)
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
