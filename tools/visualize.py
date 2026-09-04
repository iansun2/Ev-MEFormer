import argparse
import mmcv
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Add the current directory to the Python search path
sys.path.insert(0, os.getcwd())

# Force the plugin to be imported so its registry entries exist
try:
    import projects.mmdet3d_plugin
except ImportError:
    pass

from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet3D Visualization (V9 - Double List Fix)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show-dir', help='directory where results will be saved', default='vis_results_final')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--limit', type=int, default=20, help='limit samples (default: 20)')
    return parser.parse_args()

def draw_lidar_bbox3d_on_img(bboxes3d, raw_img, lidar2img, color=(0, 255, 0), thickness=2):
    img = raw_img.copy()
    bboxes3d = bboxes3d.to('cpu')
    corners_3d = bboxes3d.corners.numpy()
    
    num_bbox = corners_3d.shape[0]
    if num_bbox == 0:
        return img
        
    points_3d = corners_3d.reshape(-1, 3)
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()
    
    points_3d = points_3d.astype(np.float32)
    lidar2img = lidar2img.astype(np.float32)

    points_3d_hom = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)], axis=1)
    points_2d_hom = points_3d_hom @ lidar2img.T
    
    z = points_2d_hom[:, 2:3]
    mask = (z > 1e-3).flatten()
    z_safe = np.maximum(z, 1e-3)
    points_2d = points_2d_hom[:, :2] / z_safe
    points_2d = points_2d.reshape(num_bbox, 8, 2)
    
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    
    for i in range(num_bbox):
        if not np.any(mask[i*8 : (i+1)*8]): continue
        cnt = points_2d[i].astype(int)
        for start, end in edges:
            try:
                cv2.line(img, tuple(cnt[start]), tuple(cnt[end]), color, thickness)
            except Exception:
                pass
    return img

def draw_lidar_bev(points, gt_bboxes=None, pred_bboxes=None, out_path=None, x_range=(-54, 54), y_range=(-54, 54)):
    plt.figure(figsize=(10, 10))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.axis('off')
    
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    mask = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & \
           (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
    points = points[mask]
    
    if points.shape[0] > 15000:
        choice = np.random.choice(points.shape[0], 15000, replace=False)
        points = points[choice]
        
    plt.scatter(points[:, 0], points[:, 1], s=0.5, c='gray', alpha=0.5)

    if gt_bboxes is not None:
        if isinstance(gt_bboxes, torch.Tensor): gt_bboxes = gt_bboxes.cpu().numpy()
        elif hasattr(gt_bboxes, 'tensor'): gt_bboxes = gt_bboxes.tensor.cpu().numpy()
        for i in range(gt_bboxes.shape[0]):
            x, y, w, l, yaw = gt_bboxes[i, 0], gt_bboxes[i, 1], gt_bboxes[i, 3], gt_bboxes[i, 4], gt_bboxes[i, 6]
            c, s = np.cos(yaw), np.sin(yaw)
            corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
            corners_rot = corners @ np.array([[c, -s], [s, c]]).T + np.array([x, y])
            plt.gca().add_patch(Polygon(corners_rot, fill=False, edgecolor='green', linewidth=2.0))
        plt.scatter([], [], c='green', marker='s', label='GT')

    if pred_bboxes is not None:
        if hasattr(pred_bboxes, 'tensor'): pred_bboxes = pred_bboxes.tensor.cpu().numpy()
        elif isinstance(pred_bboxes, torch.Tensor): pred_bboxes = pred_bboxes.cpu().numpy()
        for i in range(pred_bboxes.shape[0]):
            x, y, w, l, yaw = pred_bboxes[i, 0], pred_bboxes[i, 1], pred_bboxes[i, 3], pred_bboxes[i, 4], pred_bboxes[i, 6]
            c, s = np.cos(yaw), np.sin(yaw)
            corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
            corners_rot = corners @ np.array([[c, -s], [s, c]]).T + np.array([x, y])
            plt.gca().add_patch(Polygon(corners_rot, fill=False, edgecolor='blue', linewidth=2.0))
        plt.scatter([], [], c='blue', marker='s', label='Pred')

    plt.legend(loc='upper right')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)

    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = mmcv.cnn.fuse_conv_bn(model)
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = model.cuda()
    model.eval()
    
    if not os.path.exists(args.show_dir):
        os.makedirs(args.show_dir)
        
    print(f'\nStart testing. Saving visualizations to {args.show_dir}...')
    
    img_norm_cfg = dict(mean=np.array([103.530, 116.280, 123.675]), std=np.array([57.375, 57.120, 58.395]), to_rgb=False)
    prog_bar = mmcv.ProgressBar(min(args.limit, len(dataset)))
    
    for i, data in enumerate(data_loader):
        if i >= args.limit: break
            
        with torch.no_grad():
            data = scatter(data, [args.gpu_id])[0]
            
            # --- Force the double-nested wrapping the model expects ---
            
            # 1. img_metas must be [[dict]]
            if 'img_metas' in data and isinstance(data['img_metas'], list):
                if len(data['img_metas']) > 0 and isinstance(data['img_metas'][0], dict):
                    data['img_metas'] = [data['img_metas']]

            # 2. img must be [Tensor]
            if 'img' in data and isinstance(data['img'], torch.Tensor):
                data['img'] = [data['img']]
            
            # 3. points must be [[Tensor]] so that points[0] is a [Tensor]
            #    that spconv can iterate over
            if 'points' in data:
                pts = data['points']
                if isinstance(pts, torch.Tensor):
                    # A bare Tensor is wrapped as [[Tensor]]
                    data['points'] = [[pts]]
                elif isinstance(pts, list):
                    # An existing [Tensor] is wrapped as [[Tensor]]
                    if len(pts) > 0 and isinstance(pts[0], torch.Tensor):
                        data['points'] = [pts]

            # 4. Capture the ground truth
            gt_bboxes = None
            if 'gt_bboxes_3d' in data:
                gt_list = data.pop('gt_bboxes_3d')
                if isinstance(gt_list, list):
                    gt_bboxes = gt_list[0]
                else:
                    gt_bboxes = gt_list

            if 'gt_labels_3d' in data:
                data.pop('gt_labels_3d')

            # Run inference
            result = model(return_loss=False, rescale=True, **data)
            
        # Parse the predictions
        if isinstance(result, list):
            pred_bboxes = result[0]['pts_bbox']['boxes_3d']
            pred_scores = result[0]['pts_bbox']['scores_3d']
        else:
            pred_bboxes = result['pts_bbox']['boxes_3d']
            pred_scores = result['pts_bbox']['scores_3d']
        
        mask = pred_scores > args.score_thr
        pred_bboxes = pred_bboxes[mask]
        
        # Prepare the data for plotting. points was wrapped as [[Tensor]] above,
        # so the tensor itself is data['points'][0][0].
        points = data['points'][0][0]
        
        img_metas = data['img_metas'][0][0] 
        lidar2imgs = img_metas['lidar2img']

        # A. BEV
        pc_range = cfg.point_cloud_range
        draw_lidar_bev(
            points, gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, 
            out_path=os.path.join(args.show_dir, f'sample_{i}_BEV.png'),
            x_range=(pc_range[0], pc_range[3]),
            y_range=(pc_range[1], pc_range[4])
        )

        # B. 2D Projection
        imgs_tensor = data['img'][0] 
        if imgs_tensor.dim() == 5:
            imgs_tensor = imgs_tensor.squeeze(0)
            
        for cam_idx in range(len(imgs_tensor)):
            img_t = imgs_tensor[cam_idx]
            img_np = img_t.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * img_norm_cfg['std'] + img_norm_cfg['mean'])
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_vis = img_np.copy()
            
            matrix = lidar2imgs[cam_idx]
            vis_img = draw_lidar_bbox3d_on_img(pred_bboxes, img_vis, matrix, color=(255, 0, 0)) # Blue
            if gt_bboxes is not None:
                vis_img = draw_lidar_bbox3d_on_img(gt_bboxes, vis_img, matrix, color=(0, 255, 0)) # Green
            
            base_name = os.path.basename(img_metas['filename'][cam_idx])
            cv2.imwrite(os.path.join(args.show_dir, f'sample_{i}_cam_{cam_idx}_aug_{base_name}'), vis_img)
        
        prog_bar.update()

    print(f'\nDone! Check folder: {args.show_dir}')

if __name__ == '__main__':
    main()