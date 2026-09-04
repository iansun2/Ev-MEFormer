# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified for EDL (Evidential Deep Learning) support with Debug Info
# ------------------------------------------------------------------------

import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class EDLBBoxCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        # [REMOVED] dim_log_range - removed; dim uses the raw output


    def encode(self):
        pass

    def decode(self, preds_dicts):
        """Decode bboxes with EDL support.
        Args:
            preds_dicts (list[list[dict]]): List of task predictions.
        """
        task_num = len(preds_dicts)
        
        pred_bbox_list = []
        pred_logits_list = []
        pred_admission_logits_list = []
        task_ids_list = []
        rv_box_mask_lists = []
        has_admission_logits = False

        # ----------------------------------------------------------
        # 1. Iterate over the tasks and gather their outputs
        # ----------------------------------------------------------
        for task_id in range(task_num):
            # Take this task's layer dict
            task_pred_dict = preds_dicts[task_id][0]

            # [REMOVED Beta EDL] use the raw logits directly. The Beta
            # distribution is gone, so cls_logits are now standard logits.
            task_pred_logits = task_pred_dict['cls_logits'][-1]

            pred_logits_list.append(task_pred_logits)

            task_admission_logits = task_pred_dict.get('v10_edlcue_admission_logits')
            if isinstance(task_admission_logits, (list, tuple)):
                task_admission_logits = task_admission_logits[0] if task_admission_logits else None
            if isinstance(task_admission_logits, torch.Tensor):
                if task_admission_logits.dim() == 4:
                    task_admission_logits = task_admission_logits[-1]
                if task_admission_logits.shape == task_pred_logits.shape:
                    pred_admission_logits_list.append(task_admission_logits)
                    has_admission_logits = True
                else:
                    pred_admission_logits_list.append(task_pred_logits)
            else:
                pred_admission_logits_list.append(task_pred_logits)

            # [EDL logic] handle the bbox (bbox_preds may be passed directly)
            if 'bbox_preds' in task_pred_dict:
                task_pred_bbox = task_pred_dict['bbox_preds'][-1]
            else:
                # Backwards-compatible path for the older split layout
                task_pred_bbox = [task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                                  task_pred_dict['dim'][-1], task_pred_dict['rot'][-1]]
                if 'vel' in task_pred_dict:
                    task_pred_bbox.append(task_pred_dict['vel'][-1])
                task_pred_bbox = torch.cat(task_pred_bbox, dim=-1)
            
            pred_bbox_list.append(task_pred_bbox)

            # Mask handling
            if "rv_box_mask" in task_pred_dict:
                rv_box_mask_lists.append(task_pred_dict["rv_box_mask"])
            else:
                # Create an all-True mask
                rv_box_mask_lists.append(task_pred_logits.new_ones(
                    task_pred_logits.shape[1], 6, task_pred_logits.shape[2]).to(torch.bool))

            # Task IDs
            task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
            task_ids_list.append(task_ids)

        # ----------------------------------------------------------
        # 2. Concatenate, keeping the upstream dimension conventions exactly
        # ----------------------------------------------------------
        # Logits: [BS, Q, Total_Cls] (dim=-1)
        all_pred_logits = torch.cat(pred_logits_list, dim=-1)  
        all_admission_logits = (
            torch.cat(pred_admission_logits_list, dim=-1)
            if has_admission_logits else None)
        
        # BBox: [BS, Task*Q, Code] (dim=1) -> note the different concat dim
        all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  
        
        # Task IDs: [BS, Q, Total_Cls] (dim=-1)
        all_task_ids = torch.cat(task_ids_list, dim=-1)  
        
        # Mask: [BS, 6, Task*Q] (dim=-1) -> the mask is assumed to follow the bbox
        all_rv_box_masks = torch.cat(rv_box_mask_lists, dim=-1)

        batch_size = all_pred_logits.shape[0]
        predictions_list = []
        
        for i in range(batch_size):
            # Build the mask
            rv_box_mask = all_rv_box_masks[i].sum(dim=0) != 0 
            
            # [Upstream compatibility] align the mask length with the bbox length
            if rv_box_mask.shape[0] != all_pred_bbox[i].shape[0]:
                # If the lengths disagree, do not use the mask to filter the
                # tensor here (that would risk an IndexError); fall back to an
                # all-open mask and let decode_single deal with it.
                box_mask = torch.ones(all_pred_bbox[i].shape[0], device=rv_box_mask.device).bool()
            else:
                box_mask = rv_box_mask

            # [Safety squeeze] drop the singleton dimensions
            c_score = all_pred_logits[i]
            if c_score.dim() == 3 and c_score.shape[0] == 1: c_score = c_score.squeeze(0)

            admission_score = all_admission_logits[i] if all_admission_logits is not None else None
            if (
                    isinstance(admission_score, torch.Tensor) and
                    admission_score.dim() == 3 and admission_score.shape[0] == 1):
                admission_score = admission_score.squeeze(0)
            
            b_pred = all_pred_bbox[i]
            if b_pred.dim() == 3 and b_pred.shape[0] == 1: b_pred = b_pred.squeeze(0)
            
            t_id = all_task_ids[i]
            if t_id.dim() == 2 and t_id.shape[0] == 1: t_id = t_id.squeeze(0)

            # Call decode_single
            predictions_list.append(self.decode_single(
                c_score, 
                b_pred, 
                t_id,
                admission_score
            ))
            
        return predictions_list

    def decode_single(self, cls_scores, bbox_preds, task_ids, admission_logits=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): [Q, Cls_Total]
            bbox_preds (Tensor): [Task*Q, Code]
            task_ids (Tensor): [Q, Cls_Total]
        """
        max_num = self.max_num
        num_query = cls_scores.shape[0]

        # 1. Sigmoid & TopK
        raw_scores = cls_scores.sigmoid()
        if (
                isinstance(admission_logits, torch.Tensor) and
                admission_logits.shape == cls_scores.shape):
            admission_scores = admission_logits.sigmoid()
        else:
            admission_scores = raw_scores
        scores, indexs = admission_scores.view(-1).topk(max_num)
        decode_topk_rank = torch.arange(
            scores.shape[0], device=scores.device, dtype=torch.long)
        raw_topk_scores = raw_scores.view(-1)[indexs]
        
        # 2. Resolve the indices
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes 
        
        # 3. Task Mapping
        task_index = task_ids[bbox_index.long(), labels.long()].long()
        final_bbox_indices = task_index * num_query + bbox_index
        
        # 4. Gather the matching bboxes (still NIG parameters at this point)
        final_bbox_indices = torch.clamp(final_bbox_indices, max=bbox_preds.shape[0]-1)
        selected_bbox_preds = bbox_preds[final_bbox_indices]
        
        # ----------------------------------------------------------
        # [EDL logic] NIG parsing.
        # Layout: center_nig(8) + height_nig(4) + dim(3) + rot(2) + vel(2) = 19
        # ----------------------------------------------------------
        is_nig = False
        uncertainty = None
        
        # The EDL output is 19-dimensional (NIG for center/height, plain for dim/rot/vel)
        if selected_bbox_preds.shape[-1] == 19:
            is_nig = True
            
            # Parse the NIG parameters according to the layout above
            ptr = 0
            gammas = []
            all_alphas = []
            all_betas = []
            all_vs = []
            
            # center: 8 -> gamma(2)
            center_nig = selected_bbox_preds[:, ptr:ptr+8].view(-1, 2, 4)
            gammas.append(center_nig[..., 0])
            all_vs.append(center_nig[..., 1])
            all_alphas.append(center_nig[..., 2])
            all_betas.append(center_nig[..., 3])
            ptr += 8
            
            # height: 4 -> gamma(1)
            height_nig = selected_bbox_preds[:, ptr:ptr+4].view(-1, 1, 4)
            gammas.append(height_nig[..., 0])
            all_vs.append(height_nig[..., 1])
            all_alphas.append(height_nig[..., 2])
            all_betas.append(height_nig[..., 3])
            ptr += 4
            
            # dim: 3 (plain regression, no NIG)
            dim = selected_bbox_preds[:, ptr:ptr+3]
            gammas.append(dim)
            ptr += 3
            
            # rot: 2 (plain, no NIG)
            rot = selected_bbox_preds[:, ptr:ptr+2]
            gammas.append(rot)
            ptr += 2
            
            # vel: 2 (plain, no NIG)
            vel = selected_bbox_preds[:, ptr:ptr+2]
            gammas.append(vel)
            ptr += 2
            
            # Assemble gamma: [N, 10].
            # Network output order: [center(2), height(1), dim(3), rot(2), vel(2)],
            # i.e. [cx, cy, cz, w, l, h, sin, cos, vx, vy] (the upstream format).
            gamma_raw = torch.cat(gammas, dim=-1)
            
            # -----------------------------------------------------------
            # [CRITICAL FIX 1] convert the normalized [0, 1] values to absolute /
            # log scale. This must match the conversion in edl_head.py's
            # _loss_single_task exactly.
            # -----------------------------------------------------------
            gamma_abs = gamma_raw.clone()
            pc_range = self.pc_range
            
            # center (0:2): normalized [0,1] -> absolute PC range
            gamma_abs[:, 0] = gamma_raw[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            gamma_abs[:, 1] = gamma_raw[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            
            # height cz (2:3): normalized [0,1] -> absolute PC range  
            gamma_abs[:, 2] = gamma_raw[:, 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
            
            # dim w, l, h (3:6): use the raw gamma directly -- the network
            # learns the log scale on its own, so no conversion is needed.
            
            # rot (6:8) and vel (8:10) are left unchanged
            
            # -----------------------------------------------------------
            # [CRITICAL FIX 2] reorder to the layout denormalize_bbox expects.
            # denormalize_bbox wants: [cx, cy, w, l, cz, h, sin, cos, vx, vy]
            # current layout:         [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            # so cz (index 2) and w, l (indices 3:5) swap places.
            # -----------------------------------------------------------
            gamma = torch.cat([
                gamma_abs[:, 0:2],   # cx, cy (unchanged)
                gamma_abs[:, 3:5],   # w, l (first two of dim, moved to index 2:4)
                gamma_abs[:, 2:3],   # cz (from height, moved to index 4:5)
                gamma_abs[:, 5:6],   # h (third of dim, moved to index 5:6)
                gamma_abs[:, 6:10],  # rot, vel (unchanged)
            ], dim=-1)

            
            # Compute the uncertainty over the NIG part only (center/height = 3 dims)
            v = torch.cat(all_vs, dim=-1)  # [N, 3]
            alpha = torch.cat(all_alphas, dim=-1)  # [N, 3]
            beta = torch.cat(all_betas, dim=-1)  # [N, 3]
            
            denom = torch.clamp(alpha - 1.0, min=1e-6)
            aleatoric = beta / denom
            epistemic = beta / (v.clamp(min=1e-6) * denom)
            
            uncertainty = dict(
                aleatoric=aleatoric.mean(dim=-1),
                epistemic=epistemic.mean(dim=-1)
            )
            
            selected_bbox_preds = gamma
        else:
            # -----------------------------------------------------------
            # [CRITICAL FIX] reordering for the 10-dimensional (non-NIG) path,
            # which uses the already-processed center/height/dim directly.
            # current layout:         [cx, cy, cz, w_log, l_log, h_log, sin, cos, vx, vy]
            # denormalize_bbox wants: [cx, cy, w_log, l_log, cz, h_log, sin, cos, vx, vy]
            # -----------------------------------------------------------
            selected_bbox_preds = torch.cat([
                selected_bbox_preds[:, 0:2],   # cx, cy (unchanged)
                selected_bbox_preds[:, 3:5],   # w_log, l_log (first two of dim, moved to index 2:4)
                selected_bbox_preds[:, 2:3],   # cz (from height, moved to index 4:5)
                selected_bbox_preds[:, 5:6],   # h_log (third of dim, moved to index 5:6)
                selected_bbox_preds[:, 6:10],  # rot, vel (unchanged)
            ], dim=-1)

        # 5. Denormalize
        boxes3d = denormalize_bbox(selected_bbox_preds, self.pc_range)

        # 6. Post-processing
        if self.score_threshold is not None:
            thresh_mask = scores > self.score_threshold
            
            if self.post_center_range is not None:
                self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
                mask = (boxes3d[..., :3] >= self.post_center_range[:3]).all(1)
                mask &= (boxes3d[..., :3] <= self.post_center_range[3:]).all(1)
                thresh_mask &= mask

            boxes3d = boxes3d[thresh_mask]
            scores = scores[thresh_mask]
            raw_topk_scores = raw_topk_scores[thresh_mask]
            labels = labels[thresh_mask]
            bbox_index = bbox_index[thresh_mask]
            task_index = task_index[thresh_mask]
            final_bbox_indices = final_bbox_indices[thresh_mask]
            decode_topk_rank = decode_topk_rank[thresh_mask]
            
            if is_nig:
                uncertainty['aleatoric'] = uncertainty['aleatoric'][thresh_mask]
                uncertainty['epistemic'] = uncertainty['epistemic'][thresh_mask]

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels,
            'raw_scores': raw_topk_scores,
            'admission_scores': scores,
            'decode_query_idx': bbox_index.long(),
            'decode_task_idx': task_index.long(),
            'decode_final_bbox_idx': final_bbox_indices.long(),
            'decode_topk_rank': decode_topk_rank.long(),
        }
        
        if is_nig:
            predictions_dict['uncertainty'] = uncertainty
            
        return predictions_dict
