# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified for EDL (Evidential Deep Learning) Support
# Verified against original MEFormer implementation
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from mmcv.runner import BaseModule
from mmdet.models import HEADS


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C // groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)


@HEADS.register_module()
class EDLNIGHead(BaseModule):
    """SeparateHead for CenterHead with EDL support.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        groups (int): Number of groups (Decoder layers).
        head_conv (int): Output channels. Default: 64.
        final_kernel (int): Kernel size for the last conv layer. Default: 1.
        init_bias (float): Initial bias. Default: -2.19.
        init_cfg (dict): Config of initialization.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 groups=1,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        # [Config safety] reject an externally supplied init_cfg so it cannot
        # interfere with the manual initialisation below.
        assert init_cfg is None, 'To prevent abnormal initialization behavior, init_cfg is not allowed to be set'
        super(EDLNIGHead, self).__init__(init_cfg=init_cfg)
        
        self.heads = heads
        self.groups = groups
        self.init_bias = init_bias

        for head in self.heads:
            # [Assumption] `classes` is already the expanded channel count
            # (e.g. center = 2 * 4 = 8).
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv1d(
                        c_in * groups,
                        head_conv * groups,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        groups=groups,
                        bias=False),
                    # Note: GroupLayerNorm1d must be available here; otherwise use BN or LN.
                    GroupLayerNorm1d(head_conv * groups, groups=groups),
                    nn.ReLU(inplace=True)
                ])
                c_in = head_conv

            # Final Prediction Layer
            conv_layers.append(
                nn.Conv1d(
                    head_conv * groups,
                    classes * groups, 
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    groups=groups,
                    bias=True))
            
            conv_layers = nn.Sequential(*conv_layers)
            self.__setattr__(head, conv_layers)

        # [CRITICAL FIX] run the initialisation explicitly rather than relying
        # on the mmcv runner to do it.
        self._force_init_edl_weights()

    def init_weights(self):
        """Default init_weights is bypassed by _force_init_edl_weights, 
        but kept for compatibility if called explicitly."""
        super().init_weights()
        self._force_init_edl_weights()

    def _force_init_edl_weights(self):
        """
        Force Initialize weights to ensure stable gradients at the beginning.
        This sets Evidence (v) to ~10.0 to prevent 1/v explosion.
        
        Note: For NIG distribution, we need:
        - v > 0 (evidence/precision)
        - alpha > 1 (shape parameter, for valid variance)
        - beta > 0 (scale parameter)
        
        Initial bias values are set such that after Softplus activation:
        - v ≈ softplus(10) + ε ≈ 10
        - alpha ≈ softplus(2) + 1 ≈ 3.1 (> 1 required)
        - beta ≈ softplus(1) + ε ≈ 1.3
        """
        # Initial values. These are bias values and pass through Softplus.
        init_v = 10.0      # softplus(10) ≈ 10
        init_alpha = 2.0   # softplus(2) + 1 ≈ 3.1 (ensures > 1)
        init_beta = 1.0    # softplus(1) ≈ 1.3
        
        # Print on rank 0 only, so distributed training does not repeat the line.
        # Note: loading a checkpoint overrides these initial values.
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            print(f"[EDL-Info] Initializing EDL Parameters (v={init_v}, α={init_alpha}, β={init_beta}) - will be overridden if loading checkpoint")

        with torch.no_grad():
            for head_name in self.heads:
                layer = self.__getattr__(head_name)
                # Take the final Conv1d layer
                final_conv = layer[-1]
                
                if head_name in ['center', 'height', 'reg']:  # dim reverted to plain regression; vel also uses plain regression
                    # These heads emit [Gamma, Nu, Alpha, Beta] interleaved
                    if final_conv.bias is not None:
                        # 1. Reset every bias to 0 (gamma starts at 0, i.e. 0.5 after sigmoid)
                        final_conv.bias.data.fill_(0.0)
                        
                        # 2. Set the biases for nu, alpha and beta.
                        # dim now uses the same sigmoid normalisation as
                        # center/height. With C output channels, they are
                        # grouped four at a time.
                        # Nu (Index 1, 5, 9...)
                        final_conv.bias.data[1::4] = init_v 
                        # Alpha (Index 2, 6, 10...)
                        final_conv.bias.data[2::4] = init_alpha
                        # Beta (Index 3, 7, 11...)
                        final_conv.bias.data[3::4] = init_beta
                        
                elif head_name == 'cls_logits':
                    # [EDL CLS] Beta distribution initialisation.
                    # Output layout: [alpha_evidence_1, beta_evidence_1, alpha_2, beta_2, ...]
                    # Init: alpha = beta ~= 1 (uniform prior) -> softplus(0) + 1 = 2.
                    # Background classes want beta > alpha, so beta_evidence starts higher.
                    if final_conv.bias is not None:
                        # Fill everything with 0 first (softplus(0) + 1 = 2, close to uniform)
                        final_conv.bias.data.fill_(0.0)
                        # alpha evidence (indices 0, 2, 4, ...) stays at 0.
                        # beta evidence (indices 1, 3, 5, ...) is raised slightly
                        # so the initial probability is low.
                        final_conv.bias.data[1::2] = 1.0  # softplus(1)+1 ≈ 2.3
                        
    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map from Transformer Decoder.
            Shape: [N, B, query, C] (Layers, Batch, Query, Channels)
        """
        N, B, query_num, c1 = x.shape
        # [Pre-process]: Flatten Layer and Batch for efficient Conv1d
        x = rearrange(x, "n b q c -> b (n c) q")

        ret_dict = dict()

        for head in self.heads:
            # Forward Pass through Conv Layers
            # Input: [B, N*C_in, Q] -> Output: [B, N*C_out, Q]
            head_output = self.__getattr__(head)(x)
            
            # 2. Restore Structure
            # [B, N*C_out, Q] -> [N, B, Q, C_out]
            head_output = rearrange(head_output, "b (n c) q -> n b q c", n=N)

            # -------------------------------------------------------
            # [Case 1] Classification: Standard Logits (NO EDL)
            # Output raw logits, to be used with Focal Loss + sigmoid
            # -------------------------------------------------------
            if head == 'cls_logits':
                # [REMOVED EDL] the Beta distribution is no longer used; raw
                # logits are emitted, matching the upstream SeparateTaskHead.
                ret_dict[head] = head_output

            # -------------------------------------------------------
            # [Case 2] Standard Regression (NO EDL): rot, vel, dim
            # -------------------------------------------------------
            elif head in ['rot', 'vel', 'dim']:  # dim, vel, rot use standard regression
                # dim: object size does not change under sensor degradation, so
                #      EDL uncertainty adds nothing.
                # rot: sin/cos are geometrically constrained, and NIG's
                #      independent-Gaussian assumption would break that.
                # vel: velocity uses plain regression.
                ret_dict[head] = head_output

            # -------------------------------------------------------
            # [Case 3] EDL Regression (Center, Size, Velocity)
            # -------------------------------------------------------
            elif head in ['center', 'height', 'reg']:  # EDL heads only
                # Assumes C_out = metric_dim * 4 (gamma, nu, alpha, beta),
                # e.g. center: 2 * 4 = 8 channels.
                
                # Reshape to separate NIG params
                # Shape: [N, B, Q, Metric_Dim, 4]
                head_output_reshaped = head_output.view(N, B, query_num, -1, 4)
                
                # Unpack the NIG parameters
                gamma_logits = head_output_reshaped[..., 0] 
                v_raw = head_output_reshaped[..., 1]
                alpha_raw = head_output_reshaped[..., 2]
                beta_raw = head_output_reshaped[..., 3]
                
                # Apply Activations constraints
                # [IMPORTANT] These are activated here, Loss function should NOT re-activate
                epsilon = 1e-6
                
                v = F.softplus(v_raw) + epsilon
                alpha = F.softplus(alpha_raw) + 1.0 + epsilon # Must > 1
                beta = F.softplus(beta_raw) + epsilon        # Must > 0

                # Re-pack and Flatten: Result is [Gamma, Nu, Alpha, Beta] interleaved
                # Shape: [N, B, Q, Metric_Dim * 4]
                nig_output = torch.stack([gamma_logits, v, alpha, beta], dim=-1).flatten(-2, -1)
                
                ret_dict[head] = nig_output

            else:
                # Fallback for any other heads
                ret_dict[head] = head_output

        return ret_dict