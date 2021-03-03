''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-18
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep flow network interface
'''

import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from libs.deep_models.depth.monodepth2.layers import FlowToPix, SSIM, get_smooth_loss


class DeepFlow():
    """DeepFlow is the Base class for deep flow network interface
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        # Basic configuration
        self.height = height
        self.width = width
        self.batch_size = 1
        self.device = torch.device('cuda')
        self.enable_finetune = False
        self.flow_scales = [1]
        
        # Layer setup
        self.flow_to_pix = FlowToPix(self.batch_size, self.height, self.width) 
        self.flow_to_pix.to(self.device)

# ========================== Methods need to be implemented =======================

    def initialize_network_model(self, weight_path, finetune):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
            finetune (bool): finetune model on the run if True
        """
        raise NotImplementedError

    def inference_flow(self, 
                    img1, img2,
                    forward_backward=False,
                    dataset="kitti"):
        """Estimate flow (1->2) and compute flow consistency
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1
            img2 (tensor [Nx3xHxW]): image 2
            foward_backward (bool): forward-backward flow consistency is used if True
            dataset (str): dataset type
        
        Returns:
            a dictionary containing
                - **forward** (tensor, [Nx2xHxW]) : forward flow
                - **backward** (tensor, [Nx2xHxW]) : backward flow
                - **flow_diff** (tensor, [NxHxWx1]) : foward-backward flow inconsistency
        """
        raise NotImplementedError

    def inference(self, img1, img2):
        """Predict optical flow for the given pairs
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (tensor, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            a dictionary containing flows at different scales, resized back to input scale 
                - **scale-N** (tensor, [Nx2xHxW]): flow from img1 to img2 at scale level-N
        """
        raise NotImplementedError

# =================================================================================

    def get_target_size(self, h, w):
        """Get the closest size that is divisible by 32
        
        Args:
            h (int): given height
            w (int): given width
        
        Returns:
            a tuple containing
                - **h** (int) : target height
                - **w** (int) : target width
        """
        h = 32 * np.array([[math.floor(h / 32), math.floor(h / 32) + 1]])
        w = 32 * np.array([[math.floor(w / 32), math.floor(w / 32) + 1]])
        ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - h / w)
        index = np.argmin(ratio)
        return h[0, index // 2], w[0, index % 2]

    def resize_dense_flow(self, flow, des_height, des_width):
        """Resized flow map with scaling
        
        Args:
            flow (tensor, [Nx2xHxW]): flow map
            des_height (int): destinated height
            des_widht (int): destinated width

        Returns:
            flow (tensor, [Nx2xH'xW']): resized flow
        """
        # get height, width ratio
        ratio_height = float(des_height / flow.size(2))
        ratio_width = float(des_width / flow.size(3))

        # interpolation
        flow = F.interpolate(
            flow, (des_height, des_width), mode='bilinear', align_corners=True)
        
        flow = torch.stack(
            [flow[:, 0, :, :] * ratio_width, flow[:, 1, :, :] * ratio_height],
            dim=1)
        return flow

    # FIXME: move this function to dataset loader
    def load_flow_file(self, flow_path):
        """load flow data from a npy file

        Args:
            flow_path (str): flow data path, npy file
        
        Returns:
            flow (array, [HxWx2]): flow data
        """
        # Load flow
        flow = np.load(flow_path)

        # resize flow
        h, w, _ = flow.shape
        if self.width is None or self.height is None:
            resize_height = h
            resize_width = w
        else:
            resize_height = self.height
            resize_width = self.width
        flow = cv2.resize(flow, (resize_width, resize_height))
        flow[..., 0] *= resize_width / w
        flow[..., 1] *= resize_height / h
        return flow

    @torch.no_grad()
    def inference_no_grad(self, img1, img2):
        """Predict optical flow for the given pairs
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (tensor, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            a dictionary containing flows at different scales, resized back to input scale 
                - **scale-N** (tensor, [Nx2xHxW]): flow from img1 to img2 at scale level-N
        """
        return self.inference(img1, img2)

    def forward_backward_consistency(self, flow1, flow2, px1on2):
        """Compute flow consistency map

        Args:
            flow1 (tensor, [Nx2xHxW]): flow map 1
            flow2 (tensor, [Nx2xHxW]): flow map 2
            px1on2 (tensor, [NxHxWx2]): projected pixel of view 1 on view 2
        
        Returns:
            flow_diff (tensor, [NxHxWx1]): flow inconsistency error map
        """
        # Warp flow2 to flow1
        warp_flow1 = F.grid_sample(-flow2, px1on2)

        # Calculate flow difference
        flow_diff = (flow1 - warp_flow1)

        # TODO: UnFlow (Meister etal. 2017) constrain is not used
        UnFlow_constrain = False
        if UnFlow_constrain:
            flow_diff = (flow_diff ** 2 - 0.01 * (flow1**2 - warp_flow1 ** 2))

        # calculate norm and reshape
        flow_diff = flow_diff.norm(dim=1, keepdim=True)
        flow_diff = flow_diff.permute(0, 2, 3, 1)
        return flow_diff

    def setup_train(self, deep_model, cfg):
        """Setup training configurations for online finetuning flow network

        Args:
            deep_model (DeepModel): DeepModel interface object
            cfg (edict): configuration dictionary for flow finetuning
        """
        self.enable_finetune = cfg.enable
        
        # basic configuration
        self.frame_ids = [0, 1]
        
        # scale
        self.flow_scales = cfg.scales
        self.num_flow_scale = len(self.flow_scales)

        # train parameter 
        deep_model.parameters_to_train += list(self.model.parameters())

        # Layer setup
        self.ssim = SSIM()
        self.ssim.to(self.device)

        # loss
        self.flow_forward_backward = True
        self.flow_consistency = cfg.loss.flow_consistency
        self.flow_smoothness = cfg.loss.flow_smoothness

    def train(self, inputs, outputs):
        """Forward operations and compute flow losses including
            - photometric loss
            - flow smoothness loss
            - flow consistency loss
        
        Args:
            inputs (dict): a dictionary containing 
            
                - **('color', 0, 0)** (tensor, [1x3xHxW]): image 0 at scale-0
                - **('color', 1, 0)** (tensor, [1x3xHxW]): image 1 at scale-0
            
            outputs (dict): a dictionary containing intermediate data including

                - **('flow', 0, 1, s)** (tensor, [Nx2xHxW]) : forward flow at scale-s
                - **('flow', 1, 0, s)** (tensor, [Nx2xHxW]) : backward flow at scale-s
                - **('flow_diff', 0, 1, s)** (tensor, [NxHxWx1]) : foward-backward flow inconsistency at scale-s
        
        Returns:
            losses (dict): a dictionary containing flow losses
                - **flow_loss** (tensor): total flow loss
                - **flow_loss/s** (tensor): flow loss at scale-s
        """
        losses = {}
        outputs.update(self.generate_images_pred_flow(inputs, outputs))
        losses.update(self.compute_flow_losses(inputs, outputs))
        return losses

    def generate_images_pred_flow(self, inputs, outputs):
        """Generate the warped (reprojected) color images using optical flow for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        source_scale = 0
        for _, f_i in enumerate(self.frame_ids[1:]):
            if f_i != "s":
                for scale in self.flow_scales:
                    # Warp image using forward flow
                    flow = outputs[("flow", 0, f_i, scale)]
                    # flow = self.resize_dense_flow(flow, self.height, self.width) # have been resized in inference()
                    pix_coords = self.flow_to_pix(flow)
                    outputs[("sample_flow", 0, f_i, scale)] = pix_coords
                    outputs[("color_flow", 0, f_i, scale)] = F.grid_sample(
                        inputs[("color", f_i, source_scale)],
                        pix_coords,
                        padding_mode="border")
                    
                    if self.flow_forward_backward:
                        # Warp image using backward flow
                        flow = outputs[("flow", f_i, 0, scale)]
                        # flow = self.resize_dense_flow(flow, self.height, self.width) # have been resized in inference()
                        pix_coords = self.flow_to_pix(flow)
                        outputs[("sample_flow", f_i, 0, scale)] = pix_coords
                        outputs[("color_flow", f_i, 0, scale)] = F.grid_sample(
                            inputs[("color", 0, source_scale)],
                            pix_coords,
                            padding_mode="border") 
        return outputs

    def compute_flow_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        if forward_backward consistency is enabled:
            warp_flow and flow_diff are saved
        """
        losses = {}
        total_loss = 0

        source_scale = 0
        for scale in self.flow_scales:
            loss = 0
            reprojection_losses = []

            """ reprojection loss """
            for frame_id in self.frame_ids[1:]:
                if frame_id != "s":
                    target = inputs[("color", 0, source_scale)]
                    pred = outputs[("color_flow", 0, frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            combined = reprojection_losses

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, _ = torch.min(combined, dim=1)

            loss += to_optimise.mean()

            """ flow smoothness loss """
            for frame_id in self.frame_ids[1:]:
                if frame_id != "s":
                    flow = outputs[("flow", 0, frame_id, scale)].norm(dim=1, keepdim=True)
                    color = inputs[("color", 0, source_scale)]
                    mean_flow = flow.mean(2, True).mean(3, True)
                    norm_flow = flow / (mean_flow + 1e-7)
                    smooth_loss = get_smooth_loss(norm_flow, color)
                    loss += self.flow_smoothness * smooth_loss / (2 ** scale)
                    
                    if self.flow_forward_backward:
                        flow = outputs[("flow", frame_id, 0, scale)].norm(dim=1, keepdim=True)
                        color = inputs[("color", frame_id, source_scale)]
                        mean_flow = flow.mean(2, True).mean(3, True)
                        norm_flow = flow / (mean_flow + 1e-7)
                        smooth_loss = get_smooth_loss(norm_flow, color)
                        loss += self.flow_smoothness * smooth_loss / (2 ** scale)

            """ flow forward-backward consistency loss """
            # if self.flow_forward_backward:
            for frame_id in self.frame_ids[1:]:
                if frame_id != "s":
                    flow_consistency_loss = outputs[("flow_diff", 0, frame_id, scale)].mean()
                    loss += self.flow_consistency * flow_consistency_loss / (2 ** scale)
            total_loss += loss
            losses["flow_loss/{}".format(scale)] = loss

        total_loss /= self.num_flow_scale
        losses["flow_loss"] = total_loss
        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
