''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-25
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep stereo network interface
'''

import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from libs.deep_models.depth.monodepth2.layers import DispToPix, SSIM, get_smooth_loss


class DeepStereo():
    """DeepStereo is the Base class for deep stereo matching network interface
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
        self.stereo_scales = [0]
        
        # Layer setup
        self.disp_to_pix = DispToPix(self.batch_size, self.height, self.width) 
        self.disp_to_pix.to(self.device)

# ========================== Methods need to be implemented =======================

    def initialize_network_model(self, weight_path, finetune):
        """initialize stereo_net model with weight_path
        
        Args:
            weight_path (str): weight path
            finetune (bool): finetune model on the run if True
        """
        raise NotImplementedError

    def inference_stereo(self, 
                    img1, img2,
                    forward_backward=False,
                    dataset="kitti"):
        """Estimate disparity (1->2) and compute stereo consistency
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1
            img2 (tensor [Nx3xHxW]): image 2
            foward_backward (bool): forward-backward disparity consistency is used if True
            dataset (str): dataset type
        
        Returns:
            a dictionary containing
                - **forward** (tensor, [Nx2xHxW]) : forward disparity (left to right)
                - **backward** (tensor, [Nx2xHxW]) : backward disparity (right to left)
                - **disp_diff** (tensor, [NxHxWx1]) : foward-backward disparity inconsistency
        """
        raise NotImplementedError

    def inference(self, img1, img2):
        """Predict disparity for the given pairs
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (tensor, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            a dictionary containing disparity at different scales, resized back to input scale 
                - **scale-N** (tensor, [Nx1xHxW]): disparity from img1 to img2 at scale level-N
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

    def resize_dense_disp(self, disp, des_height, des_width):
        """Resized disparity map with scaling
        
        Args:
            disp (tensor, [Nx1xHxW]): disparity map
            des_height (int): destinated height
            des_widht (int): destinated width

        Returns:
            disp (tensor, [Nx1xH'xW']): resized disp
        """
        # get height, width ratio
        ratio_height = float(des_height / disp.size(2))
        ratio_width = float(des_width / disp.size(3))

        # interpolation
        disp = F.interpolate(
            disp, (des_height, des_width), mode='bilinear', align_corners=True)
        
        disp = disp * ratio_width
        return disp

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

    def forward_backward_consistency(self, disp1, disp2, px1on2):
        """Compute disparity consistency map

        Args:
            disp1 (tensor, [Nx1xHxW]): disparity map 1
            disp2 (tensor, [Nx1xHxW]): disparity map 2
            px1on2 (tensor, [NxHxWx2]): projected pixel of view 1 on view 2
        
        Returns:
            disp_diff (tensor, [Nx1xHxW]): disparity inconsistency error map
        """
        # Warp disp2 to disp1
        warp_disp1 = F.grid_sample(disp2, px1on2)

        # Calculate disp difference
        disp_diff = (disp1 - warp_disp1).abs()

        return disp_diff
    
    def disps_to_depth(self, disps, fx, b, diff_thre, depth_range):
        """

        Args:
            - disps (dict): predicted disp data. disps[('l', 'r')] is disps from left to right.
            
                - **disps(l, r)** (array, [1xHxW]): disps from left to right
                - **disps(r, l)** (array, [1xHxW]): disps from left to right
                - **disps(l, r, diff)** (array, [1xHxW]): disp difference of left
            
            - fx (float): focal length in x-direction
            - b (float): stereo baseline
            - diff_thre (float): disparity difference threshold that will be invalid depths
            - depth_range (list): [min_depth, max_depth]
            
            Returns:
                - raw_depth (array, [HxW]): raw depths
                - depth (array, [HxW]): filtered depth
        """
        disp = -disps[('l', 'r')]

        # convert disparity to depth
        raw_depth = fx * b / (disp + 1e-6)
        depth = raw_depth.copy()

        # get mask
        thre_mask = disps[('l', 'r', 'diff')] > diff_thre
        range_mask = np.logical_or((depth < depth_range[0]), (depth > depth_range[1]))
        mask = np.logical_or(thre_mask, range_mask)
        depth[mask] = 0.

        return raw_depth[0], depth[0]



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
                    pix_coords = self.disp_to_pix(flow)
                    outputs[("sample_flow", 0, f_i, scale)] = pix_coords
                    outputs[("color_flow", 0, f_i, scale)] = F.grid_sample(
                        inputs[("color", f_i, source_scale)],
                        pix_coords,
                        padding_mode="border")
                    
                    if self.flow_forward_backward:
                        # Warp image using backward flow
                        flow = outputs[("flow", f_i, 0, scale)]
                        # flow = self.resize_dense_flow(flow, self.height, self.width) # have been resized in inference()
                        pix_coords = self.disp_to_pix(flow)
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
