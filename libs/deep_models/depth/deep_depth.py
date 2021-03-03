''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-23
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep depth network interface
'''

import torch
import torch.nn.functional as nnFunc

from libs.deep_models.depth.monodepth2.layers import SSIM, get_smooth_loss, BackprojectDepth, disp_to_depth
from libs.geometry.reprojection import Reprojection

class DeepDepth():
    """This is the Base class for deep depth network interface
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
        self.device = torch.device('cuda')
        self.enable_finetune = False
        self.depth_scales = [0]
    
# ========================== Methods need to be implemented =======================

    def initialize_network_model(self, weight_path, dataset, finetune):
        """initialize network and load pretrained model
        
        Args:
            weight_path (str): a directory stores the pretrained models.
                - **encoder.pth**: encoder model
                - **depth.pth**: depth decoder model
            dataset (str): dataset setup for min/max depth [kitti, tum]
            finetune (bool): finetune model on the run if True
        """
        raise NotImplementedError
    
    def inference(self, img):
        """Depth prediction

        Args:
            img (tensor, [Nx3HxW]): image 

        Returns:
            a dictionary containing depths and disparities at different scales, resized back to input scale

                - **depth** (dict): depth predictions, each element is **scale-N** (tensor, [Nx1xHxW]): depth predictions at scale-N
                - **disp** (dict): disparity predictions, each element is **scale-N** (tensor, [Nx1xHxW]): disparity predictions at scale-N
        """
        raise NotImplementedError

    def inference_depth(self, img):
        """Depth prediction

        Args:
            img (tensor, [Nx3HxW]): image 

        Returns:
            depth (tensor, [Nx1xHxW]): depth prediction at highest resolution
        """
        raise NotImplementedError
# =================================================================================

    @torch.no_grad()
    def inference_no_grad(self, img):
        """Depth prediction

        Args:
            img (tensor, [Nx3HxW]): image 

        Returns:
            a dictionary containing depths at different scales, resized back to input scale
                - **scale-N** (tensor, [Nx1xHxW]): depth predictions at scale-N
        """
        return self.inference(img)

    def setup_train(self, deep_model, cfg):
        """Setup training configurations for online finetuning depth network

        Args:
            deep_model (DeepModel): DeepModel interface object
            cfg (edict): configuration dictionary for depth finetuning
        """
        self.enable_finetune = cfg.enable
        
        # basic configuration
        self.frame_ids = [0, 1]
        
        # scale
        self.depth_scales = cfg.scales
        self.num_depth_scale = len(self.depth_scales)

        # train parameter 
        deep_model.parameters_to_train += list(self.model.parameters())

        # Layer setup
        self.ssim = SSIM()
        self.ssim.to(self.device)
        self.reproj = Reprojection(self.height, self.width).to(self.device)
        self.backproject_depth = BackprojectDepth(1, self.height, self.width)
        self.backproject_depth.to(self.device)

        # loss
        self.apperance_loss = cfg.loss.apperance_loss
        self.disparity_smoothness = cfg.loss.disparity_smoothness
        self.depth_consistency = cfg.loss.depth_consistency

    def train(self, inputs, outputs):
        """Forward operations and compute losses including
            - photometric loss
            - disparity smoothness loss
        
        Args:
            inputs (dict): a dictionary containing 

                - **('color', 0, 0)** (tensor, [1x3xHxW]): image 0 at scale-0
                - **('color', 1, 0)** (tensor, [1x3xHxW]): image 1 at scale-0
                - **(K, 0)** (tensor, [1x4x4]): camera intrinsics at scale-0
                - **(inv_K, 0)** (tensor, [1x4x4]): inverse camera intrinsics at scale-0
            
            outputs (dict): a dictionary containing intermediate data including

                - **('depth', 1, s)** (tensor, [Nx1xHxW]) : predicted depth of frame-1 at scale-s
                - **('pose_T', 1, 0)** (tensor, [1x4x4]): relative pose from view-1 to view-0
        
        Returns:
            losses (dict): a dictionary containing depth losses
                - **depth_loss** (tensor): total depth loss
                - **depth_loss/s** (tensor): depth loss at scale-s
        """
        losses = {}
        # reprojection
        outputs.update(self.reprojection(inputs, outputs))

        # image warping 
        outputs.update(self.warp_image(inputs, outputs))

        # apperance loss (multiview)
        losses.update(self.compute_depth_loss(inputs, outputs))
        if self.depth_consistency != 0:
            losses.update(self.compute_depth_consistency_losses(inputs, outputs))
        
        return losses

    def reprojection(self, inputs, outputs):
        """Given a depth map, camera intrinsics and extrinsics, 
        find the reprojection of the pixels
        new keys: ('reproj_xy', 1, frame_id, s): reprojection from frame-1 to frame_id at scale-s
        """
        K = inputs[('K', 0)]
        inv_K = inputs[('inv_K', 0)]
        for s in self.depth_scales:
            for frame_id in self.frame_ids[:1]:
                depth = outputs[('depth', 1, s)]
                
                # reprojection
                T = outputs[('pose_T', 1, frame_id)] # from frame-1 to frame_id
                outputs[('reproj_xy', 1, frame_id, s)] = self.reproj(depth, T, K, inv_K)
        return outputs

    def warp_image(self, inputs, outputs):
        """Generate warped images on different depth-scale 
        i.e. depths with different scale but images are with scale-0
        new keys: ('warp_img', 1, frame_id, s): image warped from frame_id to frame-1
        """
        for s in self.depth_scales:
            for frame_id in self.frame_ids[:1]:
                # image warping
                img_ref = inputs[("color", frame_id, 0)]
                outputs[('warp_img', 1, frame_id, s)] = nnFunc.grid_sample(
                                    img_ref, outputs[('reproj_xy', 1, frame_id, s)], 
                                    mode='bilinear', padding_mode='border')
        
        return outputs

    def compute_depth_loss(self, inputs, outputs):
        """compute depth loss (photometric + SSIM + disparity smoothness)
        """
        losses = {}
        total_loss = 0

        source_scale = 0
        target = inputs[("color", 1, source_scale)]
        for scale in self.depth_scales:
            loss = 0
            reprojection_losses = []

            """ reprojection loss """
            for frame_id in self.frame_ids[:1]:
                pred = outputs[("warp_img", 1, frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[:1]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, _ = torch.min(combined, dim=1)

            loss += to_optimise.mean() * self.apperance_loss

            """ disparity smoothness loss """
            disp = outputs[('disp', 1, scale)]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, target)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["reproj_sm_loss/{}".format(scale)] = loss

        total_loss /= self.num_depth_scale
        losses["reproj_sm_loss"] = total_loss
        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_depth_consistency_losses(self, inputs, outputs):
        """Compute temporal depth consistency loss
        """
        total_loss = 0
        losses = {}
        source_scale = 0
        for scale in self.depth_scales:
            loss = 0
            depth_losses = []

            # Get depth and 3D points of frame_0
            ref_depth = outputs[("depth", 1, scale)]
            cam_points = self.backproject_depth(
                    ref_depth, inputs[("inv_K", source_scale)])

            for i, frame_id in enumerate(self.frame_ids[:1]):
                if frame_id != "s":
                    # Get relative pose
                    T = outputs[("pose_T", 1, frame_id)]

                    # Warp depth
                    warp_disp = nnFunc.grid_sample(
                        outputs[("disp",  frame_id, scale)],
                        outputs[("reproj_xy", 1, frame_id, scale)],
                        padding_mode="border")
                    _, warp_depth = disp_to_depth(warp_disp[:, 0], 0.1, 100)
                    
                    # Reproject ref_depth
                    transformed_cam_points = torch.matmul(T[:, :3, :], cam_points)
                    transformed_cam_points = transformed_cam_points.view(1, 3, self.height, self.width)
                    proj_depth = transformed_cam_points[:, 2, :, :]

                    # Compute depth difference
                    depth_diff = (1/(proj_depth+1e-6) - 1/(warp_depth+1e-6)).abs()
                    depth_losses.append(depth_diff)

            # Compute depth loss (min)
            combined = torch.cat(depth_losses, 1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            
            loss += to_optimise.mean()
            total_loss += loss * self.depth_consistency
            losses["depth_consistency_loss/{}".format(scale)] = loss
        losses["depth_consistency_loss"] = total_loss
        return losses
