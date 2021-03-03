''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: DeepModel initializes different deep networks and provide forward interfaces.
'''

import numpy as np
import os
import PIL.Image as pil
import torch
import torch.optim as optim
from torchvision import transforms

from .depth.monodepth2.monodepth2 import Monodepth2DepthNet
from .flow.lite_flow_net.lite_flow import LiteFlow
from .flow.hd3.hd3_flow import HD3Flow
from .pose.monodepth2.monodepth2 import Monodepth2PoseNet
from libs.deep_models.depth.monodepth2.layers import FlowToPix, PixToFlow, SSIM, get_smooth_loss
from libs.general.utils import mkdir_if_not_exists

class DeepModel():
    """DeepModel initializes different deep networks and provide forward interfaces.
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration dictionary
        """
        self.cfg = cfg
        self.finetune_cfg = self.cfg.online_finetune
        self.device = torch.device('cuda')

    def initialize_models(self):
        """intialize multiple deep models
        """

        ''' optical flow '''
        self.flow = self.initialize_deep_flow_model()

        ''' single-view depth '''
        if self.cfg.depth.depth_src is None:
            if self.cfg.depth.deep_depth.pretrained_model is not None:
                self.depth = self.initialize_deep_depth_model()
            else:
                assert False, "No precomputed depths nor pretrained depth model"
        
        ''' two-view pose '''
        if self.cfg.deep_pose.enable:
            if self.cfg.deep_pose.pretrained_model is not None:
                self.pose = self.initialize_deep_pose_model()
            else:
                assert False, "No pretrained pose model"

    def initialize_deep_flow_model(self):
        """Initialize optical flow network
        
        Returns:
            flow_net (nn.Module): optical flow network
        """
        if self.cfg.deep_flow.network == 'liteflow':
            flow_net = LiteFlow(self.cfg.image.height, self.cfg.image.width)
            enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.flow.enable
            flow_net.initialize_network_model(
                    weight_path=self.cfg.deep_flow.flow_net_weight,
                    finetune=enable_finetune,
                    )
        elif self.cfg.deep_flow.network == 'hd3':
            flow_net = HD3Flow(self.cfg.image.height, self.cfg.image.width)
            enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.flow.enable
            flow_net.initialize_network_model(
                    weight_path=self.cfg.deep_flow.flow_net_weight,
                    finetune=enable_finetune,
                    )
        else:
            assert False, "Invalid flow network [{}] is provided.".format(
                                self.cfg.deep_flow.network
                                )
        return flow_net

    def initialize_deep_depth_model(self):
        """Initialize single-view depth model

        Returns:
            depth_net (nn.Module): single-view depth network
        """
        if self.cfg.depth.deep_depth.network == 'monodepth2':
            depth_net = Monodepth2DepthNet(self.cfg.image.height, self.cfg.image.width)
            enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.depth.enable
            depth_net.initialize_network_model(
                    weight_path=self.cfg.depth.deep_depth.pretrained_model,
                    dataset=self.cfg.dataset,
                    finetune=enable_finetune)
        else:
            assert False, "Invalid depth network [{}] is provided.".format(
                                self.cfg.depth.deep_depth.network
                                )
        return depth_net
    
    def initialize_deep_pose_model(self):
        """Initialize two-view pose model

        Returns:
            pose_net (nn.Module): two-view pose network
        """
        pose_net = Monodepth2PoseNet()
        enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.pose.enable
        pose_net.initialize_network_model(
            weight_path=self.cfg.deep_pose.pretrained_model,
            dataset=self.cfg.dataset,
            finetune=enable_finetune,
            )
        return pose_net

    def setup_train(self):
        """Setup training configurations for online finetuning
        """
        # Basic configuration
        self.img_cnt = 0
        self.frame_ids = [0, 1]

        # Optimization
        self.learning_rate = self.finetune_cfg.lr
        self.parameters_to_train = []

        # flow train setup
        if self.finetune_cfg.flow.enable:
            self.flow.setup_train(self, self.finetune_cfg.flow)
        
        # depth train setup
        if self.finetune_cfg.depth.enable:
            self.depth.setup_train(self, self.finetune_cfg.depth)
        
        # pose train setup
        if self.finetune_cfg.pose.enable:
            self.pose.setup_train(self, self.finetune_cfg.pose)
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)

    def forward_flow(self, in_cur_data, in_ref_data, forward_backward):
        """Optical flow network forward interface, a forward inference.

        Args:
            in_cur_data (dict): current data
            in_ref_data (dict): reference data
            forward_backward (bool): use forward-backward consistency if True
        
        Returns:
            flows (dict): predicted flow data. flows[(id1, id2)] is flows from id1 to id2.

                - **flows(id1, id2)** (array, 2xHxW): flows from id1 to id2
                - **flows(id2, id1)** (array, 2xHxW): flows from id2 to id1
                - **flows(id1, id2, 'diff)** (array, 1xHxW): flow difference of id1
        """
        # Preprocess image
        cur_imgs = np.transpose((in_cur_data['img'])/255, (2, 0, 1))
        ref_imgs = np.transpose((in_ref_data['img'])/255, (2, 0, 1))
        cur_imgs = torch.from_numpy(cur_imgs).unsqueeze(0).float().cuda()
        ref_imgs = torch.from_numpy(ref_imgs).unsqueeze(0).float().cuda()

        # Forward pass
        flows = {}

        # Flow inference
        batch_flows = self.flow.inference_flow(
                                img1=ref_imgs,
                                img2=cur_imgs,
                                forward_backward=forward_backward,
                                dataset=self.cfg.dataset)
        
        # Save flows at current view
        src_id = in_ref_data['id']
        tgt_id = in_cur_data['id']
        flows[(src_id, tgt_id)] = batch_flows['forward'].detach().cpu().numpy()[0]
        if forward_backward:
            flows[(tgt_id, src_id)] = batch_flows['backward'].detach().cpu().numpy()[0]
            flows[(src_id, tgt_id, "diff")] = batch_flows['flow_diff'].detach().cpu().numpy()[0]
        return flows

    def forward_depth(self, imgs):
        """Depth network forward interface, a forward inference.

        Args:
            imgs (list): list of images, each element is a [HxWx3] array

        Returns:
            depth (array, [HxW]): depth map of imgs[0]
        """
        # Preprocess
        img_tensor = []
        for img in imgs:
            input_image = pil.fromarray(img)
            input_image = input_image.resize((self.depth.feed_width, self.depth.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            img_tensor.append(input_image)
        img_tensor = torch.cat(img_tensor, 0)
        img_tensor = img_tensor.cuda()
        
        # Inference
        pred_depth = self.depth.inference_depth(img_tensor)
        depth = pred_depth.detach().cpu().numpy()[0,0] 
        return depth

    def forward_pose(self, imgs):
        """Depth network forward interface, a forward inference.

        Args:
            imgs (list): list of images, each element is a [HxWx3] array

        Returns:
            pose (array, [4x4]): relative pose from img2 to img1
        """
        # Preprocess
        img_tensor = []
        for img in imgs:
            input_image = pil.fromarray(img)
            input_image = input_image.resize((self.depth.feed_width, self.depth.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            img_tensor.append(input_image)
        img_tensor = torch.cat(img_tensor, 1)
        img_tensor = img_tensor.cuda()

        # Prediction
        pred_poses = self.pose.inference_pose(img_tensor)
        pose = pred_poses.detach().cpu().numpy()[0]
        return pose

    def finetune(self, img1, img2, pose, K, inv_K):
        """Finetuning deep models

        Args:
            img1 (array, [HxWx3]): image 1 (reference)
            img2 (array, [HxWx3]): image 2 (current)
            pose (array, [4x4]): relative pose from view-2 to view-1 (from DF-VO)
            K (array, [3x3]): camera intrinsics
            inv_K (array, [3x3]): inverse camera intrinsics
        """
        # preprocess data
        # images
        img1 = np.transpose((img1)/255, (2, 0, 1))
        img2 = np.transpose((img2)/255, (2, 0, 1))
        img1 = torch.from_numpy(img1).unsqueeze(0).float().cuda()
        img2 = torch.from_numpy(img2).unsqueeze(0).float().cuda()

        # camera intrinsics
        K44 = np.eye(4)
        K44[:3, :3] = K.copy()
        K = torch.from_numpy(K44).unsqueeze(0).float().cuda()
        K44[:3, :3] = inv_K.copy()
        inv_K = torch.from_numpy(K44).unsqueeze(0).float().cuda()

        # pose
        if self.finetune_cfg.depth.pose_src == 'DF-VO':
            pose = torch.from_numpy(pose).unsqueeze(0).float().cuda()
            pose[:, :3, 3] /= 5.4
        elif self.finetune_cfg.depth.pose_src == 'deep_pose':
            pose = self.pose.pred_pose
        elif self.finetune_cfg.depth.pose_src == 'DF-VO2':
            deep_pose_scale = torch.norm(self.pose.pred_pose[:, :3, 3].clone())
            pose = torch.from_numpy(pose).unsqueeze(0).float().cuda()
            pose[:, :3, 3] /= torch.norm(pose[:, :3, 3])
            pose[:, :3, 3] *= deep_pose_scale
        
        if self.finetune_cfg.num_frames is None or self.img_cnt < self.finetune_cfg.num_frames:
            ''' data preparation '''
            losses = {'loss': 0}
            inputs = {
                ('color', 0, 0): img1,
                ('color', 1, 0): img2,
                ('K', 0): K,
                ('inv_K', 0): inv_K,
            }
            outputs = {}

            ''' loss computation '''
            # flow
            if self.finetune_cfg.flow.enable:
                assert self.cfg.deep_flow.forward_backward, "forward-backward option has to be True for finetuning"
                for s in self.flow.flow_scales:
                    outputs.update(
                        {
                            ('flow', 0, 1, s):  self.flow.forward_flow[s],
                            ('flow', 1, 0, s):  self.flow.backward_flow[s],
                            ('flow_diff', 0, 1, s):  self.flow.flow_diff[s]
                        }
                    )

                losses.update(self.flow.train(inputs, outputs))
                losses["loss"] += losses["flow_loss"]
            
            # depth and pose
            if self.finetune_cfg.depth.enable:
                # add predicted depths
                for s in self.depth.depth_scales:
                    outputs.update(
                        {
                            ('depth', 1, s): self.depth.pred_depths[s][:1],
                            ('disp', 1, s): self.depth.pred_disps[s][:1],
                            ('depth', 0, s): self.depth.pred_depths[s][1:],
                            ('disp', 0, s): self.depth.pred_disps[s][1:]
                        }
                    )

                # add predicted poses
                outputs.update(
                    {
                        ('pose_T', 1, 0): pose
                    }
                )
                
                losses.update(self.depth.train(inputs, outputs))
                losses["loss"] += losses["reproj_sm_loss"]
                if self.depth.depth_consistency != 0:
                    losses["loss"] += losses["depth_consistency_loss"]
            
            ''' backward '''
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            
            self.img_cnt += 1

        else:
            # reset flow model to eval mode
            if self.finetune_cfg.flow.enable:
                self.flow.model.eval()
            
            # reset depth model to eval mode
            if self.finetune_cfg.depth.enable:
                self.depth.model.eval()
            
            # reset pose model to eval mode
            if self.finetune_cfg.pose.enable:
                self.pose.model.eval()

    def save_model(self):
        """Save deep models
        """
        save_folder = os.path.join(self.cfg.directory.result_dir, "deep_models", self.cfg.seq)
        mkdir_if_not_exists(save_folder)

        # Save Flow model
        model_name = "flow"
        model = self.flow.model
        ckpt_path = os.path.join(save_folder, "{}.pth".format(model_name))
        torch.save(model.state_dict(), ckpt_path)
