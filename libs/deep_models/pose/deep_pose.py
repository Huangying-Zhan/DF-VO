''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-04
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep pose network interface
'''

import torch


class DeepPose():
    """DeepPose is the Base class for deep pose network interface
    """
    def __init__(self):
        # Basic configuration
        self.device = torch.device('cuda')
        self.enable_finetune = False

# ========================== Methods need to be implemented =======================
    def initialize_network_model(self, weight_path):
        """initialize network and load pretrained model

        Args:
            weight_path (str): directory stores pretrained models
                - **pose_encoder.pth**: encoder model; 
                - **pose.pth**: pose decoder model
            dataset (str): dataset setup
        """
        raise NotImplementedError

    def inference(self, imgs):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair
        
        Returns:
            pose (tensor, [Nx4x4]): relative pose from img2 to img1
        """
        raise NotImplementedError

    def inference_pose(self, img):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair
        
        Returns:
            pose (tensor, [Nx4x4]): relative pose from img2 to img1
        """
        raise NotImplementedError
# =================================================================================

    @torch.no_grad()
    def inference_no_grad(self, imgs):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair
        
        Returns:
            pose (tensor, [Nx4x4]): relative pose from img2 to img1
        """
        return self.inference(imgs)
    
    def setup_train(self, deep_model, cfg):
        """Setup training configurations for online finetuning depth network

        Args:
            deep_model (DeepModel): DeepModel interface object
            cfg (edict): configuration dictionary for depth finetuning
        """
        self.enable_finetune = cfg.enable
        
        # train parameter 
        deep_model.parameters_to_train += list(self.model.parameters())
    