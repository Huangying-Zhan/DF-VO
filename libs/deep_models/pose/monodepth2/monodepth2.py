''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-02
@LastEditors: Huangying Zhan
@Description: This is the interface for Monodepth2 pose network
'''

import numpy as np
import os
import PIL.Image as pil
import torch
from torchvision import transforms

from .pose_decoder import PoseDecoder
from ..deep_pose import DeepPose
from libs.deep_models.depth.monodepth2.resnet_encoder import ResnetEncoder
from libs.deep_models.depth.monodepth2.layers import transformation_from_parameters



class Monodepth2PoseNet(DeepPose):
    """This is the interface for Monodepth2 pose network
    """
    def __init__(self, *args, **kwargs):
        super(Monodepth2PoseNet, self).__init__(*args, **kwargs)

        # Online finetuning configuration
        if self.pose_cfg is not None:
            self.finetune = self.pose_cfg.online_finetune.enable
        else:
            self.finetune = False
    
    def initialize_network_model(self, weight_path, height, width, dataset='kitti'):
        """initialize network and load pretrained model

        Args:
            weight_path (str): directory stores pretrained models
                - **pose_encoder.pth**: encoder model; 
                - **pose.pth**: pose decoder model
            height (int): image height
            width (int): image width
            dataset (str): dataset setup
        """
        device = torch.device('cuda')

        # initilize network
        self.encoder = ResnetEncoder(18, False, 2)
        self.pose_decoder = PoseDecoder(
                self.encoder.num_ch_enc, 1, 2)

        print("==> Initialize Pose-CNN with [{}]".format(weight_path))
        # loading pretrained model (encoder)
        encoder_path = os.path.join(weight_path, 'pose_encoder.pth')
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()

        # loading pretrained model (pose-decoder)
        pose_decoder_path = os.path.join(weight_path, 'pose.pth')
        loaded_dict = torch.load(pose_decoder_path, map_location=device)
        self.pose_decoder.load_state_dict(loaded_dict)
        self.pose_decoder.to(device)
        self.pose_decoder.eval()

        # image size
        self.feed_height = height
        self.feed_width = width

        # dataset parameters
        if 'kitti' in dataset:
            self.stereo_baseline = 5.4
        elif dataset == 'tum':
            self.stereo_baseline = 1.
        else:
            self.stereo_baseline = 5.4

    @torch.no_grad()
    def inference_no_grad(self, imgs):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair
        
        Returns:
            pose (tensor, [Nx4x4]): relative pose from img2 to img1
        """
        return self.inference(imgs)

    def inference(self, imgs):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair
        
        Returns:
            pose (tensor, [Nx4x4]): relative pose from img2 to img1
        """
        features = self.encoder(imgs)
        axisangle, translation = self.pose_decoder([features])
        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        pose[:, :3, 3] = self.stereo_baseline * pose[:, :3, 3]

        return pose
