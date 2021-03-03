''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-25
@LastEditors: Please set LastEditors
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
        self.enable_finetune = False
    
    def initialize_network_model(self, weight_path, dataset, finetune):
        """initialize network and load pretrained model

        Args:
            weight_path (str): directory stores pretrained models
                - **pose_encoder.pth**: encoder model; 
                - **pose.pth**: pose decoder model
            dataset (str): dataset setup
            finetune (bool): finetune model on the run if True
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

        # loading pretrained model (pose-decoder)
        pose_decoder_path = os.path.join(weight_path, 'pose.pth')
        loaded_dict = torch.load(pose_decoder_path, map_location=device)
        self.pose_decoder.load_state_dict(loaded_dict)
        self.pose_decoder.to(device)

        # concatenate encoders and decoders
        self.model = torch.nn.Sequential(self.encoder, self.pose_decoder)

        if finetune:
            self.encoder.train()
            self.pose_decoder.train()
        else:
            self.encoder.eval()
            self.pose_decoder.eval()
        
        # image size
        self.feed_height = 192
        self.feed_width = 640

        # dataset parameters
        if 'kitti' in dataset:
            self.stereo_baseline_multiplier = 5.4
        elif 'tum' in dataset:
            self.stereo_baseline_multiplier = 1.
        elif 'robotcar' in dataset:
            self.stereo_baseline_multiplier = 5.4
        else:
            self.stereo_baseline_multiplier = 1.

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

        return pose

    def inference_pose(self, img):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair
        
        Returns:
            pose (tensor, [Nx4x4]): relative pose from img2 to img1
        """
        if self.enable_finetune:
            predictions = self.inference(img)
        else:
            predictions = self.inference_no_grad(img)
        self.pred_pose = predictions

        # summarize pose predictions for DF-VO
        pose = self.pred_pose[:1].clone()
        pose[:, :3, 3] *= self.stereo_baseline_multiplier # monodepth2 assumes 0.1 unit baseline
        return pose
