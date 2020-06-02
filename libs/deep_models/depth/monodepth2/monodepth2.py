''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-02
@LastEditors: Huangying Zhan
@Description: This is the interface for Monodepth2 depth network
'''

import numpy as np
import os
# import PIL.Image as pil
import sys
import torch
# from torchvision import transforms

from .depth_decoder import DepthDecoder
from .layers import disp_to_depth
from .resnet_encoder import ResnetEncoder
from ..deep_depth import DeepDepth


class Monodepth2DepthNet(DeepDepth):
    """This is the interface for Monodepth2 depth network
    """
    def __init__(self, *args, **kwargs):
        super(Monodepth2DepthNet, self).__init__(*args, **kwargs)

        self.scales = [0]
        
        # Online finetuning configuration
        if self.depth_cfg is not None:
            self.finetune = self.depth_cfg.online_finetune.enable
        else:
            self.finetune = False
        
    def initialize_network_model(self, weight_path, dataset='kitti'):
        """initialize network and load pretrained model
        
        Args:
            weight_path (str): a directory stores the pretrained models.
                - **encoder.pth**: encoder model
                - **depth.pth**: depth decoder model
            dataset (str): dataset setup for min/max depth [kitti, tum]
        """
        device = torch.device("cuda")

        # initilize network
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        print("==> Initialize Depth-CNN with [{}]".format(weight_path))
        # loading pretrained model (encoder)
        encoder_path = os.path.join(weight_path, 'encoder.pth')
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)

        # loading pretrained model (depth-decoder)
        depth_decoder_path = os.path.join(weight_path, 'depth.pth')
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(device)

        if self.finetune:
            self.encoder.train()
            self.depth_decoder.train()
            self.setup_train()
        else:
            self.encoder.eval()
            self.depth_decoder.eval()

        # image size
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']

        # dataset parameters
        if 'kitti' in dataset:
            self.min_depth = 0.1
            self.max_depth = 100
            self.stereo_baseline = 5.4
        elif 'tum' in dataset:
            self.min_depth = 0.1
            self.max_depth = 10
            self.stereo_baseline = 1
        else:
            self.min_depth = 0.1
            self.max_depth = 100
            self.stereo_baseline = 5.4

    def setup_train(self):
        """Setup training configurations for online finetuning
        """
        # Basic configuration
        self.img_cnt = 0
        self.frame_ids = [0, 1]

        if self.depth_cfg.online_finetune.scale == 'single':
            self.scales = [0]
        elif self.depth_cfg.online_finetune.scale == 'multi':
            self.scales = [0, 1, 2, 3]
        # self.num_flow_scale = len(self.flow_scales)

        # # Layer setup
        # self.ssim = SSIM()
        # self.ssim.to(self.device)
        
        # # Optimization
        # self.learning_rate = self.flow_cfg.online_finetune.lr
        # self.parameters_to_train = []
        # self.parameters_to_train += list(self.model.parameters())
        # self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)

        # # loss
        # self.flow_forward_backward = True
        # self.flow_consistency = self.flow_cfg.online_finetune.loss.flow_consistency
        # self.flow_smoothness = self.flow_cfg.online_finetune.loss.flow_smoothness

    @torch.no_grad()
    def inference_no_grad(self, img):
        """Depth prediction

        Args:
            img (tensor, [Nx3HxW]): image 

        Returns:
            a dictionary containing depths at different scales, resized back to input scale
                - **scale-N** (tensor, [Nx1xHxW]): depth predictions at scale-N
        """
        # device = torch.device('cuda')
        # feed_width = self.feed_width
        # feed_height = self.feed_height

        # # Preprocess
        # input_image = pil.fromarray(img)
        _, _, original_height, original_width = img.shape
        # input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        # input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        # input_image = input_image.to(device)

        # Prediction
        features = self.encoder(img)
        pred_disps = self.depth_decoder(features)

        outputs = {}
        for s in self.scales:
            disp = pred_disps[('disp', s)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode='bilinear', align_corners=False)

            scaled_disp, _ = disp_to_depth(disp_resized, self.min_depth, self.max_depth)
            outputs[s] = self.stereo_baseline / scaled_disp
            
        return outputs
