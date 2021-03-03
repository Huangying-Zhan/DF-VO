''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-04
@LastEditors: Huangying Zhan
@Description: This is the interface for LiteFlowNet
'''

import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from .lite_flow_net import LiteFlowNet
from ..deep_flow import DeepFlow


class LiteFlow(DeepFlow):
    """LiteFlow is the interface for LiteFlowNet. 
    """

    def __init__(self, *args, **kwargs):
        super(LiteFlow, self).__init__(*args, **kwargs)
        # FIXME: half-flow issue
        self.half_flow = False
        
    def initialize_network_model(self, weight_path, finetune):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
            finetune (bool): finetune model on the run if True
        """
        if weight_path is not None:
            print("==> Initialize LiteFlowNet with [{}]: ".format(weight_path))
            # Initialize network
            self.model = LiteFlowNet().cuda()

            # Load model weights
            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint)

            if finetune:
                self.model.train()
            else:
                self.model.eval()
        else:
            assert False, "No LiteFlowNet pretrained model is provided."

    def inference(self, img1, img2):
        """Predict optical flow for the given pairs
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (tensor, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            a dictionary containing flows at different scales, resized back to input scale 
                - **scale-N** (tensor, [Nx2xHxW]): flow from img1 to img2 at scale level-N
        """
        # get shape
        _, _, h, w = img1.shape
        th, tw = self.get_target_size(h, w)

        # forward pass
        flow_inputs = [img1, img2]
        resized_img_list = [
                            F.interpolate(
                                img, (th, tw), mode='bilinear', align_corners=True)
                            for img in flow_inputs
                        ]
        output = self.model(resized_img_list)

        # Post-process output
        flows = {}
        for s in self.flow_scales:
            flows[s] = self.resize_dense_flow(
                                output[s],
                                h, w)
            if self.half_flow:
                flows[s] /= 2.
        return flows

    def inference_flow(self, 
                    img1, img2,
                    forward_backward=False,
                    dataset='kitti'):
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
        # flow net inference to get flows
        if forward_backward:
            input_img1 = torch.cat((img1, img2), dim=0)
            input_img2 = torch.cat((img2, img1), dim=0)
        else:
            input_img1 = img1
            input_img2 = img2
        
        # inference with/without gradient
        if self.enable_finetune:
            combined_flow_data = self.inference(input_img1, input_img2)
        else:
            combined_flow_data = self.inference_no_grad(input_img1, input_img2)
        
        self.forward_flow = {}
        self.backward_flow = {}
        self.flow_diff = {}
        self.px1on2 = {}
        for s in self.flow_scales:
            self.forward_flow[s] = combined_flow_data[s][0:1]
            if forward_backward:
                self.backward_flow[s] = combined_flow_data[s][1:2]

            # sampled flow
            # Get sampling pixel coordinates
            self.px1on2[s] = self.flow_to_pix(self.forward_flow[s])

            # Forward-Backward flow consistency check
            if forward_backward:
                # get flow-consistency error map
                self.flow_diff[s] = self.forward_backward_consistency(
                                    flow1=self.forward_flow[s],
                                    flow2=self.backward_flow[s],
                                    px1on2=self.px1on2[s])
        
        # summarize flow data and flow difference for DF-VO
        flows = {}
        flows['forward'] = self.forward_flow[1].clone()
        if forward_backward:
            flows['backward'] = self.backward_flow[1].clone()
            flows['flow_diff'] = self.flow_diff[1].clone()
        return flows
