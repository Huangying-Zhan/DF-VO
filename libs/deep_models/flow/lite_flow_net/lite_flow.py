''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-20
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

from libs.general.utils import image_grid

from .lite_flow_net import LiteFlowNet
from ..deep_flow import DeepFlow


class LiteFlow(DeepFlow):
    """LiteFlow is the interface for LiteFlowNet
    """

    def __init__(self, *args, **kwargs):
        super(LiteFlow, self).__init__(args, **kwargs)
        
    def initialize_network_model(self, weight_path):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
        """
        if weight_path is not None:
            print("==> Initialize LiteFlowNet with [{}]: ".format(weight_path))
            # Initialize network
            self.model = LiteFlowNet().cuda()

            # Load model weights
            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        else:
            assert False, "No LiteFlowNet pretrained model is provided."

    @torch.no_grad()
    def inference(self, img1, img2):
        """Predict optical flow for the given pairs
        
        Args:
            img1 (array, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (array, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            flow (array, [Nx2xHxW]): flow from img1 to img2
        """
        # Convert to torch array:cuda
        img1 = torch.from_numpy(img1).float().cuda()
        img2 = torch.from_numpy(img2).float().cuda()

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
        scale_factor = 1
        flow = self.resize_dense_flow(
                                output[1] * scale_factor,
                                h, w)
        return flow.detach().cpu().numpy()

    def inference_flow(self, 
                    img1, img2,
                    flow_dir,
                    forward_backward=False,
                    dataset='kitti'):
        """Estimate flow (1->2) and form keypoints
        
        Args:
            img1 (array [Nx3xHxW]): image 1
            img2 (array [Nx3xHxW]): image 2
            flow_dir (str): if directory is given, img1 and img2 become list of img ids
            foward_backward (bool): forward-backward flow consistency is used if True
            dataset (str): dataset type
        
        Returns:
            a dictionary containing
                - **forward** (array [Nx2xHxW]) : foward flow
                - **backward** (array [Nx2xHxW]) : backward flow
                - **flow_diff** (array [NxHxWx1]) : foward-backward flow inconsistency
        """
        # Get flow data
        # if precomputed flow is provided, load precomputed flow
        if flow_dir is not None:
            if forward_backward:
                flow_data, back_flow_data = self.load_precomputed_flow(
                                img1=img1,
                                img2=img2,
                                flow_dir=flow_dir,
                                dataset=dataset,
                                forward_backward=forward_backward
                                )
            else:
                flow_data = self.load_precomputed_flow(
                                img1=img1,
                                img2=img2,
                                flow_dir=flow_dir,
                                dataset=dataset,
                                forward_backward=forward_backward
                                )
        # flow net inference to get flows
        else:
            # FIXME: combined images (batch_size>1) forward performs slightly different from batch_size=1 case
            if forward_backward:
                input_img1 = np.concatenate([img1, img2], axis=0)
                input_img2 = np.concatenate([img2, img1], axis=0)
            else:
                input_img1 = img1
                input_img2 = img2
            combined_flow_data = self.inference(input_img1, input_img2)
            flow_data = combined_flow_data[0:1]
            if forward_backward:
                back_flow_data = combined_flow_data[1:2]

        # Compute keypoint map
        n, _, h, w = flow_data.shape
        tmp_flow_data = np.transpose(flow_data, (0, 2, 3, 1))
        kp1 = image_grid(h, w)
        kp1 = np.repeat(np.expand_dims(kp1, 0), n, axis=0)
        kp2 = kp1 + tmp_flow_data

        # Forward-Backward flow consistency check
        if forward_backward:
            # get flow-consistency error map
            flow_diff = self.forward_backward_consistency(
                                flow1=flow_data,
                                flow2=back_flow_data,
                                px_coord_2=kp2)


        # summarize flow data and flow difference
        flows = {}
        flows['forward'] = flow_data
        if forward_backward:
            flows['backward'] = back_flow_data
            flows['flow_diff'] = flow_diff
        return flows
