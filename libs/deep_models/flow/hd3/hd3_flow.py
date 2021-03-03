'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 1970-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-25
@LastEditors: Huangying Zhan
@Description: This is the interface for HD3FlowNet
'''

from collections import OrderedDict
import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from .hd3model import HD3Model
from ..deep_flow import DeepFlow


def model_state_dict_parallel_convert(state_dict, mode):
    """convert model state dict between single/parallel
    Args:
        state_dict (dict): state dict
        mode (str):
            - to_single
            - to_parallel
    Returns:
        new_state_dict (dict)
    """
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict


def model_state_dict_convert_auto(state_dict, gpu_ids):
    """convert model state dict between single/parallel (auto)
    Args:
        state_dict (dict): state dict
        gpu_ids (list): gpu ids
    Returns:
        new_state_dict (dict)
    """
    for k, v in state_dict.items():
        if (k[0:7] == 'module.' and len(gpu_ids) >= 2) or (k[0:7] != 'module.' and len(gpu_ids) == 1):
            return state_dict
        elif k[0:7] == 'module.' and len(gpu_ids) == 1:
            return model_state_dict_parallel_convert(state_dict, mode='to_single')
        elif k[0:7] != 'module.' and len(gpu_ids) >= 2:
            return model_state_dict_parallel_convert(state_dict, mode='to_parallel')
        else:
            raise Exception('Error in model_state_dict_convert_auto')



class HD3Flow(DeepFlow):
    """HD3Flow is the interface for HD3FlowNet. 
    """

    def __init__(self, *args, **kwargs):
        super(HD3Flow, self).__init__(*args, **kwargs)
        
    def initialize_network_model(self, weight_path, finetune):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
            finetune (bool): finetune model on the run if True
        """
        if weight_path is not None:
            print("==> Initialize HD3Net with [{}]: ".format(weight_path))
            # Initialize network
            self.model = HD3Model(
                            task="flow",
                            encoder="dlaup",
                            decoder="hda",
                            corr_range=[4, 4, 4, 4, 4],
                            context=False
                            ).cuda()

            # Load model weights
            checkpoint = torch.load(weight_path)
            # self.model = torch.nn.DataParallel(self.model).cuda()
            checkpoint = model_state_dict_convert_auto(checkpoint['state_dict'], [0])
            self.model.load_state_dict(checkpoint, strict=True)

            if finetune:
                self.model.train()
            else:
                self.model.eval()
        else:
            assert False, "No HD3Net pretrained model is provided."

    def get_target_size(self, H, W):
        h = 64 * np.array([[math.floor(H / 64), math.floor(H / 64) + 1]])
        w = 64 * np.array([[math.floor(W / 64), math.floor(W / 64) + 1]])
        ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - H / W)
        index = np.argmin(ratio)
        return h[0, index // 2], w[0, index % 2]

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
        corr_range = [4, 4, 4, 4, 4]

        # forward pass
        flow_inputs = [img1, img2]
        resized_img_list = [
                            F.interpolate(
                                img, (th, tw), mode='bilinear', align_corners=True)
                            for img in flow_inputs
                        ]
        output = self.model(img_list=resized_img_list, get_vect=True)

        # Post-process output
        scale_factor = 1 / 2**(7 - len(corr_range))
        flows = {}
        for s in self.flow_scales:
            flows[s] = self.resize_dense_flow(
                                output['vect'] * scale_factor,
                                h, w)
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
