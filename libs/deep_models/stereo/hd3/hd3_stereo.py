''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-06-23
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

from libs.deep_models.flow.hd3.hd3model import HD3Model
from ..deep_stereo import DeepStereo


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



class HD3Stereo(DeepStereo):
    """HD3Stereo is the interface for HD3StereoNet. 
    """

    def __init__(self, *args, **kwargs):
        super(HD3Stereo, self).__init__(*args, **kwargs)
        
    def initialize_network_model(self, weight_path, finetune):
        """initialize stereo_net model with weight_path
        
        Args:
            weight_path (str): weight path
            finetune (bool): finetune model on the run if True
        """
        if weight_path is not None:
            print("==> Initialize HD3Net with [{}]: ".format(weight_path))
            # Initialize network
            self.model = HD3Model(
                            task="stereo",
                            encoder="dlaup",
                            decoder="hda",
                            corr_range=[4, 4, 4, 4, 4, 4],
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
        """Predict disparity for the given pairs
        
        Args:
            img1 (tensor, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (tensor, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            a dictionary containing disparity at different scales, resized back to input scale 
                - **scale-N** (tensor, [Nx1xHxW]): disparity from img1 to img2 at scale level-N
        """
        # get shape
        _, _, h, w = img1.shape
        th, tw = self.get_target_size(h, w)
        corr_range = [4, 4, 4, 4, 4, 4]

        # forward pass
        disp_inputs = [img1, img2]
        resized_img_list = [
                            F.interpolate(
                                img, (th, tw), mode='bilinear', align_corners=True)
                            for img in disp_inputs
                        ]
        output = self.model(img_list=resized_img_list, get_vect=True)

        # Post-process output
        scale_factor = 1 / 2**(7 - len(corr_range))
        disps = {}
        for s in self.stereo_scales:
            disps[s] = self.resize_dense_disp(
                                output['vect'] * scale_factor,
                                h, w)
        return disps

    def inference_stereo(self, 
                    img1, img2,
                    forward_backward=False,
                    dataset='kitti'):
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
        # flip imgs
        flip_img1 = torch.flip(img1, [3])
        flip_img2 = torch.flip(img2, [3])

        # stereo net inference to get disps
        if forward_backward:
            input_img1 = torch.cat((img1, flip_img2), dim=0)
            input_img2 = torch.cat((img2, flip_img1), dim=0)
        else:
            input_img1 = img1
            input_img2 = img2
        
        # inference with/without gradient
        if self.enable_finetune:
            combined_disp_data = self.inference(input_img1, input_img2)
        else:
            combined_disp_data = self.inference_no_grad(input_img1, input_img2)
        
        self.forward_disp = {}
        self.backward_disp = {}
        self.disp_diff = {}
        self.px1on2 = {}
        for s in self.stereo_scales:
            self.forward_disp[s] = combined_disp_data[s][0:1]
            if forward_backward:
                self.backward_disp[s] = torch.flip(combined_disp_data[s][1:2], [3])

            # Get sampling pixel coordinates
            self.px1on2[s] = self.disp_to_pix(self.forward_disp[s])

            # Forward-Backward flow consistency check
            if forward_backward:
                # get flow-consistency error map
                self.disp_diff[s] = self.forward_backward_consistency(
                                    disp1=self.forward_disp[s],
                                    disp2=self.backward_disp[s],
                                    px1on2=self.px1on2[s])
        
        # summarize flow data and flow difference for DF-VO
        disps = {}
        disps['forward'] = self.forward_disp[0].clone()
        if forward_backward:
            disps['backward'] = self.backward_disp[0].clone()
            disps['disp_diff'] = self.disp_diff[0].clone()
        return disps
