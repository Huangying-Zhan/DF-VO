''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-26
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep flow network interface
'''

import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F


class DeepFlow():
    """DeepFlow is the Base class for deep flow network interface
    """
    def __init__(self, height, width, cfg=None):
        """
        Args:
            height (int): image height
            width (int): image width
            cfg (edict): deep flow configuration dictionary (only required when online finetuning)
        """
        self.height = height
        self.width = width
        
        self.flow_cfg = cfg

    def initialize_network_model(self, weight_path):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
        """
        raise NotImplementedError

    def get_target_size(self, h, w):
        """Get the closest size that is divisible by 32
        
        Args:
            h (int): given height
            w (int): given width
        
        Returns:
            a tuple containing
                - **h** (int) : target height
                - **w** (int) : target width
        """
        h = 32 * np.array([[math.floor(h / 32), math.floor(h / 32) + 1]])
        w = 32 * np.array([[math.floor(w / 32), math.floor(w / 32) + 1]])
        ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - h / w)
        index = np.argmin(ratio)
        return h[0, index // 2], w[0, index % 2]

    def resize_dense_flow(self, flow, des_height, des_width):
        """Resized flow map with scaling
        
        Args:
            flow (tensor, [Nx2xHxW]): flow map
            des_height (int): destinated height
            des_widht (int): destinated width

        Returns:
            flow (tensor, [Nx2xH'xW']): resized flow
        """
        # get height, width ratio
        ratio_height = float(des_height / flow.size(2))
        ratio_width = float(des_width / flow.size(3))

        # interpolation
        flow = F.interpolate(
            flow, (des_height, des_width), mode='bilinear', align_corners=True)
        
        flow = torch.stack(
            [flow[:, 0, :, :] * ratio_width, flow[:, 1, :, :] * ratio_height],
            dim=1)
        return flow

    # FIXME: move this function to dataset loader
    def load_flow_file(self, flow_path):
        """load flow data from a npy file

        Args:
            flow_path (str): flow data path, npy file
        
        Returns:
            flow (array, [HxWx2]): flow data
        """
        # Load flow
        flow = np.load(flow_path)

        # resize flow
        h, w, _ = flow.shape
        if self.width is None or self.height is None:
            resize_height = h
            resize_width = w
        else:
            resize_height = self.height
            resize_width = self.width
        flow = cv2.resize(flow, (resize_width, resize_height))
        flow[..., 0] *= resize_width / w
        flow[..., 1] *= resize_height / h
        return flow

    # FIXME: move this function to dataset loader
    def load_precomputed_flow(self, img1, img2, flow_dir, dataset, forward_backward):
        """Load precomputed optical flow

        Args:
            img1 (list): list of img1 id
            img2 (list): list of img2 id
            flow_dir (str): directory to read flow
            dataset (str): dataset type [kitti, tum-1/2/3]
            forward_backward (bool): load backward flow if True
        """
        flow_data = []
        for i in range(len(img1)):
            # Get flow npy file
            if "kitti" in dataset:
                flow_path = os.path.join(
                            flow_dir,
                            "{:06d}".format(img2[i]),
                            "{:06d}.npy".format(img1[i]),
                            )
            elif "tum" in dataset:
                flow_path = os.path.join(
                            flow_dir,
                            "{:.6f}".format(img2[i]),
                            "{:.6f}.npy".format(img1[i]),
                            )
            assert os.path.isfile(flow_path), "wrong flow path: [{}]".format(flow_path)

            # Load and process flow data
            flow = self.load_flow_file(flow_path)
            flow_data.append(flow)
        flow_data = np.asarray(flow_data)
        flow_data = np.transpose(flow_data, (0, 3, 1, 2))

        # get backward flow data
        if forward_backward:
            back_flow_data = []
            for i in range(len(img1)):
                if "kitti" in dataset:
                    flow_path = os.path.join(
                                    flow_dir,
                                    "{:06d}".format(img1[i]),
                                    "{:06d}.npy".format(img2[i]),
                                    )
                elif "tum" in dataset:
                    flow_path = os.path.join(
                                    flow_dir,
                                    "{:.6f}".format(img1[i]),
                                    "{:.6f}.npy".format(img2[i]),
                                    )
                assert os.path.isfile(flow_path), "wrong flow path"

                flow = self.load_flow_file(flow_path)
                back_flow_data.append(flow)
            back_flow_data = np.asarray(back_flow_data)
            back_flow_data = np.transpose(back_flow_data, (0, 3, 1, 2))
            return flow_data, back_flow_data
        else:
            return flow_data

    @torch.no_grad()
    def inference(self, img1, img2):
        """Predict optical flow for the given pairs
        
        Args:
            img1 (array, [Nx3xHxW]): image 1; intensity [0-1]
            img2 (array, [Nx3xHxW]): image 2; intensity [0-1]
        
        Returns:
            flow (array, [Nx2xHxW]): flow from img1 to img2
        """
        raise NotImplementedError

    def inference_flow(self, 
                    img1, img2,
                    flow_dir,
                    forward_backward=False,
                    dataset="kitti"):
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
        raise NotImplementedError

    def forward_backward_consistency(self, flow1, flow2, px_coord_2):
        """Compute flow consistency map

        Args:
            flow1 (array, [Nx2xHxW]): flow map 1
            flow2 (array, [Nx2xHxW]): flow map 2
            px_coord_2 (array [NxHxWx2]): pixel coordinate in view 2
        Returns:
            flow_diff (array, [NxHxWx1]): flow inconsistency error map
        """
        # copy flow data to GPU
        flow1 = torch.from_numpy(flow1).float().cuda()
        flow2 = torch.from_numpy(flow2).float().cuda()

        # Normalize sampling pixel coordinates
        _, _, h, w = flow1.shape
        norm_px_coord = px_coord_2.copy()
        norm_px_coord[:, :, :, 0] = px_coord_2[:,:,:,0] / (w-1)
        norm_px_coord[:, :, :, 1] = px_coord_2[:,:,:,1] / (h-1)
        norm_px_coord = (norm_px_coord * 2) - 1
        norm_px_coord = torch.from_numpy(norm_px_coord).float().cuda()

        # Warp flow2 to flow1
        warp_flow1 = F.grid_sample(-flow2, norm_px_coord)

        # Calculate flow difference
        flow_diff = (flow1 - warp_flow1)

        # TODO: UnFlow (Meister etal. 2017) constrain is not used
        UnFlow_constrain = False
        if UnFlow_constrain:
            flow_diff = (flow_diff ** 2 - 0.01 * (flow1**2 - warp_flow1 ** 2))

        # copy flow_diff to cpu
        flow_diff = flow_diff.norm(dim=1, keepdim=True)
        flow_diff = flow_diff.permute(0, 2, 3, 1)
        flow_diff = flow_diff.detach().cpu().numpy()
        return flow_diff
