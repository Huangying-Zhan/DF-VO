''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-03-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-28
@LastEditors: Huangying Zhan
@Description: 
'''

import torch
import torch.nn as nn

from libs.deep_models.depth.monodepth2.layers import PixToFlow
from libs.geometry.reprojection import Reprojection

class RigidFlow(nn.Module):
    """Layer to compute rigid flow given depth and camera motion
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(RigidFlow, self).__init__()
        # basic configuration
        self.height = height
        self.width = width
        self.device = torch.device('cuda')

        # layer setup
        self.pix2flow = PixToFlow(1, self.height, self.width) 
        self.pix2flow.to(self.device)

        self.reprojection = Reprojection(self.height, self.width)

    def forward(self, depth, T, K, inv_K, normalized=True):
        """Forward pass
        
        Args:
            depth (tensor, [Nx1xHxW]): depth map 
            T (tensor, [Nx4x4]): transformation matrice
            inv_K (tensor, [Nx4x4]): inverse camera intrinsics
            K (tensor, [Nx4x4]): camera intrinsics
            normalized (bool): 
                
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]

        Returns:
            flow (NxHxWx2): rigid flow
        """
        xy = self.reprojection(depth, T, K, inv_K, normalized)
        flow = self.pix2flow(xy)

        return flow

