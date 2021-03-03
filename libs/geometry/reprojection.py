''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: Layer to transform pixel coordinates from one view to another view via
    backprojection, transformation in 3D, and projection
'''

import torch
import torch.nn as nn

from libs.geometry.backprojection import Backprojection
from libs.geometry.transformation3d import Transformation3D
from libs.geometry.projection import Projection


class Reprojection(nn.Module):
    """Layer to transform pixel coordinates from one view to another view via
    backprojection, transformation in 3D, and projection
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Reprojection, self).__init__()

        # layers
        self.backproj = Backprojection(height, width)
        self.transform = Transformation3D()
        self.project = Projection(height, width)

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
            xy (NxHxWx2): pixel coordinates
        """
        points3d = self.backproj(depth, inv_K)
        points3d_trans = self.transform(points3d, T)
        xy = self.project(points3d_trans, K, normalized) 
        return xy
