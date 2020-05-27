''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: Layer to transform 3D points given transformation matrice
'''

import torch
import torch.nn as nn


class Transformation3D(nn.Module):
    """Layer to transform 3D points given transformation matrice
    """
    def __init__(self):
        super(Transformation3D, self).__init__()

    def forward(self, points, T):
        """Forward pass
        
        Args:
            points (tensor, [Nx4x(HxW)]): 3D points in homogeneous coordinates
            T (tensor, [Nx4x4]): transformation matrice
        
        Returns:
            transformed_points (tensor, [Nx4x(HxW)]): 3D points in homogeneous coordinates
        """
        transformed_points = torch.matmul(T, points)
        return transformed_points