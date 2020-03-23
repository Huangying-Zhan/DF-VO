# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file.

import torch
import torch.nn as nn


class Transformation3D(nn.Module):
    """Layer which transform 3D points
    """
    def __init__(self):
        super(Transformation3D, self).__init__()

    def forward(self, points, T):
        """
        Args:
            points (Nx4x(HxW)): 3D points in homogeneous coordinates
            T (Nx4x4): transformation matrice
        Returns:
            transformed_points (Nx4x(HxW)): 3D points in homogeneous coordinates
        """
        transformed_points = torch.matmul(T, points)
        return transformed_points