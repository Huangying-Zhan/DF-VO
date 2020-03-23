# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file.

import torch
import torch.nn as nn

from .backprojection import Backprojection
from .transformation3d import Transformation3D
from .projection import Projection


class Reprojection(nn.Module):
    """Layer to transform pixel coordinates from one view to another view via
    backprojection, transformation in 3D, and projection
    """
    def __init__(self, height, width):
        """
        Args:
            height (int)
            width (int)
        """
        super(Reprojection, self).__init__()

        # layers
        self.backproj = Backprojection(height, width)
        self.transform = Transformation3D()
        self.project = Projection(height, width)

    def forward(self, depth, T, K, inv_K):
        """
        Args:
            depths (Nx1xHxW): depth map
            T (Nx4x4): transformation matrice
            K (Nx4x4): camera intrinsics
            inv_K (Nx4x4): inverse camera intrinsics
        Returns:
            xy (NxHxWx2): pixel coordinates
        """
        points3d = self.backproj(depth, inv_K)
        points3d_trans = self.transform(points3d, T)
        xy = self.project(points3d_trans, K) 
        return xy
