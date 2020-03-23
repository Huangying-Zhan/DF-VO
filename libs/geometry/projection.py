# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file.

import torch
import torch.nn as nn


class Projection(nn.Module):
    """Layer which projects 3D points into a camera view
    """
    def __init__(self, height, width, eps=1e-7):
        super(Projection, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points3d, K, normalized=True):
        """
        Args:
            points3d (Nx4x(HxW)): 3D points in homogeneous coordinates
            K (Nx4x4): camera intrinsics
            normalized (bool): 
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]
        Returns:
            xy (NxHxWx2): pixel coordinates
        """
        # projection
        points2d = torch.matmul(K[:, :3, :], points3d)

        # convert from homogeneous coordinates
        xy = points2d[:, :2, :] / (points2d[:, 2:3, :] + self.eps)
        xy = xy.view(points3d.shape[0], 2, self.height, self.width)
        xy = xy.permute(0, 2, 3, 1)

        # normalization
        if normalized:
            xy[..., 0] /= self.width - 1
            xy[..., 1] /= self.height - 1
            xy = (xy - 0.5) * 2
        return xy