''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: DepthConsistency computes depth consistency between depth maps
'''

import numpy as np
import torch
import torch.nn.functional as nnFunc

from libs.geometry.backprojection import Backprojection
from libs.geometry.reprojection import Reprojection

class DepthConsistency():
    """DepthConsistency computes depth consistency between depth maps
    """

    def __init__(self, cfg, cam_intrinsics):
        self.cfg = cfg
        self.cam_intrinsics = cam_intrinsics

        # Deep layers
        h, w = self.cfg.image.height, self.cfg.image.width
        self.backproj = Backprojection(h, w).cuda()
        self.reproj = Reprojection(h, w).cuda()

    def prepare_depth_consistency_data(self, cur_data, ref_data):
        """Prepare data for computing depth consistency
        
        Returns:
            a dictionary containing
                - **inv_K** (tensor, [1x4x4]): inverse camera intrinsics
                - **K** (tensor, [1x4x4]): camera intrinsics
                - **depth** (tensor, 1x1xHxW): depth map
                - **pose_T** (tensor, [1x4x4]): relative pose
                - **cur_id** (int): index of current frame
                - **ref_id** (int): index of reference frame
        """
        data = {}

        # camera intrinsics
        data[('inv_K')] = np.eye(4)
        data[('inv_K')][:3, :3] = self.cam_intrinsics.inv_mat
        data[('inv_K')] = torch.from_numpy(data[('inv_K')]).unsqueeze(0).float().cuda()

        data[('K')] = np.eye(4)
        data[('K')][:3, :3] = self.cam_intrinsics.mat
        data[('K')] = torch.from_numpy(data[('K')]).unsqueeze(0).float().cuda()

        # current depth
        data[('depth', cur_data['id'])] = torch.from_numpy(cur_data['raw_depth']).unsqueeze(0).unsqueeze(0).float().cuda()

        # id
        data['cur_id'] = cur_data['id']
        data['ref_id'] = ref_data['id']

        # reference depth
        data[('depth', data['ref_id'])] = torch.from_numpy(ref_data['raw_depth']).unsqueeze(0).unsqueeze(0).float().cuda()

        # pose
        data[('pose_T', cur_data['id'], data['ref_id'])] = torch.from_numpy(ref_data['deep_pose']).unsqueeze(0).float().cuda()
        
        return data

    def warp_and_reproj_depth(self, inputs):
        """Get reprojected depths and warp depths;
        - reproj_depth: the reference depths in source view;
        - warp_depth: the warped reference depths in source view

        Args:
            inputs (dict): a dictionary containing

                - **inv_K** (tensor, [1x4x4]): inverse camera intrinsics
                - **K** (tensor, [1x4x4]): camera intrinsics
                - **depth** (tensor, 1x1xHxW): depth map
                - **pose_T** (tensor, [1x4x4]): relative pose
                - **cur_id** (int): index of current frame
                - **ref_id** (int): index of reference frame
        
        Returns:
            a dictionary containing
                - **('warp_depth', cur_id, ref_id)** (tensor, [1x1xHxW]): synthesized current depth map
                - **('reproj_depth', cur_id, ref_id)** (tensor, [1x1xHxW]): transformed current depth map
        """
        outputs = {}
        K = inputs['K']
        inv_K = inputs['inv_K']

        # Get depth and 3D points of frame_0
        cur_depth = inputs[('depth', inputs['cur_id'])]
        cam_points = self.backproj(cur_depth, inv_K)


        n, _, h, w = cur_depth.shape

        T = inputs[("pose_T", inputs['cur_id'], inputs['ref_id'])]

        # reprojection
        reproj_xy = self.reproj(cur_depth, T, K, inv_K)

        # Warp src depth to tgt ref view
        outputs[('warp_depth', inputs['cur_id'], inputs['ref_id'])] = nnFunc.grid_sample(
            inputs[("depth",  inputs['ref_id'])],
            reproj_xy,
            padding_mode="border")
            
        # Reproject cur_depth
        transformed_cam_points = torch.matmul(T[:, :3, :], cam_points)
        transformed_cam_points = transformed_cam_points.view(n, 3, h, w)
        proj_depth = transformed_cam_points[:, 2:, :, :]
        outputs[('reproj_depth', inputs['cur_id'], inputs['ref_id'])] = proj_depth
        return outputs

    def compute_depth_diff(self, depth_data, inputs):
        """Compute depth difference
        
        Args:
            depth_data: a dictionary containing

                - **('warp_depth', 0, frame_id)** (tensor, [1x1xHxW]): warped depth map
                - **('reproj_depth', 0, frame_id)** (tensor, [1x1xHxW]): reprojected depth map
            
            inputs: a dictionary containing
            
                - **cur_id** (int): index of current frame
                - **ref_id** (int): index of reference frame
        
        Returns:
            outputs (dict), a dictionary containing
                - **(depth_diff, cur_id, ref_id)** (array, [HxW]): depth difference
        """
        outputs = {}
        warp_depth = depth_data[('warp_depth', inputs['cur_id'], inputs['ref_id'])]
        reproj_depth = depth_data[('reproj_depth', inputs['cur_id'], inputs['ref_id'])]
        depth_diff = (warp_depth - reproj_depth).abs()

        method = "depth_ratio"
        if method == "sc":
            depth_sum = (warp_depth + reproj_depth).abs()
            depth_diff = (depth_diff / depth_sum).clamp(0, 1).cpu().numpy()[0,0]
        elif method == "depth_ratio":
            depth_diff = (depth_diff / reproj_depth).clamp(0, 1).cpu().numpy()[0,0]
        else:
            depth_diff = depth_diff.cpu().numpy()[0,0]

        outputs[('depth_diff', inputs['cur_id'], inputs['ref_id'])] = depth_diff
        return outputs

    def compute(self, cur_data, ref_data):
        """Compute depth consistency using CNN pose and CNN depths
        New data added to ref_data
            - **depth_diff** (array, [HxW]): depth consistency data
        """
        # compute depth consistency
        inputs = self.prepare_depth_consistency_data(cur_data, ref_data)
        depth_outputs = self.warp_and_reproj_depth(inputs)
        depth_consistency = self.compute_depth_diff(depth_outputs, inputs)

        ref_data['depth_diff'] = depth_consistency[('depth_diff', cur_data['id'], ref_data['id'])]
