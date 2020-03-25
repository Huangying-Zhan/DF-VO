# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.


import numpy as np
import torch
import torch.nn.functional as nnFunc

from libs.deep_layers import DeepLayer

class DepthConsistency():
    def __init__(self, cfg, cam_intrinsics):
        self.cfg = cfg
        self.cam_intrinsics = cam_intrinsics

        # Deep layers
        self.layers = DeepLayer(self.cfg)
        self.layers.initialize_layers()

    def prepare_depth_consistency_data(self, cur_data, ref_data):
        """Prepare data for computing depth consistency
        Returns
            - data (dict): 
                - inv_K
                - K
                - depth
                - pose_T
                - id
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

        for ref_id in ref_data['id']:
            # reference depth
            data[('depth', ref_id)] = torch.from_numpy(ref_data['raw_depth'][ref_id]).unsqueeze(0).unsqueeze(0).float().cuda()

            # pose
            data[('pose_T', cur_data['id'], ref_id)] = torch.from_numpy(ref_data['deep_pose'][ref_id]).unsqueeze(0).float().cuda()
        
        return data

    def warp_and_reproj_depth(self, inputs):
        """Get reprojected depths and warp depths
        - reproj_depth: the reference depths in source view 
        - warp_depth: the warped reference depths in source view

        Args:
            inputs:
                - (inv_K) (Nx4x4)
                - (K) (Nx4x4)
                - (depth, 0) (Nx1xHxW): current depth
                - (depth, ref_id) (Nx1xHxW): reference inv.depth
                - (pose_T, 0, ref_id) (Nx4x4): rel. pose from 0 to ref_id
                - cur_id
                - ref_id
        Returns:
            outputs:
                - ('warp_depth', 0, frame_id)
                - ('reproj_depth', 0, frame_id)
        """
        outputs = {}
        K = inputs['K']
        inv_K = inputs['inv_K']

        # Get depth and 3D points of frame_0
        cur_depth = inputs[('depth', inputs['cur_id'])]
        cam_points = self.layers.backproj(cur_depth, inv_K)


        n, _, h, w = cur_depth.shape

        for frame_id in inputs['ref_id']:
            T = inputs[("pose_T", inputs['cur_id'], frame_id)]

            # reprojection
            reproj_xy = self.layers.reproj(cur_depth, T, K, inv_K)

            # Warp src depth to tgt ref view
            outputs[('warp_depth', inputs['cur_id'], frame_id)] = nnFunc.grid_sample(
                inputs[("depth",  frame_id)],
                reproj_xy,
                padding_mode="border")
                
            # Reproject cur_depth
            transformed_cam_points = torch.matmul(T[:, :3, :], cam_points)
            transformed_cam_points = transformed_cam_points.view(n, 3, h, w)
            proj_depth = transformed_cam_points[:, 2:, :, :]
            outputs[('reproj_depth', inputs['cur_id'], frame_id)] = proj_depth
        return outputs

    def compute_depth_diff(self, depth_data, inputs):
        """
        inputs:
            depth_data:
                - ('warp_depth', cur_id, ref_id)
                - ('reproj_depth', cur_id, ref_id)
            inputs:
                - (inv_K) (Nx4x4)
                - (K) (Nx4x4)
                - (depth, 0) (Nx1xHxW): current depth
                - (depth, ref_id) (Nx1xHxW): reference inv.depth
                - (pose_T, 0, ref_id) (Nx4x4): rel. pose from 0 to ref_id
                - cur_id
                - ref_id
        """
        outputs = {}
        for ref_id in inputs['ref_id']:
            warp_depth = depth_data[('warp_depth', inputs['cur_id'], ref_id)]#.cpu().numpy()[0,0]
            reproj_depth = depth_data[('reproj_depth', inputs['cur_id'], ref_id)]#.cpu().numpy()[0,0]
            depth_diff = (warp_depth - reproj_depth).abs()

            method = "depth_ratio"
            if method == "sc":
                depth_sum = (warp_depth + reproj_depth).abs()
                depth_diff = (depth_diff / depth_sum).clamp(0, 1).cpu().numpy()[0,0]
            elif method == "depth_ratio":
                depth_diff = (depth_diff / reproj_depth).clamp(0, 1).cpu().numpy()[0,0]
            else:
                depth_diff = depth_diff.cpu().numpy()[0,0]

            outputs[('depth_diff', inputs['cur_id'], ref_id)] = depth_diff
        return outputs

    def compute(self, cur_data, ref_data):
        """Compute depth consistency using CNN pose and CNN depths
        New data added to ref_data
            - depth_diff
        """
        # compute depth consistency
        inputs = self.prepare_depth_consistency_data(cur_data, ref_data)
        depth_outputs = self.warp_and_reproj_depth(inputs)
        depth_consistency = self.compute_depth_diff(depth_outputs, inputs)

        ref_data['depth_diff'] = {}
        for ref_id in ref_data['id']:
            ref_data['depth_diff'][ref_id] = depth_consistency[('depth_diff', cur_data['id'], ref_id)]
