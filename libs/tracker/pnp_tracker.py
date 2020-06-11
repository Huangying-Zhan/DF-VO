''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-10
@LastEditors: Huangying Zhan
@Description: PnP Tracker to estimate camera motion from given 3D-2D correspondences
'''

import cv2
import copy
import numpy as np
import torch

from libs.geometry.camera_modules import SE3
from libs.geometry.ops_3d import unprojection_kp
from libs.geometry.rigid_flow import RigidFlow
from libs.general.utils import image_grid
from libs.matching.kp_selection import opt_rigid_flow_kp


class PnpTracker():
    """ PnP Tracker to estimate camera motion from given 3D-2D correspondences 
    """
    def __init__(self, cfg, cam_intrinsics):
        """ 
        Args:
            cfg (edict): configuration dictionary
            cam_intrinsics (Intrinsics): camera intrinsics
        """
        self.cfg = cfg
        self.cam_intrinsics = cam_intrinsics

        # Rigid flow data
        if self.cfg.kp_selection.rigid_flow_kp.enable:
            self.K = np.eye(4)
            self.inv_K = np.eye(4)
            self.K[:3, :3] = cam_intrinsics.mat
            self.inv_K[:3, :3] = cam_intrinsics.inv_mat
            self.K = torch.from_numpy(self.K).float().unsqueeze(0).cuda()
            self.inv_K = torch.from_numpy(self.inv_K).float().unsqueeze(0).cuda()
            self.rigid_flow_layer = RigidFlow(self.cfg.image.height, self.cfg.image.width).cuda()

    def compute_pose_3d2d(self, kp1, kp2, depth_1, is_iterative):
        """Compute pose from 3d-2d correspondences

        Args:
            kp1 (array, [Nx2]): keypoints for view-1
            kp2 (array, [Nx2]): keypoints for view-2
            depth_1 (array, [HxW]): depths for view-1
            is_iterative (bool): is iterative stage
        
        Returns:
            a dictionary containing
                - **pose** (SE3): relative pose from view-2 to view-1
                - **kp1** (array, [Nx2]): filtered keypoints for view-1
                - **kp2** (array, [Nx2]): filtered keypoints for view-2
        """
        outputs = {}
        height, width = depth_1.shape

        # Filter keypoints outside image region
        x_idx = (kp2[:, 0] >= 0) * (kp2[:, 0] < width) 
        kp1 = kp1[x_idx]
        kp2 = kp2[x_idx]
        y_idx = (kp2[:, 1] >= 0) * (kp2[:, 1] < height) 
        kp1 = kp1[y_idx]
        kp2 = kp2[y_idx]

        # Filter keypoints outside depth range
        kp1_int = kp1.astype(np.int)
        kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
        non_zero_mask = (kp_depths != 0)
        depth_range_mask = (kp_depths < self.cfg.depth.max_depth) * (kp_depths > self.cfg.depth.min_depth)
        valid_kp_mask = non_zero_mask * depth_range_mask

        kp1 = kp1[valid_kp_mask]
        kp2 = kp2[valid_kp_mask]

        # Get 3D coordinates for kp1
        XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], self.cam_intrinsics)

        # initialize ransac setup
        best_rt = []
        best_inlier = 0
        max_ransac_iter = self.cfg.pnp_tracker.ransac.repeat if is_iterative else 3
        
        for _ in range(max_ransac_iter):
            # shuffle kp (only useful when random seed is fixed)	
            new_list = np.arange(0, kp2.shape[0], 1)	
            np.random.shuffle(new_list)
            new_XYZ = XYZ_kp1.copy()[new_list]
            new_kp2 = kp2.copy()[new_list]

            if new_kp2.shape[0] > 4:
                # PnP solver
                flag, r, t, inlier = cv2.solvePnPRansac(
                    objectPoints=new_XYZ,
                    imagePoints=new_kp2,
                    cameraMatrix=self.cam_intrinsics.mat,
                    distCoeffs=None,
                    iterationsCount=self.cfg.pnp_tracker.ransac.iter,
                    reprojectionError=self.cfg.pnp_tracker.ransac.reproj_thre,
                    )
                
                # save best pose estimation
                if flag and inlier.shape[0] > best_inlier:
                    best_rt = [r, t]
                    best_inlier = inlier.shape[0]
        
        # format pose
        pose = SE3()
        if len(best_rt) != 0:
            r, t = best_rt
            pose.R = cv2.Rodrigues(r)[0]
            pose.t = t
        pose.pose = pose.inv_pose

        # save output
        outputs['pose'] = pose
        outputs['kp1'] = kp1
        outputs['kp2'] = kp2

        return outputs
    
    def compute_rigid_flow_kp(self, cur_data, ref_data, pose):
        """compute keypoints from optical-rigid flow consistency

        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            pose (SE3): SE3 pose
        """
        rigid_pose = copy.deepcopy(pose)
        ref_data['rigid_flow_pose'] = SE3(rigid_pose.inv_pose)
         # kp selection
        kp_sel_outputs = self.kp_selection_good_depth(cur_data, ref_data, 
                                self.cfg.pnp_tracker.iterative_kp.score_method,
                                )
        ref_data['kp_depth'] = kp_sel_outputs['kp1_depth'][0]
        cur_data['kp_depth'] = kp_sel_outputs['kp2_depth'][0]
        ref_data['kp_depth_uniform'] = kp_sel_outputs['kp1_depth_uniform'][0]
        cur_data['kp_depth_uniform'] = kp_sel_outputs['kp2_depth_uniform'][0]
        cur_data['rigid_flow_mask'] = kp_sel_outputs['rigid_flow_mask']


    def kp_selection_good_depth(self, cur_data, ref_data, rigid_kp_score_method):
        """Choose valid kp from a series of operations

        Args:
            cur_data (dict): current data 
            ref_data (dict): reference data
            rigid_kp_method (str) : [uniform, best]
            rigid_kp_score_method (str): [opt_flow, rigid_flow]
        
        Returns:
            a dictionary containing
                
                - **kp1_depth** (array, [Nx2]): keypoints in view-1, best in terms of score_method
                - **kp2_depth** (array, [Nx2]): keypoints in view-2, best in terms of score_method
                - **kp1_depth_uniform** (array, [Nx2]): keypoints in view-1, uniformly sampled
                - **kp2_depth_uniform** (array, [Nx2]): keypoints in view-2, uniformly sampled
                - **rigid_flow_mask** (array, [HxW]): rigid-optical flow consistency 
        """
        outputs = {}

        # initialization
        h, w = cur_data['depth'].shape

        kp1 = image_grid(h, w)
        kp1 = np.expand_dims(kp1, 0)
        tmp_flow_data = np.transpose(np.expand_dims(ref_data['flow'], 0), (0, 2, 3, 1))
        kp2 = kp1 + tmp_flow_data

        """ opt-rigid flow consistent kp selection """
        if self.cfg.kp_selection.rigid_flow_kp.enable:
            ref_data['rigid_flow_diff'] = {}
            # compute rigid flow
            rigid_flow_pose = ref_data['rigid_flow_pose'].pose

            # Compute rigid flow
            pose_tensor = torch.from_numpy(rigid_flow_pose).float().unsqueeze(0).cuda()
            depth = torch.from_numpy(ref_data['raw_depth']).float().unsqueeze(0).unsqueeze(0).cuda()
            rigid_flow_tensor = self.rigid_flow_layer(
                                depth,
                                pose_tensor,
                                self.K,
                                self.inv_K,
                                normalized=False,
            )
            rigid_flow = rigid_flow_tensor.detach().cpu().numpy()[0]

            # compute optical-rigid flow difference
            rigid_flow_diff = np.linalg.norm(
                                rigid_flow - ref_data['flow'],
                                axis=0)
            ref_data['rigid_flow_diff'] = np.expand_dims(rigid_flow_diff, 2)

            # get depth-flow consistent kp
            outputs.update(
                    opt_rigid_flow_kp(
                        kp1=kp1,
                        kp2=kp2,
                        ref_data=ref_data,
                        cfg=self.cfg,
                        outputs=outputs,
                        score_method=rigid_kp_score_method
                        )
                )

        return outputs
