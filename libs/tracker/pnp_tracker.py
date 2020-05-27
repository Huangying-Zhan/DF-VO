''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: PnP Tracker to estimate camera motion from given 3D-2D correspondences
'''

import cv2
import numpy as np

from libs.geometry.camera_modules import SE3
from libs.geometry.ops_3d import unprojection_kp

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

    def compute_pose_3d2d(self, kp1, kp2, depth_1):
        """Compute pose from 3d-2d correspondences

        Args:
            kp1 (array, [Nx2]): keypoints for view-1
            kp2 (array, [Nx2]): keypoints for view-2
            depth_1 (array, [HxW]): depths for view-1
        
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
        max_ransac_iter = self.cfg.PnP.ransac.repeat
        
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
                    iterationsCount=self.cfg.PnP.ransac.iter,
                    reprojectionError=self.cfg.PnP.ransac.reproj_thre,
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
