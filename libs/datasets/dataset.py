# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.


import numpy as np

from libs.camera_modules import SE3, Intrinsics

class Dataset():
    def __init__(self, cfg):
        self.cfg = cfg

        # read camera intrinsics
        K_params = self.get_intrinsics_param()
        self.cam_intrinsics = Intrinsics(K_params)

        # get data directories
        self.data_dir = self.get_data_dir()
        
        # synchronize timestamps
        self.synchronize_timestamps()

        # get gt poses (for visualization comparison purpose)
        if self.cfg.directory.gt_pose_dir is not None:
            self.gt_poses = self.get_gt_poses()
        else:
            self.gt_poses = {0: np.eye(4)}

    def __len__(self):
        return len(self.rgb_d_pose_pair)

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        raise NotImplementedError
    
    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        mainly for TUM-RGBD dataset
        Returns:
            rgb_d_pose_pair (dict):
                - rgb_timestamp: {depth: depth_timestamp, pose: pose_timestamp}
        """
        raise NotImplementedError

    def get_data_dir(self):
        """Get data directory

        Returns:
            data_dir (dict):
                - img (str): image data directory
                - (optional) depth (str): depth data direcotry or None
                - (optional) depth_src (str): depth data type [gt/None]
        """
        raise NotImplementedError

    def get_gt_poses(self):
        """load ground-truth poses
        Returns:
            gt_poses (dict): each pose is 4x4 array
        """
        raise NotImplementedError

    def get_timestamp(self, img_id):
        """get timestamp for the query img_id
        Args:
            img_id (int): query image id
        Returns:
            timestamp (int): timestamp for query image
        """
        raise NotImplementedError

    def get_image(self, timestamp):
        """get image data given the image timestamp
        Args:
            timestamp (int): timestamp for the image
        Returns:
            img (CxHxW): image data
        """
        raise NotImplementedError
    
    def get_depth(self, timestamp):
        """get GT/precomputed depth data given the timestamp
        Args:
            timestamp (int): timestamp for the depth
        Returns:
            depth (HxW): depth data
        """
        raise NotImplementedError

    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as KITTI odometry file format
        Args:
            txt (str): pose text file path
            poses (array dict): poses, each pose is 4x4 array
            format (str): trajectory format
                - kitti: 12 parameters
                - tum: timestamp tx ty tz qx qy qz qw
        """
        raise NotImplementedError