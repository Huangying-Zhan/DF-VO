''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: This is the Base class for dataset loader.
'''

import numpy as np

from libs.geometry.camera_modules import Intrinsics

class Dataset():
    """This is the Base class for dataset loader.
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration edict
        """
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
            intrinsics_param (list): [cx, cy, fx, fy]
        """
        raise NotImplementedError
    
    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        
        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        raise NotImplementedError

    def get_data_dir(self):
        """Get data directory

        Returns:
            a dictionary containing
                - **img** (str) : image data directory
                - (optional) **depth** (str) : depth data direcotry or None
                - (optional) **depth_src** (str) : depth data type [gt/None]
        """
        raise NotImplementedError

    def get_gt_poses(self):
        """Get ground-truth poses
        
        Returns:
            gt_poses (dict): each pose is a [4x4] array
        """
        raise NotImplementedError

    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

        Args:
            img_id (int): query image id

        Returns:
            timestamp (int): timestamp for query image
        """
        raise NotImplementedError

    def get_image(self, timestamp):
        """Get image data given the image timestamp

        Args:
            timestamp (int): timestamp for the image
            
        Returns:
            img (array, [CxHxW]): image data
        """
        raise NotImplementedError
    
    def get_depth(self, timestamp):
        """Get GT/precomputed depth data given the timestamp

        Args:
            timestamp (int): timestamp for the depth

        Returns:
            depth (array, [HxW]): depth data
        """
        raise NotImplementedError

    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as KITTI odometry file format

        Args:
            txt (str): pose text file path
            poses (dict): poses, each pose is a [4x4] array
        """
        raise NotImplementedError