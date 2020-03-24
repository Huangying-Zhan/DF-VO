# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

from libs.camera_modules import SE3, Intrinsics

class Dataset():
    def __init__(self, cfg):
        self.cfg = cfg

        # read camera intrinsics
        K_params = self.get_intrinsics_param()
        self.cam_intrinsics = Intrinsics(K_params)

        # get data direcotries
        self.data_dir = self.get_data_dir()
        
        # synchronize timestamps
        self.synchronize_timestamps()

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
