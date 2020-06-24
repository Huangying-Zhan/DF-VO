''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-13
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-25
@LastEditors: Please set LastEditors
@Description: Dataset loaders for Oxford Robotcar Driving Sequence
'''

import numpy as np
from glob import glob
import os

from .dataset import Dataset
from libs.general.utils import *


class OxfordRobotCar(Dataset):
    """Base class of dataset loaders for OxfordRobotCar Driving Sequence
    """

    def __init__(self, *args, **kwargs):
        super(OxfordRobotCar, self).__init__(*args, **kwargs)
    
    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        
        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        # Load timestamps
        timestamp_txt = os.path.join(self.cfg.directory.img_seq_dir,
                                    self.cfg.seq,
                                    "stereo.timestamps")
        timestamps = np.loadtxt(timestamp_txt)[:, 0].astype(np.int)


        self.rgb_d_pose_pair = {}
        len_seq = len(glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext))))
        for cnt, i in enumerate(range(20, len_seq)):
            self.rgb_d_pose_pair[timestamps[i]] = {}
            self.rgb_d_pose_pair[timestamps[i]]['depth'] = i
            self.rgb_d_pose_pair[timestamps[i]]['pose'] = i
    
    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

        Args:
            img_id (int): query image id

        Returns:
            timestamp (int): timestamp for query image
        """
        return sorted(list(self.rgb_d_pose_pair.keys()))[img_id]

    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as KITTI odometry file format

        Args:
            txt (str): pose text file path
            poses (dict): poses, each pose is a [4x4] array
        """
        global_poses_arr = convert_SE3_to_arr(poses)
        save_traj(traj_txt, global_poses_arr, format='kitti')

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (list): [cx, cy, fx, fy]
        """
        # Reference image size
        self.height = 960
        self.width = 1280

        # Crop
        self.x_crop = np.array([0.1, 0.9]) 
        self.y_crop = np.array([0.0, 0.8]) 
        
        self.height = int((self.y_crop[1] - self.y_crop[0]) * self.height)
        self.width = int((self.x_crop[1] - self.x_crop[0]) * self.width)

        intrinsic_txt = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            "robotcar-dataset-sdk",
                            "models",
                            "stereo_narrow_left.txt"
                            )
        K_params = np.loadtxt(intrinsic_txt)
        K = np.eye(3)
        K[0, 0] = K_params[0,0] # fx
        K[0, 2] = K_params[0,2] # cx
        K[1, 1] = K_params[0,1] - int(self.width * self.x_crop[0]) # fy
        K[1, 2] = K_params[0,3] - int(self.height * self.y_crop[0]) # cy


        K[0] *= (self.cfg.image.width / self.width)
        K[1] *= (self.cfg.image.height / self.height)

        intrinsics_param = [K[0,2], K[1,2], K[0,0], K[1,1]]
        return intrinsics_param
    
    def get_data_dir(self):
        """Get data directory

        Returns:
            a dictionary containing
                - **img** (str) : image data directory
                - (optional) **depth** (str) : depth data direcotry or None
                - (optional) **depth_src** (str) : depth data type [gt/None]
        """
        data_dir = {"depth": None, "depth_src": None}

        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq,
                            "stereo",
                            "centre"
                            )
        data_dir['img'] = os.path.join(img_seq_dir)
 
        return data_dir
    
    def get_image(self, timestamp):
        """Get image data given the image timestamp

        Args:
            timestamp (int): timestamp for the image
            
        Returns:
            img (array, [CxHxW]): image data
        """
        img_path = os.path.join(self.data_dir['img'], 
                            "{:016d}.{}".format(timestamp, self.cfg.image.ext)
                            )
        crop = np.zeros((2,2))
        crop[0] = self.y_crop
        crop[1] = self.x_crop
        img = read_image(img_path, self.cfg.image.height, self.cfg.image.width, crop)
        return img
    
    ''' 
        These functions are not necessary to run DF-VO. 
        However, if you want to use RGB-D data, get_depth() is required.
        If you have gt poses as well for comparison, get_gt_poses() is required.
    ''' 
    def get_depth(self, timestamp):
        """Get GT/precomputed depth data given the timestamp

        Args:
            timestamp (int): timestamp for the depth

        Returns:
            depth (array, [HxW]): depth data
        """
        raise NotImplementedError

    def get_gt_poses(self):
        """Load ground-truth poses
        
        Returns:
            gt_poses (dict): each pose is a [4x4] array
        """
        raise NotImplementedError
    