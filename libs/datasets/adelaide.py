'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-13
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-14
@LastEditors: Huangying Zhan
@Description: Dataset loader for Adelaide Driving Sequence (Huangying driving camera)
'''
# Copyright (C) Huangying Zhan 2020. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

import numpy as np
from glob import glob
import os

from .dataset import Dataset
from libs.utils import *


class Adelaide(Dataset):
    def __init__(self, *args, **kwargs):
        super(Adelaide, self).__init__(*args, **kwargs)
        return
    
    def synchronize_timestamps(self):
        self.rgb_d_pose_pair = {}
        len_seq = len(glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext))))
        for i in range(len_seq):
            self.rgb_d_pose_pair[i] = {}
            self.rgb_d_pose_pair[i]['depth'] = i
            self.rgb_d_pose_pair[i]['pose'] = i
    
    def get_timestamp(self, img_id):
        """get timestamp for the query img_id
        Args:
            img_id (int): query image id
        Returns:
            timestamp (int): timestamp for query image
        """
        return img_id

    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as KITTI odometry file format
        Args:
            txt (str): pose text file path
            poses (array dict): poses, each pose is 4x4 array
            format (str): trajectory format
                - kitti: 12 parameters
                - tum: timestamp tx ty tz qx qy qz qw
        """
        global_poses_arr = convert_SE3_to_arr(poses)
        save_traj(traj_txt, global_poses_arr, format="kitti")


class AdelaideHY(Adelaide):
    def __init__(self, *args, **kwargs):
        self.height = 256
        self.width = 832
        super(AdelaideHY, self).__init__(*args, **kwargs)
    
    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        K = np.loadtxt(os.path.join(img_seq_dir, "cam.txt"))
        K[0] /= (self.cfg.image.width / self.width)
        K[1] /= (self.cfg.image.height / self.height)

        intrinsics_param = [K[0,2], K[1,2], K[0,0], K[1,1]]
        return intrinsics_param
    
    def get_data_dir(self):
        """Get data directory

        Returns:
            data_dir (dict):
                - img (str): image data directory
                - (optional) depth (str): depth data direcotry or None
                - (optional) depth_src (str): depth data type [gt/None]
        """
        data_dir = {"depth": None, "depth_src": None}

        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        data_dir['img'] = os.path.join(img_seq_dir)
 
        return data_dir
    
    def get_gt_poses(self):
        """ load ground-truth poses
        Returns:
            gt_poses (dict): each pose is 4x4 array
        """
        raise NotImplementedError
    
    def get_image(self, timestamp):
        """get image data given the image timestamp
        Args:
            timestamp (int): timestamp for the image
        Returns:
            img (CxHxW): image data
        """
        img_path = os.path.join(self.data_dir['img'], 
                            "{:06d}.{}".format(timestamp, self.cfg.image.ext)
                            )
        img = read_image(img_path, self.cfg.image.height, self.cfg.image.width)
        return img
    
    def get_depth(self, timestamp):
        """get GT/precomputed depth data given the timestamp
        Args:
            timestamp (int): timestamp for the depth
        Returns:
            depth (HxW): depth data
        """
        raise NotImplementedError


class AdelaideHY2(Adelaide):
    def __init__(self, *args, **kwargs):
        self.height = 512
        self.width = 1664
        super(AdelaideHY2, self).__init__(*args, **kwargs)
    
    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        K = np.loadtxt(os.path.join(img_seq_dir, "cam.txt"))
        K[0] *= (self.cfg.image.width / self.width)
        K[1] *= (self.cfg.image.height / self.height)

        intrinsics_param = [K[0,2], K[1,2], K[0,0], K[1,1]]
        return intrinsics_param
    
    def get_data_dir(self):
        """Get data directory

        Returns:
            data_dir (dict):
                - img (str): image data directory
                - (optional) depth (str): depth data direcotry or None
                - (optional) depth_src (str): depth data type [gt/None]
        """
        data_dir = {"depth": None, "depth_src": None}

        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        data_dir['img'] = os.path.join(img_seq_dir)
 
        return data_dir
    
    def get_gt_poses(self):
        """ load ground-truth poses
        Returns:
            gt_poses (dict): each pose is 4x4 array
        """
        raise NotImplementedError
    
    def get_image(self, timestamp):
        """get image data given the image timestamp
        Args:
            timestamp (int): timestamp for the image
        Returns:
            img (CxHxW): image data
        """
        img_path = os.path.join(self.data_dir['img'], 
                            "{:06d}.{}".format(timestamp, self.cfg.image.ext)
                            )
        img = read_image(img_path, self.cfg.image.height, self.cfg.image.width)
        return img
    
    def get_depth(self, timestamp):
        """get GT/precomputed depth data given the timestamp
        Args:
            timestamp (int): timestamp for the depth
        Returns:
            depth (HxW): depth data
        """
        raise NotImplementedError
    