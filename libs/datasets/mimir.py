''''''
'''
@Author: Olaya Alvarez Tunon (olaya.tunon@gmail.com)
@Date: 2022-10-31
@LastEditTime: 2022-10-31
@LastEditors: Olaya Alvarez Tunon
@Description: Dataset loaders for EuRoC Sequence
'''

import copy
from glob import glob
import os
import re
import yaml 
import pandas as pd
from scipy.spatial.transform import Rotation as R

from tools.evaluation.tum_tool.associate import associate

from .dataset import Dataset
from libs.general.utils import *

class MIMIR(Dataset):
    ''' Dataset loader for EuRoC sequence
    '''
    def __init__(self, *args, **kwargs):
        super(MIMIR, self).__init__(*args, **kwargs)
        self.gt_timestamps = []
        # update gt poses for sync pairs
        if self.cfg.directory.gt_pose_dir is not None:
            self.update_gt_pose()

    def update_gt_pose(self):
        """Update GT pose according to sync pairs
        """
        # Update gt pose
        self.tmp_gt_poses = {}
        sorted_timestamps = sorted(list(self.rgb_d_pose_pair.keys()))
        gt_pose_0_time = self.rgb_d_pose_pair[sorted_timestamps[0]]['pose']
        gt_pose_0 = self.gt_poses[gt_pose_0_time]
        
        i = 0
        for rgb_stamp in sorted(list(self.rgb_d_pose_pair.keys())):
            if self.rgb_d_pose_pair[rgb_stamp].get('pose', -1) != -1:
                self.tmp_gt_poses[i] = np.linalg.inv(gt_pose_0) @ self.gt_poses[self.rgb_d_pose_pair[rgb_stamp]['pose']]
            else:
                self.tmp_gt_poses[i] = np.eye(4)
            i += 1
        self.gt_poses = copy.deepcopy(self.tmp_gt_poses)
        
    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        
        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        self.rgb_d_pose_pair = {}

        # associate rgb-depth-pose timestamp pair
        rgb_list = glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext)))
        # depth_list = read_file_list(os.path.join(self.cfg.directory.gt_pose_dir,'data.csv'))
        pose_dict = self.get_gt_poses()
        rgb_dict = {}
        for i,dir in enumerate(rgb_list):
            timestamp = [int(s) for s in re.findall(r'\b\d+\b', dir)][0]
            aux = {timestamp: dir}
            rgb_dict.update(aux)
            self.rgb_d_pose_pair[timestamp] = {}

        # associate rgb-pose
        matches = associate(
            first_list=rgb_dict,
            second_list=pose_dict,
            offset=0,
            max_difference=0.02
        )

        for match in matches:
            rgb_stamp = match[0]
            pose_stamp = match[1]
            self.rgb_d_pose_pair[rgb_stamp].update({'pose':pose_stamp, 'depth':pose_stamp})
            # self.rgb_d_pose_pair[rgb_stamp]['pose'] = pose_stamp
            # self.rgb_d_pose_pair[rgb_stamp]['depth'] = pose_stamp #fake

        # Clear pairs without depth
        to_del_pair = []
        for rgb_stamp in self.rgb_d_pose_pair:
            if self.rgb_d_pose_pair[rgb_stamp].get('depth', -1) == -1:
                to_del_pair.append(rgb_stamp)
        for rgb_stamp in to_del_pair:
            del(self.rgb_d_pose_pair[rgb_stamp])
       
        # # Clear pairs without pose
        to_del_pair = []
        tmp_rgb_d_pose_pair = copy.deepcopy(self.rgb_d_pose_pair)
        for rgb_stamp in tmp_rgb_d_pose_pair:
            if self.rgb_d_pose_pair[rgb_stamp].get('pose', -1) == -1:
                to_del_pair.append(rgb_stamp)
        for rgb_stamp in to_del_pair:
            del(self.rgb_d_pose_pair[rgb_stamp])
        
        # timestep
        timestep = 1
        to_del_pair = []
        for cnt, rgb_stamp in enumerate(self.rgb_d_pose_pair):
            if cnt % timestep != 0:
                to_del_pair.append(rgb_stamp)
        for rgb_stamp in to_del_pair:
            del(self.rgb_d_pose_pair[rgb_stamp])

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (list): [cx, cy, fx, fy]
        """

        # Read MIMIR's YAML file
        with open(os.path.join(self.cfg.directory.img_seq_dir, "sensor.yaml"), 'r') as stream:
            sensor = yaml.safe_load(stream)
        intrinsics = sensor['intrinsics']
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]

        intrinsics_param = [cx, cy, fx, fy]
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
        img_seq_dir = self.cfg.directory.img_seq_dir
 
        data_dir['img'] = img_seq_dir
 
        return data_dir

    def get_image(self, timestamp):
        """Get image data given the image timestamp

        Args:
            timestamp (int): timestamp for the image
            
        Returns:
            img (array, [CxHxW]): image data
        """
        img_path = os.path.join(self.data_dir['img'], 
                            "{:06d}.{}".format(timestamp, self.cfg.image.ext)
                            )
        img = read_image(img_path, self.cfg.image.height, self.cfg.image.width)
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
        annotations = os.path.join(self.cfg.directory.gt_pose_dir,'data.csv')
        gt_poses = load_poses_from_txt_euroc(annotations)

        return gt_poses

    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

        Args:
            img_id (int): query image id

        Returns:
            timestamp (int): timestamp for query image
        """
        return sorted(list(self.rgb_d_pose_pair.keys()))[img_id]

    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as TUM odometry file format

        Args:
            txt (str): pose text file path
            poses (dict): poses, each pose is a [4x4] array
        """
        timestamps = sorted(list(self.rgb_d_pose_pair.keys()))
        global_poses_arr = convert_SE3_to_arr(poses, timestamps)
        save_traj(traj_txt, global_poses_arr, format='tum')