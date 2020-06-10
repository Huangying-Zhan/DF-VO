''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-10
@LastEditors: Huangying Zhan
@Description: Dataset loaders for Kinect captures (TUM RBG-D format)
'''

import copy
from glob import glob
import os

from tools.evaluation.tum_tool.associate import associate, read_file_list

from .dataset import Dataset
from libs.general.utils import *


class Kinect(Dataset):
    """Dataset loader for Kinect RBG-D dataset
    """

    def __init__(self, *args, **kwargs):
        super(Kinect, self).__init__(*args, **kwargs)

        # update gt poses for sync pairs
        if self.cfg.directory.gt_pose_dir is not None:
            self.update_gt_pose()

    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        
        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        self.pose_file_name = 'keyframe_trajectory_mono.txt'
        self.rgb_d_pose_pair = {}

        # associate rgb-depth-pose timestamp pair
        rgb_list = read_file_list(self.data_dir['img'] +'/../rgb.txt')
        depth_list = read_file_list(self.data_dir['img'] +'/../depth.txt')

        pose_list = read_file_list(self.data_dir['img'] +'/../{}'.format(self.pose_file_name))

        for i in rgb_list:
            self.rgb_d_pose_pair[i] = {}

        # associate rgb-d
        matches = associate(
            first_list=rgb_list,
            second_list=depth_list,
            offset=0,
            max_difference=0.02
        )
        for match in matches:
            rgb_stamp = match[0]
            depth_stamp = match[1]
            self.rgb_d_pose_pair[rgb_stamp]['depth'] = depth_stamp
        
        # associate rgb-pose
        matches = associate(
            first_list=rgb_list,
            second_list=pose_list,
            offset=0,
            max_difference=0.02
        )
        
        for match in matches:
            rgb_stamp = match[0]
            pose_stamp = match[1]
            self.rgb_d_pose_pair[rgb_stamp]['pose'] = pose_stamp
        
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
    
    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (list): [cx, cy, fx, fy]
        """
        # tum_intrinsics = {
        #         'tum-1': [318.6, 255.3, 517.3, 516.5],  # fr1
        #         'tum-2': [325.1, 249.7, 520.9, 521.0],  # fr2
        #         'tum-3': [320.1, 247.6, 535.4, 539.2],  # fr3
        #     }
        # intrinsics_param = tum_intrinsics[self.cfg.dataset]
        raw_img_h = 1080.0
        raw_img_w = 1920.0

        new_w = 640
        new_h = 480
        line_split = [972.34, 532.64, 1032.66, 1033.17] 
        intrinsics_param = [
                        line_split[0]/raw_img_w*new_w,
                        line_split[1]/raw_img_h*new_h,
                        line_split[2]/raw_img_w*new_w,
                        line_split[3]/raw_img_h*new_h,
                        ]
        return intrinsics_param
    
    def get_data_dir(self):
        """Get data directory

        Returns:
            a dictionary containing
                - **img** (str) : image data directory
                - (optional) **depth** (str) : depth data direcotry or None
                - (optional) **depth_src** (str) : depth data type [gt/None]
        """
        data_dir = {}

        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        data_dir['img'] = os.path.join(img_seq_dir, "rgb")
        
        # get depth data directory
        data_dir['depth_src'] = self.cfg.depth.depth_src

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = "{}/{}/depth".format(
                                self.cfg.directory.depth_dir, self.cfg.seq
                                )
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
        else:
            assert False, "Wrong depth src [{}] is given.".format(data_dir['depth_src'])
 
        return data_dir

    def get_gt_poses(self):
        """Get ground-truth poses
        
        Returns:
            gt_poses (dict): each pose is a [4x4] array
        """
        annotations = os.path.join(
                            self.cfg.directory.gt_pose_dir,
                            self.cfg.seq,
                            self.pose_file_name
                            )
        gt_poses = load_poses_from_txt_tum(annotations)
        return gt_poses
    
    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

        Args:
            img_id (int): query image id

        Returns:
            timestamp (int): timestamp for query image
        """
        return sorted(list(self.rgb_d_pose_pair.keys()))[img_id]
    
    def get_image(self, timestamp):
        """Get image data given the image timestamp

        Args:
            timestamp (int): timestamp for the image
            
        Returns:
            img (array, [CxHxW]): image data
        """
        img_path = os.path.join(self.data_dir['img'], 
                            "{}.{}".format(int(timestamp/0.1), self.cfg.image.ext)
                            )
        img = read_image(img_path, self.cfg.image.height, self.cfg.image.width)
        return img
    
    def get_depth(self, timestamp):
        """Get GT/precomputed depth data given the timestamp

        Args:
            timestamp (int): timestamp for the depth

        Returns:
            depth (array, [HxW]): depth data
        """
        img_id = self.rgb_d_pose_pair[timestamp]['depth']

        if self.data_dir['depth_src'] == "gt":
            img_name = "{}.png".format(int(img_id/0.1))
            scale_factor = 5000
        else:
            assert False, "Proper depth loader should be defined."
        
        img_h, img_w = self.cfg.image.height, self.cfg.image.width
        depth_path = os.path.join(self.data_dir['depth'], img_name)
        depth = read_depth(depth_path, scale_factor, [img_h, img_w])
        return depth

    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as TUM odometry file format

        Args:
            txt (str): pose text file path
            poses (dict): poses, each pose is a [4x4] array
        """
        timestamps = sorted(list(self.rgb_d_pose_pair.keys()))
        global_poses_arr = convert_SE3_to_arr(poses, timestamps)
        save_traj(traj_txt, global_poses_arr, format='tum')

