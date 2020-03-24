# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

import copy
from glob import glob
import os

from .dataset import Dataset
from tool.evaluation.tum_tool.associate import associate, read_file_list


class TUM(Dataset):
    def __init__(self, *args, **kwargs):
        super(TUM, self).__init__(*args, **kwargs)
        return

    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        mainly for TUM-RGBD dataset
        Returns:
            rgb_d_pose_pair (dict):
                - rgb_timestamp: {depth: depth_timestamp, pose: pose_timestamp}
        """
        self.rgb_d_pose_pair = {}

        # associate rgb-depth-pose timestamp pair
        rgb_list = read_file_list(self.data_dir['img'] +"/../rgb.txt")
        depth_list = read_file_list(self.data_dir['img'] +"/../depth.txt")
        pose_list = read_file_list(self.data_dir['img'] +"/../groundtruth.txt")

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
            if self.rgb_d_pose_pair[rgb_stamp].get("depth", -1) == -1:
                to_del_pair.append(rgb_stamp)
        for rgb_stamp in to_del_pair:
            del(self.rgb_d_pose_pair[rgb_stamp])
        
        # # Clear pairs without pose
        to_del_pair = []
        tmp_rgb_d_pose_pair = copy.deepcopy(self.rgb_d_pose_pair)
        for rgb_stamp in tmp_rgb_d_pose_pair:
            if self.rgb_d_pose_pair[rgb_stamp].get("pose", -1) == -1:
                to_del_pair.append(rgb_stamp)
        for rgb_stamp in to_del_pair:
            del(tmp_rgb_d_pose_pair[rgb_stamp])
        
        # timestep
        timestep = 5
        to_del_pair = []
        for cnt, rgb_stamp in enumerate(self.rgb_d_pose_pair):
            if cnt % timestep != 0:
                to_del_pair.append(rgb_stamp)
        for rgb_stamp in to_del_pair:
            del(self.rgb_d_pose_pair[rgb_stamp])
    
    def update_gt_pose(self):
        """update GT pose according to sync pairs
        """
        # Update gt pose
        self.tmp_gt_poses = {}
        gt_pose_0_time = tmp_rgb_d_pose_pair[sorted(list(tmp_rgb_d_pose_pair.keys()))[0]]['pose']
        gt_pose_0 = self.gt_poses[gt_pose_0_time]
        
        i = 0
        for rgb_stamp in sorted(list(self.rgb_d_pose_pair.keys())):
            if self.rgb_d_pose_pair[rgb_stamp].get("pose", -1) != -1:
                self.tmp_gt_poses[i] = np.linalg.inv(gt_pose_0) @ self.gt_poses[self.rgb_d_pose_pair[rgb_stamp]['pose']]
            else:
                self.tmp_gt_poses[i] = np.eye(4)
            i += 1
        self.gt_poses = copy.deepcopy(self.tmp_gt_poses)
    
    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        tum_intrinsics = {
                "tum-1": [318.6, 255.3, 517.3, 516.5],  # fr1
                "tum-2": [325.1, 249.7, 520.9, 521.0],  # fr2
                "tum-3": [320.1, 247.6, 535.4, 539.2],  # fr3
            }
        intrinsics_param = tum_intrinsics[self.cfg.dataset]
        return intrinsics_param
    
    def get_data_dir(self):
        """Get data directory

        Returns:
            data_dir (dict):
                - img (str): image data directory
                - (optional) depth (str): depth data direcotry or None
                - (optional) depth_src (str): depth data type [gt/None]
        """
        data_dir = {}

        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        data_dir['img'] = os.path.join(img_seq_dir, "rgb")
        
        # get depth data directory
        depth_src_cases = {
            0: "gt",
            1: "pred",
            None: None
            }
        data_dir['depth_src'] = depth_src_cases[self.cfg.depth.depth_src]

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = "{}/{}/depth".format(
                                self.cfg.directory.depth_dir, self.cfg.seq
                                )
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
 
        return data_dir

