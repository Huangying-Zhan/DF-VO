''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: Dataset loaders for KITTI Driving Sequence
'''

from glob import glob
import os

from .dataset import Dataset
from libs.general.utils import *


class KITTI(Dataset):
    """Base class of dataset loaders for KITTI Driving Sequence
    """
    
    def __init__(self, *args, **kwargs):
        super(KITTI, self).__init__(*args, **kwargs)
        self.stereo_baseline = 0.54
        return

    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        
        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        self.rgb_d_pose_pair = {}
        len_seq = len(glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext))))
        for i in range(len_seq):
            self.rgb_d_pose_pair[i] = {}
            self.rgb_d_pose_pair[i]['depth'] = i
            self.rgb_d_pose_pair[i]['pose'] = i
    
    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

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
            poses (dict): poses, each pose is a [4x4] array
        """
        global_poses_arr = convert_SE3_to_arr(poses)
        save_traj(traj_txt, global_poses_arr, format='kitti')


class KittiOdom(KITTI):
    """Dataset loader for KITTI Oodmetry Split
    """
    
    def __init__(self, *args, **kwargs):
        super(KittiOdom, self).__init__(*args, **kwargs)

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (list): [cx, cy, fx, fy]
        """
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        intrinsics_param = load_kitti_odom_intrinsics(
                        os.path.join(img_seq_dir, "calib.txt"),
                        self.cfg.image.height, self.cfg.image.width
                        )[2]
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
        data_dir['img'] = os.path.join(img_seq_dir, "image_2")

        # get depth data directory
        data_dir['depth_src'] = self.cfg.depth.depth_src

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = "{}/gt/{}/".format(
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
                            "{}.txt".format(self.cfg.seq)
                            )
        gt_poses = load_poses_from_txt(annotations)
        return gt_poses
    
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
     
    def get_depth(self, timestamp):
        """Get GT/precomputed depth data given the timestamp

        Args:
            timestamp (int): timestamp for the depth

        Returns:
            depth (array, [HxW]): depth data
        """
        img_id = self.rgb_d_pose_pair[timestamp]['depth']

        if self.data_dir['depth_src'] == "gt":
            img_name = "{:010d}.png".format(img_id)
            scale_factor = 500
        elif self.data_dir['depth_src'] == "pred":
            img_name = "depth/{:06d}.png".format(img_id)
            scale_factor = 1000
        
        img_h, img_w = self.cfg.image.height, self.cfg.image.width
        depth_path = os.path.join(self.data_dir['depth'], img_name)
        depth = read_depth(depth_path, scale_factor, [img_h, img_w])
        return depth

class KittiRaw(KITTI):
    """Dataset loader for KITTI Raw dataset
    """
    
    def __init__(self, *args, **kwargs):
        super(KittiRaw, self).__init__(*args, **kwargs)

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (list): [cx, cy, fx, fy]
        """
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq[:10],
                            )
        intrinsics_param = load_kitti_raw_intrinsics(
                        os.path.join(img_seq_dir, "calib_cam_to_cam.txt"),
                        self.cfg.image.height, self.cfg.image.width
                        )[2]
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

        img_seq_dir = os.path.join(
                        self.cfg.directory.img_seq_dir,
                        self.cfg.seq[:10],
                        self.cfg.seq
                        )
        data_dir['img'] = os.path.join(img_seq_dir, "image_02/data")
        
        # get depth data directory
        data_dir['depth_src'] = self.cfg.depth.depth_src

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = os.path.join(self.cfg.directory.depth_dir, self.cfg.seq)
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
        else:
            assert False, "Wrong depth src [{}] is given.".format(data_dir['depth_src'])
       
        return data_dir

    def get_gt_poses(self):
        """Get ground-truth poses from oxts data
        
        Returns:
            gt_poses (dict): each pose is a [4x4] array
        """
        seq_date = self.cfg.seq[:10]
        gps_info_dir =  os.path.join(
                            self.cfg.directory.gt_pose_dir,
                            seq_date,
                            self.cfg.seq,
                            "oxts/data"
                            )
        gt_poses = load_poses_from_oxts(gps_info_dir)
        return gt_poses

    def get_image(self, timestamp):
        """Get image data given the image timestamp

        Args:
            timestamp (int): timestamp for the image
            
        Returns:
            img (array, [CxHxW]): image data
        """
        img_path = os.path.join(self.data_dir['img'], 
                            "{:010d}.{}".format(timestamp, self.cfg.image.ext)
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
            img_name = "{:010d}.png".format(img_id)
            scale_factor = 500
        
        img_h, img_w = self.cfg.image.height, self.cfg.image.width
        depth_path = os.path.join(self.data_dir['depth'], img_name)
        depth = read_depth(depth_path, scale_factor, [img_h, img_w])
        return depth
