# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

from glob import glob
import os

from .dataset import Dataset
from libs.utils import load_kitti_odom_intrinsics, load_kitti_raw_intrinsics


class KITTI(Dataset):
    def __init__(self, *args, **kwargs):
        super(KITTI, self).__init__(*args, **kwargs)
        return

    def synchronize_timestamps(self):
        self.rgb_d_pose_pair = {}
        len_seq = len(glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext))))
        for i in range(len_seq):
            self.rgb_d_pose_pair[i] = {}
            self.rgb_d_pose_pair[i]['depth'] = i
            self.rgb_d_pose_pair[i]['pose'] = i

class KittiOdom(KITTI):
    def __init__(self, *args, **kwargs):
        super(KittiOdom, self).__init__(*args, **kwargs)

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
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
        data_dir['img'] = os.path.join(img_seq_dir, "image_2")

        # get depth data directory
        depth_src_cases = {
            0: "gt",
            1: "pred",
            None: None
            }
        data_dir['depth_src'] = depth_src_cases[self.cfg.depth.depth_src]

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = "{}/gt/{}/".format(
                                self.cfg.directory.depth_dir, self.cfg.seq
                            )
        elif data_dir['depth_src'] == "pred":
            data_dir['depth'] = "{}/{}/".format(
                                self.cfg.directory.depth_dir, self.cfg.seq
                            )
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
 
        return data_dir


class KittiRaw(KITTI):
    def __init__(self, *args, **kwargs):
        super(KittiRaw, self).__init__(*args, **kwargs)

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
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

        img_seq_dir = os.path.join(
                        self.cfg.directory.img_seq_dir,
                        self.cfg.seq[:10],
                        self.cfg.seq
                        )
        data_dir['img'] = os.path.join(img_seq_dir, "image_02/data")
        
        # get depth data directory
        depth_src_cases = {
            0: "gt",
            1: "pred",
            None: None
            }
        data_dir['depth_src'] = depth_src_cases[self.cfg.depth.depth_src]

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = os.path.join(self.cfg.directory.depth_dir, self.cfg.seq)
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
       
        return data_dir

