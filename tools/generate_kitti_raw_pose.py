''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-20
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: This program generates ground truth pose of KITTI Raw dataset
'''

import argparse
import os

from libs.general.utils import load_poses_from_oxts, save_traj
from libs.general.utils import mkdir_if_not_exists


def argument_parsing():
    """Argument parsing

    Returns:
        args (args): arguments
    """
    parser = argparse.ArgumentParser(description='Ground truth pose generation for KITTI Raw dataset')
    parser.add_argument('--result_dir', type=str, 
                        default="dataset/kitti_raw_pose",
                        help="Result directory")
    parser.add_argument('--data_dir', type=str, 
                        default="dataset/kitti_raw",
                        help="Raw dataset directory")
    parser.add_argument('--seqs', 
                        nargs="+",
                        help="sequences to be processed",
                        default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # argument parsing
    args = argument_parsing()

    for seq in args.seqs:
        # get gps data dir
        gps_info_dir =  os.path.join(
                    args.data_dir,
                    seq[:10],
                    seq,
                    "oxts/data"
                    )
        
        # load poses
        gt_poses = load_poses_from_oxts(gps_info_dir)

        # save poses
        traj_txt = os.path.join(args.result_dir, "{}.txt".format(seq))
        save_traj(traj_txt, gt_poses, format="kitti")
