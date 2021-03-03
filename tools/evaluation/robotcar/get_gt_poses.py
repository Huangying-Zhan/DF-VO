''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 1970-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-08
@Description: Get GT poses (KITTI format) for robotcar
'''

import numpy as np
import os

from sdk_python.interpolate_poses import interpolate_vo_poses
from libs.general.utils import save_traj, mkdir_if_not_exists

for seq in [
"2014-05-06-12-54-54",
"2014-05-06-13-09-52",
"2014-05-06-13-14-58",
"2014-05-06-13-17-51",
"2014-05-14-13-46-12",
"2014-05-14-13-50-20",
"2014-05-14-13-53-47",
"2014-05-14-13-59-05",
"2014-06-25-16-22-15",]:

    # seq = "2014-05-06-13-14-58"
    dataset_dir = "dataset/robotcar"
    time_offset = 20

    result_dir = "dataset/robotcar/gt_poses_20"

    # Load data
    timestamp_txt = os.path.join(dataset_dir, seq, "stereo.timestamps")
    timestamps = np.loadtxt(timestamp_txt)[:, 0].astype(np.int)
    origin_timestamp = list(timestamps)

    raw_vo_path = os.path.join(dataset_dir, seq, "vo/vo.csv")

    poses = interpolate_vo_poses(raw_vo_path, origin_timestamp, origin_timestamp[time_offset])

    # transformation 
    T = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    poses_dict = {}
    for i in range(time_offset, len(poses)):
        # poses_dict[i-time_offset] = np.asarray(poses[i])
        poses_dict[i-time_offset] = T @ np.asarray(poses[i]) @ np.linalg.inv(T)

    mkdir_if_not_exists(result_dir)
    save_traj(os.path.join(result_dir, "{}.txt".format(seq)), poses_dict)