import os

from libs.general.utils import load_poses_from_oxts, save_traj
from libs.general.utils import mkdir_if_not_exists

seqs = ["2011_09_26_drive_0005_sync",
"2011_09_26_drive_0009_sync",
"2011_09_26_drive_0011_sync",
"2011_09_26_drive_0013_sync",
"2011_09_26_drive_0014_sync",
"2011_09_26_drive_0015_sync",
"2011_09_26_drive_0018_sync",
"2011_09_29_drive_0004_sync",
"2011_10_03_drive_0047_sync",]

img_seq_dir = "dataset/kitti_raw"
result_dir = "dataset/kitti_raw_pose"

for seq in seqs:
    gps_info_dir =  os.path.join(
                img_seq_dir,
                seq[:10],
                seq,
                "oxts/data"
                )
    gt_poses = load_poses_from_oxts(gps_info_dir)
    traj_txt = os.path.join(result_dir, "{}.txt".format(seq))
    save_traj(traj_txt, gt_poses, format="kitti")