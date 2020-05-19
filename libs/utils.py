# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.getcwd())
from tools.evaluation.tum_tool.pose_evaluation_utils import quat2mat
from .kitti_raw_utils import generate_pose


def read_image(path, h, w):
    """read image data and convert to RGB
    Args:
        path (str): image path
    Returns:
        img (HxWx3 array): image data
    """
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    return img


def read_depth(path, scale, target_size=None):
    """Read depth png and resize it if necessary
    Args:
        path (str): depth png path
        scale (float): scaling factor for reading png
        target_szie (int list): [target_height, target_width]
    Return:
        depth (HxW array): depth map
    """
    depth = cv2.imread(path, -1) / scale
    if target_size is not None:
        img_h, img_w = target_size
        depth = cv2.resize(depth,
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                        )
    return depth


def save_depth_png(depth, png, png_scale):
    """save depth map
    Args:
        depth (HxW): depth map
        png (str): path for saving depth map PNG file
        png_scale (float): scaling factor for saving PNG file
    """
    depth = np.clip(depth, 0, 65535 / png_scale)
    depth = (depth * png_scale).astype(np.uint16)
    cv2.imwrite(png, depth)
    

def preprocess_depth(depth, crop, depth_range):
    """preprocess depth map
    Args:
        depth (HxW array): depth map
        crop (list): normalized crop regions. non-cropped regions set to 0
            - [[y0, y1], [x0, x1]]
        depth_range (float list): [min_depth, max_depth]
    Returns:
        depth (HxW array): processed depth map
    """
    min_depth, max_depth = depth_range
    h, w = depth.shape
    y0, y1 = int(h*crop[0][0]), int(h*crop[0][1])
    x0, x1 = int(w*crop[1][0]), int(w*crop[1][1])
    depth_mask = np.zeros((h, w))
    depth_mask[y0:y1, x0:x1] = 1
    depth_range_mask = (depth < max_depth) * (depth > min_depth)
    valid_mask = depth_mask * depth_range_mask
    depth = depth * valid_mask
    return depth


def image_shape(img):
    """return image shape
    Args:
        img (HxWx(c) array): image
    Returns:
        h,w,c (int): if image has 3 channels
        h,w,1 (int): if image has 2 channels
    """
    if len(img.shape) == 3:
        return img.shape
    elif len(img.shape) == 2:
        h, w = img.shape
        return h, w, 1


def skew(x):
    """Create skew-symetric matrix from a vector
    Args:
        x: 1D vector
    Returns:
        M (3x3 np.array): skew-symetric matrix
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, x[0]],
                     [x[1], x[0], 0]])


def load_poses_from_txt(file_name):
    """ Load absolute camera poses from text file
    Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)

    Args:
        file_name: txt file path
    Returns:
        poses: dictionary of poses, each pose is 4x4 array
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ")]
        withIdx = int(len(line_split) == 13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses


def load_poses_from_oxts(oxts_dir):
    """ Load absolute camera poses from oxts files
    
    Args:
        oxts_dir: directory stores oxts data
    Returns:
        poses: dictionary of poses, each pose is 4x4 array
    """
    poses = {}
    len_seq = len(glob(os.path.join(oxts_dir, "*.txt")))
    for i in range(len_seq):
        poses[i] = generate_pose(oxts_dir, i, do_flip=False)
    return poses


def load_poses_from_txt_tum(file_name):
    """ Load absolute camera poses from text file (tum format)
    Each line in the file should follow the following structure
        timestamp tx ty tz qx qy qz qw

    Args:
        file_name: txt file path
    Returns:
        poses: dictionary of poses, each pose is 4x4 array
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        line_split = line.split(" ")

        # Comments
        if line_split[0] == "#":
            continue
        
        # Get raw data
        line_split = [float(i) for i in line.split(" ")]
        P = np.eye(4)
        timestamp, tx, ty, tz, qx, qy, qz, qw = line_split

        # quat -> Rotation matrix
        P[:3, :3] = quat2mat([qw, qx, qy, qz])
        P[:3, 3] = np.asarray([tx, ty, tz])
        
        poses[timestamp] = P
    
    pose_0 = poses[list(poses.keys())[0]]
    for timestamp in poses:
        poses[timestamp] = np.linalg.inv(pose_0) @ poses[timestamp]
    return poses


def load_kitti_odom_intrinsics(file_name, new_h, new_w):
    """Load kitti odometry data intrinscis
    Args:
        file_name (str): txt file path
    Returns:
        params (dict): each element contains [cx, cy, fx, fy]
            - 0: [cx, cy, fx, fy]_cam0
            - 1: [cx, cy, fx, fy]_cam1
            - ...
    """
    raw_img_h = 370.0
    raw_img_w = 1226.0
    intrinsics = {}
    with open(file_name, 'r') as f:
        s = f.readlines()
        for cnt, line in enumerate(s):
            line_split = [float(i) for i in line.split(" ")[1:]] 
            intrinsics[cnt] = [
                            line_split[2]/raw_img_w*new_w,
                            line_split[6]/raw_img_h*new_h,
                            line_split[0]/raw_img_w*new_w,
                            line_split[5]/raw_img_h*new_h,
                            ]
    return intrinsics


def load_kitti_raw_intrinsics(file_name, new_h, new_w):
    """Load kitti raw data intrinscis
    Args:
        file_name (str): txt file path
    Returns:
        params (dict): each element contains [cx, cy, fx, fy]
            - 0: [cx, cy, fx, fy]_cam0
            - 1: [cx, cy, fx, fy]_cam1
            - ...
    """
    raw_img_h = 370.0
    raw_img_w = 1226.0
    intrinsics = {}
    with open(file_name, 'r') as f:
        s = f.readlines()
        for line in s:
            if "P_rect" in line:
                line_split = [float(i) for i in line.split(" ")[1:]]
                cnt = int(line.split(":")[0][-2:])
                intrinsics[cnt] = [
                                line_split[2]/raw_img_w*new_w,
                                line_split[6]/raw_img_h*new_h,
                                line_split[0]/raw_img_w*new_w,
                                line_split[5]/raw_img_h*new_h,
                                ]
    return intrinsics


def image_grid(h, w):
    """Generate regular image grid
    Args:
        h (int): image height
        w (int): image width
    Returns:
        grid (HxWx2): regular image grid contains [x,y]
    """
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xv, yv = np.meshgrid(x, y)
    grid = np.transpose(np.stack([xv, yv]), (1, 2, 0))
    return grid


def convert_SE3_to_arr(SE3_dict, timestamps=None):
    """Convert SE3 dictionary to array dictionary
    Args:
        SE3_dict (SE3 dict): SE3 dictionary
        timestamps (float list): timestamp list
    Returns:
        poses_dict (array dict): each pose contains 4x4 array
    """
    poses_dict = {}
    if timestamps is None:
        key_list = sorted(list(SE3_dict.keys()))
    else:
        key_list = timestamps
    for cnt, i in enumerate(SE3_dict):
        poses_dict[key_list[cnt]] = SE3_dict[i].pose
    return poses_dict


def save_traj(txt, poses, format="kitti"):
    """Save trajectory (absolute poses) as KITTI odometry file format
    Args:
        txt (str): pose text file path
        poses (array dict): poses, each pose is 4x4 array
        format (str): trajectory format
            - kitti: 12 parameters
            - tum: timestamp tx ty tz qx qy qz qw
    """
    with open(txt, "w") as f:
        for i in poses:
            pose = poses[i]
            if format == "kitti":
                pose = pose.flatten()[:12]
                line_to_write = " ".join([str(j) for j in pose])
            elif format == "tum":
                qw, qx, qy, qz = rot2quat(pose[:3, :3])
                tx, ty, tz = pose[:3, 3]
                line_to_write = " ".join([
                                    str(i), 
                                    str(tx), str(ty), str(tz),
                                    str(qx), str(qy), str(qz), str(qw)]
                                    )
            f.writelines(line_to_write+"\n")
    print("Trajectory saved.")