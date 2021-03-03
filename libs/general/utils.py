''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-25
@LastEditors: Huangying Zhan
@Description: utils.py contains varies methods for general purpose
'''

import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

from tools.evaluation.tum_tool.pose_evaluation_utils import quat2mat, rot2quat

from .kitti_raw_utils import generate_pose


def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    
    Args:
        path (str): directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(path, h, w, crop=None):
    """read image data and convert to RGB

    Args:
        path (str): image path
        h (int): final image height
        w (int): final image width
        crop (array, [2x2]): [[y_crop_0, y_crop_1],[x_crop_0, x_crop_1]]
    
    Returns:
        img (array, [HxWx3]): image data
    """
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop is not None:
        img_h, img_w, _ = img.shape
        y0, y1 = int(img_h * crop[0][0]), int(img_h * crop[0][1])
        x0, x1 = int(img_w * crop[1][0]), int(img_w * crop[1][1])
        img = img[y0:y1, x0:x1]
    img = cv2.resize(img, (w, h))
    return img


def read_depth(path, scale, target_size=None):
    """Read depth png and resize it if necessary

    Args:
        path (str): depth png path
        scale (float): scaling factor for reading png
        target_size (list): [target_height, target_width]
    
    Returns:
        depth (array, [HxW]): depth map
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
        depth (array, [HxW]): depth map
        png (str): path for saving depth map PNG file
        png_scale (float): scaling factor for saving PNG file
    """
    depth = np.clip(depth, 0, 65535 / png_scale)
    depth = (depth * png_scale).astype(np.uint16)
    cv2.imwrite(png, depth)
    

def preprocess_depth(depth, crop, depth_range):
    """preprocess depth map with cropping and capping range

    Args:
        depth (array, [HxW]): depth map
        crop (list): normalized crop regions [[y0, y1], [x0, x1]]. non-cropped regions set to 0. 
        depth_range (list): a list with float numbers [min_depth, max_depth]
    
    Returns:
        depth (array, [HxW]): processed depth map
    """
    # set cropping region
    min_depth, max_depth = depth_range
    h, w = depth.shape
    y0, y1 = int(h*crop[0][0]), int(h*crop[0][1])
    x0, x1 = int(w*crop[1][0]), int(w*crop[1][1])
    depth_mask = np.zeros((h, w))
    depth_mask[y0:y1, x0:x1] = 1

    # set range mask
    depth_range_mask = (depth < max_depth) * (depth > min_depth)
    
    # set invalid pixel to zero depth
    valid_mask = depth_mask * depth_range_mask
    depth = depth * valid_mask
    return depth


def image_shape(img):
    """return image shape

    Args:
        img (array, [HxWx(c) or HxW]): image
    
    Returns:
        a tuple containing
            - **h** (int) : image height
            - **w** (int) : image width
            - **c** (int) : image channel
    """
    if len(img.shape) == 3:
        return img.shape
    elif len(img.shape) == 2:
        h, w = img.shape
        return h, w, 1


def skew(x):
    """Create skew-symetric matrix from a vector

    Args:
        x (list): 1D vector with 3 elements
    
    Returns:
        M (array, [3x3]): skew-symetric matrix
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
        file_name (str): txt file path
    
    Returns:
        poses (dict): dictionary of poses, each pose is a [4x4] array
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
        oxts_dir (str): directory stores oxts data

    Returns:
        poses (dict): dictionary of poses, each pose is a [4x4] array
    """
    poses = {}
    len_seq = len(glob(os.path.join(oxts_dir, "*.txt")))

    # check if directory is correct
    assert len_seq != 0, "Wrong path is given: [{}]".format(oxts_dir)

    for i in range(len_seq):
        poses[i] = generate_pose(oxts_dir, i, do_flip=False)
    return poses


def load_poses_from_txt_tum(file_name):
    """ Load absolute camera poses from text file (tum format)
    Each line in the file should follow the following structure
        timestamp tx ty tz qx qy qz qw

    Args:
        file_name (str): txt file path
    
    Returns:
        poses (dict): dictionary of poses, each pose is a [4x4] array
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
        intrinsics (dict): each element contains [cx, cy, fx, fy]
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
    """
    raw_img_h = 370.0
    raw_img_w = 1226.0
    intrinsics = {}
    with open(file_name, 'r') as f:
        s = f.readlines()
        for line in s:
            if 'P_rect' in line:
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
        grid (array, [HxWx2]): regular image grid contains [x,y]
    """
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xv, yv = np.meshgrid(x, y)
    grid = np.transpose(np.stack([xv, yv]), (1, 2, 0))
    return grid


def convert_SE3_to_arr(SE3_dict, timestamps=None):
    """Convert SE3 dictionary to array dictionary

    Args:
        SE3_dict (dict): a dictionary containing SE3s
        timestamps (list): a list of timestamps
    
    Returns:
        poses_dict (dict): each pose contains a [4x4] array
    """
    poses_dict = {}
    if timestamps is None:
        key_list = sorted(list(SE3_dict.keys()))
    else:
        key_list = timestamps
    for cnt, i in enumerate(SE3_dict):
        poses_dict[key_list[cnt]] = SE3_dict[i].pose
    return poses_dict


def save_traj(txt, poses, format='kitti'):
    """Save trajectory (absolute poses) as KITTI odometry file format

    Args:
        txt (str): pose text file path
        poses (dict): poses, each pose is a [4x4] array
        format (str): trajectory format [kitti, tum]. 
            - **kitti**: timestamp [12 parameters]; 
            - **tum**: timestamp tx ty tz qx qy qz qw
    """
    with open(txt, 'w') as f:
        for i in poses:
            pose = poses[i]
            if format == 'kitti':
                pose = pose.flatten()[:12]
                line_to_write = str(i) + " "
                line_to_write += " ".join([str(j) for j in pose])
            elif format == 'tum':
                qw, qx, qy, qz = rot2quat(pose[:3, :3])
                tx, ty, tz = pose[:3, 3]
                line_to_write = " ".join([
                                    str(i), 
                                    str(tx), str(ty), str(tz),
                                    str(qx), str(qy), str(qz), str(qw)]
                                    )
            f.writelines(line_to_write+"\n")
    print("Trajectory saved.")
