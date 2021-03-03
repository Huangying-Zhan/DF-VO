''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-20
@LastEditors: Huangying Zhan
@Description: Provides helper methods for loading and parsing KITTI Raw data
'''
import numpy as np
from collections import namedtuple
import os

from .kitti_utils import *


OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' +
                            'roll, pitch, yaw, ' +
                            'vn, ve, vf, vl, vu, ' +
                            'ax, ay, az, af, al, au, ' +
                            'wx, wy, wz, wf, wl, wu, ' +
                            'pos_accuracy, vel_accuracy, ' +
                            'navstat, numsats, ' +
                            'posmode, velmode, orimode')


def generate_pose(seq, frame_idx, do_flip):
    """Get pose for a frame in a sequence

    Args:
        seq (str): sequence oxts_dir directory
        frame_idx (int): frame index
        do_flip (bool): flip sequence horizontally
        
    Returns:
        pose (array, [4x4]): absolute pose w.r.t frame-0
    """
    # Read oxts data
    oxts_files = [
            os.path.join(seq, "{:010}.txt".format(0)),
            os.path.join(seq, "{:010}.txt".format(frame_idx))
            ]
    oxts_packets = []
    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                data = OxtsPacket(*line)
                oxts_packets.append(data)

    # get absolute pose w.r.t frame-0
    gps_poses = poses_from_oxts(oxts_packets)

    # convert from GPS coordinate system to camera coordinate system
    # - Camera:   x: right,   y: down,  z: forward
    # - GPS/IMU:  x: forward, y: left,  z: up
    T = np.eye(4)
    T[:3, :3] = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]]
                )
    T_01 = np.linalg.inv(gps_poses[0]) @ gps_poses[1]
    # pose = (T @ gps_poses[0]) @ np.linalg.inv(T @ gps_poses[1])
    pose = T @ T_01 @ np.linalg.inv(T)

    if do_flip:
        pose[:3, :3] = flip_rotation(pose[:3, :3])
        pose[0, 3] = -pose[0, 3]

    return pose


def flip_rotation(R):
    """Transform rotation when there is a flipping of image along x-axis

    Args:
        R (array, [3x3]): rotation matrix

    Returns:
        new_R (array, [3x3]): new rotation matrix
    """
    theta_x = np.arctan2(R[2,1], R[2,2])
    theta_y = np.arctan2(-R[2,0], np.linalg.norm([R[2,1], R[2,2]]))
    theta_z = np.arctan2(R[1,0], R[0,0])

    R_x = np.asarray([[1, 0, 0],
                      [0, np.cos(theta_x), -np.sin(theta_x)],
                      [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.asarray([[np.cos(theta_y), 0, np.sin(theta_y)],
                      [0, 1, 0],
                      [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.asarray([[np.cos(theta_z), -np.sin(theta_z), 0],
                      [np.sin(theta_z), np.cos(theta_z), 0],
                      [0, 0, 1]])
    new_R = np.linalg.inv(R_z) @ np.linalg.inv(R_y) @ R_x
    return new_R


def poses_from_oxts(oxts_packets):
    """Helper method to compute SE(3) pose matrices from OXTS packets.
    
    Args:
        oxts_packets (namedtuple): oxts data
    
    Returns:
        poses (list): list of sensor poses
    """
    er = 6378137.  # earth radius (approx.) in meters

    # compute scale from first lat value
    scale = np.cos(oxts_packets[0].lat * np.pi / 180.)

    t_0 = []    # initial position
    poses = []  # list of poses computed from oxts
    for packet in oxts_packets:
        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * \
            np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # We want the initial position to be the origin, but keep the ENU
        # coordinate system
        if len(t_0) == 0:
            t_0 = t

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        # poses.append(transform_from_rot_trans(R, t - t_0))
        poses.append(transform_from_rot_trans(R, t))

    return poses