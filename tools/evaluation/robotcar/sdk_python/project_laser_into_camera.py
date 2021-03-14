################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, help='Directory containing images')
parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
parser.add_argument('--image_idx', type=int, help='Index of image to display')

args = parser.parse_args()

model = CameraModel(args.models_dir, args.image_dir)

extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None

poses_type = re.search('(vo|ins|rtk)\.csv', args.poses_file).group(1)
if poses_type in ['ins', 'rtk']:
    with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
else:
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle


timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
if not os.path.isfile(timestamps_path):
    timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

timestamp = 0
with open(timestamps_path) as timestamps_file:
    for i, line in enumerate(timestamps_file):
        if i == args.image_idx:
            timestamp = int(line.split(' ')[0])

pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file, args.extrinsics_dir,
                                           timestamp - 1e7, timestamp + 1e7, timestamp)

pointcloud = np.dot(G_camera_posesource, pointcloud)

image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
image = load_image(image_path, model)

uv, depth = model.project(pointcloud, image.shape)

plt.imshow(image)
plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
plt.xlim(0, image.shape[1])
plt.ylim(image.shape[0], 0)
plt.xticks([])
plt.yticks([])
plt.show()
