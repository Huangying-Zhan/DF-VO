''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 1970-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-02
@LastEditors: Huangying Zhan
@Description: This tool undistort Oxford Robotcar sequences
'''


import argparse
import cv2
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel
from tqdm import tqdm

from libs.general.utils import mkdir_if_not_exists


parser = argparse.ArgumentParser(description='Undistort images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--result_dir', type=str, default=None, help='directory to save undistorted images')

args = parser.parse_args()

# create result directory
mkdir_if_not_exists(args.result_dir)

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
if not os.path.isfile(timestamps_path):
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
  if not os.path.isfile(timestamps_path):
      raise IOError("Could not find timestamps file")

model = None
if args.models_dir:
    model = CameraModel(args.models_dir, args.dir)

current_chunk = 0
timestamps_file = open(timestamps_path).readlines()
for line in tqdm(timestamps_file):
    tokens = line.split()
    datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
    chunk = int(tokens[1])

    filename = os.path.join(args.dir, tokens[0] + '.png')
    if not os.path.isfile(filename):
        if chunk != current_chunk:
            print("Chunk " + str(chunk) + " not found")
            current_chunk = chunk
        continue

    current_chunk = chunk

    img = load_image(filename, model)
    # plt.imshow(img)
    # plt.xlabel(datetime)
    # plt.xticks([])
    # plt.yticks([])
    # plt.pause(0.01)

    # save image
    img_path = os.path.join(args.result_dir, tokens[0] + '.jpg')
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
