# Copyright (C) Huangying Zhan 2019. All rights reserved.
#
# This software is licensed under the terms of the DF-VO licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from vo_modules import VisualOdometry as VO
from libs.general.utils import *
from libs.utils import load_kitti_odom_intrinsics

""" Argument Parsing """
parser = argparse.ArgumentParser(description='VO system')
parser.add_argument("-s", "--seq", type=str,
                    default=None, help="sequence")
parser.add_argument("-c", "--configuration", type=str,
                    default=None,
                    help="custom configuration file")
parser.add_argument("-d", "--default_configuration", type=str,
                    default="options/kitti/kitti_default_configuration.yml",
                    help="default configuration files")
args = parser.parse_args()

""" Read configuration """
# Read configuration
default_config_file = args.default_configuration
config_files = [default_config_file, args.configuration]
cfg = merge_cfg(config_files)
if args.seq is not None:
    cfg.seq = args.seq
cfg.seq = str(cfg.seq)

# Double check result directory
continue_flag = input("Save result in {}? [y/n]".format(cfg.result_dir))
if continue_flag == "y":
    mkdir_if_not_exists(cfg.result_dir)
else:
    exit()

""" basic setup """
# Random seed
SEED = cfg.seed
np.random.seed(SEED)

""" Main """
vo = VO(cfg)
vo.setup()
vo.main()

# Save configuration file
cfg_path = os.path.join(cfg.result_dir, "configuration_{}.yml".format(cfg.seq))
save_cfg(config_files, file_path=cfg_path)
