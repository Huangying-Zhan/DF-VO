# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

from easydict import EasyDict as edict
import os
import shutil
import yaml


def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    Args:
        path: directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_yaml(filename):
    """Load yaml file as a dictionary item
    Args:
        filename (str): .yaml file path
    Returns:
        cfg (dict): configuration
    """
    if filename is not None:
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def copy_file(src_file, tgt_file):
    """Copy a file
    Args:
        src_file (str): source file
        tgt_file (str): target file
    """
    shutil.copyfile(src_file, tgt_file)
