''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-02
@LastEditors: Huangying Zhan
@Description: This API runs DF-VO.
'''

import argparse
import numpy as np
import os
import random
import sys
import torch

sys.path.insert(0, os.getcwd())

from libs.dfvo import DFVO
from libs.general.utils import mkdir_if_not_exists
from libs.general.configuration import ConfigLoader


config_loader = ConfigLoader()

def read_cfgs():
    """Parse arguments and laod configurations

    Returns
    -------
    args : args
        arguments
    cfg : edict
        configuration dictionary
    """
    ''' Argument Parsing '''
    parser = argparse.ArgumentParser(description='VO system')
    parser.add_argument("-s", "--seq", 
                        default=None, help="sequence")
    parser.add_argument("-d", "--default_configuration", type=str, 
                        default="options/kitti/kitti_default_configuration.yml",
                        help="default configuration files")
    parser.add_argument("-c", "--configuration", type=str,
                        default=None,
                        help="custom configuration file")
    parser.add_argument("--no_confirm", action="store_true",
                        help="no confirmation questions")
    args = parser.parse_args()

    ''' Read configuration '''
    # read default and custom config, merge cfgs
    config_files = [args.default_configuration, args.configuration]
    cfg = config_loader.merge_cfg(config_files)
    if args.seq is not None:
        if cfg.dataset == "kitti_odom":
            cfg.seq = "{:02}".format(int(args.seq))
        else:
            cfg.seq = args.seq
    cfg.seq = str(cfg.seq)

    ''' double check result directory '''
    if args.no_confirm:
        mkdir_if_not_exists(cfg.directory.result_dir)
        cfg.no_confirm = True
    else:
        cfg.no_confirm = False
        continue_flag = input("Save result in {}? [y/n]".format(cfg.directory.result_dir))
        if continue_flag == "y":
            mkdir_if_not_exists(cfg.directory.result_dir)
        else:
            exit()
    return args, cfg


if __name__ == '__main__':
    # Read config
    args, cfg = read_cfgs()

    # Set random seed
    SEED = cfg.seed
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)

    # setup DFVO
    vo = DFVO(cfg)
    vo.main()

    # Save configuration file
    cfg_path = os.path.join(cfg.directory.result_dir, 'configuration_{}.yml'.format(cfg.seq))
    config_loader.save_cfg([args.default_configuration, args.configuration], file_path=cfg_path)
