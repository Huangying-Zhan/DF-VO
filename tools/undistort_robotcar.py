'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 1970-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-02
@LastEditors: Huangying Zhan
@Description: This tool undistort Oxford Robotcar sequences
'''

import argparse

from tools.evaluation.robotcar.sdk_python.image import load_image
from tools.evaluation.robotcar.sdk_python.camera_model import CameraModel



def argument_parsing():
    """Argument parsing

    Returns:
        args (args): arguments
    """
    parser = argparse.ArgumentParser(description='KITTI Odometry evaluation')
    parser.add_argument('--data_dir', type=str,
                        default="dataset/robotcar/raw_data/",
                        help="GT Pose directory containing gt pose txt files")
    parser.add_argument('--result', type=str, required=True,
                        default="dataset/robotcar/"
                        help="Result directory")
    parser.add_argument('--seqs', 
                        nargs="+",
                        help="sequences to be undistorted",
                        default=None)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    # argument parsing
    args = argument_parsing()

    # initialize evaluation tool
    eval_tool = KittiEvalOdom() 

    continue_flag = input("Evaluate result in [{}]? [y/n]".format(args.result))
    if continue_flag == "y":
        eval_tool.eval(
            args.gt,
            args.result,
            alignment=args.align,
            seqs=args.seqs,
            )
    else: