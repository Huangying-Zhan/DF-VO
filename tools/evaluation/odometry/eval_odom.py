''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: This program evaluate KITTI odometry result
'''

import argparse

from tools.evaluation.odometry.kitti_odometry import KittiEvalOdom


def argument_parsing():
    """Argument parsing

    Returns:
        args (args): arguments
    """
    parser = argparse.ArgumentParser(description='KITTI Odometry evaluation')
    parser.add_argument('--result', type=str, required=True,
                        help="Result directory")
    parser.add_argument('--gt', type=str,
                        default="dataset/kitti_odom/gt_poses/",
                        help="GT Pose directory containing gt pose txt files")
    parser.add_argument('--align', type=str, 
                        choices=['scale', 'scale_7dof', '7dof', '6dof'],
                        default=None,
                        help="alignment type")
    parser.add_argument('--seqs', 
                        nargs="+",
                        help="sequences to be evaluated",
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
        print("Double check the path!")
