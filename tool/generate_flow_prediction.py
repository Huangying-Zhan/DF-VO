'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-07
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-12
@LastEditors: Huangying Zhan
@Description: This program generate optical flow prediction for KITTI Flow 2012/2015
'''

import argparse
import cv2
from glob import glob
# from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.misc
from tqdm import tqdm

from libs.matching.deep_flow import LiteFlow
from libs.general.utils import *


""" Argument Parsing """
parser = argparse.ArgumentParser(description='VO system')
parser.add_argument("--result", type=str, required=True, 
                    help="Result output directory, RESULT/data will be created")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["kitti2012", "kitti2015"],
                    help="Dataset choice: kitti2012, kitti2015")
parser.add_argument("--test", action="store_true",
                    help="Test testing split. If not set, training split")
parser.add_argument("--model", type=str, required=True,
                    help="Model weight path")
parser.add_argument("--flow_mask_thre" , type=float,
                    default=None,
                    help="Forward-backward flow consistency mask threshold. If non-zero, mask is used")
args = parser.parse_args()


""" Read configuration """
# Create result directory
dirs = {}
dirs['result'] = args.result
mkdir_if_not_exists(os.path.join(dirs['result'], "data"))

# Get dataset directory
dirs['img_data'] = {
                    "kitti2012": "dataset/kitti_flow_2012/{}/colored_0",
                    "kitti2015": "dataset/kitti_flow_2015/{}/image_2",
                    }[args.dataset]

if args.test:
    dirs['img_data'] = dirs['img_data'].format("testing")
else:
    dirs['img_data'] = dirs['img_data'].format("training")

# Basic setup
h = 370
w = 1226

def initialize_deep_flow_model(h, w, weight):
    """Initialize optical flow network
    Args:
        h (int): image height
        w (int): image width
    Returns:
        flow_net: optical flow network
    """
    flow_net = LiteFlow(h, w)
    flow_net.initialize_network_model(
            weight_path=weight
            )
    return flow_net

def read_image(path):
    """read image data and convert to RGB
    Args:
        path (str): image path
    Returns:
        img (HxWx3 array): image data
    """
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (w, h))
    return img

def get_img_idxs(dataset, is_test):
    """Get image paths
    Args:
        dataset: 
            - kitti2012: All kitti-2012 image
            - kitti2015: All kitti-2015 image
        is_test (bool): Use testing set if true
    Returns:
        img_idx (list): Indexs of test image
    """
    if dataset == "kitti2012":
        if is_test:
            raise NotImplementedError
        else:
            return [i for i in range(194)]
    
    elif dataset == "kitti2015":
        if is_test:
            raise NotImplementedError
        else:
            return [i for i in range(200)]


# initalize network
flow_net = initialize_deep_flow_model(h, w, args.model)

img_idxs = get_img_idxs(args.dataset, args.test)

for i in tqdm(img_idxs):
    # get image paths
    img1_path = os.path.join(dirs['img_data'] , "{:06}_10.png".format(i))
    img2_path = os.path.join(dirs['img_data'] , "{:06}_11.png".format(i))

    # load image
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    
    cur_imgs = [np.transpose((img1)/255, (2, 0, 1))]
    ref_imgs = [np.transpose((img2)/255, (2, 0, 1))]
    ref_imgs = np.asarray(ref_imgs)
    cur_imgs = np.asarray(cur_imgs)

    """ prediction """
    flows = {}
    # Flow inference
    batch_flows = flow_net.inference_flow(
                            img1=cur_imgs[0:1],
                            img2=ref_imgs[0:1],
                            flow_dir=None,
                            forward_backward=True,
                            dataset="kitti")
        
    flows = batch_flows['forward'][0].copy()

    # save prediction
    _, h, w = flows.shape
    flows3 = np.ones((h, w, 3))
    
    if args.flow_mask_thre is not None:
        flow_mask = (batch_flows['flow_diff'][0,:,:,0] < args.flow_mask_thre) * 1
        flows3[:, :, 0] = flow_mask
    flows3[:, :, 2] = flows[0] * 64 + 2**15
    flows3[:, :, 1] = flows[1] * 64 + 2**15
    flows3 = flows3.astype(np.uint16)

    out_png = os.path.join(dirs['result'], 'data', '{:06}_10.png'.format(i))
    cv2.imwrite(out_png, flows3)
