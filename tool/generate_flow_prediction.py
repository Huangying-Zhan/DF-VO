import numpy as np
import cv2
import os
from glob import glob
import scipy.misc
from tqdm import tqdm

from libs.matching.deep_flow import LiteFlow
# from libs.utils import *
from libs.general.utils import *

from matplotlib import pyplot as plt

# directory settings
dirs = {}
# dirs['result'] = "result/flow/liteflownet_self_kitti/data"
dirs['result'] = "result/flow/liteflownet_self_bestN/data"
dirs['img_data'] = "dataset/kitti_flow_2012/training/colored_0"
mkdir_if_not_exists(dirs['result'])
use_mask = True

# setup
h = 370
w = 1226
weight = "model_zoo/optical_flow/UnLiteFlowNet/kitti_odom/mono_640x192/flow.pth"
# weight = "model_zoo/optical_flow/LiteFlowNet/network-default.pytorch"

def initialize_deep_flow_model(h, w, weight):
    """Initialize optical flow network
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

# initalize network
flow_net = initialize_deep_flow_model(h, w, weight)

for i in tqdm(range(194)):
    # get image paths
    img1_path = os.path.join(dirs['img_data'] , "{:06}_10.png".format(i))
    img2_path = os.path.join(dirs['img_data'] , "{:06}_11.png".format(i))

    # load image
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    # prediction
    cur_imgs = [np.transpose((img1)/255, (2, 0, 1))]
    ref_imgs = [np.transpose((img2)/255, (2, 0, 1))]
    ref_imgs = np.asarray(ref_imgs)
    cur_imgs = np.asarray(cur_imgs)

    # Forward pass
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
    
    if use_mask:
        delta = 0.1
        flow_mask = (batch_flows['flow_diff'][0,:,:,0] < delta) * 1
        flows3[:, :, 0] = flow_mask
    flows3[:, :, 2] = flows[0] * 64 + 2**15
    flows3[:, :, 1] = flows[1] * 64 + 2**15
    flows3 = flows3.astype(np.uint16)

    out_png = os.path.join(dirs['result'], '{:06}_10.png'.format(i))
    cv2.imwrite(out_png, flows3)