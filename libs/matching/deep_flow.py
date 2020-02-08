# Copyright (C) Huangying Zhan. All rights reserved.
#
# This software is licensed under the terms of the DF-VO licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from ..flowlib.flowlib import read_flow, flow_to_image
from ..utils import image_grid

# import LiteFlowNet modules
sys.path.insert(0, "deep_depth/monodepth2")
from deep_depth.monodepth2.networks.lite_flow_net.lite_flow_net import LiteFlowNet
del sys.path[0]


def resize_dense_vector(vec, des_height, des_width):
    ratio_height = float(des_height / vec.size(2))
    ratio_width = float(des_width / vec.size(3))
    vec = F.interpolate(
        vec, (des_height, des_width), mode='bilinear', align_corners=True)
    if vec.size(1) == 1:
        vec = vec * ratio_width
    else:
        vec = torch.stack(
            [vec[:, 0, :, :] * ratio_width, vec[:, 1, :, :] * ratio_height],
            dim=1)
    return vec


def uniform_bestN_selection(flow_diff, num_col, num_row, N):
    """select best-N from uniform regions in the image
    Args:
        flow_diff (1xHxWx1)
        num_col (int)
        num_row (int)
        N (int)
    Returns:
        sel_kps (4xN): keypoint locations
    """
    n, h, w, _ = flow_diff.shape
    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    for row in range(num_row):
        for col in range(num_col):
            x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
            x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

            tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
            tmp_kp_list = np.where(tmp_flow_diff[:] > 0)
            sel_list = np.argpartition(tmp_flow_diff[tmp_kp_list], n_best)[:n_best]
            sel_kps.append(convert_idx_to_global_coord(sel_list, tmp_kp_list, x0))

    sel_kps = np.asarray(sel_kps)
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))
    return sel_kps


def uniform_filtered_bestN_selection(flow_diff, num_col, num_row, N, thre):
    """select best-N kp from uniform regions in the image. bad kps are filtered by thre
    Args:
        flow_diff (1xHxWx1)
        num_col (int)
        num_row (int)
        N (int)
        thre (float)
    Returns:
        sel_kps (4xK): keypoint locations
    """
    n, h, w, _ = flow_diff.shape
    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    for row in range(num_row):
        for col in range(num_col):
            x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
            x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

            tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
            tmp_kp_list = np.where(tmp_flow_diff[:] < thre)
            num_to_pick = min(n_best, len(tmp_kp_list[0]))
            if num_to_pick <= n_best:
                sel_list = np.argpartition(tmp_flow_diff[tmp_kp_list], num_to_pick-1)[:num_to_pick]
            else:
                sel_list = np.argpartition(tmp_flow_diff[tmp_kp_list], num_to_pick)[:num_to_pick]
            
            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
            for i in range(sel_global_coords.shape[1]):
                sel_kps.append(sel_global_coords[:, i:i+1])

    sel_kps = np.asarray(sel_kps)
    assert sel_kps.shape[0]!=0, "sampling threshold is too small."
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))
    return sel_kps


def convert_idx_to_global_coord(local_idx, local_kp_list, x0):
    coord = [local_kp_list[0][local_idx], local_kp_list[1][local_idx], local_kp_list[2][local_idx], local_kp_list[3][local_idx]]
    coord = np.asarray(coord)
    coord[1] += x0[0] # h
    coord[2] += x0[1] # w
    return coord


class LiteFlow():
    def __init__(self, h=None, w=None):
        self.height = h
        self.width = w

    def initialize_network_model(self, weight_path):
        """initialize flow_net model
        Args:
            weight_path (str): weight path
        """
        if weight_path is not None:
            print("==> initialize LiteFlowNet with [{}]: ".format(weight_path))
            # Initialize network
            self.model = LiteFlowNet().cuda()

            # Load model weights
            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint)
            self.model.eval()

        # FIXME: hardcode for network-default.pytorch
        if "network-default.pytorch" in weight_path:
            self.half_flow = True
        else:
            self.half_flow = False

    def get_target_size(self, H, W):
        h = 32 * np.array([[math.floor(H / 32), math.floor(H / 32) + 1]])
        w = 32 * np.array([[math.floor(W / 32), math.floor(W / 32) + 1]])
        ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - H / W)
        index = np.argmin(ratio)
        return h[0, index // 2], w[0, index % 2]

    def load_flow_file(self, flow_path):
        """load flow data from a npy file
        Args:
            flow_path (str): flow data path, npy file
        Returns:
            flow (HxWx2 array): flow data
        """
        # Load flow
        flow = np.load(flow_path)

        # resize flow
        h, w, _ = flow.shape
        if self.width is None or self.height is None:
            resize_height = h
            resize_width = w
        else:
            resize_height = self.height
            resize_width = self.width
        flow = cv2.resize(flow, (resize_width, resize_height))
        flow[..., 0] *= resize_width / w
        flow[..., 1] *= resize_height / h
        return flow

    def load_precomputed_flow(self, img1, img2, flow_dir, dataset, forward_backward):
        """Load precomputed optical flow
        Args:
            img1: list of img1 id
            img2: list of img2 id
            flow_dir (str): directory to read flow
            dataset (str): dataset type
                - kitti
                - tum-1/2/3
            forward_backward (bool): load backward flow if True
        """
        flow_data = []
        for i in range(len(img1)):
            # Get flow npy file
            if dataset == "kitti":
                flow_path = os.path.join(
                            flow_dir,
                            "{:06d}".format(img2[i]),
                            "{:06d}.npy".format(img1[i]),
                            )
            elif "tum" in dataset:
                flow_path = os.path.join(
                            flow_dir,
                            "{:.6f}".format(img2[i]),
                            "{:.6f}.npy".format(img1[i]),
                            )
            assert os.path.isfile(flow_path), "wrong flow path: [{}]".format(flow_path)

            # Load and process flow data
            flow = self.load_flow_file(flow_path)
            flow_data.append(flow)
        flow_data = np.asarray(flow_data)
        flow_data = np.transpose(flow_data, (0, 3, 1, 2))

        # get backward flow data
        if forward_backward:
            back_flow_data = []
            for i in range(len(img1)):
                if dataset == "kitti":
                    flow_path = os.path.join(
                                    flow_dir,
                                    "{:06d}".format(img1[i]),
                                    "{:06d}.npy".format(img2[i]),
                                    )
                elif "tum" in dataset:
                    flow_path = os.path.join(
                                    flow_dir,
                                    "{:.6f}".format(img1[i]),
                                    "{:.6f}.npy".format(img2[i]),
                                    )
                assert os.path.isfile(flow_path), "wrong flow path"

                flow = self.load_flow_file(flow_path)
                back_flow_data.append(flow)
            back_flow_data = np.asarray(back_flow_data)
            back_flow_data = np.transpose(back_flow_data, (0, 3, 1, 2))
            return flow_data, back_flow_data
        else:
            return flow_data

    @torch.no_grad()
    def inference(self, img1, img2):
        """Predict optical flow for the given pairs
        Args:
            img1 (Nx3xHxW numpy array): image 1; intensity [0-1]
            img2 (Nx3xHxW numpy array): image 2; intensity [0-1]
        Returns:
            flow (Nx2xHxW numpy array): flow from img1 to img2
        """
        # Convert to torch array:cuda
        img1 = torch.from_numpy(img1).float().cuda()
        img2 = torch.from_numpy(img2).float().cuda()

        _, _, h, w = img1.shape
        th, tw = self.get_target_size(h, w)

        # forward pass
        flow_inputs = [img1, img2]
        resized_img_list = [
                            F.interpolate(
                                img, (th, tw), mode='bilinear', align_corners=True)
                            for img in flow_inputs
                        ]
        output = self.model(resized_img_list)

        # Post-process output
        scale_factor = 1
        flow = resize_dense_vector(
                                output[1] * scale_factor,
                                h, w)
        return flow.detach().cpu().numpy()

    def inference_kp(self, 
                    img1, img2, 
                    flow_dir, 
                    img_crop, 
                    kp_list, 
                    forward_backward=False, 
                    N_list=None, N_best=None,
                    kp_sel_method=None,
                    dataset="kitti"):
        """Estimate flow (1->2) and form keypoints
        Args:
            img1 (Nx3xHxW numpy array): image 1
            img2 (Nx3xHxW numpy array): image 2
            flow_dir (str): if not None:
                - img1: list of img1 id
                - img2: list of img2 id
                - flow_dir: directory to read flow
            img_crop (float list): [[y0, y1],[x0, x1]] in normalized range
            kp_list (int list): list of keypoint index
            foward_backward (bool): forward-backward flow consistency is used if True
            N_list (int): number of keypoint in regular list
            N_best (int): number of keypoint in best-N list
            kp_sel_method (str): method for selecting best-N keypoint
                - bestN: best-N kp over the whole image
                - uniform_bestN: uniformly divide the whole images into 100 pieces 
                                 and select best-N/100 from each piece
            dataset (str): dataset type
        Returns:
            kp1_best (BxNx2 array): best-N keypoints in img1
            kp2_best (BxNx2 array): best-N keypoints in img2
            kp1_list (BxNx2 array): N keypoints in kp_list in img1
            kp2_list (BxNx2 array): N keypoints in kp_list in img2
        """
        # Get flow data
        # if precomputed flow is provided, load precomputed flow
        if flow_dir is not None:
            if self.half_flow:
                self.half_flow = False
            if forward_backward:
                flow_data, back_flow_data = self.load_precomputed_flow(
                                img1=img1,
                                img2=img2,
                                flow_dir=flow_dir,
                                dataset=dataset,
                                forward_backward=forward_backward
                                )
            else:
                flow_data = self.load_precomputed_flow(
                                img1=img1,
                                img2=img2,
                                flow_dir=flow_dir,
                                dataset=dataset,
                                forward_backward=forward_backward
                                )
        # flow net inference to get flows
        else:
            # FIXME: combined images (batch_size>1) forward performs slightly different from batch_size=1 case
            if forward_backward:
                input_img1 = np.concatenate([img1, img2], axis=0)
                input_img2 = np.concatenate([img2, img1], axis=0)
            else:
                input_img1 = img1
                input_img2 = img2
            combined_flow_data = self.inference(input_img1, input_img2)
            flow_data = combined_flow_data[0:1]
            if forward_backward:
                back_flow_data = combined_flow_data[1:2]

            # flow_data = self.inference(img1, img2)
            # if forward_backward:
            #     back_flow_data = self.inference(img2, img1)

        if self.half_flow:
            flow_data /= 2.
            if forward_backward:
                back_flow_data /= 2.

        # Compute keypoint map
        n, _, h, w = flow_data.shape
        tmp_flow_data = np.transpose(flow_data, (0, 2, 3, 1))
        kp1 = image_grid(h, w)
        kp1 = np.repeat(np.expand_dims(kp1, 0), n, axis=0)
        kp2 = kp1 + tmp_flow_data

        # initialize output keypoint data
        kp1_best = np.zeros((n, N_best, 2)) - 1 # initialize as -1
        kp2_best = np.zeros((n, N_best, 2)) - 1 # initialize as -1
        kp1_list = np.zeros((n, N_list, 2))
        kp2_list = np.zeros((n, N_list, 2))

        # Forward-Backward flow consistency check
        if forward_backward:
            # get flow-consistency error map
            flow_diff = self.forward_backward_consistency(
                                flow1=flow_data,
                                flow2=back_flow_data,
                                px_coord_2=kp2)

            # get best-N keypoints
            if kp_sel_method == "bestN":
                tmp_kp_list = np.where(flow_diff > 0)
                sel_list = np.argpartition(flow_diff[tmp_kp_list], N_best)[:N_best]
                sel_kps = convert_idx_to_global_coord(sel_list, tmp_kp_list, [0, 0])
            elif kp_sel_method == "uniform_bestN":
                sel_kps = uniform_bestN_selection(
                                flow_diff=flow_diff, 
                                num_col=10,
                                num_row=10,
                                N=N_best)
            elif kp_sel_method == "uniform_filtered_bestN":
                sel_kps = uniform_filtered_bestN_selection(
                                flow_diff=flow_diff,
                                num_col=10,
                                num_row=10,
                                N=N_best,
                                thre=0.1)
            kp1_best[:,:sel_kps.shape[1]] = kp1[:, sel_kps[1], sel_kps[2]]
            kp2_best[:,:sel_kps.shape[1]] = kp2[:, sel_kps[1], sel_kps[2]]

        # Get uniform sampled keypoints
        y0, y1 = 0, h
        x0, x1 = 0, w
        if img_crop is not None:
            y0, y1 = int(h*img_crop[0][0]), int(h*img_crop[0][1])
            x0, x1 = int(w*img_crop[1][0]), int(w*img_crop[1][1])

            kp1 = kp1[:, y0:y1, x0:x1]
            kp2 = kp2[:, y0:y1, x0:x1]

        kp1_list = kp1.reshape(n, -1, 2)
        kp2_list = kp2.reshape(n, -1, 2)

        if kp_list is not None:
            kp1_list = np.transpose(kp1_list, (1,0,2))
            kp2_list = np.transpose(kp2_list, (1,0,2))
            kp1_list = kp1_list[kp_list]
            kp2_list = kp2_list[kp_list]
            kp1_list = np.transpose(kp1_list, (1,0,2))
            kp2_list = np.transpose(kp2_list, (1,0,2))

        # summarize flow data and flow difference
        flows = {}
        flows['forward'] = flow_data
        if forward_backward:
            flows['backward'] = back_flow_data
            flows['flow_diff'] = flow_diff
        # return kp1, kp2, flows
        return kp1_best, kp2_best, kp1_list, kp2_list, flows

    def forward_backward_consistency(self, flow1, flow2, px_coord_2):
        """Compute flow consistency map
        Args:
            flow1 (Nx2xHxW array): flow map 1
            flow2 (Nx2xHxW array): flow map 2
            px_coord_2 (NxHxWx2 array): pixel coordinate in view 2
        Returns:
            flow_diff (NxHxWx1): flow inconsistency error map
        """
        # copy flow data to GPU
        flow1 = torch.from_numpy(flow1).float().cuda()
        flow2 = torch.from_numpy(flow2).float().cuda()

        # Normalize sampling pixel coordinates
        _, _, h, w = flow1.shape
        norm_px_coord = px_coord_2.copy()
        norm_px_coord[:, :, :, 0] = px_coord_2[:,:,:,0] / (w-1)
        norm_px_coord[:, :, :, 1] = px_coord_2[:,:,:,1] / (h-1)
        norm_px_coord = (norm_px_coord * 2) - 1
        norm_px_coord = torch.from_numpy(norm_px_coord).float().cuda()

        # Warp flow2 to flow1
        warp_flow1 = F.grid_sample(-flow2, norm_px_coord)

        # Calculate flow difference
        flow_diff = (flow1 - warp_flow1)

        # TODO: UnFlow (Meister etal. 2017) constrain is not used
        UnFlow_constrain = False
        if UnFlow_constrain:
            flow_diff = (flow_diff ** 2 - 0.01 * (flow1**2 - warp_flow1 ** 2))

        # cppy flow_diff to cpu
        flow_diff = flow_diff.norm(dim=1, keepdim=True)
        flow_diff = flow_diff.permute(0, 2, 3, 1)
        flow_diff = flow_diff.detach().cpu().numpy()
        return flow_diff
