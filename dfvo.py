# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import multiprocessing as mp
# import torch.multiprocessing as mp
import numpy as np
import os
from sklearn import linear_model
from time import time
import torch
import torch.nn.functional as nnFunc
from tqdm import tqdm

import libs.datasets as Dataset
from libs.deep_depth.monodepth2 import Monodepth2DepthNet
from libs.deep_pose.monodepth2 import Monodepth2PoseNet
from libs.geometry.ops_3d import *
from libs.geometry.backprojection import Backprojection
from libs.geometry.reprojection import Reprojection
from libs.geometry.find_Ess_mat import find_Ess_mat
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timers
from libs.matching.deep_flow import LiteFlow
from libs.matching.kp_selection import uniform_filtered_bestN, bestN, sampled_kp, good_depth_kp
from libs.camera_modules import SE3, Intrinsics
from libs.utils import *

from tool.evaluation.tum_tool.pose_evaluation_utils import rot2quat


class DFVO():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        # configuration
        self.cfg = cfg

        # tracking stage
        self.tracking_stage = 0

        # predicted global poses
        self.global_poses = {0: SE3()}

        # window size and keyframe step
        self.window_size = 2
        self.keyframe_step = 1

        # visualization interface
        self.drawer = FrameDrawer(self.cfg.visualization)

        # timer
        self.initialize_timer()
        
        # reference data and current data
        self.initialize_data()
        
        # multiprocessing
        if self.cfg.use_multiprocessing:
            self.p = mp.Pool(5)

    def initialize_timer(self):
        self.timers = Timers()
        self.timers.add(["img_reading",
                         "Depth-CNN",
                         "tracking",
                         "Ess. Mat.",
                         "Flow-CNN",
                         "visualization",
                         "visualization_traj",
                         "visualization_match",
                         "visualization_flow",
                         "visualization_depth",
                         "visualization_masks",
                         "visualization_save_img", ])

    def initialize_data(self):
        self.ref_data = {
                        'id': [],
                        'timestamp': {},
                        'img': {},
                        'depth': {},
                        'raw_depth': {},
                        'pose': {},
                        'kp': {},
                        'kp_best': {},
                        'kp_list': {},
                        'pose_back': {},
                        'kp_back': {},
                        'flow': {},  # from ref->cur
                        'flow_diff': {},  # flow-consistency-error of ref->cur
                        'inliers': {}
                        }
        self.cur_data = {
                        'id': 0,
                        'timestamp': 0,
                        'img': np.zeros(1),
                        'depth': np.zeros(1),
                        'pose': np.eye(4),
                        'kp': np.zeros(1),
                        'kp_best': np.zeros(1),
                        'kp_list': np.zeros(1),
                        'pose_back': np.eye(4),
                        'kp_back': np.zeros(1),
                        'flow': {},  # from cur->ref
                        }

    def get_tracking_method(self, method_idx):
        """Get tracking method
        Args:
            method_idx (int): tracking method index
                - 0: 2d-2d
                - 1: 3d-2d
                - 2: 3d-3d
                - 3: hybrid
        Returns:
            track_method (str): tracking method
        """
        tracking_method_cases = {
            0: "2d-2d",
            1: "3d-2d",
            2: "3d-3d",
            3: "hybrid"
        }
        return tracking_method_cases[method_idx]

    def get_feat_track_methods(self, method_idx):
        """Get feature tracking method
        Args:
            method_idx (int): feature tracking method index
                - 1: deep_flow
        Returns:
            feat_track_method (str): feature tracking method
        """
        feat_track_methods = {
            1: "deep_flow",
        }
        return feat_track_methods[self.cfg.feature_tracking_method]

    def get_img_depth_dir(self):
        """Get image data directory and (optional) depth data directory

        Returns:
            img_data_dir (str): image data directory
            depth_data_dir (str): depth data directory / None
            depth_src (str): depth data type
                - gt
                - None
        """
        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        if self.cfg.dataset == "kitti_odom":
            img_data_dir = os.path.join(img_seq_dir, "image_2")
        elif self.cfg.dataset == "kitti_raw":
            img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq[:10],
                            self.cfg.seq
                            )
            img_data_dir = os.path.join(img_seq_dir, "image_02/data")
        elif "tum" in self.cfg.dataset:
            img_data_dir = os.path.join(img_seq_dir, "rgb")
        else:
            warn_msg = "Wrong dataset [{}] is given.".format(self.cfg.dataset)
            warn_msg += "\n Choose from [kitti, tum-1/2/3]"
            assert False, warn_msg
        
        # get depth data directory
        depth_src_cases = {
            0: "gt",
            1: "pred",
            None: None
            }
        depth_src = depth_src_cases[self.cfg.depth.depth_src]

        if self.cfg.dataset == "kitti_odom":
            if depth_src == "gt":
                depth_data_dir = "{}/gt/{}/".format(
                                    self.cfg.directory.depth_dir, self.cfg.seq
                                )
            elif depth_src == "pred":
                depth_data_dir = "{}/{}/".format(
                                    self.cfg.directory.depth_dir, self.cfg.seq
                                )
            elif depth_src is None:
                depth_data_dir = None
        elif self.cfg.dataset == "kitti_raw":
            if depth_src == "gt":
                depth_data_dir = os.path.join(self.cfg.directory.depth_dir, self.cfg.seq)
            elif depth_src is None:
                depth_data_dir = None
        elif "tum" in self.cfg.dataset:
            if depth_src == "gt":
                depth_data_dir = "{}/{}/depth".format(
                                    self.cfg.directory.depth_dir, self.cfg.seq
                                    )
            elif depth_src is None:
                depth_data_dir = None
 
        return img_data_dir, depth_data_dir, depth_src

    def generate_kp_samples(self, img_h, img_w, crop, N):
        """generate keypoint samples according to image height, width
        and cropping scheme

        Args:
            img_h (int): image height
            img_w (int): image width
            crop (list): normalized cropping ratio
                - [[y0, y1],[x0, x1]]
            N (int): number of keypoint

        Returns:
            kp_list (N array): keypoint list
        """
        y0, y1 = crop[0]
        y0, y1 = int(y0 * img_h), int(y1 * img_h)
        x0, x1 = crop[1]
        x0, x1 = int(x0 * img_w), int(x1 * img_w)
        total_num = (x1-x0) * (y1-y0) - 1
        kp_list = np.linspace(0, total_num, N, dtype=np.int)
        return kp_list

    def initialize_deep_flow_model(self):
        """Initialize optical flow network
        Returns:
            flow_net: optical flow network
        """
        if self.cfg.deep_flow.network == "liteflow":
            flow_net = LiteFlow(self.cfg.image.height, self.cfg.image.width)
            flow_net.initialize_network_model(
                    weight_path=self.cfg.deep_flow.flow_net_weight
                    )
        else:
            assert False, "Invalid flow network [{}] is provided.".format(
                                self.cfg.deep_flow.network
                                )
        return flow_net

    def initialize_deep_depth_model(self):
        """Initialize single-view depth model
        Returns:
            depth_net: single-view depth network
        """
        depth_net = Monodepth2DepthNet()
        depth_net.initialize_network_model(
                weight_path=self.cfg.depth.pretrained_model,
                dataset=self.cfg.dataset)
        return depth_net
    
    def initialize_deep_pose_model(self):
        """Initialize two-view pose model
        Returns:
            pose_net: two-view pose network
        """
        pose_net = Monodepth2PoseNet()
        pose_net.initialize_network_model(
            weight_path=self.cfg.pose_net.pretrained_model,
            height=self.cfg.image.height,
            width=self.cfg.image.width,
            dataset=self.cfg.dataset
            )
        return pose_net

    def get_gt_poses(self):
        """load ground-truth poses
        Returns:
            gt_poses (dict): each pose is 4x4 array
        """
        if self.cfg.directory.gt_pose_dir is not None:
            if self.cfg.dataset == "kitti_odom":
                annotations = os.path.join(
                                    self.cfg.directory.gt_pose_dir,
                                    "{}.txt".format(self.cfg.seq)
                                    )
                gt_poses = load_poses_from_txt(annotations)
            elif self.cfg.dataset == "kitti_raw":
                # gps_info_dir =  os.path.join(
                #             self.cfg.directory.gt_pose_dir,
                #             self.cfg.seq,
                #             "oxts/data"
                #             )
                # gt_poses = load_poses_from_oxts(gps_info_dir)
                annotations = os.path.join(
                                    self.cfg.directory.gt_pose_dir,
                                    "{}.txt".format(self.cfg.seq)
                                    )
                gt_poses = load_poses_from_txt(annotations)
            elif "tum" in self.cfg.dataset:
                annotations = os.path.join(
                                    self.cfg.directory.gt_pose_dir,
                                    self.cfg.seq,
                                    "groundtruth.txt"
                                    )
                gt_poses = load_poses_from_txt_tum(annotations)
            return gt_poses
        else:
            return {0: np.eye(4)}

    def setup(self):
        """Reading configuration and setup, including
        - Get tracking method
        - Get feature tracking method
        - Generate keypoint sampling scheme
        - Deep networks
        - Deep layers
        - Load GT poses
        - Set drawer
        """
        # get tracking method
        self.tracking_method = self.get_tracking_method(self.cfg.tracking_method)

        # feature tracking method
        self.feature_tracking_method = self.get_feat_track_methods(
                                            self.cfg.feature_tracking_method
                                            )

        # intialize dataset
        datasets = {
            "kitti_odom": Dataset.KittiOdom,
            "kitti_raw": Dataset.KittiRaw,
            "tum": Dataset.TUM
        }
        self.dataset = datasets[self.cfg.dataset](self.cfg)

        # generate keypoint sampling scheme
        self.uniform_kp_list = None
        if (self.cfg.deep_flow.num_kp is not None and self.feature_tracking_method == "deep_flow"):
            self.uniform_kp_list = self.generate_kp_samples(
                                        img_h=self.cfg.image.height,
                                        img_w=self.cfg.image.width,
                                        crop=self.cfg.crop.flow_crop,
                                        N=self.cfg.deep_flow.num_kp
                                        )

        # Deep networks
        self.deep_models = {}
        # optical flow
        if self.feature_tracking_method == "deep_flow":
            self.deep_models['flow'] = self.initialize_deep_flow_model()

            # allow to read precomputed flow instead of network inference
            # for speeding up testing time
            if self.cfg.deep_flow.precomputed_flow is not None:
                self.cfg.deep_flow.precomputed_flow = self.cfg.deep_flow.precomputed_flow.replace("{}", self.cfg.seq)

        # single-view depth
        if self.dataset.data_dir['depth_src'] is None:
            if self.cfg.depth.pretrained_model is not None:
                self.deep_models['depth'] = self.initialize_deep_depth_model()
            else:
                assert False, "No precomputed depths nor pretrained depth model"
        
        # two-view pose
        if self.cfg.pose_net.enable:
            if self.cfg.pose_net.pretrained_model is not None:
                self.deep_models['pose'] = self.initialize_deep_pose_model()
            else:
                assert False, "No pretrained pose model"

        # Deep layers
        self.backproj = Backprojection(self.cfg.image.height, self.cfg.image.width).cuda()
        self.reproj = Reprojection(self.cfg.image.height, self.cfg.image.width).cuda()

        # Load GT pose
        self.gt_poses = self.get_gt_poses()

        # Set drawer
        self.drawer.get_traj_init_xy(
                        vis_h=self.drawer.h,
                        vis_w=self.drawer.w/5*2,
                        gt_poses=self.gt_poses)

    def load_depth(self, depth_seq_dir, img_id, depth_src,
                   resize=None, dataset="kitti_odom"):
        """Load depth map for different source
        Args:
            depth_seq_dir (str): depth sequence dir
            img_id (int): depth image id
            depth_src (str): depth src type
                - gt
            resize (int list): [target_height, target_width]
            dataset (str):
                - kitti_odom
                - kitti_raw
                - tum
        Returns:
            depth (HxW array): depth map
        """
        if dataset == "kitti_odom":
            if depth_src == "gt":
                img_id = "{:010d}.png".format(img_id)
                scale_factor = 500
            elif depth_src == "pred":
                img_id = "depth/{:06d}.png".format(img_id)
                scale_factor = 1000
        elif dataset == "kitti_raw":
            if depth_src == "gt":
                img_id = "{:010d}.png".format(img_id)
                scale_factor = 500
        elif "tum" in dataset:
            if depth_src == "gt":
                img_id = "{:.6f}.png".format(img_id)
                scale_factor = 5000

        img_h, img_w = resize
        depth_path = os.path.join(depth_seq_dir, img_id)
        depth = read_depth(depth_path, scale_factor, [img_h, img_w])
        return depth

    def compute_pose_2d2d(self, kp_ref, kp_cur):
        """Compute the pose from view2 to view1
        Args:
            kp_ref (Nx2 array): keypoints for reference view
            kp_cur (Nx2 array): keypoints for current view
        Returns:
            outputs (dict):
                - pose (SE3): relative pose from current to reference view
                - best_inliers (N boolean array): inlier mask
        """
        principal_points = (self.dataset.cam_intrinsics.cx, self.dataset.cam_intrinsics.cy)

        # validity check
        valid_cfg = self.cfg.compute_2d2d_pose.validity
        valid_case = True

        # initialize ransac setup
        R = np.eye(3)
        t = np.zeros((3,1))
        best_Rt = [R, t]
        best_inlier_cnt = 0
        max_ransac_iter = self.cfg.compute_2d2d_pose.ransac.repeat
        best_inliers = np.ones((kp_ref.shape[0], 1)) == 1

        if valid_cfg.method == "flow":
            # check flow magnitude
            avg_flow = np.mean(np.linalg.norm(kp_ref-kp_cur, axis=1))
            valid_case = avg_flow > valid_cfg.thre
        
        elif valid_cfg.method == "homo_ratio":
            # Find homography
            H, H_inliers = cv2.findHomography(
                        kp_cur,
                        kp_ref,
                        method=cv2.RANSAC,
                        confidence=0.99,
                        ransacReprojThreshold=0.2,
                        )
        
        if valid_case:
            num_valid_case = 0
            for i in range(max_ransac_iter): # repeat ransac for several times for stable result
                # shuffle kp_cur and kp_ref (only useful when random seed is fixed)	
                new_list = np.arange(0, kp_cur.shape[0], 1)	
                np.random.shuffle(new_list)
                new_kp_cur = kp_cur.copy()[new_list]
                new_kp_ref = kp_ref.copy()[new_list]

                start_time = time()
                E, inliers = cv2.findEssentialMat(
                            new_kp_cur,
                            new_kp_ref,
                            focal=self.dataset.cam_intrinsics.fx,
                            pp=principal_points,
                            method=cv2.RANSAC,
                            prob=0.99,
                            threshold=self.cfg.compute_2d2d_pose.ransac.reproj_thre,
                            )
                
                # check homography inlier ratio
                if valid_cfg.method == "homo_ratio":
                    H_inliers_ratio = H_inliers.sum()/(H_inliers.sum()+inliers.sum())
                    valid_case = H_inliers_ratio < valid_cfg.thre
                    # print("valid: {}; ratio: {}".format(valid_case, H_inliers_ratio))

                    # inlier check
                    inlier_check = inliers.sum() > best_inlier_cnt
                elif valid_cfg.method == "flow":
                    cheirality_cnt, R, t, _ = cv2.recoverPose(E, new_kp_cur, new_kp_ref,
                                            focal=self.dataset.cam_intrinsics.fx,
                                            pp=principal_points)
                    valid_case = cheirality_cnt > kp_cur.shape[0]*0.1
                    
                    # inlier check
                    inlier_check = inliers.sum() > best_inlier_cnt and cheirality_cnt > kp_cur.shape[0]*0.05

                # save best_E
                if inlier_check:
                    best_E = E
                    best_inlier_cnt = inliers.sum()

                    revert_new_list = np.zeros_like(new_list)
                    for cnt, i in enumerate(new_list):
                        revert_new_list[i] = cnt
                    best_inliers = inliers[list(revert_new_list)]
                num_valid_case += (valid_case * 1)

            major_valid = num_valid_case > (max_ransac_iter/2)
            if major_valid:
                cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, kp_cur, kp_ref,
                                        focal=self.dataset.cam_intrinsics.fx,
                                        pp=principal_points,
                                        )
                self.timers.timers["Ess. Mat."].append(time()-start_time)

                # cheirality_check
                if cheirality_cnt > kp_cur.shape[0]*0.1:
                    best_Rt = [R, t]

        R, t = best_Rt
        pose = SE3()
        pose.R = R
        pose.t = t
        outputs = {"pose": pose, "inliers": best_inliers[:,0]==1}
        return outputs

    def compute_pose_2d2d_mp(self, kp_ref, kp_cur):
        """Compute the pose from view2 to view1 (multiprocessing ver)
        Args:
            kp_ref (Nx2 array): keypoints for reference view
            kp_cur (Nx2 array): keypoints for current view
        Returns:
            outputs (dict):
                - pose (SE3): relative pose from current to reference view
                - best_inliers (N boolean array): inlier mask
        """
        start_time = time()
        principal_points = (self.dataset.cam_intrinsics.cx, self.dataset.cam_intrinsics.cy)

        # validity check
        valid_cfg = self.cfg.compute_2d2d_pose.validity
        valid_case = True

        # initialize ransac setup
        R = np.eye(3)
        t = np.zeros((3,1))
        best_Rt = [R, t]
        max_ransac_iter = self.cfg.compute_2d2d_pose.ransac.repeat

        if valid_cfg.method == "flow":
            # check flow magnitude
            avg_flow = np.mean(np.linalg.norm(kp_ref-kp_cur, axis=1))
            valid_case = avg_flow > valid_cfg.thre
        
        elif valid_cfg.method == "homo_ratio":
            # Find homography
            H, H_inliers = cv2.findHomography(
                        kp_cur,
                        kp_ref,
                        method=cv2.RANSAC,
                        confidence=0.99,
                        ransacReprojThreshold=0.2,
                        )
        
        if valid_case:
            inputs_mp = []
            outputs_mp = []
            for i in range(max_ransac_iter):
                # shuffle kp_cur and kp_ref
                new_list = np.arange(0, kp_cur.shape[0], 1)
                np.random.shuffle(new_list)
                new_kp_cur = kp_cur.copy()[new_list]
                new_kp_ref = kp_ref.copy()[new_list]

                inputs = {}
                inputs['kp_cur'] = new_kp_cur
                inputs['kp_ref'] = new_kp_ref
                inputs['H_inliers'] = H_inliers
                inputs['cfg'] = self.cfg
                inputs['cam_intrinsics'] = self.dataset.cam_intrinsics
                inputs_mp.append(inputs)
            outputs_mp = self.p.map(find_Ess_mat, inputs_mp)

            # Gather result
            num_valid_case = 0
            best_inlier_cnt = 0
            best_inliers = np.ones((kp_ref.shape[0])) == 1
            for outputs in outputs_mp:
                num_valid_case += outputs['valid_case']
                if outputs['inlier_cnt'] > best_inlier_cnt:
                    best_E = outputs['E']
                    best_inlier_cnt = outputs['inlier_cnt']
                    best_inliers = outputs['inlier']

            # Recover pose
            major_valid = num_valid_case > (max_ransac_iter/2)
            if major_valid:
                cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, new_kp_cur, new_kp_ref,
                                        focal=self.dataset.cam_intrinsics.fx,
                                        pp=principal_points,)

                # cheirality_check
                if cheirality_cnt > kp_cur.shape[0]*0.1:
                    best_Rt = [R, t]

        R, t = best_Rt
        pose = SE3()
        pose.R = R
        pose.t = t
        self.timers.timers["Ess. Mat."].append(time()-start_time)

        outputs = {"pose": pose, "inliers": best_inliers}
        return outputs

    def compute_pose_3d2d(self, kp1, kp2, depth_1):
        """Compute pose from 3d-2d correspondences
        Args:
            kp1 (Nx2 array): keypoints for view-1
            kp2 (Nx2 array): keypoints for view-2
            depth_1 (HxW array): depths for view-1
        Returns:
            pose (SE3): relative pose from view-2 to view-1
            kp1 (Nx2 array): filtered keypoints for view-1
            kp2 (Nx2 array): filtered keypoints for view-2
        """
        height, width = depth_1.shape

        # Filter keypoints outside image region
        x_idx = (kp2[:, 0] >= 0) * (kp2[:, 0] < width) 
        kp1 = kp1[x_idx]
        kp2 = kp2[x_idx]
        y_idx = (kp2[:, 1] >= 0) * (kp2[:, 1] < height) 
        kp1 = kp1[y_idx]
        kp2 = kp2[y_idx]

        # Filter keypoints outside depth range
        kp1_int = kp1.astype(np.int)
        kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
        non_zero_mask = (kp_depths != 0)
        depth_range_mask = (kp_depths < self.cfg.depth.max_depth) * (kp_depths > self.cfg.depth.min_depth)
        valid_kp_mask = non_zero_mask * depth_range_mask

        kp1 = kp1[valid_kp_mask]
        kp2 = kp2[valid_kp_mask]

        # Get 3D coordinates for kp1
        XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], self.dataset.cam_intrinsics)

        # initialize ransac setup
        best_rt = []
        best_inlier = 0
        max_ransac_iter = self.cfg.PnP.ransac.repeat
        
        for i in range(max_ransac_iter):
            # shuffle kp (only useful when random seed is fixed)	
            new_list = np.arange(0, kp2.shape[0], 1)	
            np.random.shuffle(new_list)
            new_XYZ = XYZ_kp1.copy()[new_list]
            new_kp2 = kp2.copy()[new_list]

            if new_kp2.shape[0] > 4:
                flag, r, t, inlier = cv2.solvePnPRansac(
                    objectPoints=new_XYZ,
                    imagePoints=new_kp2,
                    cameraMatrix=self.dataset.cam_intrinsics.mat,
                    distCoeffs=None,
                    iterationsCount=self.cfg.PnP.ransac.iter,
                    reprojectionError=self.cfg.PnP.ransac.reproj_thre,
                    )
                if flag and inlier.shape[0] > best_inlier:
                    best_rt = [r, t]
                    best_inlier = inlier.shape[0]
        pose = SE3()
        if len(best_rt) != 0:
            r, t = best_rt
            pose.R = cv2.Rodrigues(r)[0]
            pose.t = t
        pose.pose = pose.inv_pose
        return pose, kp1, kp2

    def update_global_pose(self, new_pose, scale):
        """update estimated poses w.r.t global coordinate system
        Args:
            new_pose (SE3)
            scale (float): scaling factor
        """
        self.cur_data['pose'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])

    def scale_recovery_single(self, cur_data, ref_data, E_pose, ref_id):
        """recover depth scale by comparing triangulated depths and CNN depths
        
        Args:
            cur_data (dict)
            ref_data (dict)
            E_pose (SE3)
        Returns:
            scale (float)
        """
        if self.cfg.translation_scale.kp_src == "kp_best":
            ref_kp = cur_data[self.cfg.translation_scale.kp_src]
            cur_kp = ref_data[self.cfg.translation_scale.kp_src][ref_id]
        elif self.cfg.translation_scale.kp_src == "kp_depth":
            ref_kp = cur_data[self.cfg.translation_scale.kp_src]
            cur_kp = ref_data[self.cfg.translation_scale.kp_src][ref_id]

        scale = self.find_scale_from_depth(
            ref_kp,
            cur_kp,
            E_pose.inv_pose, 
            self.cur_data['depth']
        )
        return scale

    def scale_recovery_iterative(self, cur_data, ref_data, E_pose, ref_id):
        """recover depth scale by comparing triangulated depths and CNN depths
        Iterative scale recovery is applied
        
        Args:
            cur_data (dict)
            ref_data (dict)
            E_pose (SE3)
        Returns:
            scale (float)
        """
        # Initialization
        scale = self.prev_scale
        delta = 0.001
        ref_data['rigid_flow_pose'] = {}

        for _ in range(5):    
            rigid_flow_pose = copy.deepcopy(E_pose)
            rigid_flow_pose.t *= scale

            ref_data['rigid_flow_pose'][ref_id] = SE3(rigid_flow_pose.inv_pose)

            # kp selection
            kp_sel_outputs = self.kp_selection_good_depth(cur_data, ref_data)
            ref_data['kp_depth'] = {}
            cur_data['kp_depth'] = kp_sel_outputs['kp1_depth'][0]
            for ref_id in ref_data['id']:
                ref_data['kp_depth'][ref_id] = kp_sel_outputs['kp2_depth'][ref_id][0]
            
            cur_data['rigid_flow_mask'] = kp_sel_outputs['rigid_flow_mask']
            
            # translation scale from triangulation v.s. CNN-depth
            if self.cfg.translation_scale.kp_src == "kp_best":
                ref_kp = cur_data[self.cfg.translation_scale.kp_src][ref_data['inliers'][ref_id]]
                cur_kp = ref_data[self.cfg.translation_scale.kp_src][ref_id][ref_data['inliers'][ref_id]]
            elif self.cfg.translation_scale.kp_src == "kp_depth":
                ref_kp = cur_data[self.cfg.translation_scale.kp_src]
                cur_kp = ref_data[self.cfg.translation_scale.kp_src][ref_id]
            new_scale = self.find_scale_from_depth(
                ref_kp,
                cur_kp,
                E_pose.inv_pose, 
                self.cur_data['depth']
            )

            delta_scale = np.abs(new_scale-scale)
            scale = new_scale
            self.prev_scale = new_scale
            
            if delta_scale < delta:
                return scale
        return scale

    def find_scale_from_depth(self, kp1, kp2, T_21, depth2):
        """Compute VO scaling factor for T_21
        Args:
            kp1 (Nx2 array): reference kp
            kp2 (Nx2 array): current kp
            T_21 (4x4 array): relative pose; from view 1 to view 2
            depth2 (HxW array): depth 2
        Returns:
            scale (float): scaling factor
        """
        # Triangulation
        img_h, img_w, _ = image_shape(depth2)
        kp1_norm = kp1.copy()
        kp2_norm = kp2.copy()

        kp1_norm[:, 0] = \
            (kp1[:, 0] - self.dataset.cam_intrinsics.cx) / self.dataset.cam_intrinsics.fx
        kp1_norm[:, 1] = \
            (kp1[:, 1] - self.dataset.cam_intrinsics.cy) / self.dataset.cam_intrinsics.fy
        kp2_norm[:, 0] = \
            (kp2[:, 0] - self.dataset.cam_intrinsics.cx) / self.dataset.cam_intrinsics.fx
        kp2_norm[:, 1] = \
            (kp2[:, 1] - self.dataset.cam_intrinsics.cy) / self.dataset.cam_intrinsics.fy

        _, X1_tri, X2_tri = triangulation(kp1_norm, kp2_norm, np.eye(4), T_21)

        # Triangulation outlier removal
        depth2_tri = convert_sparse3D_to_depth(kp2, X2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0

        # common mask filtering
        non_zero_mask_pred2 = (depth2 > 0)
        non_zero_mask_tri2 = (depth2_tri > 0)
        valid_mask2 = non_zero_mask_pred2 * non_zero_mask_tri2

        # if self.cfg.debug and False:
        #     # print("max: ", depth2_tri.max())
        #     # print("median: ", np.median(depth2_tri))
        #     f = plt.figure("depth2_tri")
        #     plt.imshow(depth2_tri, vmin=0, vmax=200)
        #     f,ax = plt.subplots(3,1, num="mask;depth2_tri, depth2")
        #     ax[0].imshow(valid_mask2*1.)
        #     ax[1].imshow(depth2_tri)
        #     ax[2].imshow(depth2)
        #     plt.show()

        # depth_pred_non_zero = np.concatenate([depth2[valid_mask2], depth1[valid_mask1]])
        # depth_tri_non_zero = np.concatenate([depth2_tri[valid_mask2], depth1_tri[valid_mask1]])
        depth_pred_non_zero = np.concatenate([depth2[valid_mask2]])
        depth_tri_non_zero = np.concatenate([depth2_tri[valid_mask2]])
        depth_ratio = depth_tri_non_zero / depth_pred_non_zero
        
        # Estimate scale (ransac)
        # if (valid_mask1.sum() + valid_mask2.sum()) > 10:
        if valid_mask2.sum() > 10:
            # RANSAC scaling solver
            ransac = linear_model.RANSACRegressor(
                        base_estimator=linear_model.LinearRegression(
                            fit_intercept=False),
                        min_samples=self.cfg.translation_scale.ransac.min_samples,
                        max_trials=self.cfg.translation_scale.ransac.max_trials,
                        stop_probability=self.cfg.translation_scale.ransac.stop_prob,
                        residual_threshold=self.cfg.translation_scale.ransac.thre
                        )
            if self.cfg.translation_scale.ransac.method == "depth_ratio":
                ransac.fit(
                    depth_ratio.reshape(-1, 1),
                    np.ones((depth_ratio.shape[0],1))
                    )
            elif self.cfg.translation_scale.ransac.method == "abs_diff":
                ransac.fit(
                    depth_tri_non_zero.reshape(-1, 1),
                    depth_pred_non_zero.reshape(-1, 1),
                )
            scale = ransac.estimator_.coef_[0, 0]

            # # scale outlier
            # if ransac.inlier_mask_.sum() / depth_ratio.shape[0] < 0.2:
            #     scale = -1
        else:
            scale = -1

        return scale

    def deep_flow_forward(self, in_cur_data, in_ref_data, forward_backward):
        """Update keypoints in cur_data and ref_data
        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            forward_backward (bool): use forward-backward consistency if True
        Returns:
            cur_data (dict): current data
            ref_data (dict): reference data
        """
        cur_data = copy.deepcopy(in_cur_data)
        ref_data = copy.deepcopy(in_ref_data)
        if self.cfg.deep_flow.precomputed_flow is None:
            # Preprocess image
            ref_imgs = []
            cur_imgs = []
            cur_img = np.transpose((cur_data['img'])/255, (2, 0, 1))
            for ref_id in ref_data['id']:
                ref_img = np.transpose((ref_data['img'][ref_id])/255, (2, 0, 1))
                ref_imgs.append(ref_img)
                cur_imgs.append(cur_img)
            ref_imgs = np.asarray(ref_imgs)
            cur_imgs = np.asarray(cur_imgs)
        else:
            # if precomputed flow is available, collect image timestamps for
            # later data reading
            ref_imgs = [ref_data['timestamp'][idx] for idx in ref_data['id']]
            cur_imgs = [cur_data['timestamp'] for i in ref_data['timestamp']]

        # Forward pass
        flows = {}
        flow_net_tracking = self.deep_models['flow'].inference_kp
        batch_size = self.cfg.deep_flow.batch_size
        num_forward = int(np.ceil(len(ref_data['id']) / batch_size))
        for i in range(num_forward):
            # Read precomputed flow / real-time flow
            batch_flows = flow_net_tracking(
                                    img1=ref_imgs[i*batch_size: (i+1)*batch_size],
                                    img2=cur_imgs[i*batch_size: (i+1)*batch_size],
                                    flow_dir=self.cfg.deep_flow.precomputed_flow,
                                    forward_backward=forward_backward,
                                    dataset=self.cfg.dataset)
            
            # Save flows at current view
            for j in range(batch_size):
                src_id = ref_data['id'][i*batch_size: (i+1)*batch_size][j]
                tgt_id = cur_data['id']
                flows[(src_id, tgt_id)] = batch_flows['forward'][j].copy()
                if forward_backward:
                    flows[(tgt_id, src_id)] = batch_flows['backward'][j].copy()
                    flows[(src_id, tgt_id, "diff")] = batch_flows['flow_diff'][j].copy()

            # Store flow
            for ref_id in ref_data['id']:
                ref_data['flow'][ref_id] = flows[(ref_data['id'][i], cur_data['id'])].copy()
                if forward_backward:
                    cur_data['flow'][ref_id] = flows[(cur_data['id'], ref_data['id'][i])].copy()
                    ref_data['flow_diff'][ref_id] = flows[(ref_data['id'][i], cur_data['id'], "diff")].copy()
        return cur_data, ref_data

    def unprojection(self, depth, cam_intrinsics):
        """Convert a depth map to XYZ
        Args:
            depth (HxW array): depth map
            cam_intrinsics (Intrinsics): camera intrinsics
        Returns:
            XYZ (HxWx3): 3D coordinates
        """
        height, width = depth.shape

        depth_mask = (depth != 0)
        depth_mask = np.repeat(np.expand_dims(depth_mask, 2), 3, axis=2)

        # initialize regular grid
        XYZ = np.ones((height, width, 3, 1))
        XYZ[:, :, 0, 0] = np.repeat(
                            np.expand_dims(np.arange(0, width), 0),
                            height, axis=0
                            )
        XYZ[:, :, 1, 0] = np.repeat(
                            np.expand_dims(np.arange(0, height), 1),
                            width, axis=1)

        inv_K = np.ones((1, 1, 3, 3))
        inv_K[0, 0] = cam_intrinsics.inv_mat
        inv_K = np.repeat(inv_K, height, axis=0)
        inv_K = np.repeat(inv_K, width, axis=1)

        XYZ = np.matmul(inv_K, XYZ)[:, :, :, 0]
        XYZ[:, :, 0] = XYZ[:, :, 0] * depth
        XYZ[:, :, 1] = XYZ[:, :, 1] * depth
        XYZ[:, :, 2] = XYZ[:, :, 2] * depth
        return XYZ

    def projection(self, XYZ, proj_mat):
        """Convert XYZ to [u,v,d]
        Args:
            XYZ (HxWx3): 3D coordinates
            proj_mat (3x3 array): camera intrinsics / projection matrix
        Returns:
            xy (HxWx2 array): projected image coordinates
        """
        if len(XYZ.shape) == 3:
            h, w, _ = XYZ.shape
            tmp_XYZ = XYZ.copy()
            tmp_XYZ[:, :, 0] /= tmp_XYZ[:, :, 2]
            tmp_XYZ[:, :, 1] /= tmp_XYZ[:, :, 2]
            tmp_XYZ[:, :, 2] /= tmp_XYZ[:, :, 2]
            tmp_XYZ = np.expand_dims(tmp_XYZ, axis=3)
            K = np.ones((1, 1, 3, 3))
            K[0, 0] = proj_mat
            K = np.repeat(K, h, axis=0)
            K = np.repeat(K, w, axis=1)

            xy = np.matmul(K, tmp_XYZ)[:, :, :2, 0]
        elif len(XYZ.shape) == 2:
            n, _ = XYZ.shape
            tmp_XYZ = XYZ.copy()
            tmp_XYZ[:, 0] /= tmp_XYZ[:, 2]
            tmp_XYZ[:, 1] /= tmp_XYZ[:, 2]
            tmp_XYZ[:, 2] /= tmp_XYZ[:, 2]
            tmp_XYZ = np.expand_dims(tmp_XYZ, axis=2)
            K = np.ones((1, 3, 3))
            K[0] = proj_mat
            K = np.repeat(K, n, axis=0)

            xy = np.matmul(K, tmp_XYZ)[:, :2, 0]
        return xy

    def transform_XYZ(self, XYZ, transformation):
        """Transform point cloud
        Args:
            XYZ (HxWx3 / Nx3): 3D coordinates
            pose (4x4 array): tranformation matrix
        Returns:
            new_XYZ (HxWx3 / Nx3): 3D coordinates
        """
        if len(XYZ.shape) == 3:
            h, w, _ = XYZ.shape
            new_XYZ = np.ones((h, w, 4, 1))
            new_XYZ[:, :, :3, 0] = XYZ

            T = np.ones((1, 1, 4, 4))
            T[0, 0] = transformation
            T = np.repeat(T, h, axis=0)
            T = np.repeat(T, w, axis=1)

            new_XYZ = np.matmul(T, new_XYZ)[:, :, :3, 0]
        elif len(XYZ.shape) == 2:
            n, _ = XYZ.shape
            new_XYZ = np.ones((n, 4, 1))
            new_XYZ[:, :3, 0] = XYZ

            T = np.ones((1, 4, 4))
            T[0] = transformation
            T = np.repeat(T, n, axis=0)

            new_XYZ = np.matmul(T, new_XYZ)[:, :3, 0]

        return new_XYZ

    def xy_to_uv(self, xy):
        """Convert image coordinates to optical flow
        Args:
            xy (HxWx2 array): image coordinates
        Returns:
            uv (HxWx2 array): x-flow and y-flow
        """
        h, w, _ = xy.shape
        img_grid = image_grid(h, w)
        uv = xy - img_grid
        return uv

    def compute_rigid_flow(self, depth, pose):
        """Compute rigid flow 
        Args:
            depth (HxW array): depth map of reference view
            pose (4x4 array): from reference to current
        Returns:
            rigid_flow (HxWx2): rigid flow [x-flow, y-flow]
        """
        XYZ_ref = self.unprojection(depth, self.dataset.cam_intrinsics)
        XYZ_cur = self.transform_XYZ(XYZ_ref, pose)
        xy = self.projection(XYZ_cur, self.dataset.cam_intrinsics.mat)
        rigid_flow = self.xy_to_uv(xy)
        return rigid_flow

    def kp_selection(self, cur_data, ref_data):
        """Choose valid kp from a series of operations
        """
        outputs = {}

        # initialization
        h, w = cur_data['depth'].shape
        ref_id = ref_data['id'][0]

        kp1 = image_grid(h, w)
        kp1 = np.expand_dims(kp1, 0)
        tmp_flow_data = np.transpose(np.expand_dims(ref_data['flow'][ref_id], 0), (0, 2, 3, 1))
        kp2 = kp1 + tmp_flow_data

        """ best-N selection """
        if self.cfg.kp_selection.uniform_filtered_bestN.enable:
            kp_sel_method = uniform_filtered_bestN
        elif self.cfg.kp_selection.bestN.enable:
            kp_sel_method = bestN
        
        outputs.update(
            kp_sel_method(
                kp1=kp1,
                kp2=kp2,
                ref_data=ref_data,
                cfg=self.cfg,
                outputs=outputs
                )
        )

        """ sampled kp selection """
        if self.cfg.kp_selection.sampled_kp.enable:
            sampled_kp_list = self.uniform_kp_list
            outputs.update(
                sampled_kp(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    kp_list=sampled_kp_list,
                    cfg=self.cfg,
                    outputs=outputs
                    )
        )

        # """ depth consistent kp selection """
        # if self.cfg.kp_selection.good_depth_kp.enable:
        #     ref_data['rigid_flow_pose'] = {}
        #     # compute rigid flow
        #     # gt_pose = np.linalg.inv(self.gt_poses[cur_data['id']]) @ self.gt_poses[ref_id] 
        #     rigid_flow_pose = ref_data['rigid_flow_pose'][ref_id]
        #     rigid_flow = self.compute_rigid_flow(
        #                                 ref_data['raw_depth'][ref_id],
        #                                 # np.linalg.inv(ref_data['deep_pose'][ref_id])
        #                                 rigid_flow_pose
        #                                 )
        #     rigid_flow_diff = np.linalg.norm(
        #                         rigid_flow - ref_data['flow'][ref_id].transpose(1,2,0),
        #                         axis=2)
        #     rigid_flow_diff = np.expand_dims(rigid_flow_diff, 2)
            
        #     # print("pose: ", np.linalg.inv(ref_data['deep_pose'][ref_id]))
        #     # print("gt_pose: ", gt_pose)
        #     # plt.figure("depth")
        #     # plt.imshow(self.ref_data['raw_depth'][ref_id])
        #     # plt.figure("rigid_flow x")
        #     # plt.imshow(rigid_flow[:,:,0])
        #     # plt.figure("rigid_flow y")
        #     # plt.imshow(rigid_flow[:,:,1])
        #     # plt.figure("ref_flow")
        #     # plt.imshow(ref_data['flow'][ref_id][0, :,:])
        #     # plt.figure("flow_diff")
        #     # plt.imshow(rigid_flow_diff, vmin=0, vmax=3)
        #     # plt.show()

        #     ref_data['rigid_flow_diff'][ref_id] = rigid_flow_diff

        #     # get depth-flow consistent kp
        #     outputs.update(
        #         good_depth_kp(
        #             kp1=kp1,
        #             kp2=kp2,
        #             ref_data=ref_data,
        #             cfg=self.cfg,
        #             outputs=outputs
        #             )
        #     )

        return outputs

    def kp_selection_good_depth(self, cur_data, ref_data):
        """Choose valid kp from a series of operations
        """
        outputs = {}

        # initialization
        h, w = cur_data['depth'].shape
        ref_id = ref_data['id'][0]

        kp1 = image_grid(h, w)
        kp1 = np.expand_dims(kp1, 0)
        tmp_flow_data = np.transpose(np.expand_dims(ref_data['flow'][ref_id], 0), (0, 2, 3, 1))
        kp2 = kp1 + tmp_flow_data

        """ depth consistent kp selection """
        if self.cfg.kp_selection.good_depth_kp.enable:
            ref_data['rigid_flow_diff'] = {}
            # compute rigid flow
            rigid_flow_pose = ref_data['rigid_flow_pose'][ref_id].pose
            rigid_flow = self.compute_rigid_flow(
                                        ref_data['raw_depth'][ref_id],  
                                        # cur_data['raw_depth'],
                                        # np.linalg.inv(ref_data['deep_pose'][ref_id])
                                        rigid_flow_pose
                                        )
            rigid_flow_diff = np.linalg.norm(
                                rigid_flow - ref_data['flow'][ref_id].transpose(1,2,0),
                                axis=2)
            rigid_flow_diff = np.expand_dims(rigid_flow_diff, 2)
            
            # print("pose: ", np.linalg.inv(ref_data['deep_pose'][ref_id]))
            # print("gt_pose: ", gt_pose)
            # plt.figure("depth")
            # plt.imshow(self.ref_data['raw_depth'][ref_id])
            # plt.figure("rigid_flow x")
            # plt.imshow(rigid_flow[:,:,0])
            # plt.figure("rigid_flow y")
            # plt.imshow(rigid_flow[:,:,1])
            # plt.figure("ref_flow")
            # plt.imshow(ref_data['flow'][ref_id][0, :,:])
            # plt.figure("flow_diff")
            # plt.imshow(rigid_flow_diff, vmin=0, vmax=3)
            # plt.show()

            ref_data['rigid_flow_diff'][ref_id] = rigid_flow_diff

            # get depth-flow consistent kp
            outputs.update(
                good_depth_kp(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    cfg=self.cfg,
                    outputs=outputs
                    )
            )

        return outputs

    def prepare_depth_consistency_data(self, cur_data, ref_data):
        """Prepare data for computing depth consistency
        Returns
            - data (dict): 
                - inv_K
                - K
                - depth
                - pose_T
        """
        data = {}

        # camera intrinsics
        data[('inv_K')] = np.eye(4)
        data[('inv_K')][:3, :3] = self.dataset.cam_intrinsics.inv_mat
        data[('inv_K')] = torch.from_numpy(data[('inv_K')]).unsqueeze(0).float().cuda()

        data[('K')] = np.eye(4)
        data[('K')][:3, :3] = self.dataset.cam_intrinsics.mat
        data[('K')] = torch.from_numpy(data[('K')]).unsqueeze(0).float().cuda()

        # current depth
        data[('depth', cur_data['id'])] = torch.from_numpy(cur_data['raw_depth']).unsqueeze(0).unsqueeze(0).float().cuda()

        for ref_id in ref_data['id']:
            # reference depth
            data[('depth', ref_id)] = torch.from_numpy(ref_data['raw_depth'][ref_id]).unsqueeze(0).unsqueeze(0).float().cuda()

            # pose
            data[('pose_T', cur_data['id'], ref_id)] = torch.from_numpy(ref_data['deep_pose'][ref_id]).unsqueeze(0).float().cuda()
        
        return data

    def warp_and_reproj_depth(self, inputs):
        """Get reprojected depths and warp depths
        - reproj_depth: the reference depths in source view 
        - warp_depth: the warped reference depths in source view

        Args:
            inputs:
                - (inv_K) (Nx4x4)
                - (K) (Nx4x4)
                - (depth, 0) (Nx1xHxW): current depth
                - (depth, ref_id) (Nx1xHxW): reference inv.depth
                - (pose_T, 0, ref_id) (Nx4x4): rel. pose from 0 to ref_id
        Returns:
            outputs:
                - ('warp_depth', 0, frame_id)
                - ('reproj_depth', 0, frame_id)
        """
        outputs = {}
        K = inputs['K']
        inv_K = inputs['inv_K']

        # Get depth and 3D points of frame_0
        cur_depth = inputs[('depth', self.cur_data['id'])]
        cam_points = self.backproj(cur_depth, inv_K)


        n, _, h, w = cur_depth.shape

        for frame_id in self.ref_data['id']:
            T = inputs[("pose_T", self.cur_data['id'], frame_id)]

            # reprojection
            reproj_xy = self.reproj(cur_depth, T, K, inv_K)

            # Warp src depth to tgt ref view
            outputs[('warp_depth', self.cur_data['id'], frame_id)] = nnFunc.grid_sample(
                inputs[("depth",  frame_id)],
                reproj_xy,
                padding_mode="border")
                
            # Reproject cur_depth
            transformed_cam_points = torch.matmul(T[:, :3, :], cam_points)
            transformed_cam_points = transformed_cam_points.view(n, 3, h, w)
            proj_depth = transformed_cam_points[:, 2:, :, :]
            outputs[('reproj_depth', self.cur_data['id'], frame_id)] = proj_depth
        return outputs

    def compute_depth_diff(self, depth_data):
        """
        inputs:
            depth_data:
                - ('warp_depth', cur_id, ref_id)
                - ('reproj_depth', cur_id, ref_id)
        """
        outputs = {}
        for ref_id in self.ref_data['id']:
            warp_depth = depth_data[('warp_depth', self.cur_data['id'], ref_id)]#.cpu().numpy()[0,0]
            reproj_depth = depth_data[('reproj_depth', self.cur_data['id'], ref_id)]#.cpu().numpy()[0,0]
            depth_diff = (warp_depth - reproj_depth).abs()

            method = "depth_ratio"
            if method == "sc":
                depth_sum = (warp_depth + reproj_depth).abs()
                depth_diff = (depth_diff / depth_sum).clamp(0, 1).cpu().numpy()[0,0]
            elif method == "depth_ratio":
                depth_diff = (depth_diff / reproj_depth).clamp(0, 1).cpu().numpy()[0,0]
            else:
                depth_diff = depth_diff.cpu().numpy()[0,0]

            outputs[('depth_diff', self.cur_data['id'], ref_id)] = depth_diff
        return outputs

    def compute_depth_consistency(self):
        """Compute depth consistency using CNN pose and CNN depths
        New data added to ref_data
            - deep_pose
            - depth_diff
        """
        self.ref_data['deep_pose'] = {}

        # Deep pose prediction
        for ref_id in self.ref_data['id']:
            # pose prediction
            pose = self.deep_models['pose'].inference(
                            self.ref_data['img'][ref_id],
                            self.cur_data['img'], 
                            )
            self.ref_data['deep_pose'][ref_id] = pose[0] # from cur->ref

        # compute depth consistency
        inputs = self.prepare_depth_consistency_data(self.cur_data, self.ref_data)
        depth_outputs = self.warp_and_reproj_depth(inputs)
        depth_consistency = self.compute_depth_diff(depth_outputs)

        self.ref_data['depth_diff'] = {}
        for ref_id in self.ref_data['id']:
            self.ref_data['depth_diff'][ref_id] = depth_consistency[('depth_diff', self.cur_data['id'], ref_id)]

    def tracking_hybrid(self):
        """Tracking using both Essential matrix and PnP
        Essential matrix for rotation (and direction);
            *** triangluate depth v.s. CNN-depth for translation scale ***
        PnP if Essential matrix fails
        """
        # First frame
        if self.tracking_stage == 0:
            # initial
            self.cur_data['pose'] = SE3(self.gt_poses[self.cur_data['id']])
            self.tracking_stage = 1
            return

        # Second to last frames
        elif self.tracking_stage >= 1:
            # Flow-net for 2D-2D correspondence
            start_time = time()
            cur_data, ref_data = self.deep_flow_forward(
                                        self.cur_data,
                                        self.ref_data,
                                        forward_backward=self.cfg.deep_flow.forward_backward)
            self.timers.timers['Flow-CNN'].append(time()-start_time)

            # Depth consistency (CNN depths + CNN pose)
            if self.cfg.kp_selection.depth_consistency.enable:
                self.compute_depth_consistency()

            # kp_selection
            kp_sel_outputs = self.kp_selection(cur_data, ref_data)

            # save selected kp
            ref_data['kp_best'] = {}
            cur_data['kp_best'] = kp_sel_outputs['kp1_best'][0]
            for ref_id in ref_data['id']:
                ref_data['kp_best'][ref_id] = kp_sel_outputs['kp2_best'][ref_id][0]
            
            if self.cfg.kp_selection.sampled_kp.enable:
                ref_data['kp_list'] = {}
                cur_data['kp_list'] = kp_sel_outputs['kp1_list'][0]
                for ref_id in ref_data['id']:
                    ref_data['kp_list'][ref_id] = kp_sel_outputs['kp2_list'][ref_id][0]
            
            # save mask
            cur_data['flow_mask'] = kp_sel_outputs['flow_mask']
            if self.cfg.kp_selection.uniform_filtered_bestN.enable:
                cur_data['valid_mask'] = kp_sel_outputs['valid_mask']
            
            if self.cfg.kp_selection.depth_consistency.enable:
                cur_data['depth_mask'] = kp_sel_outputs['depth_mask']
            
            # Pose estimation
            for ref_id in ref_data['id']:
                # Initialize hybrid pose
                hybrid_pose = SE3()

                # Essential matrix pose
                outputs_2d2d = self.compute_pose_2d2d(
                                cur_data[self.cfg.compute_2d2d_pose.kp_src],
                                ref_data[self.cfg.compute_2d2d_pose.kp_src][ref_id]) # pose: from cur->ref
                E_pose = outputs_2d2d['pose']

                # Rotation
                hybrid_pose.R = E_pose.R

                # save inliers
                ref_data['inliers'][ref_id] = outputs_2d2d['inliers']

                # scale recovery
                if np.linalg.norm(E_pose.t) != 0:
                    if self.cfg.translation_scale.method == "single":
                        scale = self.scale_recovery_single(cur_data, ref_data, E_pose, ref_id)
                    
                    elif self.cfg.translation_scale.method == "iterative":
                        scale = self.scale_recovery_iterative(cur_data, ref_data, E_pose, ref_id)
                    
                    if scale != -1:
                        hybrid_pose.t = E_pose.t * scale
                    
                # PnP if Essential matrix fail
                if np.linalg.norm(E_pose.t) == 0 or scale == -1:
                    pnp_pose, _, _ \
                        = self.compute_pose_3d2d(
                                    cur_data[self.cfg.PnP.kp_src],
                                    ref_data[self.cfg.PnP.kp_src][ref_id],
                                    ref_data['depth'][ref_id]
                                    ) # pose: from cur->ref
                    # use PnP pose instead of E-pose
                    hybrid_pose = pnp_pose
                    self.tracking_mode = "PnP"
                ref_data['pose'][ref_id] = copy.deepcopy(hybrid_pose)

            self.ref_data = copy.deepcopy(ref_data)
            self.cur_data = copy.deepcopy(cur_data)

            # update global poses
            pose = self.ref_data['pose'][self.ref_data['id'][-1]]
            self.update_global_pose(pose, 1)

            self.tracking_stage += 1
            del(ref_data)
            del(cur_data)

    def update_ref_data(self, ref_data, cur_data, window_size, kf_step=1):
        """Update reference data
        Args:
            ref_data (dict): reference data
                - e.g.
                    ref_data:
                    {
                        id: [0, 1, 2]
                        img: {0: I0, 1:I1, 2:I2}
                        ...
                    }
            cur_data (dict): current data
                - e.g.
                    cur_data:
                    {
                        id: 3
                        img: I3
                        ...
                    }
            cur_id (int): current image id
            window_size (int): number of frames in the window
        Returns:
            ref_data (dict): reference data
        """
        for key in cur_data:
            if key == "id":
                ref_data['id'].append(cur_data['id'])
                if len(ref_data['id']) > window_size - 1:
                    del(ref_data['id'][0])
            else:
                if ref_data.get(key, -1) == -1:
                    ref_data[key] = {}
                ref_data[key][cur_data['id']] = cur_data[key]
                if len(ref_data[key]) > window_size - 1:
                    drop_id = np.min(list(ref_data[key].keys()))
                    del(ref_data[key][drop_id])
        # Delete unused flow
        ref_data['flow'] = {}
        cur_data['flow'] = {}
        ref_data['flow_diff'] = {}
        return ref_data, cur_data

    def main(self):
        """ Initialization """
        len_seq = len(self.dataset.rgb_d_pose_pair)
        self.prev_scale = 1

        # Main
        print("==> Start VO")
        main_start_time = time()
        if self.cfg.no_confirm:
            start_frame = 0
        else:
            start_frame = int(input("Start with frame: "))

        for img_id in tqdm(range(start_frame, len_seq)):
        # for img_id in range(start_frame, len_seq):
        #     print("cur frame: ", img_id)
            self.tracking_mode = "Ess. Mat."

            """ Data reading """
            start_time = time()

            # Initialize ids and timestamps
            self.cur_data['id'] = img_id

            # if self.cfg.dataset == "kitti_odom":
            if "kitti" in self.cfg.dataset:
                self.cur_data['timestamp'] = img_id
            elif "tum" in self.cfg.dataset:
                self.cur_data['timestamp'] = sorted(list(self.dataset.rgb_d_pose_pair.keys()))[img_id]
            
            # Reading image
            if self.cfg.dataset == "kitti_odom":
                img = read_image(self.dataset.data_dir['img']+"/{:06d}.{}".format(img_id, self.cfg.image.ext), 
                                    self.cfg.image.height, self.cfg.image.width)
            elif self.cfg.dataset == "kitti_raw":
                img = read_image(self.dataset.data_dir['img']+"/{:010d}.{}".format(img_id, self.cfg.image.ext), 
                                    self.cfg.image.height, self.cfg.image.width)
            elif "tum" in self.cfg.dataset:
                img = read_image(self.dataset.data_dir['img']+"/{:.6f}.{}".format(self.cur_data['timestamp'], self.cfg.image.ext),
                                    self.cfg.image.height, self.cfg.image.width)
            img_h, img_w, _ = image_shape(img)
            self.cur_data['img'] = img
            self.timers.timers["img_reading"].append(time()-start_time)

            # Reading/Predicting depth
            if self.dataset.data_dir['depth_src'] is not None:
                self.cur_data['raw_depth'] = self.load_depth(
                                self.dataset.data_dir['depth'], 
                                self.dataset.rgb_d_pose_pair[self.cur_data['timestamp']]['depth'], 
                                self.dataset.data_dir['depth_src'], 
                                [img_h, img_w], 
                                dataset=self.cfg.dataset,
                                )
            else:
                start_time = time()
                self.cur_data['raw_depth'] = \
                        self.deep_models['depth'].inference(img=self.cur_data['img'])
                self.cur_data['raw_depth'] = cv2.resize(self.cur_data['raw_depth'],
                                                    (img_w, img_h),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
                self.timers.timers['Depth-CNN'].append(time()-start_time)
            self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            """ Visual odometry """
            start_time = time()
            if self.tracking_method == "hybrid":
                self.tracking_hybrid()
            else:
                raise NotImplementedError
            self.timers.timers["tracking"].append(time()-start_time)

            """ Visualization """
            start_time = time()
            self=self.drawer.main(self)
            self.timers.timers["visualization"].append(time()-start_time)

            """ Update reference and current data """
            self.ref_data, self.cur_data = self.update_ref_data(
                                    self.ref_data,
                                    self.cur_data,
                                    self.window_size,
                                    self.keyframe_step
            )

        print("=> Finish!")
        """ Display & Save result """
        # Output experiement information
        print("---- time breakdown ----")
        print("total runtime: {}".format(time() - main_start_time))
        for key in self.timers.timers.keys():
            if len(self.timers.timers[key]) != 0:
                print("{} : {}".format(key, np.asarray(self.timers.timers[key]).mean()))

        # Save trajectory map
        print("Save VO map.")
        map_png = "{}/map.png".format(self.cfg.result_dir)
        cv2.imwrite(map_png, self.drawer.data['traj'])

        # Save trajectory txt
        traj_txt = "{}/{}.txt".format(self.cfg.result_dir, self.cfg.seq)
        if "kitti" in self.cfg.dataset:
            global_poses_arr = convert_SE3_to_arr(self.global_poses)
            save_traj(traj_txt, global_poses_arr, format="kitti")
        elif "tum" in self.cfg.dataset:
            timestamps = sorted(list(self.dataset.rgb_d_pose_pair.keys()))
            global_poses_arr = convert_SE3_to_arr(self.global_poses, timestamps)
            save_traj(traj_txt, global_poses_arr, format="tum")
