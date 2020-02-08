# Copyright (C) Huangying Zhan 2019. All rights reserved.
#
# This software is licensed under the terms of the DF-VO licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import cv2
import copy
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn import linear_model
from time import time
from tqdm import tqdm

from libs.deep_depth.monodepth2 import Monodepth2DepthNet
from libs.geometry.ops_3d import *
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timers
from libs.matching.deep_flow import LiteFlow
from libs.camera_modules import SE3, Intrinsics
from libs.utils import *
from tool.evaluation.tum_tool.associate import associate, read_file_list
from tool.evaluation.tum_tool.pose_evaluation_utils import rot2quat


class VisualOdometry():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        # camera intrinsics
        self.cam_intrinsics = Intrinsics()

        # predicted global poses
        self.global_poses = {0: SE3()}

        # tracking stage
        self.tracking_stage = 0

        # configuration
        self.cfg = cfg

        # window size and keyframe step
        self.window_size = 2
        self.keyframe_step = 1

        # visualization interface
        self.initialize_visualization_drawer()

        # timer
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
                         "visualization_save_img", ])

        # reference data and current data
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

    def initialize_visualization_drawer(self):
        visual_h = self.cfg.visualization.window_h
        visual_w = self.cfg.visualization.window_w
        self.drawer = FrameDrawer(visual_h, visual_w)

        self.drawer.assign_data(
                    item="traj",
                    top_left=[0, 0], 
                    bottom_right=[int(visual_h), int(visual_w)],
                    )

        self.drawer.assign_data(
                    item="match_temp",
                    top_left=[int(visual_h/4*0), int(visual_w/4*2)], 
                    bottom_right=[int(visual_h/4*1), int(visual_w/4*4)],
                    )
        
        self.drawer.assign_data(
                    item="match_side",
                    top_left=[int(visual_h/4*1), int(visual_w/4*2)], 
                    bottom_right=[int(visual_h/4*2), int(visual_w/4*4)],
                    )
        
        self.drawer.assign_data(
                    item="depth",
                    top_left=[int(visual_h/4*2), int(visual_w/4*2)], 
                    bottom_right=[int(visual_h/4*3), int(visual_w/4*3)],
                    )
        
        self.drawer.assign_data(
                    item="flow1",
                    top_left=[int(visual_h/4*2), int(visual_w/4*3)], 
                    bottom_right=[int(visual_h/4*3), int(visual_w/4*4)],
                    )
        
        self.drawer.assign_data(
                    item="flow2",
                    top_left=[int(visual_h/4*3), int(visual_w/4*2)], 
                    bottom_right=[int(visual_h/4*4), int(visual_w/4*3)],
                    )
        
        self.drawer.assign_data(
                    item="flow_diff",
                    top_left=[int(visual_h/4*3), int(visual_w/4*3)], 
                    bottom_right=[int(visual_h/4*4), int(visual_w/4*4)],
                    )

    def get_intrinsics_param(self, dataset):
        """Read intrinsics parameters for each dataset
        Args:
            dataset (str): dataset
                - kitti
                - tum-1/2/3
        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        # Kitti
        if dataset == "kitti":
            img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
            intrinsics_param = load_kitti_odom_intrinsics(
                            os.path.join(img_seq_dir, "calib.txt"),
                            self.cfg.image.height, self.cfg.image.width
                            )[2]
        # TUM
        elif "tum" in dataset:
            tum_intrinsics = {
                "tum-1": [318.6, 255.3, 517.3, 516.5],  # fr1
                "tum-2": [325.1, 249.7, 520.9, 521.0],  # fr2
                "tum-3": [320.1, 247.6, 535.4, 539.2],  # fr3
            }
            intrinsics_param = tum_intrinsics[dataset]
        return intrinsics_param

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
        if self.cfg.dataset == "kitti":
            img_data_dir = os.path.join(img_seq_dir, "image_2")
        elif "tum" in self.cfg.dataset:
            img_data_dir = os.path.join(img_seq_dir, "rgb")
        else:
            warn_msg = "Wrong dataset [{}] is given.".format(self.cfg.dataset)
            warn_msg += "\n Choose from [kitti, tum-1/2/3]"
            assert False, warn_msg
        
        # get depth data directory
        depth_src_cases = {
            0: "gt",
            None: None
            }
        depth_src = depth_src_cases[self.cfg.depth.depth_src]

        if self.cfg.dataset == "kitti":
            if depth_src == "gt":
                depth_data_dir = "{}/gt/{}/".format(
                                    self.cfg.directory.depth_dir, self.cfg.seq
                                )
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

    def get_gt_poses(self):
        """load ground-truth poses
        Returns:
            gt_poses (dict): each pose is 4x4 array
        """
        if self.cfg.directory.gt_pose_dir is not None:
            if self.cfg.dataset == "kitti":
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

    def setup(self):
        """Reading configuration and setup, including
        - Get camera intrinsics
        - Get tracking method
        - Get feature tracking method
        - Get image & (optional depth) data
        - Generate keypoint sampling scheme
        - Deep networks
        - Load GT poses
        - Set drawer
        """
        # read camera intrinsics
        intrinsics_param = self.get_intrinsics_param(self.cfg.dataset)
        self.cam_intrinsics = Intrinsics(intrinsics_param)

        # get tracking method
        self.tracking_method = self.get_tracking_method(self.cfg.tracking_method)

        # feature tracking method
        self.feature_tracking_method = self.get_feat_track_methods(
                                            self.cfg.feature_tracking_method
                                            )

        # get image and depth data directory
        self.img_path_dir, self.depth_seq_dir, self.depth_src = self.get_img_depth_dir()

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
        if self.depth_src is None:
            if self.cfg.depth.pretrained_model is not None:
                self.deep_models['depth'] = self.initialize_deep_depth_model()
            else:
                assert False, "No precomputed depths nor pretrained depth model"

        # Load GT pose
        self.gt_poses = self.get_gt_poses()

        # Set drawer
        self.drawer.get_traj_init_xy(
                        vis_h=self.drawer.h,
                        vis_w=self.drawer.h,
                        gt_poses=self.gt_poses)

    def load_depth(self, depth_seq_dir, img_id, depth_src,
                   resize=None, dataset="kitti"):
        """Load depth map for different source
        Args:
            depth_seq_dir (str): depth sequence dir
            img_id (int): depth image id
            depth_src (str): depth src type
                - gt
            resize (int list): [target_height, target_width]
            dataset (str):
                - kitti
                - tum
        Returns:
            depth (HxW array): depth map
        """
        if dataset == "kitti":
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
            pose (SE3): relative pose from current to reference view
            best_inliers (N boolean array): inlier mask
        """
        principal_points = (self.cam_intrinsics.cx, self.cam_intrinsics.cy)

        # validity check
        valid_cfg = self.cfg.compute_2d2d_pose.validity
        valid_case = True

        # initialize ransac setup
        best_Rt = []
        best_inlier_cnt = 0
        max_ransac_iter = self.cfg.compute_2d2d_pose.ransac.repeat
        best_inliers = np.ones((kp_ref.shape[0])) == 1

        if valid_cfg.method == "flow+chei":
            # check flow magnitude
            avg_flow = np.mean(np.linalg.norm(kp_ref-kp_cur, axis=1))
            min_flow = self.cfg.compute_2d2d_pose.min_flow
            valid_case = avg_flow > min_flow
        if valid_case:
            for i in range(max_ransac_iter): # repeat ransac for several times for stable result
                # shuffle kp_cur and kp_ref (only useful when random seed is fixed)
                new_list = np.random.randint(0, kp_cur.shape[0], (kp_cur.shape[0]))
                new_kp_cur = kp_cur.copy()[new_list]
                new_kp_ref = kp_ref.copy()[new_list]

                start_time = time()
                E, inliers = cv2.findEssentialMat(
                            new_kp_cur,
                            new_kp_ref,
                            focal=self.cam_intrinsics.fx,
                            pp=principal_points,
                            method=cv2.RANSAC,
                            prob=0.99,
                            threshold=self.cfg.compute_2d2d_pose.ransac.reproj_thre,
                            )
                
                # check homography inlier ratio
                if valid_cfg.method == "homo_ratio":
                    # Find homography
                    H, H_inliers = cv2.findHomography(
                                new_kp_cur,
                                new_kp_ref,
                                method=cv2.RANSAC,
                                confidence=0.99,
                                ransacReprojThreshold=0.2,
                                )
                    H_inliers_ratio = H_inliers.sum()/(H_inliers.sum()+inliers.sum())
                    valid_case = H_inliers_ratio < 0.25
                
                if valid_case:
                    cheirality_cnt, R, t, _ = cv2.recoverPose(E, new_kp_cur, new_kp_ref,
                                            focal=self.cam_intrinsics.fx,
                                            pp=principal_points,)
                    self.timers.timers["Ess. Mat."].append(time()-start_time)
                    
                    # check best inlier cnt
                    if valid_cfg.method == "flow+chei":
                        inlier_check = inliers.sum() > best_inlier_cnt and cheirality_cnt > 50
                    elif valid_cfg.method == "homo_ratio":
                        inlier_check = inliers.sum() > best_inlier_cnt
                    else:
                        assert False, "wrong cfg for compute_2d2d_pose.validity.method"
                    
                    if inlier_check:
                        best_Rt = [R, t]
                        best_inlier_cnt = inliers.sum()
                        best_inliers = inliers
            if len(best_Rt) == 0:
                R = np.eye(3)
                t = np.zeros((3, 1))
                best_Rt = [R, t]
        else:
            R = np.eye(3)
            t = np.zeros((3,1))
            best_Rt = [R, t]
        R, t = best_Rt
        pose = SE3()
        pose.R = R
        pose.t = t
        return pose, best_inliers

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
        XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], self.cam_intrinsics)

        # initialize ransac setup
        best_rt = []
        best_inlier = 0
        max_ransac_iter = self.cfg.PnP.ransac.repeat
        
        for i in range(max_ransac_iter):
            # shuffle kp_cur and kp_ref (only useful when random seed is fixed)
            new_list = np.random.randint(0, kp2.shape[0], (kp2.shape[0]))
            new_XYZ = XYZ_kp1.copy()[new_list]
            new_kp2 = kp2.copy()[new_list]

            if new_kp2.shape[0] > 4:
                flag, r, t, inlier = cv2.solvePnPRansac(
                    objectPoints=new_XYZ,
                    imagePoints=new_kp2,
                    cameraMatrix=self.cam_intrinsics.mat,
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
            (kp1[:, 0] - self.cam_intrinsics.cx) / self.cam_intrinsics.fx
        kp1_norm[:, 1] = \
            (kp1[:, 1] - self.cam_intrinsics.cy) / self.cam_intrinsics.fy
        kp2_norm[:, 0] = \
            (kp2[:, 0] - self.cam_intrinsics.cx) / self.cam_intrinsics.fx
        kp2_norm[:, 1] = \
            (kp2[:, 1] - self.cam_intrinsics.cy) / self.cam_intrinsics.fy

        _, _, X2_tri = triangulation(kp1_norm, kp2_norm, np.eye(4), T_21)

        # Triangulation outlier removal
        depth2_tri = convert_sparse3D_to_depth(kp2, X2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0

        # common mask filtering
        non_zero_mask_pred = (depth2 > 0)
        non_zero_mask_tri = (depth2_tri > 0)
        valid_mask = non_zero_mask_pred * non_zero_mask_tri

        depth_pred_non_zero = depth2[valid_mask]
        depth_tri_non_zero = depth2_tri[valid_mask]
        
        # Estimate scale (ransac)
        if valid_mask.sum() > 50: #self.cfg.translation_scale.ransac.min_samples:
            # RANSAC scaling solver
            ransac = linear_model.RANSACRegressor(
                        base_estimator=linear_model.LinearRegression(
                            fit_intercept=False),
                        min_samples=self.cfg.translation_scale.ransac.min_samples,
                        max_trials=self.cfg.translation_scale.ransac.max_trials,
                        stop_probability=self.cfg.translation_scale.ransac.stop_prob,
                        residual_threshold=self.cfg.translation_scale.ransac.thre
                        )
            ransac.fit(
                    depth_tri_non_zero.reshape(-1, 1),
                    depth_pred_non_zero.reshape(-1, 1)
                    )
            scale = ransac.estimator_.coef_[0, 0]
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

        # Regular sampling
        kp_list_regular = self.uniform_kp_list
        kp_ref_regular = np.zeros((len(ref_data['id']), len(kp_list_regular), 2))
        num_kp_regular = len(kp_list_regular)

        # Best-N sampling
        kp_ref_best = np.zeros((len(ref_data['id']), self.cfg.deep_flow.num_kp, 2))
        num_kp_best = self.cfg.deep_flow.num_kp
        
        # Forward pass
        flows = {}
        flow_net_tracking = self.deep_models['flow'].inference_kp
        batch_size = self.cfg.deep_flow.batch_size
        num_forward = int(np.ceil(len(ref_data['id']) / batch_size))
        for i in range(num_forward):
            # Read precomputed flow / real-time flow
            batch_kp_ref_best, batch_kp_cur_best, batch_kp_ref_regular, batch_kp_cur_regular, batch_flows = flow_net_tracking(
                                    img1=ref_imgs[i*batch_size: (i+1)*batch_size],
                                    img2=cur_imgs[i*batch_size: (i+1)*batch_size],
                                    kp_list=kp_list_regular,
                                    img_crop=self.cfg.crop.flow_crop,
                                    flow_dir=self.cfg.deep_flow.precomputed_flow,
                                    N_list=num_kp_regular,
                                    N_best=num_kp_best,
                                    kp_sel_method=self.cfg.deep_flow.kp_sel_method,
                                    forward_backward=forward_backward,
                                    dataset=self.cfg.dataset)
            
            # Save keypoints at current view
            kp_ref_best[i*batch_size:(i+1)*batch_size] = batch_kp_cur_best.copy() # each kp_ref_best saves best-N kp at cur-view
            kp_ref_regular[i*batch_size:(i+1)*batch_size] = batch_kp_cur_regular.copy() # each kp_ref_list saves regular kp at cur-view

            # Save keypoints at reference view
            for j in range(batch_size):
                src_id = ref_data['id'][i*batch_size: (i+1)*batch_size][j]
                tgt_id = cur_data['id']
                flows[(src_id, tgt_id)] = batch_flows['forward'][j].copy()
                if forward_backward:
                    flows[(tgt_id, src_id)] = batch_flows['backward'][j].copy()
                    flows[(src_id, tgt_id, "diff")] = batch_flows['flow_diff'][j].copy()

        # Store kp
        cur_data['kp_best'] = batch_kp_ref_best[0].copy() # cur_data save each kp at ref-view (i.e. regular grid)
        cur_data['kp_list'] = batch_kp_ref_regular[0].copy() # cur_data save each kp at ref-view (i.e. regular grid)
        cur_data['kp_best'] = cur_data['kp_best'][cur_data['kp_best'][:,0]!=-1] # remove invalid kp

        for i, ref_id in enumerate(ref_data['id']):
            ref_data['kp_best'][ref_id] = kp_ref_best[i].copy()
            ref_data['kp_list'][ref_id] = kp_ref_regular[i].copy()
            ref_data['kp_best'][ref_id] = ref_data['kp_best'][ref_id][ref_data['kp_best'][ref_id][:,0]!=-1] # remove invalid kp

            # Store flow
            ref_data['flow'][ref_id] = flows[(ref_data['id'][i], cur_data['id'])].copy()
            if forward_backward:
                cur_data['flow'][ref_id] = flows[(cur_data['id'], ref_data['id'][i])].copy()
                ref_data['flow_diff'][ref_id] = flows[(ref_data['id'][i], cur_data['id'], "diff")].copy()
        return cur_data, ref_data

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

            for ref_id in self.ref_data['id']:
                # Compose hybrid pose
                hybrid_pose = SE3()

                # FIXME: add if statement for deciding which kp to use
                # Essential matrix pose
                E_pose, _ = self.compute_pose_2d2d(
                                cur_data['kp_best'],
                                ref_data['kp_best'][ref_id]) # pose: from cur->ref

                # Rotation
                hybrid_pose.R = E_pose.R

                # translation scale from triangulation v.s. CNN-depth
                if np.linalg.norm(E_pose.t) != 0:
                    scale = self.find_scale_from_depth(
                        cur_data[self.cfg.translation_scale.kp_src], ref_data[self.cfg.translation_scale.kp_src][ref_id],
                        E_pose.inv_pose, self.cur_data['depth']
                    )
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
                # ref_data['pose'][ref_id] = hybrid_pose

            self.ref_data = copy.deepcopy(ref_data)
            self.cur_data = copy.deepcopy(cur_data)

            # copy keypoint for visualization
            self.ref_data['kp'] = copy.deepcopy(ref_data['kp_best'])
            self.cur_data['kp'] = copy.deepcopy(cur_data['kp_best'])
            
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
                ref_data[key][cur_data['id']] = cur_data[key]
                if len(ref_data[key]) > window_size - 1:
                    drop_id = np.min(list(ref_data[key].keys()))
                    del(ref_data[key][drop_id])
        # Delete unused flow
        ref_data['flow'] = {}
        cur_data['flow'] = {}
        ref_data['flow_diff'] = {}
        return ref_data, cur_data

    def synchronize_rgbd_pose_pairs(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        mainly for TUM-RGBD dataset
        Returns:
            rgb_d_pose_pair (dict):
                - rgb_timestamp: {depth: depth_timestamp, pose: pose_timestamp}
        """
        rgb_d_pose_pair = {}
        # KITTI 
        if self.cfg.dataset == "kitti":
            len_seq = len(self.gt_poses)
            for i in range(len_seq):
                rgb_d_pose_pair[i] = {}
                rgb_d_pose_pair[i]['depth'] = i
                rgb_d_pose_pair[i]['pose'] = i

        # TUM
        elif "tum" in self.cfg.dataset:        
            # associate rgb-depth-pose timestamp pair
            rgb_list = read_file_list(self.img_path_dir +"/../rgb.txt")
            depth_list = read_file_list(self.img_path_dir +"/../depth.txt")
            pose_list = read_file_list(self.img_path_dir +"/../groundtruth.txt")

            for i in rgb_list:
                rgb_d_pose_pair[i] = {}

            # associate rgb-d
            matches = associate(
                first_list=rgb_list,
                second_list=depth_list,
                offset=0,
                max_difference=0.02
            )
            for match in matches:
                rgb_stamp = match[0]
                depth_stamp = match[1]
                rgb_d_pose_pair[rgb_stamp]['depth'] = depth_stamp
            
            # associate rgb-pose
            matches = associate(
                first_list=rgb_list,
                second_list=pose_list,
                offset=0,
                max_difference=0.02
            )
            
            for match in matches:
                rgb_stamp = match[0]
                pose_stamp = match[1]
                rgb_d_pose_pair[rgb_stamp]['pose'] = pose_stamp
            
            # Clear pairs without depth
            to_del_pair = []
            for rgb_stamp in rgb_d_pose_pair:
                if rgb_d_pose_pair[rgb_stamp].get("depth", -1) == -1:
                    to_del_pair.append(rgb_stamp)
            for rgb_stamp in to_del_pair:
                del(rgb_d_pose_pair[rgb_stamp])
            
            # # Clear pairs without pose
            to_del_pair = []
            tmp_rgb_d_pose_pair = copy.deepcopy(rgb_d_pose_pair)
            for rgb_stamp in tmp_rgb_d_pose_pair:
                if rgb_d_pose_pair[rgb_stamp].get("pose", -1) == -1:
                    to_del_pair.append(rgb_stamp)
            for rgb_stamp in to_del_pair:
                del(tmp_rgb_d_pose_pair[rgb_stamp])
            
            # timestep
            timestep = 5
            to_del_pair = []
            for cnt, rgb_stamp in enumerate(rgb_d_pose_pair):
                if cnt % timestep != 0:
                    to_del_pair.append(rgb_stamp)
            for rgb_stamp in to_del_pair:
                del(rgb_d_pose_pair[rgb_stamp])
            
            len_seq = len(rgb_d_pose_pair)
            
            # Update gt pose
            self.tmp_gt_poses = {}
            gt_pose_0_time = tmp_rgb_d_pose_pair[sorted(list(tmp_rgb_d_pose_pair.keys()))[0]]['pose']
            gt_pose_0 = self.gt_poses[gt_pose_0_time]
            
            i = 0
            for rgb_stamp in sorted(list(rgb_d_pose_pair.keys())):
                if rgb_d_pose_pair[rgb_stamp].get("pose", -1) != -1:
                    self.tmp_gt_poses[i] = np.linalg.inv(gt_pose_0) @ self.gt_poses[rgb_d_pose_pair[rgb_stamp]['pose']]
                else:
                    self.tmp_gt_poses[i] = np.eye(4)
                i += 1
            self.gt_poses = copy.deepcopy(self.tmp_gt_poses)
        return rgb_d_pose_pair

    def main(self):
        """ Initialization """
        # Synchronize rgb-d-pose pair
        self.rgb_d_pose_pair = self.synchronize_rgbd_pose_pairs()
        len_seq = len(self.rgb_d_pose_pair)

        # Main
        print("==> Start VO")
        main_start_time = time()
        start_frame = int(input("Start with frame: "))

        for img_id in tqdm(range(start_frame, len_seq)):
            self.tracking_mode = "Ess. Mat."

            """ Data reading """
            start_time = time()

            # Initialize ids and timestamps
            self.cur_data['id'] = img_id

            if self.cfg.dataset == "kitti":
                self.cur_data['timestamp'] = img_id
            elif "tum" in self.cfg.dataset:
                self.cur_data['timestamp'] = sorted(list(self.rgb_d_pose_pair.keys()))[img_id]
            
            # Reading image
            if self.cfg.dataset == "kitti":
                img = read_image(self.img_path_dir+"/{:06d}.{}".format(img_id, self.cfg.image.ext), 
                                    self.cfg.image.height, self.cfg.image.width)
            elif "tum" in self.cfg.dataset:
                img = read_image(self.img_path_dir+"/{:.6f}.{}".format(self.cur_data['timestamp'], self.cfg.image.ext),
                                    self.cfg.image.height, self.cfg.image.width)
            img_h, img_w, _ = image_shape(img)
            self.cur_data['img'] = img
            self.timers.timers["img_reading"].append(time()-start_time)

            # Reading/Predicting depth
            if self.depth_src is not None:
                self.cur_data['raw_depth'] = self.load_depth(
                                self.depth_seq_dir, 
                                self.rgb_d_pose_pair[self.cur_data['timestamp']]['depth'], 
                                self.depth_src, 
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
        if self.cfg.dataset == "kitti":
            global_poses_arr = convert_SE3_to_arr(self.global_poses)
            save_traj(traj_txt, global_poses_arr, format="kitti")
        elif "tum" in self.cfg.dataset:
            timestamps = sorted(list(self.rgb_d_pose_pair.keys()))
            global_poses_arr = convert_SE3_to_arr(self.global_poses, timestamps)
            save_traj(traj_txt, global_poses_arr, format="tum")
