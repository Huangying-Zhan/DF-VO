''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-06
@LastEditors: Huangying Zhan
@Description: This file contains Essential matrix based tracker
'''

import cv2
import copy
import multiprocessing as mp
import numpy as np
from sklearn import linear_model
import torch


from .gric import *
from libs.geometry.camera_modules import SE3
from libs.geometry.ops_3d import *
from libs.geometry.rigid_flow import RigidFlow
from libs.general.utils import image_shape, image_grid
from libs.matching.kp_selection import opt_rigid_flow_kp

def find_Ess_mat(inputs):
    """Find essetial matrix 

    Args:
        a dictionary containing

            - **kp_cur** (array, [Nx2]): keypoints at current view
            - **kp_ref** (array, [Nx2]): keypoints at reference view
            - **H_inliers** (array, [N]): boolean inlier mask 
            - **cfg** (edict): configuration dictionary related to pose estimation from 2D-2D matches
            - **cam_intrinsics** (Intrinsics): camera intrinsics

    Returns:
        a dictionary containing
            - **E** (array, [3x3]): essential matrix
            - **valid_case** (bool): validity of the solution
            - **inlier_cnt** (int): number of inliners
            - **inlier** (array, [N]): boolean inlier mask
        
    """
    # inputs
    kp_cur = inputs['kp_cur']
    kp_ref = inputs['kp_ref']
    H_inliers = inputs['H_inliers']
    cfg = inputs['cfg']
    cam_intrinsics = inputs['cam_intrinsics']

    # initialization
    valid_cfg = cfg.e_tracker.validity
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)
    fx = cam_intrinsics.fx

    # compute Ess
    E, inliers = cv2.findEssentialMat(
                        kp_cur,
                        kp_ref,
                        focal=fx,
                        pp=principal_points,
                        method=cv2.RANSAC,
                        prob=0.99,
                        threshold=cfg.e_tracker.ransac.reproj_thre,
                        )
    # check homography inlier ratio
    if valid_cfg.method == "homo_ratio":
        H_inliers_ratio = H_inliers.sum()/(H_inliers.sum()+inliers.sum())
        valid_case = H_inliers_ratio < valid_cfg.thre
    elif valid_cfg.method == "flow":
        cheirality_cnt, R, t, _ = cv2.recoverPose(E, kp_cur, kp_ref,
                                focal=cam_intrinsics.fx,
                                pp=principal_points,)
        valid_case = cheirality_cnt > kp_cur.shape[0]*0.05
    elif valid_cfg.method == "GRIC":
        H_gric = inputs['H_gric']
        # get F from E
        K = cam_intrinsics.mat
        F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
        E_res = compute_fundamental_residual(F, kp_cur, kp_ref)

        E_gric = calc_GRIC(
            res=E_res,
            sigma=0.8,
            n=kp_cur.shape[0],
            model='EMat'
        )
        valid_case = H_gric > E_gric
    
    # gather output
    outputs = {}
    outputs['E'] = E
    outputs['valid_case'] = valid_case
    outputs['inlier_cnt'] = inliers.sum()
    outputs['inlier'] = inliers

    return outputs


def get_E_from_pose(pose):
    """Recover essential matrix from a pose
    
    Args:
        pose (SE3): SE3
    
    Returns:
        E (array, [3x3]): essential matrix
    """
    R = pose.R
    t = pose.t / np.linalg.norm(pose.t)
    t_ssym = np.zeros((3,3))
    t_ssym[0, 1] = - t[2,0]
    t_ssym[0, 2] = t[1,0]
    t_ssym[1, 2] = - t[0,0]
    t_ssym[2, 1] = t[0,0]
    t_ssym[2, 0] = - t[1,0]
    t_ssym[1, 0] = t[2,0]

    E = t_ssym @ R
    return E

class EssTracker():
    def __init__(self, cfg, cam_intrinsics, timers):
        """
        Args:
            cfg (edict): configuration dictionary
            cam_intrinsics (Intrinsics): camera intrinsics
            timers (Timer): timers
        """
        self.cfg = cfg
        self.prev_scale = 0
        self.prev_pose = SE3()
        self.cam_intrinsics = cam_intrinsics

        # multiprocessing (not used since doesn't speed up much)
        # if self.cfg.use_multiprocessing:
        #     self.p = mp.Pool(2)
        
        # Rigid flow data
        if self.cfg.kp_selection.rigid_flow_kp.enable:
            self.K = np.eye(4)
            self.inv_K = np.eye(4)
            self.K[:3, :3] = cam_intrinsics.mat
            self.inv_K[:3, :3] = cam_intrinsics.inv_mat
            self.K = torch.from_numpy(self.K).float().unsqueeze(0).cuda()
            self.inv_K = torch.from_numpy(self.inv_K).float().unsqueeze(0).cuda()
            self.rigid_flow_layer = RigidFlow(self.cfg.image.height, self.cfg.image.width).cuda()
        
        # FIXME: For debug
        self.timers = timers

    def compute_pose_2d2d(self, kp_ref, kp_cur, is_iterative):
        """Compute the pose from view2 to view1
        
        Args:
            kp_ref (array, [Nx2]): keypoints for reference view
            kp_cur (array, [Nx2]): keypoints for current view
            cam_intrinsics (Intrinsics): camera intrinsics
            is_iterative (bool): is iterative stage
        
        Returns:
            a dictionary containing
                - **pose** (SE3): relative pose from current to reference view
                - **best_inliers** (array, [N]): boolean inlier mask
        """
        principal_points = (self.cam_intrinsics.cx, self.cam_intrinsics.cy)

        # validity check
        valid_cfg = self.cfg.e_tracker.validity
        valid_case = True

        # initialize ransac setup
        R = np.eye(3)
        t = np.zeros((3,1))
        best_Rt = [R, t]
        best_inlier_cnt = 0
        max_ransac_iter = self.cfg.e_tracker.ransac.repeat if is_iterative else 3
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
        elif valid_cfg.method == "GRIC":
            if kp_cur.shape[0] > 10:
                self.timers.start('GRIC-H', 'E-tracker')
                self.timers.start('find H', 'E-tracker')
                H, H_inliers = cv2.findHomography(
                            kp_cur,
                            kp_ref,
                            method=cv2.RANSAC,
                            confidence=0.99,
                            ransacReprojThreshold=1,
                            )
                self.timers.end('find H')

                H_res = compute_homography_residual(H, kp_cur, kp_ref)
                H_gric = calc_GRIC(
                            res=H_res,
                            sigma=0.8,
                            n=kp_cur.shape[0],
                            model="HMat"
                )
                self.timers.end('GRIC-H')
            else:
                valid_case = False

        
        if valid_case:
            num_valid_case = 0
            self.timers.start('find-Ess (full)', 'E-tracker')
            for i in range(max_ransac_iter): # repeat ransac for several times for stable result
                # shuffle kp_cur and kp_ref (only useful when random seed is fixed)	
                new_list = np.arange(0, kp_cur.shape[0], 1)	
                np.random.shuffle(new_list)
                new_kp_cur = kp_cur.copy()[new_list]
                new_kp_ref = kp_ref.copy()[new_list]

                self.timers.start('find-Ess', 'E-tracker')
                E, inliers = cv2.findEssentialMat(
                            new_kp_cur,
                            new_kp_ref,
                            focal=self.cam_intrinsics.fx,
                            pp=principal_points,
                            method=cv2.RANSAC,
                            prob=0.99,
                            threshold=self.cfg.e_tracker.ransac.reproj_thre,
                            )
                self.timers.end('find-Ess')

                # check homography inlier ratio
                if valid_cfg.method == "homo_ratio":
                    H_inliers_ratio = H_inliers.sum()/(H_inliers.sum()+inliers.sum())
                    valid_case = H_inliers_ratio < valid_cfg.thre
                    # print("valid: {} ratio: {}".format(valid_case, H_inliers_ratio))

                    # inlier check
                    inlier_check = inliers.sum() > best_inlier_cnt
                elif valid_cfg.method == "flow":
                    cheirality_cnt, R, t, _ = cv2.recoverPose(E, new_kp_cur, new_kp_ref,
                                            focal=self.cam_intrinsics.fx,
                                            pp=principal_points)
                    valid_case = cheirality_cnt > kp_cur.shape[0]*0.1
                    
                    # inlier check
                    inlier_check = inliers.sum() > best_inlier_cnt and cheirality_cnt > kp_cur.shape[0]*0.05               
                elif valid_cfg.method == "GRIC":
                    self.timers.start('GRIC-E', 'E-tracker')
                    # get F from E
                    K = self.cam_intrinsics.mat
                    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
                    E_res = compute_fundamental_residual(F, new_kp_cur, new_kp_ref)

                    E_gric = calc_GRIC(
                        res=E_res,
                        sigma=0.8,
                        n=kp_cur.shape[0],
                        model='EMat'
                    )
                    valid_case = H_gric > E_gric

                    # inlier check
                    inlier_check = inliers.sum() > best_inlier_cnt
                    self.timers.end('GRIC-E')

                # save best_E
                if inlier_check:
                    best_E = E
                    best_inlier_cnt = inliers.sum()

                    revert_new_list = np.zeros_like(new_list)
                    for cnt, i in enumerate(new_list):
                        revert_new_list[i] = cnt
                    best_inliers = inliers[list(revert_new_list)]
                num_valid_case += (valid_case * 1)

            self.timers.end('find-Ess (full)')
            major_valid = num_valid_case > (max_ransac_iter/2)
            if major_valid:
                self.timers.start('recover pose', 'E-tracker')
                cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, kp_cur, kp_ref,
                                        focal=self.cam_intrinsics.fx,
                                        pp=principal_points,
                                        )
                self.timers.end('recover pose')

                # cheirality_check
                if cheirality_cnt > kp_cur.shape[0]*0.1:
                    best_Rt = [R, t]

        R, t = best_Rt
        pose = SE3()
        pose.R = R
        pose.t = t
        outputs = {"pose": pose, "inliers": best_inliers[:,0]==1}
        return outputs

    # def compute_pose_2d2d_mp(self, kp_ref, kp_cur):
    #     """Compute the pose from view2 to view1 (multiprocessing version)
    #     Speed doesn't change much
        
    #     Args:
    #         kp_ref (array, [Nx2]): keypoints for reference view
    #         kp_cur (array, [Nx2]): keypoints for current view
        
    #     Returns:
    #         a dictionary containing
    #             - **pose** (SE3): relative pose from current to reference view
    #             - **best_inliers** (array, [N]): boolean inlier mask
    #     """
    #     principal_points = (self.cam_intrinsics.cx, self.cam_intrinsics.cy)

    #     # validity check
    #     valid_cfg = self.cfg.e_tracker.validity
    #     valid_case = True

    #     # initialize ransac setup
    #     R = np.eye(3)
    #     t = np.zeros((3,1))
    #     best_Rt = [R, t]
    #     max_ransac_iter = self.cfg.e_tracker.ransac.repeat

    #     if valid_cfg.method == "flow":
    #         # check flow magnitude
    #         avg_flow = np.mean(np.linalg.norm(kp_ref-kp_cur, axis=1))
    #         valid_case = avg_flow > valid_cfg.thre
        
    #     elif valid_cfg.method == "homo_ratio":
    #         # Find homography
    #         H, H_inliers = cv2.findHomography(
    #                     kp_cur,
    #                     kp_ref,
    #                     method=cv2.RANSAC,
    #                     confidence=0.99,
    #                     ransacReprojThreshold=0.2,
    #                     )

    #     elif valid_cfg.method == "GRIC":
    #         self.timers.start('GRIC-H', 'E-tracker')
    #         self.timers.start('find H', 'E-tracker')
    #         H, H_inliers = cv2.findHomography(
    #                     kp_cur,
    #                     kp_ref,
    #                     method=cv2.RANSAC,
    #                     # method=cv2.LMEDS,
    #                     confidence=0.99,
    #                     ransacReprojThreshold=1,
    #                     )
    #         self.timers.end('find H')

    #         H_res = compute_homography_residual(H, kp_cur, kp_ref)
    #         H_gric = calc_GRIC(
    #                     res=H_res,
    #                     sigma=0.8,
    #                     n=kp_cur.shape[0],
    #                     model="HMat"
    #         )
    #         self.timers.end('GRIC-H')

    #     if valid_case:
    #         inputs_mp = []
    #         outputs_mp = []
    #         for i in range(max_ransac_iter):
    #             # shuffle kp_cur and kp_ref
    #             new_list = np.arange(0, kp_cur.shape[0], 1)
    #             np.random.shuffle(new_list)
    #             new_kp_cur = kp_cur.copy()[new_list]
    #             new_kp_ref = kp_ref.copy()[new_list]

    #             inputs = {}
    #             inputs['kp_cur'] = new_kp_cur
    #             inputs['kp_ref'] = new_kp_ref
    #             inputs['H_inliers'] = H_inliers
    #             inputs['cfg'] = self.cfg
    #             inputs['cam_intrinsics'] = self.cam_intrinsics
    #             if valid_cfg.method == "GRIC":
    #                 inputs['H_gric'] = H_gric
    #             inputs_mp.append(inputs)
    #         outputs_mp = self.p.map(find_Ess_mat, inputs_mp)

    #         # Gather result
    #         num_valid_case = 0
    #         best_inlier_cnt = 0
    #         best_inliers = np.ones((kp_ref.shape[0])) == 1
    #         for outputs in outputs_mp:
    #             num_valid_case += outputs['valid_case']
    #             if outputs['inlier_cnt'] > best_inlier_cnt:
    #                 best_E = outputs['E']
    #                 best_inlier_cnt = outputs['inlier_cnt']
    #                 best_inliers = outputs['inlier']

    #         # Recover pose
    #         major_valid = num_valid_case > (max_ransac_iter/2)
    #         if major_valid:
    #             cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, new_kp_cur, new_kp_ref,
    #                                     focal=self.cam_intrinsics.fx,
    #                                     pp=principal_points,)

    #             # cheirality_check
    #             if cheirality_cnt > kp_cur.shape[0]*0.1:
    #                 best_Rt = [R, t]

    #     R, t = best_Rt
    #     pose = SE3()
    #     pose.R = R
    #     pose.t = t

    #     outputs = {"pose": pose, "inliers": best_inliers[:, 0]==1}
    #     return outputs

    def compute_rigid_flow_kp(self, cur_data, ref_data, pose):
        """compute keypoints from optical-rigid flow consistency

        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            pose (SE3): SE3 pose
        """
        rigid_pose = copy.deepcopy(pose)
        ref_data['rigid_flow_pose'] = SE3(rigid_pose.inv_pose)
         # kp selection
        kp_sel_outputs = self.kp_selection_good_depth(cur_data, ref_data, 
                                self.cfg.e_tracker.iterative_kp.score_method
                                )
        ref_data['kp_depth'] = kp_sel_outputs['kp1_depth'][0]
        cur_data['kp_depth'] = kp_sel_outputs['kp2_depth'][0]
        ref_data['kp_depth_uniform'] = kp_sel_outputs['kp1_depth_uniform'][0]
        cur_data['kp_depth_uniform'] = kp_sel_outputs['kp2_depth_uniform'][0]
        cur_data['rigid_flow_mask'] = kp_sel_outputs['rigid_flow_mask']

    def scale_recovery(self, cur_data, ref_data, E_pose, is_iterative):
        """recover depth scale

        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            E_pose (SE3): SE3 pose
            is_iterative (bool): is iterative stage
        
        Returns:
            a dictionary containing
                - **scale** (float): estimated scaling factor
                - **cur_kp_depth** (array, [Nx2]): keypoints at current view
                - **ref_kp_depth** (array, [Nx2]): keypoints at referenceview
                - **rigid_flow_mask** (array, [HxW]): rigid flow mask

        """
        outputs = {}

        if self.cfg.scale_recovery.method == "simple":
            scale = self.scale_recovery_simple(cur_data, ref_data, E_pose, is_iterative)
        
        elif self.cfg.scale_recovery.method == "iterative":
            iter_outputs = self.scale_recovery_iterative(cur_data, ref_data, E_pose)
            scale = iter_outputs['scale']
            outputs['cur_kp_depth'] = iter_outputs['cur_kp']
            outputs['ref_kp_depth'] = iter_outputs['ref_kp']
            outputs['rigid_flow_mask'] = iter_outputs['rigid_flow_mask']
        else:
            assert False, "Wrong scale recovery method [{}] used.".format(self.cfg.scale_recovery.method)
        
        outputs['scale'] = scale
        return outputs

    def scale_recovery_simple(self, cur_data, ref_data, E_pose, is_iterative):
        """recover depth scale by comparing triangulated depths and CNN depths
        
        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            E_pose (SE3): SE3 pose
        
        Returns:
            a dictionary containing
                - **scale** (float): estimated scaling factor
                - **cur_kp_depth** (array, [Nx2]): keypoints at current view
                - **ref_kp_depth** (array, [Nx2]): keypoints at referenceview
                - **rigid_flow_mask** (array, [HxW]): rigid flow mask

        Returns:
            scale (float)
        """
        if is_iterative:
            cur_kp = cur_data[self.cfg.scale_recovery.iterative_kp.kp_src]
            ref_kp = ref_data[self.cfg.scale_recovery.iterative_kp.kp_src]
        else:
            cur_kp = cur_data[self.cfg.scale_recovery.kp_src]
            ref_kp = ref_data[self.cfg.scale_recovery.kp_src]

        scale = self.find_scale_from_depth(
            ref_kp,
            cur_kp,
            E_pose.inv_pose, 
            cur_data['depth']
        )
        return scale

    def scale_recovery_iterative(self, cur_data, ref_data, E_pose):
        """recover depth scale by comparing triangulated depths and CNN depths
        Iterative scale recovery is applied
        
        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            E_pose (SE3): SE3 pose
        
        Returns:
            a dictionary containing
                - **scale** (float): estimated scaling factor
                - **cur_kp** (array, [Nx2]): keypoints at current view
                - **ref_kp** (array, [Nx2]): keypoints at referenceview
                - **rigid_flow_mask** (array, [HxW]): rigid flow mask
        """
        outputs = {}

        # Initialization
        scale = self.prev_scale
        delta = 0.001

        for _ in range(5):    
            rigid_flow_pose = copy.deepcopy(E_pose)
            rigid_flow_pose.t *= scale

            ref_data['rigid_flow_pose'] = SE3(rigid_flow_pose.inv_pose)

            # kp selection
            kp_sel_outputs = self.kp_selection_good_depth(cur_data, ref_data, 
                                    self.cfg.scale_recovery.iterative_kp.score_method
                                    )
            ref_data['kp_depth'] = kp_sel_outputs['kp1_depth_uniform'][0]
            cur_data['kp_depth'] = kp_sel_outputs['kp2_depth_uniform'][0]
            
            cur_data['rigid_flow_mask'] = kp_sel_outputs['rigid_flow_mask']

            # translation scale from triangulation v.s. CNN-depth
            cur_kp = cur_data[self.cfg.scale_recovery.kp_src]
            ref_kp = ref_data[self.cfg.scale_recovery.kp_src]

            new_scale = self.find_scale_from_depth(
                ref_kp,
                cur_kp,
                E_pose.inv_pose, 
                cur_data['depth']
            )

            delta_scale = np.abs(new_scale-scale)
            scale = new_scale
            self.prev_scale = new_scale

            # Get outputs
            outputs['scale'] = scale
            outputs['cur_kp'] = cur_data['kp_depth']
            outputs['ref_kp'] = ref_data['kp_depth']
            outputs['rigid_flow_mask'] = cur_data['rigid_flow_mask']
            
            if delta_scale < delta:
                return outputs
        return outputs

    def find_scale_from_depth(self, kp1, kp2, T_21, depth2):
        """Compute VO scaling factor for T_21

        Args:
            kp1 (array, [Nx2]): reference kp
            kp2 (array, [Nx2]): current kp
            T_21 (array, [4x4]): relative pose; from view 1 to view 2
            depth2 (array, [HxW]): depth 2
        
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

        self.timers.start('triangulation', 'scale_recovery')
        _, _, X2_tri = triangulation(kp1_norm, kp2_norm, np.eye(4), T_21)

        # Triangulation outlier removal
        depth2_tri = convert_sparse3D_to_depth(kp2, X2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0
        self.timers.end('triangulation')

        # common mask filtering
        non_zero_mask_pred2 = (depth2 > 0)
        non_zero_mask_tri2 = (depth2_tri > 0)
        valid_mask2 = non_zero_mask_pred2 * non_zero_mask_tri2

        depth_pred_non_zero = np.concatenate([depth2[valid_mask2]])
        depth_tri_non_zero = np.concatenate([depth2_tri[valid_mask2]])
        depth_ratio = depth_tri_non_zero / depth_pred_non_zero
        
        # Estimate scale (ransac)
        if valid_mask2.sum() > 10:
            # RANSAC scaling solver
            self.timers.start('scale ransac', 'scale_recovery')
            ransac = linear_model.RANSACRegressor(
                        base_estimator=linear_model.LinearRegression(
                            fit_intercept=False),
                        min_samples=self.cfg.scale_recovery.ransac.min_samples,
                        max_trials=self.cfg.scale_recovery.ransac.max_trials,
                        stop_probability=self.cfg.scale_recovery.ransac.stop_prob,
                        residual_threshold=self.cfg.scale_recovery.ransac.thre
                        )
            if self.cfg.scale_recovery.ransac.method == "depth_ratio":
                ransac.fit(
                    depth_ratio.reshape(-1, 1),
                    np.ones((depth_ratio.shape[0],1))
                    )
            elif self.cfg.scale_recovery.ransac.method == "abs_diff":
                ransac.fit(
                    depth_tri_non_zero.reshape(-1, 1),
                    depth_pred_non_zero.reshape(-1, 1),
                )
            scale = ransac.estimator_.coef_[0, 0]

            self.timers.end('scale ransac')

        else:
            scale = -1
       
        return scale

    def kp_selection_good_depth(self, cur_data, ref_data, rigid_kp_score_method):
        """Choose valid kp from a series of operations

        Args:
            cur_data (dict): current data
            ref_data (dict): reference data
            rigid_kp_score_method (str): [opt_flow, rigid_flow]
        
        Returns:
            a dictionary containing
                
                - **kp1_depth** (array, [Nx2]): keypoints in view-1
                - **kp2_depth** (array, [Nx2]): keypoints in view-2
                - **rigid_flow_mask** (array, [HxW]): rigid-optical flow consistency 
        """
        outputs = {}

        # initialization
        h, w = cur_data['depth'].shape

        kp1 = image_grid(h, w)
        kp1 = np.expand_dims(kp1, 0)
        tmp_flow_data = np.transpose(np.expand_dims(ref_data['flow'], 0), (0, 2, 3, 1))
        kp2 = kp1 + tmp_flow_data

        """ opt-rigid flow consistent kp selection """
        if self.cfg.kp_selection.rigid_flow_kp.enable:
            # compute rigid flow
            rigid_flow_pose = ref_data['rigid_flow_pose'].pose

            # Compute rigid flow
            pose_tensor = torch.from_numpy(rigid_flow_pose).float().unsqueeze(0).cuda()
            depth = torch.from_numpy(ref_data['raw_depth']).float().unsqueeze(0).unsqueeze(0).cuda()
            rigid_flow_tensor = self.rigid_flow_layer(
                                depth,
                                pose_tensor,
                                self.K,
                                self.inv_K,
                                normalized=False,
            )
            rigid_flow = rigid_flow_tensor.detach().cpu().numpy()[0]

            # compute optical-rigid flow difference
            rigid_flow_diff = np.linalg.norm(
                                rigid_flow - ref_data['flow'],
                                axis=0)
            ref_data['rigid_flow_diff'] = np.expand_dims(rigid_flow_diff, 2)

            # get depth-flow consistent kp
            outputs.update(
                    opt_rigid_flow_kp(
                        kp1=kp1,
                        kp2=kp2,
                        ref_data=ref_data,
                        cfg=self.cfg,
                        outputs=outputs,
                        score_method=rigid_kp_score_method
                        )
                )

        return outputs
