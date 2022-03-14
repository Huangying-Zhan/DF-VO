''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: DF-VO core program
'''

import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm

from libs.geometry.camera_modules import SE3
import libs.datasets as Dataset
from libs.deep_models.deep_models import DeepModel
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timer
from libs.matching.keypoint_sampler import KeypointSampler
from libs.matching.depth_consistency import DepthConsistency
from libs.tracker import EssTracker, PnpTracker
from libs.general.utils import *



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

        # reference data and current data
        self.initialize_data()

        self.setup()

    def setup(self):
        """Reading configuration and setup, including

            - Timer
            - Dataset
            - Tracking method
            - Keypoint Sampler
            - Deep networks
            - Deep layers
            - Visualizer
        """
        # timer
        self.timers = Timer()

        # intialize dataset
        self.dataset = Dataset.datasets[self.cfg.dataset](self.cfg)
        
        # get tracking method
        self.tracking_method = self.cfg.tracking_method
        self.initialize_tracker()

        # initialize keypoint sampler
        self.kp_sampler = KeypointSampler(self.cfg)
        
        # Deep networks
        self.deep_models = DeepModel(self.cfg)
        self.deep_models.initialize_models()
        if self.cfg.online_finetune.enable:
            self.deep_models.setup_train()
        
        # Depth consistency
        if self.cfg.kp_selection.depth_consistency.enable:
            self.depth_consistency_computer = DepthConsistency(self.cfg, self.dataset.cam_intrinsics)

        # visualization interface
        self.drawer = FrameDrawer(self.cfg.visualization)
        
    def initialize_data(self):
        """initialize data of current view and reference view
        """
        self.ref_data = {}
        self.cur_data = {}

    def initialize_tracker(self):
        """Initialize tracker
        """
        if self.tracking_method == 'hybrid':
            self.e_tracker = EssTracker(self.cfg, self.dataset.cam_intrinsics, self.timers)
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'PnP':
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'deep_pose':
            return
        else:
            assert False, "Wrong tracker is selected, choose from [hybrid, PnP, deep_pose]"

    def update_global_pose(self, new_pose, scale=1.):
        """update estimated poses w.r.t global coordinate system

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        self.cur_data['pose'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])

    def tracking(self):
        """Tracking using both Essential matrix and PnP
        Essential matrix for rotation and translation direction;
            *** triangluate depth v.s. CNN-depth for translation scale ***
        PnP if Essential matrix fails
        """
        # First frame
        if self.tracking_stage == 0:
            # initial pose
            if self.cfg.directory.gt_pose_dir is not None:
                self.cur_data['pose'] = SE3(self.dataset.gt_poses[self.cur_data['id']])
            else:
                self.cur_data['pose'] = SE3()
            self.ref_data['motion'] = SE3()
            return

        # Second to last frames
        elif self.tracking_stage >= 1:
            ''' keypoint selection '''
            if self.tracking_method in ['hybrid', 'PnP']:
                # Depth consistency (CNN depths + CNN pose)
                if self.cfg.kp_selection.depth_consistency.enable:
                    self.depth_consistency_computer.compute(self.cur_data, self.ref_data)

                # kp_selection
                self.timers.start('kp_sel', 'tracking')
                kp_sel_outputs = self.kp_sampler.kp_selection(self.cur_data, self.ref_data)
                if kp_sel_outputs['good_kp_found']:
                    self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)
                self.timers.end('kp_sel')

            ''' Pose estimation '''
            # Initialize hybrid pose
            hybrid_pose = SE3()
            E_pose = SE3()

            if not(kp_sel_outputs['good_kp_found']):
                print("No enough good keypoints, constant motion will be used!")
                pose = self.ref_data['motion']
                self.update_global_pose(pose, 1)
                return 


            ''' E-tracker '''
            if self.tracking_method in ['hybrid']:
                # Essential matrix pose
                self.timers.start('E-tracker', 'tracking')
                e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.kp_src],
                                self.cur_data[self.cfg.e_tracker.kp_src],
                                not(self.cfg.e_tracker.iterative_kp.enable)) # pose: from cur->ref
                E_pose = e_tracker_outputs['pose']
                self.timers.end('E-tracker')

                # Rotation
                hybrid_pose.R = E_pose.R

                # save inliers
                self.ref_data['inliers'] = e_tracker_outputs['inliers']

                # scale recovery
                if np.linalg.norm(E_pose.t) != 0:
                    self.timers.start('scale_recovery', 'tracking')
                    scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, False)
                    scale = scale_out['scale']
                    if self.cfg.scale_recovery.kp_src == 'kp_depth':
                        self.cur_data['kp_depth'] = scale_out['cur_kp_depth']
                        self.ref_data['kp_depth'] = scale_out['ref_kp_depth']
                        self.cur_data['rigid_flow_mask'] = scale_out['rigid_flow_mask']
                    if scale != -1:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('scale_recovery')

                # Iterative keypoint refinement
                if np.linalg.norm(E_pose.t) != 0 and self.cfg.e_tracker.iterative_kp.enable:
                    self.timers.start('E-tracker iter.', 'tracking')
                    # Compute refined keypoint
                    self.e_tracker.compute_rigid_flow_kp(self.cur_data,
                                                         self.ref_data,
                                                         hybrid_pose)

                    e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                self.cur_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                True) # pose: from cur->ref
                    E_pose = e_tracker_outputs['pose']

                    # Rotation
                    hybrid_pose.R = E_pose.R

                    # save inliers
                    self.ref_data['inliers'] = e_tracker_outputs['inliers']

                    # scale recovery
                    if np.linalg.norm(E_pose.t) != 0 and self.cfg.scale_recovery.iterative_kp.enable:
                        scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, True)
                        scale = scale_out['scale']
                        if scale != -1:
                            hybrid_pose.t = E_pose.t * scale
                    else:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('E-tracker iter.')

            ''' PnP-tracker '''
            if self.tracking_method in ['PnP', 'hybrid']:
                # PnP if Essential matrix fail
                if np.linalg.norm(E_pose.t) == 0 or scale == -1:
                    self.timers.start('pnp', 'tracking')
                    pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.kp_src],
                                    self.ref_data['depth'],
                                    not(self.cfg.pnp_tracker.iterative_kp.enable)
                                    ) # pose: from cur->ref
                    
                    # Iterative keypoint refinement
                    if self.cfg.pnp_tracker.iterative_kp.enable:
                        self.pnp_tracker.compute_rigid_flow_kp(self.cur_data, self.ref_data, pnp_outputs['pose'])
                        pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                                    self.ref_data['depth'],
                                    True
                                    ) # pose: from cur->ref

                    self.timers.end('pnp')

                    # use PnP pose instead of E-pose
                    hybrid_pose = pnp_outputs['pose']
                    self.tracking_mode = "PnP"

            ''' Deep-tracker '''
            if self.tracking_method in ['deep_pose']:
                hybrid_pose = SE3(self.ref_data['deep_pose'])
                self.tracking_mode = "DeepPose"

            ''' Summarize data '''
            # update global poses
            self.ref_data['pose'] = copy.deepcopy(hybrid_pose)
            self.ref_data['motion'] = copy.deepcopy(hybrid_pose)
            pose = self.ref_data['pose']
            self.update_global_pose(pose, 1)

    def update_data(self, ref_data, cur_data):
        """Update data
        
        Args:
            ref_data (dict): reference data
            cur_data (dict): current data
        
        Returns:
            ref_data (dict): updated reference data
            cur_data (dict): updated current data
        """
        for key in cur_data:
            if key == "id":
                ref_data['id'] = cur_data['id']
            else:
                if ref_data.get(key, -1) is -1:
                    ref_data[key] = {}
                ref_data[key] = cur_data[key]
        
        # Delete unused flow to avoid data leakage
        ref_data['flow'] = None
        cur_data['flow'] = None
        ref_data['flow_diff'] = None
        return ref_data, cur_data

    def load_raw_data(self):
        """load image data and (optional) GT/precomputed depth data
        """
        # Reading image
        self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

        # Reading/Predicting depth
        if self.dataset.data_dir['depth_src'] is not None:
            self.cur_data['raw_depth'] = self.dataset.get_depth(self.cur_data['timestamp'])
    
    def deep_model_inference(self):
        """deep model prediction
        """
        if self.tracking_method in ['hybrid', 'PnP']:
            # Single-view Depth prediction
            if self.dataset.data_dir['depth_src'] is None:
                self.timers.start('depth_cnn', 'deep inference')
                if self.tracking_stage > 0 and \
                    self.cfg.online_finetune.enable and self.cfg.online_finetune.depth.enable:
                        img_list = [self.cur_data['img'], self.ref_data['img']]
                else:
                    img_list = [self.cur_data['img']]

                self.cur_data['raw_depth'] = \
                    self.deep_models.forward_depth(imgs=img_list)
                self.cur_data['raw_depth'] = cv2.resize(self.cur_data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
                self.timers.end('depth_cnn')
            self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            # Two-view flow
            if self.tracking_stage >= 1:
                self.timers.start('flow_cnn', 'deep inference')
                flows = self.deep_models.forward_flow(
                                        self.cur_data,
                                        self.ref_data,
                                        forward_backward=self.cfg.deep_flow.forward_backward)
                
                # Store flow
                self.ref_data['flow'] = flows[(self.ref_data['id'], self.cur_data['id'])].copy()
                if self.cfg.deep_flow.forward_backward:
                    self.cur_data['flow'] = flows[(self.cur_data['id'], self.ref_data['id'])].copy()
                    self.ref_data['flow_diff'] = flows[(self.ref_data['id'], self.cur_data['id'], "diff")].copy()
                
                self.timers.end('flow_cnn')
            
        # Relative camera pose
        if self.tracking_stage >= 1 and self.cfg.deep_pose.enable:
            self.timers.start('pose_cnn', 'deep inference')
            # Deep pose prediction
            pose = self.deep_models.forward_pose(
                        [self.ref_data['img'], self.cur_data['img']] 
                        )
            self.ref_data['deep_pose'] = pose # from cur->ref
            self.timers.end('pose_cnn')

    def main(self):
        """Main program
        """
        print("==> Start DF-VO")
        print("==> Running sequence: {}".format(self.cfg.seq))

        if self.cfg.no_confirm:
            start_frame = 0
        else:
            start_frame = int(input("Start with frame: "))

        for img_id in tqdm(range(start_frame, len(self.dataset), self.cfg.frame_step)):
            self.timers.start('DF-VO')
            self.tracking_mode = "Ess. Mat."

            """ Data reading """
            # Initialize ids and timestamps
            self.cur_data['id'] = img_id
            self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)

            # Read image data and (optional) precomputed depth data
            self.timers.start('data_loading')
            self.load_raw_data()
            self.timers.end('data_loading')

            # Deep model inferences
            self.timers.start('deep_inference')
            self.deep_model_inference()
            self.timers.end('deep_inference')

            """ Visual odometry """
            self.timers.start('tracking')
            self.tracking()
            self.timers.end('tracking')

            """ Online Finetuning """
            if self.tracking_stage >= 1 and self.cfg.online_finetune.enable:
                self.deep_models.finetune(self.ref_data['img'], self.cur_data['img'],
                                      self.ref_data['pose'].pose,
                                      self.dataset.cam_intrinsics.mat,
                                      self.dataset.cam_intrinsics.inv_mat)

            """ Visualization """
            if self.cfg.visualization.enable:
                self.timers.start('visualization')
                self.drawer.main(self)
                self.timers.end('visualization')

            """ Update reference and current data """
            self.ref_data, self.cur_data = self.update_data(
                                    self.ref_data,
                                    self.cur_data,
            )

            self.tracking_stage += 1

            self.timers.end('DF-VO')

        print("=> Finish!")



        """ Display & Save result """
        print("The result is saved in [{}].".format(self.cfg.directory.result_dir))
        # Save trajectory map
        print("Save VO map.")
        map_png = "{}/map.png".format(self.cfg.directory.result_dir)
        cv2.imwrite(map_png, self.drawer.data['traj'])

        # Save trajectory txt
        traj_txt = "{}/{}.txt".format(self.cfg.directory.result_dir, self.cfg.seq)
        self.dataset.save_result_traj(traj_txt, self.global_poses)

        # save finetuned model
        if self.cfg.online_finetune.enable and self.cfg.online_finetune.save_model:
            self.deep_models.save_model()

        # Output experiement information
        self.timers.time_analysis()
