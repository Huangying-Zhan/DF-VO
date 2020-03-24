# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.

import cv2
import matplotlib as mpl
import numpy as np
import os
from time import time

from ..flowlib.flowlib import flow_to_image
from .utils import mkdir_if_not_exists


def draw_match_temporal(img1, kp1, img2, kp2, N):
    """Draw matches temporally
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    r1, g1, b1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
    r2, g2, b2 = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    out_img = img2.copy()

    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                                            dtype=np.int
                                            )
    for i in kp_list:
        center1 = (kp1[i][0].astype(np.int), kp1[i][1].astype(np.int))
        center2 = (kp2[i][0].astype(np.int), kp2[i][1].astype(np.int))

        color = np.random.randint(0, 255, 3)
        color = tuple([int(i) for i in color])

        cv2.line(out_img, center1, center2, color, 2)
    return out_img


def draw_match_side(img1, kp1, img2, kp2, N, inliers):
    """Draw matches on 2 sides
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
        inliers (Nx1 array): boolean mask for inlier (not used)
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                            dtype=np.int
                            )
    
    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(cv_kp1))]

    # inlier/outlier plot option
    if inliers is not None:
        inlier_mask = inliers[kp_list].ravel().tolist()
        inlier_draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = inlier_mask, # draw only inliers
                    flags = 2)
        
        outlier_mask = (inliers==0)[kp_list].ravel().tolist()
        outlier_draw_params = dict(matchColor = (255,0,0), # draw matches in red color
                    singlePointColor = None,
                    matchesMask = outlier_mask, # draw only inliers
                    flags = 2)
        out_img1 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, outImg=out_img, **inlier_draw_params)
        out_img2 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, outImg=out_img, **outlier_draw_params)
        out_img = cv2.addWeighted(out_img1, 0.5, out_img2, 0.5, 0)
    else:
        out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)
    return out_img


class FrameDrawer():
    """
    Attributes
        h (int): drawer image height
        w (int): drawer image width
        img (hxwx3): drawer image
        data (dict): linking between item and img
        display (dict): options to display items
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): visualization configuration
        """
        # intialize drawer size
        self.cfg = cfg
        self.h = cfg.window_h
        self.w = cfg.window_w
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # initialize data and data assignment
        self.data = {}
        self.display = {}
        self.initialize_drawer()
    
    def initialize_drawer(self):
        """Initialize drawer by assigning items to the drawer
        """
        visual_h = self.h
        visual_w = self.w

        self.assign_data(
                    item="traj",
                    top_left=[0, 0], 
                    bottom_right=[int(visual_h), int(visual_w)],
                    )

        self.assign_data(
                    item="match_temp",
                    top_left=[int(visual_h/4*0), int(visual_w/5*2)], 
                    bottom_right=[int(visual_h/4*1), int(visual_w/5*5)],
                    )
        
        self.assign_data(
                    item="match_side",
                    top_left=[int(visual_h/4*1), int(visual_w/5*2)], 
                    bottom_right=[int(visual_h/4*2), int(visual_w/5*5)],
                    )
        
        self.assign_data(
                    item="depth",
                    top_left=[int(visual_h/4*2), int(visual_w/5*2)], 
                    bottom_right=[int(visual_h/4*3), int(visual_w/5*3)],
                    )
        
        self.assign_data(
                    item="flow1",
                    top_left=[int(visual_h/4*2), int(visual_w/5*3)], 
                    bottom_right=[int(visual_h/4*3), int(visual_w/5*4)],
                    )
        
        self.assign_data(
                    item="flow2",
                    top_left=[int(visual_h/4*2), int(visual_w/5*4)], 
                    bottom_right=[int(visual_h/4*3), int(visual_w/5*5)],
                    )
        
        self.assign_data(
                    item="depth_mask",
                    top_left=[int(visual_h/4*3), int(visual_w/5*2)], 
                    bottom_right=[int(visual_h/4*4), int(visual_w/5*3)],
                    )
        
        self.assign_data(
                    item="rigid_flow_mask",
                    top_left=[int(visual_h/4*3), int(visual_w/5*2)], 
                    bottom_right=[int(visual_h/4*4), int(visual_w/5*3)],
                    )

        self.assign_data(
                    item="flow_mask",
                    top_left=[int(visual_h/4*3), int(visual_w/5*3)], 
                    bottom_right=[int(visual_h/4*4), int(visual_w/5*4)],
                    )

        self.assign_data(
                    item="valid_mask",
                    top_left=[int(visual_h/4*3), int(visual_w/5*4)], 
                    bottom_right=[int(visual_h/4*4), int(visual_w/5*5)],
                    )

    def assign_data(self, item, top_left, bottom_right):
        """assign data to the drawer image
        Args:
            top_left (list): [y, x] position of top left corner
            bottom_right (list): [y, x] position of bottom right corner
            item (str): item name
        """
        self.data[item] = self.img[
                                    top_left[0]:bottom_right[0],
                                    top_left[1]:bottom_right[1]
                                    ]
        self.display[item] = True

    def update_data(self, item, data):
        """update drawer content
        Args:
            item (str): item to be updated
            data (HxWx3 array): content to be updated, RGB format
        """
        data_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        vis_h, vis_w, _ = self.data[item].shape
        self.data[item][...] = cv2.resize(data_bgr, (vis_w, vis_h))

    def update_display(self, item):
        """update display option by inversing the current setup
        Args:
            item (str): item to be updated
        """
        self.display[item] = not(self.display[item])

    def get_traj_init_xy(self, vis_h, vis_w, gt_poses):
        """Get [x,y] of initial pose
        Args:
            vis_h (int): visualization image height
            vis_w (int): visualization image width
            gt_poses (dict): ground truth poses
        Returns:
            [x_off, y_off] (int): x,y offset of initial pose
        """
        if len(gt_poses) != 1:
            # Get max and min X,Z; [x,y] of
            gt_Xs = []
            gt_Zs = []
            for cnt, i in enumerate(gt_poses):
                trueX, trueY, trueZ = gt_poses[i][:3, 3]
                gt_Xs.append(trueX)
                gt_Zs.append(trueZ)
                if cnt == 0:
                    x0 = trueX
                    z0 = trueZ
            min_x, max_x = np.min(gt_Xs), np.max(gt_Xs)
            min_z, max_z = np.min(gt_Zs), np.max(gt_Zs)

            # Get ratio
            ratio_x = (x0 - min_x)/(max_x-min_x)
            ratio_z = (z0 - min_z)/(max_z-min_z)

            # Get offset (only using [0.2:0.8])
            crop = [0.2, 0.8]
            x_off = int(vis_w * (crop[1]-crop[0]) * ratio_x + vis_w * crop[0])
            y_off = int(vis_h * crop[1] - vis_h * (crop[1]-crop[0]) * (ratio_z))
            self.traj_x0, self.traj_y0 = x_off, y_off
        else:
            self.traj_x0, self.traj_y0 = int(vis_w * 0.5), int(vis_h * 0.5)
        # return x_off, y_off

    def interface(self):
        key = cv2.waitKey(10) or 0xff

        # pause
        if key == ord('p'):
            while True:
                key2 = cv2.waitKey(1) or 0xff

                # Match_temp
                if key2 == ord('1'):
                    self.display['match_temp'] = not(self.display['match_temp'])
                    print("Match(1): {}".format(self.display['match_temp']))

                # Match side
                if key2 == ord('2'):
                    self.display['match_side'] = not(self.display['match_side'])
                    print("Match(2): {}".format(self.display['match_side']))

                # depth
                if key2 == ord('3'):
                    self.display['depth'] = not(self.display['depth'])
                    print("depth: {}".format(self.display['depth']))

                if key2 == ord('4'):
                    self.display['flow1'] = not(self.display['flow1'])
                    self.display['flow2'] = not(self.display['flow2'])
                    self.display['flow_diff'] = not(self.display['flow_diff'])
                    print("flow: {}".format(self.display['flow1']))

                # Continue
                if key2 == ord('c'):
                    return

        # Match_temp
        if key == ord('1'):
            self.display['match_temp'] = not(self.display['match_temp'])
            print("Match(1): {}".format(self.display['match_temp']))

        # Match side
        if key == ord('2'):
            self.display['match_side'] = not(self.display['match_side'])
            print("Match(2): {}".format(self.display['match_side']))

        # depth
        if key == ord('3'):
            self.display['depth'] = not(self.display['depth'])
            print("depth: {}".format(self.display['depth']))

        # flow
        if key == ord('4'):
            self.display['flow1'] = not(self.display['flow1'])
            self.display['flow2'] = not(self.display['flow2'])
            self.display['flow_diff'] = not(self.display['flow_diff'])
            print("flow: {}".format(self.display['flow1']))

    def draw_traj(self, pred_poses, gt_poses, traj_cfg, tracking_mode):
        """draw trajectory and related information
        Args:
            pred_poses (dict): predicted poses w.r.t world coordinate system
        """
        traj = self.data["traj"]
        latest_id = max(pred_poses.keys())

        # draw scales
        draw_scale = traj_cfg.draw_scale
        mono_scale = traj_cfg.mono_scale
        pred_draw_scale = draw_scale * mono_scale

        # Draw GT trajectory
        if traj_cfg.vis_gt_traj:
            trueX, trueY, trueZ = gt_poses[latest_id][:3, 3]
            true_x = int(trueX*draw_scale) + self.traj_x0
            true_y = -(int(trueZ*draw_scale)) + self.traj_y0
            cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 1)
        
        # Draw prediction trajectory
        cur_t = pred_poses[latest_id].t[:, 0]
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        draw_x = int(x*pred_draw_scale) + self.traj_x0
        draw_y = -(int(z*pred_draw_scale)) + self.traj_y0
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
        
        # Draw coordinate information
        cv2.rectangle(traj, (10, 20), (int(self.w/2), 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj, text, (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        
        # Draw tracking mode
        cv2.rectangle(traj, (0, self.h-50), (350, self.h), (0, 0, 0), -1)
        text = "Tracking mode: {}".format(tracking_mode)
        cv2.putText(traj, text, (20, self.h-20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # Draw interface text
        text = "p/c: pause/continue;  1: flow; 2: match; 3: depth; 4: flow vis."
        cv2.putText(traj, text, (20, self.h-40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    def main(self, vo):
        start_time = time()
        # Trajectory visualization
        if vo.cfg.visualization.trajectory.vis_traj:
            vo.drawer.draw_traj(
                    pred_poses=vo.global_poses,
                    gt_poses=vo.gt_poses,
                    traj_cfg=vo.cfg.visualization.trajectory,
                    tracking_mode=vo.tracking_mode
                    )

        # Draw correspondence
        tmp_start_time = time()
        if vo.tracking_stage > 1:
            ref_id = vo.ref_data['id'][-1]

            # Set number of kp to be visualized
            if (vo.cur_data[vo.cfg.visualization.kp_src].shape[0] < vo.cfg.visualization.match.kp_num or \
                    vo.ref_data[vo.cfg.visualization.kp_src][ref_id].shape[0] < vo.cfg.visualization.match.kp_num or\
                        vo.cfg.visualization.match.kp_num == -1):
                vis_kp_num = min(vo.cur_data[vo.cfg.visualization.kp_src].shape[0], vo.ref_data[vo.cfg.visualization.kp_src][ref_id].shape[0])
            else:
                vis_kp_num = vo.cfg.visualization.match.kp_num

            if vo.drawer.display['match_side'] and\
                vo.cfg.visualization.match.vis_side.enable:
                # Set keypoints
                vis_kp_ref = vo.ref_data[vo.cfg.visualization.kp_src][ref_id]
                vis_kp_cur = vo.cur_data[vo.cfg.visualization.kp_src]

                if vo.cfg.visualization.match.vis_side.inlier_plot:
                    inliers=vo.ref_data['inliers'][ref_id]
                else:
                    inliers=None
                vis_match_side = draw_match_side(
                    img2=vo.ref_data['img'][ref_id],
                    kp2=vis_kp_cur,
                    img1=vo.cur_data['img'],
                    kp1=vis_kp_ref,
                    N=vis_kp_num,
                    inliers=inliers
                    )
                vo.drawer.update_data("match_side", vis_match_side)
            else:
                h, w, c = vo.drawer.data["match_side"][...].shape
                vo.drawer.data["match_side"][...] = np.zeros((h,w,c))

            # Draw temporal flow
            if vo.drawer.display['match_temp'] and \
                vo.cfg.visualization.match.vis_temp.enable:
                
                # Set keypoints
                vis_kp_ref = vo.ref_data[vo.cfg.visualization.kp_src][ref_id]
                vis_kp_cur = vo.cur_data[vo.cfg.visualization.kp_src]
                
                vis_match_temp = draw_match_temporal(
                        img1=vo.ref_data['img'][ref_id],
                        kp1=vis_kp_ref,
                        img2=vo.cur_data['img'],
                        kp2=vis_kp_cur,
                        N=vis_kp_num
                        )
                vo.drawer.update_data("match_temp", vis_match_temp)
            else:
                h, w, c = vo.drawer.data["match_temp"][...].shape
                vo.drawer.data["match_temp"][...] = np.zeros((h,w,c))

        vo.timers.timers["visualization_match"].append(time()-tmp_start_time)

        # Visualize flow (forward; backward) and flow inconsistency
        tmp_start_time = time()
        if vo.drawer.display['flow1'] and vo.cfg.visualization.flow.vis_full_flow and vo.tracking_stage > 1:
            vis_flow = vo.ref_data['flow'][vo.ref_data['id'][0]].transpose(1,2,0)
            vis_flow = flow_to_image(vis_flow)
            vo.drawer.update_data("flow1", vis_flow)
        else:
            h, w, c = vo.drawer.data["flow1"][...].shape
            vo.drawer.data["flow1"][...] = np.zeros((h,w,c))

        if vo.drawer.display['flow2'] and vo.cfg.visualization.flow.vis_back_flow and vo.tracking_stage > 1:
            vis_flow = vo.cur_data['flow'][vo.ref_data['id'][0]].transpose(1,2,0)
            vis_flow = flow_to_image(vis_flow)
            vo.drawer.update_data("flow2", vis_flow)
        else:
            h, w, c = vo.drawer.data["flow2"][...].shape
            vo.drawer.data["flow2"][...] = np.zeros((h,w,c))

        vo.timers.timers["visualization_flow"].append(time()-tmp_start_time)

        # Visualize full depth
        tmp_start_time = time()
        if vo.drawer.display['depth'] and \
              (vo.cfg.visualization.depth.vis_full_depth or vo.cfg.visualization.depth.vis_full_disp):
            if vo.cfg.visualization.depth.use_tracking_depth:
                tmp_vis_depth = vo.cur_data['depth']
            else:
                tmp_vis_depth = vo.cur_data['raw_depth']
                    
            if vo.cfg.visualization.depth.vis_full_depth:
                vis_depth = tmp_vis_depth
                normalizer = mpl.colors.Normalize(vmin=0, vmax=vo.cfg.depth.max_depth)
                mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(vis_depth)[:, :, :3] * 255).astype(np.uint8)
                vo.drawer.update_data("depth", colormapped_im)
            
            # Visualize full inverse depth
            if vo.cfg.visualization.depth.vis_full_disp:
                vis_depth = 1/(tmp_vis_depth+1e-3)
                vis_depth[tmp_vis_depth==0] = 0
                if "kitti" in vo.cfg.dataset:
                    vmax = 0.3
                elif "tum" in vo.cfg.dataset:
                    vmax = 5
                normalizer = mpl.colors.Normalize(vmin=0, vmax=vmax)
                mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(vis_depth)[:, :, :3] * 255).astype(np.uint8)
                vo.drawer.update_data("depth", colormapped_im)
        else:
            h, w, c = vo.drawer.data["depth"][...].shape
            vo.drawer.data["depth"][...] = np.zeros((h,w,c))
        vo.timers.timers["visualization_depth"].append(time()-tmp_start_time)

        # visualize masks
        if vo.tracking_stage > 1 and vo.cfg.visualization.mask.vis_masks:
            tmp_start_time = time()

            if vo.cfg.kp_selection.flow_consistency.enable:
                # normalizer = mpl.colors.Normalize(vmin=0, vmax=vo.cfg.kp_selection.flow_consistency.thre)
                normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
                mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='jet')
                mask = vo.cur_data['flow_mask']
                colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
                vo.drawer.update_data("flow_mask", colormapped_im)
                
                if vo.cfg.kp_selection.uniform_filtered_bestN.enable:
                    normalizer = mpl.colors.Normalize(vmin=0, vmax=vo.cfg.kp_selection.flow_consistency.thre)
                    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='gray_r')
                    mask = vo.cur_data['valid_mask']
                    colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
                    vo.drawer.update_data("valid_mask", colormapped_im)

            if vo.cfg.kp_selection.depth_consistency.enable:
                normalizer = mpl.colors.Normalize(vmin=0, vmax=vo.cfg.kp_selection.depth_consistency.thre)
                mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='jet')
                mask = vo.cur_data['depth_mask']
                colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
                vo.drawer.update_data("depth_mask", colormapped_im)
            
            if vo.cur_data.get('rigid_flow_mask', -1) is not -1:
                normalizer = mpl.colors.Normalize(vmin=0, vmax=vo.cfg.kp_selection.rigid_flow_consistency.thre)
                mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='jet')
                mask = vo.cur_data['rigid_flow_mask']
                colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
                vo.drawer.update_data("rigid_flow_mask", colormapped_im)

            vo.timers.timers["visualization_masks"].append(time()-tmp_start_time)

        # Save visualization result
        if vo.cfg.visualization.save_img:
            img_dir_path = os.path.join(
                vo.cfg.result_dir, "img_{}".format(vo.cfg.seq))
            mkdir_if_not_exists(img_dir_path)
            img_path = os.path.join(img_dir_path, "{:06d}.jpg".format(vo.cur_data['id']))
            cv2.imwrite(img_path, vo.drawer.img)
        
        cv2.imshow('DF-VO', vo.drawer.img)
        cv2.waitKey(1)

        vo.drawer.interface()

        vo.timers.timers["visualization"].append(time()-start_time)
        return vo