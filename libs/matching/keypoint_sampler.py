# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.


# import math
import numpy as np

from libs.utils import image_grid
from .kp_selection import *

class KeypointSampler():
    def __init__(self, cfg):
        self.cfg = cfg
        self.kps = {}

        # feature tracking method
        self.feature_tracking_method = self.get_feat_track_methods(
                                            self.cfg.kp_selection.feature_tracking_method
                                            )

        # generate uniform kp list
        if self.cfg.kp_selection.sampled_kp.enable:
            self.kps['uniform'] = self.generate_kp_samples(
                                        img_h=self.cfg.image.height,
                                        img_w=self.cfg.image.width,
                                        crop=self.cfg.crop.flow_crop,
                                        N=self.cfg.deep_flow.num_kp
                                        )

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
        return feat_track_methods[method_idx]

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
            outputs.update(
                sampled_kp(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    kp_list=self.kps['uniform'],
                    cfg=self.cfg,
                    outputs=outputs
                    )
        )
        return outputs

    def update_kp_data(self, cur_data, ref_data, kp_sel_outputs):
        """update cur_data and ref_data with the kp_selection output
        """
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
