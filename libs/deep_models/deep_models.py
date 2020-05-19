''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-20
@LastEditors: Huangying Zhan
@Description: DeepModel initializes different deep networks and provide forward interfaces.
'''

import numpy as np

from .flow.lite_flow_net.lite_flow import LiteFlow
from .depth.monodepth2.monodepth2 import Monodepth2DepthNet
# FIXME: update correct path
# from libs.deep_pose.monodepth2 import Monodepth2PoseNet

class DeepModel():
    """DeepModel initializes different deep networks and provide forward interfaces.

    TODO:
        add forward_depth()
        
        add forward_pose()

    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration dictionary
        """
        self.cfg = cfg
        
    def initialize_models(self):
        """intialize multiple deep models
        """

        ''' optical flow '''
        self.flow = self.initialize_deep_flow_model()

        # allow reading precomputed flow instead of network inference for speeding up debug
        if self.cfg.deep_flow.precomputed_flow is not None:
            self.cfg.deep_flow.precomputed_flow = self.cfg.deep_flow.precomputed_flow.replace("{}", self.cfg.seq)

        ''' single-view depth '''
        if self.cfg.depth.depth_src is None:
            if self.cfg.depth.pretrained_model is not None:
                self.depth = self.initialize_deep_depth_model()
            else:
                assert False, "No precomputed depths nor pretrained depth model"
        
        ''' two-view pose '''
        if self.cfg.pose_net.enable:
            if self.cfg.pose_net.pretrained_model is not None:
                self.pose = self.initialize_deep_pose_model()
            else:
                assert False, "No pretrained pose model"

    def initialize_deep_flow_model(self):
        """Initialize optical flow network
        
        Returns:
            flow_net (network): optical flow network
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
            depth_net (network): single-view depth network
        """
        depth_net = Monodepth2DepthNet()
        depth_net.initialize_network_model(
                weight_path=self.cfg.depth.pretrained_model,
                dataset=self.cfg.dataset)
        return depth_net
    
    def initialize_deep_pose_model(self):
        """Initialize two-view pose model

        Returns:
            pose_net (network): two-view pose network
        """
        pose_net = Monodepth2PoseNet()
        pose_net.initialize_network_model(
            weight_path=self.cfg.pose_net.pretrained_model,
            height=self.cfg.image.height,
            width=self.cfg.image.width,
            dataset=self.cfg.dataset
            )
        return pose_net

    def forward_flow(self, in_cur_data, in_ref_data, forward_backward):
        """Optical flow network forward interface, a forward inference.

        Args:
            in_cur_data (dict): current data
            in_ref_data (dict): reference data
            forward_backward (bool): use forward-backward consistency if True
        
        Returns:
            flows (dict): predicted flow data. flows[(id1, id2)] is flows from id1 to id2.
        """
        # Preprocess image
        ref_imgs = []
        cur_imgs = []
        cur_img = np.transpose((in_cur_data['img'])/255, (2, 0, 1))
        for ref_id in in_ref_data['id']:
            ref_img = np.transpose((in_ref_data['img'][ref_id])/255, (2, 0, 1))
            ref_imgs.append(ref_img)
            cur_imgs.append(cur_img)
        ref_imgs = np.asarray(ref_imgs)
        cur_imgs = np.asarray(cur_imgs)

        # Forward pass
        flows = {}
        batch_size = self.cfg.deep_flow.batch_size
        num_forward = int(np.ceil(len(in_ref_data['id']) / batch_size))
        for i in range(num_forward):
            # Flow inference
            batch_flows = self.flow.inference_flow(
                                    img1=ref_imgs[i*batch_size: (i+1)*batch_size],
                                    img2=cur_imgs[i*batch_size: (i+1)*batch_size],
                                    flow_dir=self.cfg.deep_flow.precomputed_flow,
                                    forward_backward=forward_backward,
                                    dataset=self.cfg.dataset)
            
            # Save flows at current view
            for j in range(batch_size):
                src_id = in_ref_data['id'][i*batch_size: (i+1)*batch_size][j]
                tgt_id = in_cur_data['id']
                flows[(src_id, tgt_id)] = batch_flows['forward'][j].copy()
                if forward_backward:
                    flows[(tgt_id, src_id)] = batch_flows['backward'][j].copy()
                    flows[(src_id, tgt_id, "diff")] = batch_flows['flow_diff'][j].copy()
        return flows

    def forward_depth(self):
        """Not implemented
        """
        raise NotImplementedError

    def forward_pose(self):
        """Not implemented
        """
        raise NotImplementedError
