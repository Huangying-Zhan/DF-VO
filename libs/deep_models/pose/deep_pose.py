''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-02
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep pose network interface
'''


class DeepPose():
    """DeepPose is the Base class for deep pose network interface
    """
    def __init__(self, cfg=None):
        """
        Args:
            cfg (edict): deep pose configuration dictionary (only required when online finetuning)
        """
        self.pose_cfg = cfg

    def initialize_network_model(self, weight_path):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
        """
        raise NotImplementedError

    def inference(self, imgs):
        raise NotImplementedError
