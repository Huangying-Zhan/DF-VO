''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-06-02
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep depth network interface
'''

class DeepDepth():
    """This is the Base class for deep depth network interface
    """
    
    def __init__(self, cfg=None):
        """
        Args:
            cfg (edict): deep depth configuration dictionary (only required when online finetuning)
        """
        self.depth_cfg = cfg

    def initialize_network_model(self, weight_path):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
        """
        raise NotImplementedError
    
    def inference(self, img):
        raise NotImplementedError
