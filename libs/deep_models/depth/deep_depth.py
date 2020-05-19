''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-20
@LastEditors: Huangying Zhan
@Description: This is the Base class for deep depth network interface
'''

class DeepDepth():
    """This is the Base class for deep depth network interface
    """
    
    def __init__(self):
        return

    def initialize_network_model(self, weight_path):
        """initialize flow_net model with weight_path
        
        Args:
            weight_path (str): weight path
        """
        raise NotImplementedError
