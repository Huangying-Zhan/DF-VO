# Copyright (C) Huangying Zhan 2019. All rights reserved.
# This software is licensed under the terms in the LICENSE file 
# which allows for non-commercial use only.


from libs.geometry.backprojection import Backprojection
from libs.geometry.reprojection import Reprojection


class DeepLayer():
    def __init__(self, cfg):
        self.cfg = cfg
        
    def initialize_layers(self):
        if self.cfg.kp_selection.depth_consistency.enable:
            self.backproj = Backprojection(self.cfg.image.height, self.cfg.image.width).cuda()
            self.reproj = Reprojection(self.cfg.image.height, self.cfg.image.width).cuda()