from .kitti import KittiOdom, KittiRaw
from .tum import TUM
from .adelaide import Adelaide
from .kinect import Kinect

datasets = {
            "kitti_odom": KittiOdom,
            "kitti_raw": KittiRaw,
            "tum-1": TUM,
            "tum-2": TUM,
            "tum-3": TUM,
            "adelaide1": Adelaide,
            "adelaide2": Adelaide,
            "kinect": Kinect
        }
