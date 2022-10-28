from .kitti import KittiOdom, KittiRaw
from .tum import TUM
from .adelaide import Adelaide
from .kinect import Kinect
from .oxford_robotcar import OxfordRobotCar
from .euroc import EUROC
from.tartanair import TartanAIR
from .mimir import MIMIR

datasets = {
            "kitti_odom": KittiOdom,
            "kitti_raw": KittiRaw,
            "tum-1": TUM,
            "tum-2": TUM,
            "tum-3": TUM,
            "adelaide1": Adelaide,
            "adelaide2": Adelaide,
            "kinect": Kinect,
            'robotcar': OxfordRobotCar,
            "euroc": EUROC,
            "tartanair": TartanAIR,
            "mimir": MIMIR,
        }
