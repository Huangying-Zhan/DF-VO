''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-27
@LastEditors: Huangying Zhan
@Description: Camera related classes
'''

import numpy as np


class SE3():
    """SE3 object consists rotation and translation components
    """
    def __init__(self, np_arr=None):
        if np_arr is None:
            self._pose = np.eye(4)
        else:
            self._pose = np_arr

    @property
    def pose(self):
        """ (array, [4x4]): camera pose 
        """
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def inv_pose(self):
        """ (array, [4x4]): inverse camera pose 
        """
        return np.linalg.inv(self._pose)

    @inv_pose.setter
    def inv_pose(self, value):
        self._pose = np.linalg.inv(value)

    @property
    def R(self):
        """ (array, [3x4]): rotation matrix
        """
        return self._pose[:3, :3]

    @R.setter
    def R(self, value):
        self._pose[:3, :3] = value

    @property
    def t(self):
        """ (array, [3x1]): translation vector
        """
        return self._pose[:3, 3:]

    @t.setter
    def t(self, value):
        self._pose[:3, 3:] = value


class Intrinsics():
    """Camera intrinsics object
    """
    def __init__(self, param=None):
        """
        Args:
            param (list): [cx, cy, fx, fy]
        """
        if param is None:
            self._mat = np.zeros((3, 3))
        else:
            cx, cy, fx, fy = param
            self._mat = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    @property
    def mat(self):
        """ (array, [3x3]): intrinsics matrix """
        return self._mat

    @mat.setter
    def mat(self, mat):
        self._mat = mat

    @property
    def inv_mat(self):
        """ (array, [3x3]): inverse intrinsics matrix """
        return np.linalg.inv(self._mat)

    @inv_mat.setter
    def inv_mat(self, mat):
        self._mat = np.linalg.inv(mat)

    @property
    def fx(self):
        """ float: focal length in x-direction """
        return self._mat[0, 0]

    @fx.setter
    def fx(self, value):
        self._mat[0, 0] = value

    @property
    def fy(self):
        """ float: focal length in y-direction """
        return self._mat[1, 1]

    @fy.setter
    def fy(self, value):
        self._mat[1, 1] = value

    @property
    def cx(self):
        """ float: principal point in x-direction """
        return self._mat[0, 2]

    @cx.setter
    def cx(self, value):
        self._mat[0, 2] = value

    @property
    def cy(self):
        """ float: principal point in y-direction """
        return self._mat[1, 2]

    @cy.setter
    def cy(self, value):
        self._mat[1, 2] = value


class PinholeCamera():
    """Pinhole camera model
    Attributes:
        height (int): image height
        width (int): image width
        SE3 (SE3): camera pose
        K (intrinsics): camera intrinsics
    """
    def __init__(self, pose=None, K=None):
        """
        Args:
            pose (array, [4x4]): camera pose
            K (list): [cx, cy, fx, fy]
        """
        self._height = 0
        self._width = 0
        self._SE3 = SE3(pose)
        self._K = Intrinsics(K)

    @property
    def height(self):
        """ (int): image height """
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        """ (int): image width """
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def SE3(self):
        """ (SE3): pose """
        return self._SE3

    @SE3.setter
    def SE3(self, SE3_obj):
        self._SE3 = SE3_obj

    @property
    def K(self):
        """ (Intrinsics): camera intrinsics """
        return self._K

    @K.setter
    def K(self, intrinsics):
        self._K = intrinsics
