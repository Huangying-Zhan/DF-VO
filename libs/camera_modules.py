# Copyright (C) Huangying Zhan 2019. All rights reserved.
#
# This software is licensed under the terms of the DF-VO licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np


class SE3():
    """SE3 object consists rotation and translation components
    Attributes:
        pose (4x4 numpy array): camera pose
        inv_pose (4x4 numpy array): inverse camera pose
        R (3x3 numpy array): Rotation component
        t (3x1 numpy array): translation component,
    """
    def __init__(self, np_arr=None):
        if np_arr is None:
            self._pose = np.eye(4)
        else:
            self._pose = np_arr

    @property
    def pose(self):
        """ pose (4x4 numpy array): camera pose """
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def inv_pose(self):
        """ inv_pose (4x4 numpy array): inverse camera pose """
        return np.linalg.inv(self._pose)

    @inv_pose.setter
    def inv_pose(self, value):
        self._pose = np.linalg.inv(value)

    @property
    def R(self):
        return self._pose[:3, :3]

    @R.setter
    def R(self, value):
        self._pose[:3, :3] = value

    @property
    def t(self):
        return self._pose[:3, 3:]

    @t.setter
    def t(self, value):
        self._pose[:3, 3:] = value


class Intrinsics():
    """Camera intrinsics object
    Attributes:
        mat (3x3 numpy array): intrinsics matrix
            [fx 0 cx]
            [0 fy cy]
            [0 0  1 ]
        inv_mat (3x3 array): perspective transformation matrix
        cx (float): principal point x
        cy (float): principal point y
        fx (float): focal length x
        fy (float): focal length y
    """
    def __init__(self, param=None):
        """
        Args:
            param: [cx, cy, fx, fy]
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
        return self._mat

    @mat.setter
    def mat(self, mat):
        self._mat = mat

    @property
    def inv_mat(self):
        return np.linalg.inv(self._mat)

    @inv_mat.setter
    def inv_mat(self, mat):
        self._mat = np.linalg.inv(mat)

    @property
    def fx(self):
        return self._mat[0, 0]

    @fx.setter
    def fx(self, value):
        self._mat[0, 0] = value

    @property
    def fy(self):
        return self._mat[1, 1]

    @fy.setter
    def fy(self, value):
        self._mat[1, 1] = value

    @property
    def cx(self):
        return self._mat[0, 2]

    @cx.setter
    def cx(self, value):
        self._mat[0, 2] = value

    @property
    def cy(self):
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
            pose (4x4 matrix): camera pose
            K (float list): [cx, cy, fx, fy]
        """
        self._height = 0
        self._width = 0
        self._SE3 = SE3(pose)
        self._K = Intrinsics(K)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def SE3(self):
        return self._SE3

    @SE3.setter
    def SE3(self, SE3_obj):
        self._SE3 = SE3_obj

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, intrinsics):
        self._K = intrinsics
