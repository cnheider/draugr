#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/10/2019
           """


def draw_axis(img, corners, rot_vecs, trans_vecs, camera_mtx, dist_coef, size=1):
    """

    :param img:
    :type img:
    :param corners:
    :type corners:
    :param rot_vecs:
    :type rot_vecs:
    :param trans_vecs:
    :type trans_vecs:
    :param camera_mtx:
    :type camera_mtx:
    :param dist_coef:
    :type dist_coef:
    :param size:
    :type size:
    :return:
    :rtype:
    """
    axis_size: numpy.ndarray = numpy.eye(3, 3) @ numpy.diag([size, size, -size])
    axis_size = axis_size.astype(numpy.float32)

    img_pts, jac = cv2.projectPoints(
        axis_size, rot_vecs, trans_vecs, camera_mtx, dist_coef
    )  # project 3D points to image plane

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
    return img


def cube_3d_matrix():
    """

    :return:
    :rtype:
    """
    return [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, -1],
        [1, 1, -1],
        [1, 0, -1],
    ]


def draw_cube(img, rot_vecs, trans_vecs, camera_mtx, dist_coef, size=6):
    """

    :param img:
    :type img:
    :param rot_vecs:
    :type rot_vecs:
    :param trans_vecs:
    :type trans_vecs:
    :param camera_mtx:
    :type camera_mtx:
    :param dist_coef:
    :type dist_coef:
    :param size:
    :type size:
    :return:
    :rtype:
    """
    cube_size = numpy.float32(cube_3d_matrix()) * size

    imgpts, jac2 = cv2.projectPoints(
        cube_size, rot_vecs, trans_vecs, camera_mtx, dist_coef
    )

    imgpts = numpy.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img
