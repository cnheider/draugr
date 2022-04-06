#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/10/2019
           """

__all__ = ["draw_cube", "draw_axis", "cube_3d_matrix"]

from draugr.opencv_utilities.windows.image import show_image
from warg import Number


def draw_axis(
    img,
    corners,
    rotation_vectors,
    translation_vectors,
    camera_matrix,
    dist_coef,
    size: Number = 1,
):
    """

    :param img:
    :type img:
    :param corners:
    :type corners:
    :param rotation_vectors:
    :type rotation_vectors:
    :param translation_vectors:
    :type translation_vectors:
    :param camera_matrix:
    :type camera_matrix:
    :param dist_coef:
    :type dist_coef:
    :param size:
    :type size:
    :return:
    :rtype:"""
    axis_size: numpy.ndarray = numpy.eye(3, 3) @ numpy.diag([size, size, -size])
    axis_size = axis_size.astype(numpy.float32)

    img_pts, jac = cv2.projectPoints(
        axis_size, rotation_vectors, translation_vectors, camera_matrix, dist_coef
    )  # project 3D points to image plane

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
    return img


def cube_3d_matrix():
    """

    :return:
    :rtype:"""
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


def draw_cube(
    img,
    rotation_vectors,
    translation_vectors,
    camera_matrix,
    dist_coef,
    size: Number = 6,
):
    """

    :param img:
    :type img:
    :param rotation_vectors:
    :type rotation_vectors:
    :param translation_vectors:
    :type translation_vectors:
    :param camera_matrix:
    :type camera_matrix:
    :param dist_coef:
    :type dist_coef:
    :param size:
    :type size:
    :return:
    :rtype:"""
    cube_size = numpy.float32(cube_3d_matrix()) * size

    img_pts, jac2 = cv2.projectPoints(
        cube_size, rotation_vectors, translation_vectors, camera_matrix, dist_coef
    )

    img_pts = numpy.int32(img_pts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), (255, 0, 0), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [img_pts[4:]], -1, (0, 0, 255), 3)

    return img


if __name__ == "__main__":
    show_image(
        draw_cube(
            numpy.zeros((50, 50)),
        )
    )
