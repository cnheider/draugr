#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

__author__ = "heider"
__doc__ = r"""

           Created on 01/02/2022
           """

__all__ = [
    "MarkerTypesEnum",
    "DistanceTypesEnum",
    "ThresholdTypeFlag",
    "HoughModeEnum",
    "WindowFlagEnum",
    "WindowPropertyFlag",
    "ContourRetrievalModeEnum",
    "MouseEventEnum",
    "MouseEventFlag",
    "FontEnum",
]

from enum import Enum, Flag

import cv2


class MarkerTypesEnum(Enum):
    cross = cv2.MARKER_CROSS  # A crosshair marker shape.
    tilted_cross = cv2.MARKER_TILTED_CROSS  # A 45 degree tilted crosshair marker shape.
    star = (
        cv2.MARKER_STAR
    )  # A star marker shape, combination of cross and tilted cross.
    diamond = cv2.MARKER_DIAMOND  # A diamond marker shape.
    square = cv2.MARKER_SQUARE  # A square marker shape.
    triangle_up = cv2.MARKER_TRIANGLE_UP  # An upwards pointing triangle marker shape.
    triangle_down = (
        cv2.MARKER_TRIANGLE_DOWN
    )  # A downwards pointing triangle marker shape.


class DistanceTypesEnum(Enum):
    user = cv2.DIST_USER  # User defined distance.
    l1 = cv2.DIST_L1  # distance = |x1-x2| + |y1-y2|
    l2 = cv2.DIST_L2  # the simple euclidean distance
    c = cv2.DIST_C  # distance = max(|x1-x2|,|y1-y2|)
    l12 = cv2.DIST_L12  # L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
    fair = cv2.DIST_FAIR  # distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
    welsch = cv2.DIST_WELSCH  # distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
    huber = cv2.DIST_HUBER  # distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345


class DistanceTransformLabelTypesEnum(Enum):
    """
    distanceTransform algorithm flags
    """

    ccomp = cv2.DIST_LABEL_CCOMP
    """each connected component of zeros in src (as well as all the non-zero pixels closest to the connected component) will be assigned the same label
    """

    pixel = cv2.DIST_LABEL_PIXEL
    """each zero pixel (and all the non-zero pixels closest to it) gets its own label."""


class DistanceTransformMasksEnum(Enum):
    """
    Mask size for distance transform.
    """

    mask_3 = cv2.DIST_MASK_3
    mask_5 = cv2.DIST_MASK_5
    mask_precise = cv2.DIST_MASK_PRECISE


class ThresholdTypeFlag(Flag):
    binary = cv2.THRESH_BINARY
    inverse_binary = cv2.THRESH_BINARY_INV
    truncate = cv2.THRESH_TRUNC
    to_zero = cv2.THRESH_TOZERO
    inverse_to_zero = cv2.THRESH_TOZERO_INV
    mask = cv2.THRESH_MASK
    otsu = cv2.THRESH_OTSU
    triangle = cv2.THRESH_TRIANGLE


class HoughModeEnum(Enum):
    standard = (
        cv2.HOUGH_STANDARD
    )  # classical or standard Hough transform. Every line is represented by two floating-point numbers (ρ,θ) , where ρ is a distance between (0,0) point and the line, and θ is the angle between x-axis and the normal to the line. Thus, the matrix must be (the created sequence will be) of CV_32FC2 type
    gradient = cv2.HOUGH_GRADIENT  # basically 21HT
    probabilistic = (
        cv2.HOUGH_PROBABILISTIC
    )  # probabilistic Hough transform (more efficient in case if the picture contains a few long linear segments). It returns line segments rather than the whole line. Each segment is represented by starting and ending points, and the matrix must be (the created sequence will be) of the CV_32SC4 type.
    multi_scale = (
        cv2.HOUGH_MULTI_SCALE
    )  # multi-scale variant of the classical Hough transform. The lines are encoded the same way as HOUGH_STANDARD.
    gradient_alternative = (
        cv2.HOUGH_GRADIENT_ALT
    )  # variation of HOUGH_GRADIENT to get better accuracy


class WindowFlagEnum(Enum):
    normal = (
        cv2.WINDOW_NORMAL
    )  # the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size.
    autosize = (
        cv2.WINDOW_AUTOSIZE
    )  # the user cannot resize the window, the size is constrainted by the image displayed.
    opengl = cv2.WINDOW_OPENGL  # window with opengl support.
    fullscreen = cv2.WINDOW_FULLSCREEN  # change the window to fullscreen.
    free_ratio = (
        cv2.WINDOW_FREERATIO
    )  # the image expends as much as it can (no ratio constraint).
    keep_ratio = cv2.WINDOW_KEEPRATIO  # the ratio of the image is respected.
    gui_expanded = cv2.WINDOW_GUI_EXPANDED  # status bar and tool bar
    gui_normal = cv2.WINDOW_GUI_NORMAL  # old fashious way


class WindowPropertyFlag(Flag):
    fullscreen = (
        cv2.WND_PROP_FULLSCREEN
    )  # fullscreen property (can be WINDOW_NORMAL or WINDOW_FULLSCREEN).
    autosize = (
        cv2.WND_PROP_AUTOSIZE
    )  # autosize property (can be WINDOW_NORMAL or WINDOW_AUTOSIZE).
    keep_ratio = (
        cv2.WND_PROP_ASPECT_RATIO
    )  # window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO).
    opengl = cv2.WND_PROP_OPENGL  # opengl support.
    visible = cv2.WND_PROP_VISIBLE  # checks whether the window exists and is visible
    topmost = (
        cv2.WND_PROP_TOPMOST
    )  # property to toggle normal window being topmost or not


class FontEnum(Enum):
    hershey_simplex = cv2.FONT_HERSHEY_SIMPLEX  # normal size sans-serif font

    hershey_plain = cv2.FONT_HERSHEY_PLAIN  # small size sans-serif font

    hershey_duplex = (
        cv2.FONT_HERSHEY_DUPLEX
    )  # normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)

    hershey_complex = cv2.FONT_HERSHEY_COMPLEX  # normal size serif font

    hershey_triplex = (
        cv2.FONT_HERSHEY_TRIPLEX
    )  # normal size serif font (more complex than FONT_HERSHEY_COMPLEX)

    hershey_complex_small = (
        cv2.FONT_HERSHEY_COMPLEX_SMALL
    )  # smaller version of FONT_HERSHEY_COMPLEX

    hershey_script_simplex = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font

    hershey_script_complex = (
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    )  # more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX

    italic = cv2.FONT_ITALIC  # flag for italic font


class ContourRetrievalModeEnum(Enum):
    """

      RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
    RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
    RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
    RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours. This full hierarchy is built and shown in the OpenCV contours.c demo.

      :return:
      :rtype:
    """

    external = cv2.RETR_EXTERNAL
    list_all = cv2.RETR_LIST
    ccomp = cv2.RETR_CCOMP
    tree = cv2.RETR_TREE


class MouseEventEnum(Enum):
    move = (
        cv2.EVENT_MOUSEMOVE
    )  # indicates that the mouse pointer has moved over the window.
    left_down = (
        cv2.EVENT_LBUTTONDOWN
    )  # indicates that the left mouse button is pressed.
    left_up = cv2.EVENT_LBUTTONUP  # indicates that left mouse button is released.
    left_double = (
        cv2.EVENT_LBUTTONDBLCLK
    )  # indicates that left mouse button is double clicked.
    right_down = (
        cv2.EVENT_RBUTTONDOWN
    )  # indicates that the right mouse button is pressed.
    right_up = cv2.EVENT_RBUTTONUP  # indicates that right mouse button is released.
    right_double = (
        cv2.EVENT_RBUTTONDBLCLK
    )  # indicates that right mouse button is double clicked.
    middle_down = (
        cv2.EVENT_MBUTTONDOWN
    )  # indicates that the middle mouse button is pressed.
    middle_up = cv2.EVENT_MBUTTONUP  # indicates that middle mouse button is released.
    middle_double = (
        cv2.EVENT_MBUTTONDBLCLK
    )  # indicates that middle mouse button is double clicked.
    wheel_ud = (
        cv2.EVENT_MOUSEWHEEL
    )  # positive and negative values mean forward and backward scrolling, respectively.
    wheel_rl = (
        cv2.EVENT_MOUSEHWHEEL
    )  # positive and negative values mean right and left scrolling, respectively.


class MouseEventFlag(Flag):
    ctrl_down = cv2.EVENT_FLAG_CTRLKEY  # indicates that CTRL Key is pressed.
    shift_down = cv2.EVENT_FLAG_SHIFTKEY  # indicates that SHIFT Key is pressed.
    alt_down = cv2.EVENT_FLAG_ALTKEY  # indicates that ALT Key is pressed.
    left_down = cv2.EVENT_FLAG_LBUTTON  # indicates that the left mouse button is down.
    right_down = (
        cv2.EVENT_FLAG_RBUTTON
    )  # indicates that the right mouse button is down.
    middle_down = (
        cv2.EVENT_FLAG_MBUTTON
    )  # indicates that the middle mouse button is down.
