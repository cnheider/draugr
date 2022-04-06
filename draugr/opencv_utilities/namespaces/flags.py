#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 01/02/2022
           """

__all__ = [
    "ThresholdTypeFlag",
    "WindowPropertyFlag",
    "DrawMatchesFlagEnum",
    "MouseEventFlag",
    "TermCriteriaFlag",
]

from enum import Flag

import cv2


class TermCriteriaFlag(Flag):
    count = (
        cv2.TERM_CRITERIA_COUNT
    )  # the maximum number of iterations or elements to compute
    eps = (
        cv2.TERM_CRITERIA_EPS
    )  # the desired accuracy or change in parameters at which the iterative algorithm stops
    max_iter = cv2.TERM_CRITERIA_MAX_ITER  # the maximum number of iterations to compute


class DrawMatchesFlagEnum(Flag):
    default = cv2.DRAW_MATCHES_FLAGS_DEFAULT
    # Output image matrix will be created (Mat::create), i.e. existing memory of output image may be reused. Two source image, matches and single keypoints will be drawn. For each keypoint only the center point will be drawn (without the circle around keypoint with keypoint size and orientation).

    over_outimg = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    # Output image matrix will not be created (Mat::create). Matches will be drawn on existing content of output image.

    not_draw_single_points = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    # Single keypoints will not be drawn.

    rich_keypoints = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # For each keypoint the circle around keypoint with keypoint size and orientation will be drawn.


class ThresholdTypeFlag(Flag):
    binary = cv2.THRESH_BINARY  # dst(x,y)={maxval0if src(x,y)>threshotherwise
    inverse_binary = (
        cv2.THRESH_BINARY_INV
    )  # dst(x,y)={0maxvalif src(x,y)>threshotherwise
    truncate = (
        cv2.THRESH_TRUNC
    )  # dst(x,y)={thresholdsrc(x,y)if src(x,y)>threshotherwise
    to_zero = cv2.THRESH_TOZERO  # dst(x,y)={src(x,y)0if src(x,y)>threshotherwise
    inverse_to_zero = (
        cv2.THRESH_TOZERO_INV
    )  # dst(x,y)={0src(x,y)if src(x,y)>threshotherwise
    mask = cv2.THRESH_MASK
    otsu = (
        cv2.THRESH_OTSU
    )  # flag, use Otsu algorithm to choose the optimal threshold value
    triangle = (
        cv2.THRESH_TRIANGLE
    )  # flag, use Triangle algorithm to choose the optimal threshold value


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
