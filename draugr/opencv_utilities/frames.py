#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/03/2020
           """

from typing import Iterable

import cv2


def frame_generator(video: cv2.VideoCapture) -> Iterable:
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
