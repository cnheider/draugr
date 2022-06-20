#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """

import cv2

from draugr.opencv_utilities import frame_generator
from draugr.opencv_utilities.windows.image import show_image
from draugr.tqdm_utilities import progress_bar

cameras = []
print("open cam1")
cameras += (frame_generator(cv2.VideoCapture(0)),)
print("open cam2")
cameras += (frame_generator(cv2.VideoCapture(1)),)
print("opened")

if __name__ == "__main__":
    for image, image2 in progress_bar(zip(*cameras)):
        show_image(image2)
        if show_image(image, wait=1):
            break
