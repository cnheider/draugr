#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """

import cv2

from draugr.opencv_utilities import frame_generator, to_gray
from draugr.opencv_utilities.windows.image import show_image
from draugr.visualisation.progress import progress_bar

if __name__ == "__main__":
    for image in progress_bar(frame_generator(cv2.VideoCapture(0))):
        gray = to_gray(image)
        show_image(gray)
        if show_image(image, wait=1):
            break  # esc to quit
