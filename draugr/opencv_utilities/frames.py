#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/03/2020
           """

from typing import Iterable

import cv2

__all__ = ["frame_generator"]


def frame_generator(video: cv2.VideoCapture) -> Iterable:
    """

    :param video:
    :type video:"""
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


if __name__ == "__main__":

    def asd():
        from tqdm import tqdm

        for image in tqdm(frame_generator(cv2.VideoCapture(0))):
            cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
            cv2.imshow("window_name", image)

            if cv2.waitKey(1) == 27:
                break  # esc to quit

    asd()
