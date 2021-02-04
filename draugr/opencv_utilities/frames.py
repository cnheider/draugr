#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/03/2020
           """

from functools import partial
from typing import Callable, Iterable, Union

import cv2

__all__ = ["frame_generator"]


from warg import identity


def frame_generator(
    video_stream: cv2.VideoCapture,
    coder: Union[None, Callable] = partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
) -> Iterable:
    """

    @param video_stream:
    @param coder:
    """
    if coder is None:
        coder = identity
    while video_stream.isOpened():
        success, frame = video_stream.read()
        if success:
            yield coder(frame)
        else:
            break


if __name__ == "__main__":

    def asd():
        from tqdm import tqdm

        for image in tqdm(frame_generator(cv2.VideoCapture(0), None)):
            cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
            cv2.imshow("window_name", image)

            if cv2.waitKey(1) == 27:
                break  # esc to quit

    asd()
