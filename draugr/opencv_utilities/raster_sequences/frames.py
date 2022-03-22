#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/03/2020
           """

from functools import partial
from typing import Callable, Iterable, Optional

import cv2

__all__ = ["frame_generator", "to_rgb"]

from warg import identity

to_rgb = partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB)


def frame_generator(
    video_stream: cv2.VideoCapture,
    coder: Optional[Callable] = identity,
) -> Iterable:
    """

    :param video_stream:
    :param coder:"""
    if coder is None:
        coder = identity
    while video_stream.isOpened():
        success, frame = video_stream.read()
        if success:
            yield coder(frame)
        else:
            break


if __name__ == "__main__":

    def asd() -> None:
        """
        :rtype: None
        """
        from draugr.opencv_utilities.windows.image import show_image
        from draugr.tqdm_utilities import progress_bar

        for image in progress_bar(frame_generator(cv2.VideoCapture(0))):
            if show_image(image, "frame", wait=1):
                break  # esc to quit

    asd()
