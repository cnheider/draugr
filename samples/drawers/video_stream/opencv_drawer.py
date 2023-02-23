#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/9/22
           """

__all__ = []

from draugr.drawers import OpencvImageStream


def asdasf() -> None:
    """
    :rtype: None
    """
    from draugr.opencv_utilities import frame_generator, AsyncVideoStream
    from draugr.visualisation.progress import progress_bar

    with AsyncVideoStream() as vc:
        with OpencvImageStream() as s:
            for i in progress_bar(frame_generator(vc), auto_total_generator=False):
                s.draw(i)


if __name__ == "__main__":
    asdasf()
