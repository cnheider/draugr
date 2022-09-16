#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/9/22
           """

__all__ = []

from draugr.drawers import ImageStreamPlot


def asdasf() -> None:
    """
    :rtype: None
    """
    import cv2
    from draugr.opencv_utilities import frame_generator
    from draugr.visualisation.progress import progress_bar
    from functools import partial
    from draugr.opencv_utilities import AsyncVideoStream

    with AsyncVideoStream() as vc:
        coder = partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB)
        with ImageStreamPlot(coder(next(vc))) as s:
            for i in progress_bar(
                frame_generator(vc, coder=coder), auto_total_generator=False
            ):
                s.draw(i)


asdasf()
