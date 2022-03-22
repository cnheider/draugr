#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-01-2021
           """

from typing import Sequence

import cv2

from draugr.drawers.drawer import Drawer
from draugr.opencv_utilities import WindowFlagEnum
from draugr.opencv_utilities.windows.default import match_return_code
from warg import drop_unused_kws, passes_kws_to

__all__ = ["OpencvImageStream"]


class OpencvImageStream(Drawer):
    """ """

    @drop_unused_kws
    @passes_kws_to(Drawer.__init__)
    def __init__(self, title: str = "", render: bool = True, **kwargs):

        super().__init__(**kwargs)
        if not render:
            return

        self.window_id = title
        cv2.namedWindow(self.window_id, WindowFlagEnum.normal.value)

    def draw(self, data: Sequence):
        """

        :param data:
        :return:"""
        if match_return_code(cv2.waitKey(1)):  # esc to quit
            cv2.destroyWindow(self.window_id)
            raise StopIteration
        else:
            cv2.imshow(self.window_id, data)


if __name__ == "__main__":

    def asdasf() -> None:
        """
        :rtype: None
        """
        from draugr.opencv_utilities import frame_generator, AsyncVideoStream
        from draugr.tqdm_utilities import progress_bar

        with AsyncVideoStream() as vc:
            with OpencvImageStream() as s:
                for i in progress_bar(frame_generator(vc), auto_total_generator=False):
                    s.draw(i)

    asdasf()
