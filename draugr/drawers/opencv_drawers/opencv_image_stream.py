#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-01-2021
           """

from typing import Sequence

import cv2
from draugr.drawers.drawer import Drawer
from warg import passes_kws_to


class OpencvImageStream(Drawer):
    """"""

    def __init__(self, title: str = "", render: bool = True, **kwargs):

        if not render:
            return

        self.window_id = title
        cv2.namedWindow(self.window_id, cv2.WINDOW_NORMAL)

    def draw(self, data: Sequence):
        """

        :param data:
        :return:"""
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyWindow(self.window_id)
            raise StopIteration
        else:
            cv2.imshow(self.window_id, data)


if __name__ == "__main__":

    def asdasf():

        from draugr.opencv_utilities import frame_generator, AsyncVideoStream
        from draugr.tqdm_utilities import progress_bar

        with AsyncVideoStream() as vc:
            with OpencvImageStream() as s:
                for i in progress_bar(
                    frame_generator(vc, coder=None), auto_total_generator=False
                ):
                    s.draw(i)

    asdasf()
