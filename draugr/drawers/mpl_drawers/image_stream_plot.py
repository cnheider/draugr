#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
               Image stream plot using matplotlib

           Created on 22-01-2021
           """

__all__ = ["ImageStreamPlot"]

from typing import Sequence

from matplotlib import pyplot
import matplotlib
from draugr.drawers.mpl_drawers.mpldrawer import MplDrawer
from warg import passes_kws_to

# pyplot.switch_backend("QtAgg")  # MacOSX, QtAgg, GTK4Agg, Gtk3Agg, TkAgg, WxAgg, Agg


class ImageStreamPlot(MplDrawer):
    """
    Image stream plot using matplotlib
    """

    @passes_kws_to(MplDrawer.__init__)
    @passes_kws_to(pyplot.imshow)
    def __init__(
        self, image, title: str = "", render: bool = True, **kwargs
    ):  # size:Tuple[int,...],
        super().__init__(render=render, **kwargs)

        if not render:
            return

        self.fig = pyplot.figure(figsize=(4, 4))
        self.im = pyplot.imshow(image, **kwargs)

        pyplot.title(title)
        pyplot.tight_layout()
        cid = self.fig.canvas.mpl_connect("key_press_event", self.close)

    def _draw(self, data: Sequence):
        """

        :param data:
        :return:"""
        self.im.set_data(data)

    @staticmethod
    def close(event: matplotlib.backend_bases.Event):
        """description"""
        if event.key == "q" or event.key == "escape":
            pyplot.close(event.canvas.figure)
            raise GeneratorExit
            # raise StopIteration


if __name__ == "__main__":

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
                for i in progress_bar(frame_generator(vc, coder=coder)):
                    s.draw(i)

    asdasf()
