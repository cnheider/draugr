#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Any, Sequence, Tuple

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/09/2019
           """

__all__ = ["MplDrawer", "MockDrawer"]

import matplotlib
from draugr.drawers.drawer import Drawer
from matplotlib import pyplot


class MplDrawer(
    # metaclass=PostInit
    Drawer
):
    """
    Real time plotting base class

    for Matplotlib"""

    # @drop_unused_kws
    def __init__(
        self,
        *,
        default_delta: float = 1 / 120,
        render: bool = True,
        placement: Tuple[int, int] = None,
        **kwargs,
    ):
        """

        :param default_delta:
        :param render:
        :param placement:
        :param kwargs:"""
        super().__init__(**kwargs)
        self.fig = None

        if not render:
            return

        if default_delta is None:  # Zero still passes
            default_delta = 1 / 120

        self._default_delta = default_delta
        self.n = 0

        """
fig_manager = pyplot.get_current_fig_manager()
geom = fig_manager.window.geometry()
x, y, dx, dy = geom.getRect()
fig_manager.window.setGeometry(*placement, dx, dy)
fig_manager.window.SetPosition((500, 0))
"""
        self.placement = placement

    """
@drop_unused_kws
def __post_init__(self,*, figure_size: Tuple[int, int] = None):
if self.fig is None:
if figure_size is None:
figure_size = (4, 4)
self.fig = pyplot.figure(figsize=figure_size)
"""

    def draw(self, data: Any, delta: float = None):
        """ """
        if not self.fig:
            raise NotImplementedError(
                "Figure was not instantiated check specialisation of MplDrawer"
            )

        self._draw(data)

        pyplot.draw()

        if self.n <= 1 and self.placement:
            self.move_figure(self.fig, *self.placement)
        self.n += 1

        if delta:  # TODO: ALLOW FOR ASYNC DRAWING
            pyplot.pause(delta)
        elif self._default_delta:
            pyplot.pause(self._default_delta)

    @staticmethod
    def move_figure(figure: pyplot.Figure, x: int = 0, y: int = 0):
        r"""
        Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if hasattr(figure.canvas.manager, "window"):
            window = figure.canvas.manager.window
            if backend == "TkAgg":
                window.wm_geometry(f"+{x:d}+{y:d}")
            elif backend == "WXAgg":
                window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                window.move(x, y)

    def __enter__(self):
        return self

    def close(self):
        """ """
        if self._verbose:
            print("mlpdrawer close was called")
        if self.fig:
            pyplot.close(self.fig)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @abstractmethod
    def _draw(self, data: Any) -> None:
        """

        :param data:
        :type data:"""
        raise NotImplementedError


class MockDrawer(MplDrawer):
    """
    Mock for drawer, accepts data but draws nothing"""

    def _draw(self, data: Sequence) -> None:
        """

        :param data:
        :type data:"""
        pass
