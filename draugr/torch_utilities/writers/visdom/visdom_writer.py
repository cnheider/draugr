#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import visdom

from draugr.writers.writer import Writer

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/14/19
           """

__all__ = ["VisdomWriter"]

# Visualisation
USE_VISDOM = False
START_VISDOM_SERVER = False
VISDOM_SERVER = "http://localhost"
if not START_VISDOM_SERVER:
    # noinspection PyRedeclaration
    VISDOM_SERVER = "http://visdom.ml"


class VisdomWriter(Writer):
    """ """

    def __init__(self):
        super().__init__()

    def _scalar(self, tag: str, value: float, step: int):
        self.server.line(
            Y=numpy.array([value]), X=numpy.array([step]), win=tag, update="append"
        )

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        # self.server.close() # close a window by id
        del self.server

    def _open(self):
        self.server = visdom.Visdom(server=VISDOM_SERVER, port=8097)
