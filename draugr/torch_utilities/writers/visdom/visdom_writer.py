#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

    def _scalar(self, tag: str, value: float, step: int):
        pass

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        pass

    def _open(self):
        pass
