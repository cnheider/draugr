#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-01-2021
           """

__all__ = ["Drawer"]

from abc import abstractmethod

from warg import drop_unused_kws


class Drawer(object):
    """description"""

    @drop_unused_kws
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    @abstractmethod
    def draw(self, *args, **kwargs):
        """description"""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
