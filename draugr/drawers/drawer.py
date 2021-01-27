#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-01-2021
           """

from abc import abstractmethod

__all__ = ["Drawer"]


class Drawer(object):
    @abstractmethod
    def draw(self, *args, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
