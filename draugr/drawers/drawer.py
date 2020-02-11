#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Sized

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/09/2019
           """

__all__ = ["Drawer", "MockDrawer"]


class Drawer:
    @abstractmethod
    def draw(self, data: Sized, delta: float = 1 / 120):
        raise NotImplementedError


class MockDrawer(Drawer):
    def draw(self, data: Sized, delta: float = 1 / 120):
        pass
