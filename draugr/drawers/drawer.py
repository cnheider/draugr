#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Sequence

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/09/2019
           """

__all__ = ["Drawer", "MockDrawer"]


class Drawer:
    """
  Real time plotting base class

  """

    @abstractmethod
    def draw(self, data: Sequence, delta: float = 1 / 120) -> None:
        """

    :param data:
    :type data:
    :param delta:
    :type delta:
    """
        raise NotImplementedError


class MockDrawer(Drawer):
    """
  Mock for drawer, accepts data but draws nothing

  """

    def draw(self, data: Sequence, delta: float = 1 / 120) -> None:
        """

    :param data:
    :type data:
    :param delta:
    :type delta:
    """
        pass
