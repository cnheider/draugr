#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["HistogramWriterMixin"]


class HistogramWriterMixin(ABC):
    """
  Writer mixin that provides an interface for 'writing' histogram
  """

    @abstractmethod
    def histogram(self, tag: str, values: list, step: int, **kwargs) -> None:
        """

    :param tag:
    :type tag:
    :param data:
    :type data:
    :param step:
    :type step:
    :param dataformats:
    :type dataformats:
    :param kwargs:
    :type kwargs:
    """
        raise NotImplementedError
