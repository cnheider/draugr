#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from abc import ABCMeta, abstractmethod
from collections import Counter

__all__ = ["Writer"]


class Writer(metaclass=ABCMeta):
    def __init__(self, *, interval: int = 1, filters=None, **kwargs):
        self._counter = Counter()
        self._interval = interval
        self.filters = filters

    def filter(self, tag: str) -> bool:
        is_in_filters = self.filters is None or tag in self.filters
        at_interval = self._counter[tag] % (self._interval + 1) == 0
        return is_in_filters and at_interval

    def __enter__(self):
        return self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close(exc_type, exc_val, exc_tb)

    def scalar(self, tag: str, value: float, step_i: int = None) -> None:
        if step_i:
            if self.filter(tag):
                self._scalar(tag, value, self._counter[tag])
            self._counter[tag] = step_i
        else:
            if self.filter(tag):
                self._scalar(tag, value, self._counter[tag])
            self._counter[tag] += 1

    def blip(self, tag: str, step_i: int = None) -> None:
        if step_i:
            self.scalar(tag, 0, step_i - 1)
            self.scalar(tag, 1, step_i)
            self.scalar(tag, 0, step_i + 1)
        else:
            self.scalar(tag, 0)
            self.scalar(tag, 1)
            self.scalar(tag, 0)

    def close(self):
        self._close()

    def open(self):
        self._open()

    @abstractmethod
    def _scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    @abstractmethod
    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        raise NotImplementedError

    @abstractmethod
    def _open(self):
        return self
