#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from abc import ABCMeta, abstractmethod
from collections import Counter


class Writer(metaclass=ABCMeta):
    def __init__(self, interval: int = 1, filters=None):
        self.counter = Counter()
        self.interval = interval
        self.filters = filters

    def filter(self, tag: str) -> bool:
        is_in_filters = self.filters is None or tag in self.filters
        at_interval = self.counter[tag] % (self.interval + 1) == 0
        return is_in_filters and at_interval

    def __enter__(self):
        return self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close(exc_type, exc_val, exc_tb)

    def scalar(self, tag: str, value: float, step_i: int = None):
        if step_i:
            if self.filter(tag):
                self._scalar(tag, value, self.counter[tag])
            self.counter[tag] = step_i
        else:
            if self.filter(tag):
                self._scalar(tag, value, self.counter[tag])
            self.counter[tag] += 1

    @abstractmethod
    def _scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    @abstractmethod
    def _close(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @abstractmethod
    def _open(self):
        return self
