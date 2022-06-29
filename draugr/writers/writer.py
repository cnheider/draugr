#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from abc import ABCMeta, abstractmethod
from collections import Counter, deque

from draugr.writers.mixins.scalar_writer_mixin import ScalarWriterMixin

__all__ = ["Writer", "global_writer", "set_global_writer"]

from typing import Any, Iterable, Optional

from warg import is_none_or_zero_or_negative_or_mod_zero
from warg import drop_unused_kws


class Writer(ScalarWriterMixin, metaclass=ABCMeta):
    """description"""

    @drop_unused_kws
    def __init__(
        self,
        *,
        interval: Optional[int] = 1,
        filters: Iterable = None,
        verbose: bool = False
    ):
        """

        :param interval:
        :param filters:
        :param verbose:"""
        self._counter = Counter()

        self._interval = interval
        self.filters = filters
        self._verbose = verbose

    def filter(self, tag: str) -> bool:
        """

            returns a boolean  value, true if to be included, False if to be excluded

            tag is in filter if not None
            and within interval for inclusion

        :param tag:
        :type tag:
        :return:
        :rtype:"""
        is_in_filters = self.filters is None or tag in self.filters
        at_interval = is_none_or_zero_or_negative_or_mod_zero(
            self._interval, self._counter[tag]
        )
        return is_in_filters and at_interval

    def __enter__(self):
        global GLOBAL_WRITER_STACK, GLOBAL_WRITER
        GLOBAL_WRITER_STACK.appendleft(self)
        GLOBAL_WRITER = self
        return self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global GLOBAL_WRITER, GLOBAL_WRITER_STACK

        if len(GLOBAL_WRITER_STACK) > 0:
            GLOBAL_WRITER_STACK.popleft()  # pop self

        if len(GLOBAL_WRITER_STACK) > 0:
            GLOBAL_WRITER = GLOBAL_WRITER_STACK.popleft()  # then previous
        else:
            GLOBAL_WRITER = None
        return self._close(exc_type, exc_val, exc_tb)

    def close(self) -> Any:
        """description"""
        self._close()

    def open(self) -> Any:
        """description"""
        self._open()

    @abstractmethod
    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        raise NotImplementedError

    @abstractmethod
    def _open(self):
        return self

    def __call__(self, *args, **kwargs):
        self.scalar(*args, *kwargs)


GLOBAL_WRITER_STACK = deque()
GLOBAL_WRITER = None


def global_writer() -> Optional[Writer]:
    """

    :return:
    :rtype:"""
    global GLOBAL_WRITER
    return GLOBAL_WRITER


def set_global_writer(writer: Writer) -> None:
    """

    :return:
    :rtype:"""
    global GLOBAL_WRITER
    # if GLOBAL_WRITER:
    # GLOBAL_WRITER_STACK TODO: push to stack if existing?

    GLOBAL_WRITER = writer
