#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from abc import ABCMeta, abstractmethod
from collections import deque

from draugr.python_utilities.counter_filter import CounterFilter
from draugr.writers.mixins.scalar_writer_mixin import ScalarWriterMixin

__all__ = ["Writer", "global_writer", "set_global_writer"]

from typing import Any, Optional, Sequence, MutableMapping


class Writer(ScalarWriterMixin, metaclass=ABCMeta):
    """description"""

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

    def __call__(self, *args: Sequence, **kwargs: MutableMapping):
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
