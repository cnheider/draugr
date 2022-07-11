#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 7/8/22
           """

__all__ = ["CounterFilter"]

from abc import ABCMeta
from collections import Counter
from typing import Optional, Iterable

from warg import drop_unused_kws, is_none_or_zero_or_negative_or_mod_zero


class CounterFilter(metaclass=ABCMeta):
    """ """

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
