#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sized, Iterable

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/10/2019
           """


def sized_batch(sized: Iterable, n: int = 32, drop_not_full: bool = True):
    if not isinstance(sized, Sized):
        sized = list(sized)
    l = len(sized)
    for ndx in range(0, l, n):
        if drop_not_full and ndx + n > l - 1:
            return
        yield sized[ndx : min(ndx + n, l)]


def generator_batch(iterable: Iterable, n: int = 32, drop_not_full: bool = True):
    b = []
    i = 0
    for a in iterable:
        b.append(a)
        i += 1
        if i >= n:
            yield b
            b.clear()
            i = 0
    if drop_not_full:
        return
    return b
