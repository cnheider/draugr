#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Iterable

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/11/2019
           """

__all__ = ["yield_and_map", "inner_map", "kw_map"]


def yield_and_map(iterable: Iterable, level: int = 0, func: callable = print) -> Any:
    """

    :param iterable:
    :type iterable:
    :param level:
    :type level:
    :param func:
    :type func:"""
    if level == 0:
        for a in iterable:
            func(a)
            yield a
    elif level == 1:
        for a in iterable:
            for b in a:
                func(b)
                yield b
    elif level == 2:
        for a in iterable:
            for b in a:
                for c in b:
                    func(c)
                    yield c


def inner_map(func: callable, iterable: Iterable, aggregate_yield: bool = True) -> Any:
    """

    :param func:
    :type func:
    :param iterable:
    :type iterable:
    :param aggregate_yield:
    :type aggregate_yield:"""
    if aggregate_yield:
        for a in iterable:
            yield [func(b) for b in a]
    else:
        for a in iterable:
            for b in a:
                yield func(b)


def kw_map(func: callable, kw: str, iterable: Iterable) -> Any:
    """

    :param func:
    :type func:
    :param kw:
    :type kw:
    :param iterable:
    :type iterable:"""
    for a in iterable:
        yield func(**{kw: a})


if __name__ == "__main__":
    a = (2, 3)
    # TODO
