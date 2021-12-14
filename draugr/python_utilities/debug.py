#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/03/2020
           """

__all__ = ["evaluate_context"]

from typing import Any, Callable, Iterable, MutableMapping, Tuple


def evaluate_context(x: Any, *args: Iterable, **kwargs: MutableMapping) -> Tuple:
    """

    :param x:
    :param args:
    :param kwargs:
    :return:"""
    if isinstance(x, Callable):
        x = x(*args, **kwargs)
    return (
        [evaluate_context(a) for a in args],
        {k: evaluate_context(v) for k, v in kwargs.items()},
        x,
        type(x),
    )


if __name__ == "__main__":
    print(evaluate_context(2, 2))
