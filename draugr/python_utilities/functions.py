#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14/01/2020
           """

__all__ = ["identity", "sink", "prod", "evaluate_context"]

from typing import Iterable, Union, Any, Callable
import operator
from functools import reduce


def identity(x: Any) -> Any:
    """
    Returns x without any modification what so ever.
    :param x:
    :return:
    """
    return x


def sink(*args, **kwargs):
    pass


def prod(iterable: Iterable[Union[int, float]]) -> Union[int, float]:
    """
  Calculate the product of the a Iterable of int or floats
  :param iterable:
  :return:
  """
    return reduce(operator.mul, iterable, 1)


def evaluate_context(x: Any, *args, **kwargs) -> Any:
    """

  :param x:
  :param args:
  :param kwargs:
  :return:
  """
    a_r = [evaluate_context(a) for a in args]
    kw_r = {k: evaluate_context(v) for k, v in kwargs.items()}
    if isinstance(x, Callable):
        x = x(*args, **kwargs)
    return a_r, kw_r, x, type(x)


if __name__ == "__main__":

    print(evaluate_context(identity, "str"))
    print(evaluate_context(identity, 2))
    print(evaluate_context(identity, 2.2))

    print(evaluate_context(prod, (2, 2)))

    print(evaluate_context(prod, (2.2, 2.2)))

    print(evaluate_context(prod, (2, 2.2)))

    print(evaluate_context(prod, (2.2, 2)))

    print(evaluate_context(sink, (2, 2), face=(2.2, 2)))

    print(evaluate_context(2, 2))
