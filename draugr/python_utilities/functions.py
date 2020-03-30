#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14/01/2020
           """

__all__ = ["identity", "sink", "prod", "collate_batch_fn", "kw_identity"]

import operator
from functools import reduce
from typing import Any, Dict, Iterable, Tuple, Union

from warg import drop_unused_kws


@drop_unused_kws
def identity(*args) -> Any:
    """
Returns args without any modification what so ever. Drops kws
:param x:
:return:
"""
    return args


def kw_identity(*args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
    """

:param args:
:param kwargs:
:return:
"""
    return args, kwargs


def collate_batch_fn(batch: Iterable) -> tuple:
    """

:param batch:
:return:
"""
    return tuple(zip(*batch))


def sink(*args, **kwargs):
    """
Returns None, but accepts everthing

:param args:
:param kwargs:
:return:
"""
    pass


def prod(iterable: Iterable[Union[int, float]]) -> Union[int, float]:
    """
Calculate the product of the a Iterable of int or floats
:param iterable:
:return:
"""
    return reduce(operator.mul, iterable, 1)
