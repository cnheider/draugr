#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable

from draugr import to_tensor
from warg import passes_kws_to, NOD

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


@passes_kws_to(to_tensor)
def to_tensor_generator(iterable: Iterable, preload_next: bool = True, **kwargs):
    """

  :param iterable:
  :param preload_next:
  :param kwargs:
  :return:
  """
    if preload_next:
        iterable_iter = iter(iterable)
        current = to_tensor(next(iterable_iter), **kwargs)
        kwargs["non_blocking"] = True
        while current is not None:
            next_ = to_tensor(next(iterable_iter), **kwargs)
            yield current
            current = next_
    else:
        for a in iterable:
            yield to_tensor(a, **kwargs)
    return
