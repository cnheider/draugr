#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

__all__ = ["replicate", "recursive_flatten"]

from typing import Sequence, Union

from warg import Number


def recursive_flatten(S: Sequence) -> Sequence:
    if not S:
        return S
    if isinstance(S[0], Sequence):
        return (*recursive_flatten(S[0]), *recursive_flatten(S[1:]))
    return (*S[:1], *recursive_flatten(S[1:]))


def replicate(x: Union[Sequence, Number], times: int = 2) -> Sequence:
    """
  if not tuple

  :param times:
  :type times:
  :param x:
  :type x:
  :return:
  :rtype:
  """
    if isinstance(x, Sequence):
        if len(x) == times:
            return x
    return (x,) * times


if __name__ == "__main__":

    print(recursive_flatten((((2,), 2), (2,), 2)))

    print(replicate(2))
    print(replicate(2, 4))

    print(replicate((2, 3)))
    print(replicate((2, 3), times=4))
