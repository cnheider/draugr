#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

__all__ = ["replicate"]

from typing import Sequence, Union

from warg import Number


def replicate(x: Union[Sequence, Number], times: int = 2) -> Sequence:
    """
    if not tuple

    :param times:
    :type times:
    :param x:
    :type x:
    :return:
    :rtype:"""
    if isinstance(x, Sequence):
        if len(x) == times:
            return x
    return (x,) * times


if __name__ == "__main__":

    def asdaa():

        print(replicate(2))
        print(replicate(2, 4))

        print(replicate((2, 3)))
        print(replicate((2, 3), times=4))
