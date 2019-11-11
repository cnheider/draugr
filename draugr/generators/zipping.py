#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def unzipper(iterable: Iterable):
    """Unzips an iterable"""
    for a in iterable:
        yield list(zip(*a))
    return


if __name__ == "__main__":
    for i in yield_and_map([0, 1, 2, 3]):
        print(2 ** i)
