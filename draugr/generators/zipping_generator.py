#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable

import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """

__all__ = ["unzipper"]


def unzipper(iterable: Iterable):
    """
  Unzips an iterable

  :param iterable:
  :return:
  """

    for a in iterable:
        yield list(zip(*a))
    return


if __name__ == "__main__":
    r = range(4)

    a = [[[*r] for _ in r] for _ in r]
    for _, (b, *c) in zip(r, unzipper(a)):
        print(b)
        print(c)
