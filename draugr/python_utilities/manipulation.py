#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

__all__ = ["recursive_flatten"]

from typing import Sequence


def recursive_flatten(seq: Sequence) -> Sequence:
    """ """
    if not seq:  # is empty Sequence
        return seq
    if isinstance(seq[0], Sequence):
        return (*recursive_flatten(seq[0]), *recursive_flatten(seq[1:]))
    return (*seq[:1], *recursive_flatten(seq[1:]))


if __name__ == "__main__":
    print(recursive_flatten((((2,), 2), (2,), 2)))
    print(recursive_flatten((([[None]], 2), (2,), 2)))
