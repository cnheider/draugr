#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/03/2020
           """

__all__ = ["compute_color_for_labels"]

from typing import Tuple

from warg import TripleNumber


def compute_color_for_labels(
    label: int, palette: TripleNumber = (2**11 - 1, 2**15 - 1, 2**20 - 1)
) -> Tuple:
    """
    Simple function that adds fixed color depending on the class"""
    return (
        *[int(((label > 0) * p * (label**2 - label + 1)) % 255) for p in palette],
    )


if __name__ == "__main__":
    for i in range(9):
        print(compute_color_for_labels(i))
