#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 05-03-2021
           """

__all__ = ["increasing_density_hatch"]

from itertools import count

increasing_density_hatch = (b * a for a in count(1) for b in ("\\", "/", "x", "."))

four_times_denser_hatch = tuple(
    b * a for a in range(1, 4 + 1) for b in ("\\", "/", "x", ".")
)

if __name__ == "__main__":
    print(tuple(four_times_denser_hatch))
