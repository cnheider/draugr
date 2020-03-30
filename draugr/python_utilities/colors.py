#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/03/2020
           """

from typing import Tuple


def RGB(R: int, G: int, B: int) -> Tuple[int, int, int]:
    assert 0 <= R <= 255 and 0 <= G <= 255 and 0 <= B <= 255
    return R, G, B


def RGBA(R: int, G: int, B: int, A: int) -> Tuple[int, int, int, int]:
    assert 0 <= R <= 255 and 0 <= G <= 255 and 0 <= B <= 255 and 0 <= A <= 255
    return R, G, B, A


def color_from_str(s: str, seperator=" ") -> Tuple[int, ...]:
    components = s.split(seperator)
    n = len(components)
    if n == 3:
        return RGB(*[int(c) for c in components])
    elif n == 4:
        return RGBA(*[int(c) for c in components])
    raise NotImplementedError("Color space not recognised")


def color_to_str(t: Tuple[int, ...], seperator=" ") -> str:
    return seperator.join([str(c) for c in t])


if __name__ == "__main__":
    a = RGB(1, 50, 100)
    b = color_to_str(a)
    c = color_from_str(b)
    print(c)
