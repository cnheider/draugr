#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30/03/2020
           """

from typing import Tuple

COLOR_RGB = Tuple[int, int, int]
COLOR_RGBA = Tuple[int, int, int, int]
COLOR_INT = Tuple[int, ...]

__all__ = ["rgb", "rgba", "RGB", "RGBA", "color_to_str", "color_from_str"]


def rgb(r: int, g: int, b: int) -> COLOR_RGB:
    """

    :param r:
    :type r:
    :param g:
    :type g:
    :param b:
    :type b:
    :return:
    :rtype:"""
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    return r, g, b


RGB = rgb


def rgba(r: int, g: int, b: int, a: int) -> COLOR_RGBA:
    """

    :param r:
    :type r:
    :param g:
    :type g:
    :param b:
    :type b:
    :param a:
    :type a:
    :return:
    :rtype:"""
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 255
    return r, g, b, a


RGBA = rgba


def color_from_str(s: str, seperator: str = " ") -> COLOR_INT:
    """

    :param s:
    :type s:
    :param seperator:
    :type seperator:
    :return:
    :rtype:"""
    components = s.split(seperator)
    n = len(components)
    if n == 3:
        return RGB(*[int(c) for c in components])
    elif n == 4:
        return RGBA(*[int(c) for c in components])
    raise NotImplementedError("Color space not recognised")


def color_to_str(t: COLOR_INT, seperator: str = " ") -> str:
    """

    :param t:
    :type t:
    :param seperator:
    :type seperator:
    :return:
    :rtype:"""
    return seperator.join([str(c) for c in t])


if __name__ == "__main__":

    def main() -> None:
        """
        :rtype: None
        """
        a = RGB(1, 50, 100)
        b = color_to_str(a)
        c = color_from_str(b)
        print(c)

    main()
