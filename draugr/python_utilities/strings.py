#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = ["indent_lines", "str_to_tuple"]

from typing import Any


def indent_lines(
    input_str: Any, indent_spaces_num: int = 2, ignore_single_lines: bool = False
) -> str:
    """

    :param ignore_single_lines:
    :type ignore_single_lines:
    :param input_str:
    :type input_str:
    :param indent_spaces_num:
    :type indent_spaces_num:
    :return:
    :rtype:"""
    if not isinstance(input_str, str):
        input_str = str(input_str)
    s = input_str.split("\n")
    indent_s = indent_spaces_num * " "
    if len(s) == 1:
        if ignore_single_lines:
            return input_str
        else:
            return f"{indent_s}{input_str}"
    first = s.pop(0)
    s = [f"{indent_s}{line}" for line in s]
    s = "\n".join(s)
    s = f"{indent_s}{first}\n{s}"
    return s


def str_to_tuple(arg):
    """Convert a series of zero or more numbers to an argument tuple"""
    return tuple(map(int, arg.split()))


if __name__ == "__main__":
    a = "slasc\nsaffasd\n2dasf"
    print(a)
    print(indent_lines(a))
