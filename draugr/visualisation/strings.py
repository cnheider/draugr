#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = ["add_indent"]


def add_indent(input_str: str, indent_spaces_num: int = 2):
    s = input_str.split("\n")
    if len(s) == 1:  # don't do anything for single-line stuff
        return input_str
    first = s.pop(0)
    indent = indent_spaces_num * " "
    s = [indent + line for line in s]
    s = "\n".join(s)
    s = indent + first + "\n" + s
    return s


if __name__ == "__main__":
    a = "slasc\nsaffasd\n2dasf"
    print(a)
    print(add_indent(a))
