#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """

from draugr.torch_utilities import global_pin_memory


def test_load_pin_memory_global():
    print(global_pin_memory(0))
    print(global_pin_memory(1))
    print(global_pin_memory(2))
    print()
    print(global_pin_memory(0, "cpu"))
    print(global_pin_memory(1, "cpu"))
    print(global_pin_memory(1, "cuda"))
    print()
    print(print(global_pin_memory(0, "cuda:0")))
    print(print(global_pin_memory(1, "cuda:0")))


if __name__ == "__main__":
    test_load_pin_memory_global()
