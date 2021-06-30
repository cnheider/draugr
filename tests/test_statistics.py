#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-12-2020
           """

from draugr.numpy_utilities import root_mean_square


def test_root_mean_square_signed(asad: int = 60):
    s = [i - asad // 2 for i in range(asad)]
    a = root_mean_square(s)
    # assert a


def test_root_mean_square_unsigned(asad: int = 60):
    s = [i + asad // 2 for i in range(asad)]
    a = root_mean_square(s)
    # assert a == mean(s)..
