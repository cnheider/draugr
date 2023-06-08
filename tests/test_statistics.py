#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-12-2020
           """

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)
from draugr.numpy_utilities import root_mean_square


def test_root_mean_square_signed(asad: int = 60):
    s = [i - asad // 2 for i in range(asad)]
    a = root_mean_square(s)
    # assert a


def test_root_mean_square_unsigned(asad: int = 60):
    s = [i + asad // 2 for i in range(asad)]
    a = root_mean_square(s)
    # assert a == mean(s)..
