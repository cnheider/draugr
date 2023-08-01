#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)
from draugr.python_utilities import recycle

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def test_recycle1():
    a = range(9)
    for i, b in zip(range(18), recycle(a)):
        assert b in a
    assert i == 17
