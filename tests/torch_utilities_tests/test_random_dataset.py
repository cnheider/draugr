#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/12/2019
           """

from draugr.torch_utilities import RandomDataset


def test_random_dataset():
    s = (5, 5)
    assert RandomDataset(s, 10)[0].shape == (5, 5)
