#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr import recycle, batched_recycle

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def test_unzipper_generator1():
    a = range(9)
    batch_size = 3
    for a, b, c in zip(range(18), batched_recycle(a, batch_size)):
        assert [b_ in a for b_ in b]
    assert i == 17
