#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15-10-2020
           """

import pyperf

runner = pyperf.Runner()
runner.timeit(
    name="sort a sorted list",
    stmt="sorted(s, key=f)",
    setup="f = lambda x: x; s = list(range(1000))",
)
