#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.metrics import MetricCollection

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


def test_a():
    stats = MetricCollection(keep_measure_history=False)
    stats2 = MetricCollection(keep_measure_history=True)

    for i in range(10):
        stats.signal.append(i)
        stats2.signal.append(i)

    print(stats)
    print(stats.signal)
    print(stats.length)
    print(stats.length.measures)
    print(stats.signal.measures)
    print(stats.signal.variance)
    print(stats.signal.calc_moving_average())
    print(stats.signal.max)
    print(stats.signal.min)
    print("\n")
    print(stats2)
    print(stats2.signal.min)
