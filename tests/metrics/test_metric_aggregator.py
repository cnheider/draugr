#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path


from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)
from draugr.metrics import MetricAggregator

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


def test_a():
    signals = MetricAggregator(keep_measure_history=False)

    for i in range(10):
        signals.append(i)

    print(signals)
    print(signals.measures)
    print(signals.variance)
    print(signals.calc_moving_average())
    print(signals.max)
    print(signals.min)
