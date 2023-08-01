#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)

from draugr.metrics import MetricSummary

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


def test_summary_construction():
    MetricSummary()
