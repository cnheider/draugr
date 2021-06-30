#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["metrics"]


def metrics(array: numpy.ndarray) -> collections.OrderedDict:
    """

    :param array:
    :type array:
    :return:
    :rtype:"""
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
