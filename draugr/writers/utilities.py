#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import os
import random

import numpy
import torch

__author__ = "cnheider"
__doc__ = ""


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def metrics(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
