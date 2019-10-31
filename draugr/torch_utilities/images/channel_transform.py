#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """


def reverse_rgb_channel_transform(inp):
    inp = inp.transpose((1, 2, 0))
    inp = inp * 255.0
    inp = numpy.clip(inp, 0, 255).astype(numpy.uint8)
    return inp


def rgb_channel_transform(inp):
    inp = inp / 255.0
    inp = numpy.clip(inp, 0, 1)
    inp = inp.transpose((2, 0, 1))
    return inp
