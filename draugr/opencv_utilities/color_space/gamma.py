#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

import numpy


def gamma_correct_fast_to_byte(image):
    return ((image ** 0.454545) * 255).astype(numpy.uint8)


def gamma_correct_float_to_byte(image, gamma=2.2):
    return ((image ** (1.0 / gamma)) * 255).astype(numpy.uint8)


def linear_correct_float_to_byte(image, gamma=2.2):
    return ((image ** gamma) * 255).astype(numpy.uint8)


def linear_correct_byte(image, gamma=2.2):
    return gamma_correct_float_to_byte(image / 255, gamma)


def gamma_correct_byte(image, gamma=2.2):
    return gamma_correct_float_to_byte(image / 255, gamma)
