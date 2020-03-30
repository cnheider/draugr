#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/03/2020
           """

__all__ = ["compute_color_for_labels"]


def compute_color_for_labels(label, palette=(2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)):
    """
Simple function that adds fixed color depending on the class
"""
    return tuple([int((p * (label ** 2 - label + 1)) % 255) for p in palette])