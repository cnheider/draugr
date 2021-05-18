#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 05-03-2021
           """

__all__ = ["line_styles", "marker_styles"]

line_styles = (
    "-",  # solid
    "--",  # dashed
    ":",  # dotted
    "-.",  # dash-dot
    (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
)
marker_styles = (
    "",  # Nothing
    # ",", # pixel
    "^",  # triangle
    "o",  # o
    "x",  # x
    "s",  # square
    # '*', # star
    "|",  # vline
)
