#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02/03/2020
           """

__all__ = ["NoData"]


class NoData(Exception):
    """

"""

    def __init__(self, msg="No Data Available"):
        Exception.__init__(self, msg)
