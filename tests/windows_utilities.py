#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30-09-2020
           """

from draugr.windows_utilities import WindowsSettings


def test_get_dark_mode():
    ws = WindowsSettings()
    dark_mode = ws.get_dark_mode()
    print(dark_mode)
