#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30-09-2020
           """

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)
from draugr.os_utilities.windows_utilities import WindowsSettings


def test_get_dark_mode():
    ws = WindowsSettings()
    dark_mode = ws.get_dark_mode()
    print(dark_mode)
    assert isinstance(dark_mode, bool)
