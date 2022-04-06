#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "heider"
__doc__ = r"""

           Created on 01/02/2022
           """

__all__ = []

from draugr.os_utilities.linux_utilities.gtk_utilities import GtkThemePreferences

if __name__ == "__main__":
    print(GtkThemePreferences().prefer_dark_mode)
