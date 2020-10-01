#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/10/2020
           """

__all__ = []

from draugr.gtk_utilities import GtkThemePreferences


def main():
    with GtkThemePreferences() as a:
        a.prefer_dark_mode = not a.prefer_dark_mode


if __name__ == "__main__":
    main()
