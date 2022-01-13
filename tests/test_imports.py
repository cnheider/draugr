#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/08/2020
           """

__all__ = []


def test_import():
    import draugr

    print(draugr.__version__)


def test_import_torch_utilities():
    import draugr.torch_utilities

    print(draugr.torch_utilities.__author__)


def test_import_numpy_utilities():
    import draugr.numpy_utilities

    print(draugr.numpy_utilities.__author__)


def test_import_multiprocessing_utilities():
    import draugr.multiprocessing_utilities

    print(draugr.multiprocessing_utilities.__author__)


def test_import_pandas_utilities():
    import draugr.pandas_utilities

    print(draugr.pandas_utilities.__author__)


def test_import_opencv_utilities():
    import draugr.opencv_utilities

    print(draugr.opencv_utilities.__author__)


def test_import_drawers():
    import draugr.drawers

    print(draugr.drawers.__author__)


def test_import_opencv_utilities():
    import draugr.opencv_utilities

    print(draugr.opencv_utilities.__author__)


def test_import_entry_points():
    import draugr.entry_points

    print(draugr.entry_points.__author__)
