#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/08/2020
           """

__all__ = []

import pytest

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)


def test_import():
    import draugr

    print(draugr.__version__)


@pytest.mark.skip
def test_import_dlib_utilities():
    import draugr.dlib_utilities

    print(draugr.dlib_utilities.__author__)


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


def test_import_writers():
    import draugr.writers

    print(draugr.writers.__author__)


def test_import_scipy_utilities():
    import draugr.scipy_utilities

    print(draugr.scipy_utilities.__author__)


@pytest.mark.skip
def test_import_matlab_utilities():
    import draugr.matlab_utilities

    print(draugr.matlab_utilities.__author__)


@pytest.mark.skip
def test_import_tensorboard_utilities():
    import draugr.tensorboard_utilities

    print(draugr.tensorboard_utilities.__author__)


def test_import_extensions():
    import draugr.extensions

    print(draugr.extensions.__author__)


def test_import_metrics():
    import draugr.metrics

    print(draugr.metrics.__author__)


def test_import_generators():
    import draugr.python_utilities.generators

    print(draugr.python_utilities.generators.__author__)


def test_import_os_utilities():
    import draugr.os_utilities

    print(draugr.os_utilities.__author__)


def test_import_python_utilities():
    import draugr.python_utilities

    print(draugr.python_utilities.__author__)


def test_import_random_utilities():
    import draugr.random_utilities

    print(draugr.random_utilities.__author__)


def test_import_stopping():
    import draugr.stopping

    print(draugr.stopping.__author__)


def test_import_visualisation():
    import draugr.visualisation

    print(draugr.visualisation.__author__)


def test_import_entry_points():
    import draugr.entry_points

    print(draugr.entry_points.__author__)
