#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

from pathlib import Path

from draugr import INCLUDE_PROJECT_READMES

if INCLUDE_PROJECT_READMES:
    with open(Path(__file__).parent / "README.md", "r") as this_init_file:
        __doc__ += this_init_file.read()
try:
    from .systemd_utilities import *
    from .user_utilities import *

    # from .gtk_utilities import *
except ImportError as ix:
    this_package_name = Path(__file__).parent.name
    this_package_reqs = (
        Path(__file__).parent.parent.parent.parent
        / "requirements"
        / f"requirements_{this_package_name}.txt"
    )
    print(
        f"Make sure requirements is installed for {this_package_name}, see {this_package_reqs}"
    )  # TODO: PARSE WHAT is missing and print
    raise ix
