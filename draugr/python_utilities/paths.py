#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/03/2020
           """

import pathlib


def ensure_existence(out: pathlib.Path):
    if not out.exists():
        out.mkdir(parents=True)
