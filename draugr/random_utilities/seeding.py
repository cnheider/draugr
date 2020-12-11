#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11-12-2020
           """

import os
import random

import numpy

__all__ = ["seed_stack"]


def seed_stack(s: int = 23) -> None:
    python_seed(s)
    numpy_seed(s)
    torch_seeed(s)


def python_seed(s: int = 2318) -> None:
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def numpy_seed(s: int = 78213) -> None:
    numpy.random.seed(s)
