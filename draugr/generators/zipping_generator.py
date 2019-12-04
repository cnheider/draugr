#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable

import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def unzipper(iterable: Iterable):
    """Unzips an iterable"""
    for a in iterable:
        yield list(zip(*a))
    return
