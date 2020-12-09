#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """

import tqdm

from warg import passes_kws_to

__all__ = ["progress_bar"]


@passes_kws_to(tqdm.tqdm)
def progress_bar(iterable, desc="#", *, leave=False, **kwargs):
    return tqdm.tqdm(iterable, desc, leave=leave, **kwargs)
