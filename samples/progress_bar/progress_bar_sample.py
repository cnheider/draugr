#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30-12-2020
           """

if __name__ == "__main__":

    from draugr.tqdm_utilities import progress_bar

    for a in progress_bar(range(100)):
        print(a)
