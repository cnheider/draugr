#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """


import os
import psutil

process = psutil.Process(os.getpid())
print(process.memory_percent())
