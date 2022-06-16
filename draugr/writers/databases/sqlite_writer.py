#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-03-2021
           """

from draugr.writers import Writer


class SqliteWriter(Writer):
    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        pass

    def _open(self):
        pass

    def _scalar(self, tag: str, value: float, step: int):
        pass
