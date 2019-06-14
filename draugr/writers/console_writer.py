#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.writers.writer import Writer

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""


class ConsoleWriter(Writer):
    def _close(self, exc_type, exc_val, exc_tb):
        pass

    def _open(self):
        return self

    def _scalar(self, tag: str, value: float, step: int):
        print(f"{step} [{tag}] {value}")


if __name__ == "__main__":

    with ConsoleWriter() as w:
        w.scalar("lol", 6)
