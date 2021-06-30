#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from draugr.writers.writer import Writer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""
__all__ = ["ConsoleWriter"]


class ConsoleWriter(Writer):
    """ """

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        pass

    def _open(self):
        return self

    def _scalar(self, tag: str, value: float, step: int):
        print(f"{step} [{tag}] {value}")


if __name__ == "__main__":

    with ConsoleWriter() as w:
        for i in range(10):
            w.scalar("lol", i)

    print()
    with ConsoleWriter(interval=0) as w:
        for i in range(10):
            w.scalar("lol", i)
    print()
    with ConsoleWriter(interval=-1) as w:
        for i in range(10):
            w.scalar("lol", i)
    print()
    with ConsoleWriter(interval=None) as w:
        for i in range(10):
            w.scalar("lol", i)
    print()
    with ConsoleWriter(interval=2) as w:
        for i in range(10):
            w.scalar("lol", i)
