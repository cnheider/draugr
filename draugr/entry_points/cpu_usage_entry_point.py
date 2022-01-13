#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 16-09-2020
           """


def main() -> None:
    """
    :rtype: None
    """
    # from draugr import TerminalPlotWriter
    from draugr.drawers import SeriesScrollPlot
    import psutil

    # psutil.virtual_memory() # gives an object with many fields

    # dict(psutil.virtual_memory()._asdict()) # you can convert that object to a dictionary

    # psutil.virtual_memory().percent # you can have the percentage of used RAM

    # psutil.virtual_memory().available * 100 / psutil.virtual_memory().total # you can calculate percentage of available memory

    s = SeriesScrollPlot(window_length=100, reverse=False, overwrite=True)
    while True:
        s.draw(psutil.cpu_percent())

    # with TerminalPlotWriter() as w:
    #  while True:


#     w.scalar("cpu", psutil.cpu_percent())


if __name__ == "__main__":
    main()
