#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

__all__ = ["MATLAB_ENGINE", "start_engine"]

from typing import Any

MATLAB_ENGINE = None


def start_engine() -> Any:
    """ """
    return
    global MATLAB_ENGINE
    if not MATLAB_ENGINE:
        from matlab import engine

        print("Starting Matlab")
        if True:
            MATLAB_ENGINE = engine.start_matlab()
        else:
            future = engine.start_matlab(background=True)
            MATLAB_ENGINE = future.result()
        print("Matlab started")
    else:
        print("Matlab already running")

    return MATLAB_ENGINE


if __name__ == "__main__":
    engine = start_engine()
    print(engine)
    print(MATLAB_ENGINE)
