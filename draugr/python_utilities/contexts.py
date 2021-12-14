#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09-02-2021
           """

import contextlib
import signal

__all__ = ["IgnoreInterruptSignal"]


class IgnoreInterruptSignal(contextlib.AbstractContextManager):
    def __enter__(self) -> bool:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # signal.getsignal() No sideeffect options for exit
        return True

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
