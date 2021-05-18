#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 12-05-2021
           """

__all__ = ["min_interval_wrapper", "min_interval_wrapper_global"]


def min_interval_wrapper(f: callable, min_interval: int = 100) -> callable:
    """
    to ensure that a function is now being called more often than max_freq, TODO: use proper naming for the interval
    :param f:
    :param min_interval:
    :return:
    """

    def s(last_exec, *, step_i, verbose: bool = False, **kwargs) -> int:
        if verbose:
            print(f"{f, last_exec, step_i, min_interval}")
        if step_i - last_exec >= min_interval:
            f(step_i=step_i, verbose=verbose, **kwargs)
            return step_i
        return last_exec

    return s


_GLOBAL_COUNTERS = {}


def min_interval_wrapper_global(f: callable, min_interval: int = 100) -> callable:
    """
    to ensure that a function is now being called more often than max_freq, TODO: use proper naming for the interval
    :param f:
    :param min_interval:
    :return:
    """

    _GLOBAL_COUNTERS[f] = 0

    def s(*, step_i, verbose: bool = False, **kwargs) -> None:
        if verbose:
            print(f"{f, _GLOBAL_COUNTERS[f], step_i, min_interval}")
        if step_i - _GLOBAL_COUNTERS[f] >= min_interval:
            _GLOBAL_COUNTERS[f] = step_i
            f(step_i=step_i, verbose=verbose, **kwargs)

    return s


if __name__ == "__main__":

    def a(step_i, **kwargs):
        print(step_i)

    def uhsud():
        b = min_interval_wrapper(a)
        c = 0
        for i in range(1000 + 1):
            c = b(c, step_i=i)

    def uhsud23():
        from random import random

        b = min_interval_wrapper_global(a)

        for i in range(1000 + 1):
            if random() > 0.8:
                b(step_i=i)

    def uhsud123():
        from random import random

        b = min_interval_wrapper_global(a, 0)

        for i in range(1000 + 1):
            if random() > 0.8 or True:
                b(step_i=i)

    uhsud123()
