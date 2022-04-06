#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 12-05-2021
           """

__all__ = [
    "min_interval_wrapper",
    "min_interval_wrapper_global",
    "max_frequency",
    "wrap_args",
]

from collections import namedtuple
from typing import MutableMapping, Tuple, Any

import wrapt


def min_interval_wrapper(f: callable, min_interval: int = 100) -> callable:
    """
    to ensure that a function is now being called more often than max_freq, TODO: use proper naming for the interval
    :param f:
    :param min_interval:
    :return:
    """

    def s(
        last_exec: int, *, step_i: int, verbose: bool = False, **kwargs: MutableMapping
    ) -> int:
        """

        :param last_exec:
        :param step_i:
        :param verbose:
        :param kwargs:
        :return:
        """
        if verbose:
            print(f"{f, last_exec, step_i, min_interval}")
        if step_i - last_exec >= min_interval:
            f(step_i=step_i, verbose=verbose, **kwargs)
            return step_i
        return last_exec

    return s


max_frequency_wrapper = min_interval_wrapper

_GLOBAL_COUNTERS = {}


def min_interval_wrapper_global(f: callable, min_interval: int = 100) -> callable:
    """
    to ensure that a function is now being called more often than max_freq, TODO: use proper naming for the interval
    :param f:
    :param min_interval:
    :return:
    """

    _GLOBAL_COUNTERS[f] = 0

    def s(*, step_i: int, verbose: bool = False, **kwargs: MutableMapping) -> None:
        """

        :param step_i:
        :param verbose:
        :param kwargs:
        """
        if verbose:
            print(f"{f, _GLOBAL_COUNTERS[f], step_i, min_interval}")
        if step_i - _GLOBAL_COUNTERS[f] >= min_interval:
            _GLOBAL_COUNTERS[f] = step_i
            f(step_i=step_i, verbose=verbose, **kwargs)

    return s


max_frequency_wrapper_global = min_interval_wrapper_global


def max_frequency(key: Any, min_interval: int = 100, verbose: bool = False) -> callable:
    """
    initially returns recallable func later bools
        :param f:
    :param min_interval:
    :return:
    """

    if key in _GLOBAL_COUNTERS:
        if verbose:
            print(f"{key, _GLOBAL_COUNTERS[key], min_interval}")
        if _GLOBAL_COUNTERS[key] >= min_interval:
            _GLOBAL_COUNTERS[key] = 1
            return True
        _GLOBAL_COUNTERS[key] += 1
        return False
    else:
        _GLOBAL_COUNTERS[key] = min_interval

        def s() -> bool:
            """

            :param step_i:
            :param verbose:
            :param kwargs:
            """
            if verbose:
                print(f"{key, _GLOBAL_COUNTERS[key], min_interval}")
            if _GLOBAL_COUNTERS[key] >= min_interval:
                _GLOBAL_COUNTERS[key] = 1
                return True
            _GLOBAL_COUNTERS[key] += 1
            return False

        return s


def wrap_args(n_tuple: namedtuple):
    """

    :param n_tuple:
    :type n_tuple:
    :return:
    :rtype:"""

    @wrapt.decorator(adapter=n_tuple)
    def wrapper(wrapped, instance, args, kwargs):
        """

        :param wrapped:
        :type wrapped:
        :param instance:
        :type instance:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:"""
        if isinstance(args[0], n_tuple):
            n = args[0]
        else:
            n = n_tuple(*args, **kwargs)
        return wrapped(n)

    return wrapper


def str_to_bool(s: str, preds: Tuple[str, ...] = ("true", "1")) -> bool:
    """


    :param preds:
    :param s:
    :return:"""
    return s.lower() in preds


str2bool = str_to_bool

if __name__ == "__main__":

    c = namedtuple("C", ("a", "b"))

    @wrap_args(c)
    def add(v):
        """

        :param v:
        :type v:
        :return:
        :rtype:"""
        return v.a + v.b

    def add2(a, b):
        """

        :param a:
        :type a:
        :param b:
        :type b:
        :return:
        :rtype:"""
        return a + b

    h = add(2, 2)
    print(h)

    j = add(c(1, 4))
    print(j)

    wq = add2(2, 4)
    print(wq)

    wc = add2(*c(4, 3))
    print(wc)

    def a(step_i, **kwargs):
        """

        :param step_i:
        :param kwargs:
        """
        print(step_i)

    def uhsud() -> None:
        """
        :rtype: None
        """
        b = min_interval_wrapper(a)
        c = 0
        for i in range(1000 + 1):
            c = b(c, step_i=i)

    def uhsud23() -> None:
        """
        :rtype: None
        """
        from random import random

        b = min_interval_wrapper_global(a)

        for i in range(1000 + 1):
            if random() > 0.8:
                b(step_i=i)

    def uhsud123() -> None:
        """
        :rtype: None
        """
        from random import random

        b = min_interval_wrapper_global(a, 0)

        for i in range(1000 + 1):
            if random() > 0.8 or True:
                b(step_i=i)

    # uhsud123()

    def iuhasd():
        from random import random

        a = 0
        for i in range(1000 + 1):
            if True:
                if max_frequency("key1", 100):
                    a += 1
                    print(i, a)

    def iuhasd2():
        from random import random

        f = max_frequency("key2", 100)
        a = 0
        for i in range(1000 + 1):
            if f():
                a += 1
                print(i, a)

    iuhasd()
    iuhasd2()
