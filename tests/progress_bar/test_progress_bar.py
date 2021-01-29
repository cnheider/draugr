#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30-12-2020
           """

from draugr.tqdm_utilities import progress_bar


def test_progress_bar():
    for a in progress_bar(range(100), notifications=False):
        pass


def test_asudhweasijdq():
    sas = [f"a{a}" for a in range(5)]

    for a in progress_bar(sas, notifications=False):
        pass


def test_asudhwea213si23jdq():
    def sas(l):
        yield from range(l)

    for a in progress_bar(sas(129), notifications=False):
        pass


def test_asudhweasi23jdq():
    def sas():
        yield from range(100)

    for a in progress_bar(sas(), notifications=False):
        pass


def test_dsad3123():
    for a in progress_bar([2.13, 8921.9123, 923], notifications=False):
        pass


def test_dsad311231223():
    for a in progress_bar(
        (
            2.13,
            8921.9123,
            923,
            821738,
            782173,
            8912738124,
            8471827,
            661262,
            1111,
            2222,
            3333,
            4444,
            5555,
        ),
        notifications=False,
    ):
        pass


def test_ds12sadad311231223():
    for a in progress_bar(
        {
            2.13j,
            8921.9123j,
            923j,
            821738j,
            782173j,
            8912738124j,
            8471827j,
            661262j,
            1111j,
            2222j,
            3333j,
            4444j,
            5555j,
        },
        notifications=False,
    ):
        pass


def test_ds12s23():
    for a in progress_bar(
        [[2.13j], [8921.9123j], [923j], [821738j], [782173j]], notifications=False
    ):
        pass


def test_ds1saijd2s23():
    a1 = range(6)
    b2 = range(6)

    for _ in progress_bar(a1, notifications=False):
        pass

    for _ in progress_bar(b2, notifications=False):
        pass


def test_dict_items():
    from time import sleep

    class exp_v:
        Test_Sets = {v: v for v in range(9)}

    for a in progress_bar(exp_v.Test_Sets.items()):
        sleep(1)


if __name__ == "__main__":
    test_progress_bar()
    test_asudhweasijdq()
    test_asudhweasi23jdq()
    test_dsad3123()
    test_dsad311231223()
    test_ds12sadad311231223()
    test_ds12s23()
    test_asudhwea213si23jdq()
    test_ds1saijd2s23()
    test_dict_items()
