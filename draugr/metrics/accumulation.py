#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 16/02/2020
           """

import math

__all__ = ["lambda_accumulator", "mean_accumulator", "total_accumulator"]

from typing import Generator

from warg import Number


def lambda_accumulator(start_value=None, lambd: float = 0.99) -> Generator:
    """

    :param start_value:
    :type start_value:
    :param lambd:
    :type lambd:
    :return:
    :rtype:"""
    assert 0 <= lambd <= 1

    def lambda_accumulator_(n: Number = start_value):
        """

        :param n:
        :type n:"""
        while True:
            new_n = yield n
            if new_n is not None:
                if n is not None:
                    n = n * lambd + new_n * (1 - lambd)
                else:
                    n = new_n

    acc = lambda_accumulator_()
    acc.send(None)

    return acc


def mean_accumulator(start_value: Number = None) -> Generator:
    """

    :param start_value:
    :type start_value:
    :return:
    :rtype:"""

    def mean_accumulator_(n: Number = start_value):
        """

        :param n:
        :type n:"""
        if n is not None:
            num = 1
        else:
            num = 0
        while True:
            new_n = yield n
            if new_n is not None:
                num += 1
                if n is not None:
                    n = n + (new_n - n) / num
                else:
                    n = new_n

    acc = mean_accumulator_()
    acc.send(None)

    return acc


def total_accumulator(start_value: Number = 0) -> Generator:
    """

    :param start_value:
    :type start_value:
    :return:
    :rtype:"""

    def total_accumulator_(total=start_value):
        """

        :param total:
        :type total:"""
        while True:
            a = yield total
            if a:
                total += a

    acc = total_accumulator_()
    acc.send(None)

    return acc


if __name__ == "__main__":
    samples = 10000

    def stest_1() -> None:
        """
        :rtype: None
        """
        lmbd_acc = lambda_accumulator()

        print(next(lmbd_acc))

        for i in range(samples):
            lmbd_acc.send(math.cos(i))
            if i < 5 or i > samples - 5:
                print(next(lmbd_acc))

    def stest_2() -> None:
        """
        :rtype: None
        """
        mean_acc = mean_accumulator()

        print(next(mean_acc))

        for i in range(samples):
            mean_acc.send(math.cos(i))
            if i < 5 or i > samples - 5:
                print(next(mean_acc))

    def stest_3() -> None:
        """
        :rtype: None
        """
        mean_acc = mean_accumulator()
        print(next(mean_acc))

        for i in range(samples):
            mean_acc.send(1)
            if i < 5 or i > samples - 5:
                print(next(mean_acc))

    def stest_4() -> None:
        """
        :rtype: None
        """
        lmbd_acc = lambda_accumulator()

        print(next(lmbd_acc))

        for i in range(samples):
            lmbd_acc.send(1)
            if i < 5 or i > samples - 5:
                print(next(lmbd_acc))

    def stest_lambda_zero() -> None:
        """
        :rtype: None
        """
        lmbd_acc = lambda_accumulator(lambd=0)

        print(next(lmbd_acc))

        for i in range(samples):
            lmbd_acc.send(1)
            if i < 5 or i > samples - 5:
                print(next(lmbd_acc))

    def stest_lambda_zero_input() -> None:
        """
        :rtype: None
        """
        lmbd_acc = lambda_accumulator()

        print(next(lmbd_acc))

        for i in range(samples):
            lmbd_acc.send(1)
            if i < 5 or i > samples - 5:
                print(next(lmbd_acc))

    def stest_5() -> None:
        """
        :rtype: None
        """
        total = total_accumulator()

        print(next(total))

        for i in range(samples):
            total.send(1)
            if i < 5 or i > samples - 5:
                print(next(total))
        print(next(total))

    def stest_6() -> None:
        """
        :rtype: None
        """
        total = total_accumulator()

        print(next(total))

        for i in range(samples):
            total.send(i)
            if i < 5 or i > samples - 5:
                print(next(total))
        print(next(total))

    stest_1()
    stest_2()
    stest_3()
    stest_4()
    stest_5()
    stest_6()
