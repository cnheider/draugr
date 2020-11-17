#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any, Generator, Iterable, Iterator

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """

__all__ = ["unzip", "unzipper"]


def unzip(iterable: Iterable) -> Iterable:
    return zip(*iterable)


def unzipper(iterable: Iterable[Iterable]) -> Iterable:
    """
    Unzips an iterable of an iterable

    Be carefully has undefined and expected behaviour

    :param iterable:
    :return:"""

    def check_next_iter(iterable: Any) -> Any:
        if isinstance(iterable, Iterable):
            try:
                a = next(iter(iterable))
                if isinstance(a, Iterable):
                    return a
            except StopIteration:
                pass

    if isinstance(iterable, Iterable):
        check_a = check_next_iter(check_next_iter(deepcopy(iterable)))
        if check_next_iter(check_a):
            for a in iterable:
                yield unzipper(a)
        elif check_a:
            for a in iterable:
                yield unzip(a)
        else:
            for i in iterable:
                yield i
    return


if __name__ == "__main__":

    def recursive_eval(node: Any):
        if isinstance(node, (Iterable, Generator, Iterator)):
            gather = []
            for i in node:
                gather.append(recursive_eval(i))
            return gather
        return node

    def aasda():
        r = range(4)

        print(0)

        a = [[[*r] for _ in r] for _ in r]
        print(a)

        print(1)

        for _, assd in zip(r, unzipper(a)):
            print()
            print(recursive_eval(assd))
            print()

        for _, (a, *_) in zip(r, unzipper(a)):
            print()
            print(recursive_eval(a))
            print()

        print(2)

    def skad23():
        print(0)
        zippy_once = zip(range(6), range(3))
        dsadsa = list(deepcopy(zippy_once))
        zippy_twice = zip(dsadsa, dsadsa)
        zippy_twice_copy = deepcopy(zippy_twice)
        asds = list(deepcopy(zippy_twice_copy))
        zippy_trice = zip(asds, asds)
        zippy_trice_copy = deepcopy(zippy_trice)

        print(1)

        for aa in zippy_twice:
            print(recursive_eval(aa))

        print(2)

        for a1 in unzip(zippy_twice_copy):
            print(recursive_eval(a1))

        print(3)

        for a1 in unzip(zippy_once):
            print(recursive_eval(a1))

        print(4)

        for a1 in zippy_trice:
            print(recursive_eval(a1))

        print(5)

        for a1 in unzip(zippy_trice_copy):
            print(recursive_eval(a1))

        print(6)

    def skad():
        print(0)
        zippy_once = zip(zip(range(6), range(3)))
        zippy_once_copy = deepcopy(zippy_once)
        dsadsa = list(deepcopy(zippy_once))
        zippy_twice = zip(dsadsa, dsadsa)
        zippy_twice_copy = deepcopy(zippy_twice)
        asds = list(deepcopy(zippy_twice_copy))
        zippy_trice = zip(asds, asds)
        zippy_trice_copy = deepcopy(zippy_trice)
        asds2323 = list(deepcopy(zippy_trice_copy))
        zippy_quad = zip(asds2323, asds2323)
        zippy_quad_copy = deepcopy(zippy_quad)

        print(1)

        for aa in zippy_twice:
            print(recursive_eval(aa))

        print(2)

        for a1 in unzipper(zippy_twice_copy):
            print(recursive_eval(a1))

        print(3)

        for a1 in zippy_once_copy:
            print(recursive_eval(a1))

        print(4)

        for a1 in unzipper(zippy_once):
            print(recursive_eval(a1))

        print(5)

        for a1 in zippy_trice:
            print(recursive_eval(a1))

        print(6)

        for a1 in unzipper(zippy_trice_copy):
            print(recursive_eval(a1))

        print(7)

        for a1 in zippy_quad:
            print(recursive_eval(a1))

        print(8)

        for a1 in unzipper(zippy_quad_copy):
            print(recursive_eval(a1))

        print(9)

    aasda()

    print()
    print("asafasdw")
    print()

    skad()
    # skad23()
