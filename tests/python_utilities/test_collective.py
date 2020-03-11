#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/03/2020
           """

from draugr import identity, prod, sink
from draugr.python_utilities.debug import evaluate_context


def test_a():
    print(evaluate_context(identity, "str"))
    print(evaluate_context(identity, 2))
    print(evaluate_context(identity, 2.2))

    print(evaluate_context(prod, (2, 2)))

    print(evaluate_context(prod, (2.2, 2.2)))

    print(evaluate_context(prod, (2, 2.2)))

    print(evaluate_context(prod, (2.2, 2)))

    print(evaluate_context(sink, (2, 2), face=(2.2, 2)))
