#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/10/22
           """

__all__ = []
from pathlib import Path


from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)
from draugr.python_utilities.iterators import BidirectionalIterator, prev


def test_bidirectional_iterator():
    a = BidirectionalIterator(iter([1, 2, 3, 4, 5, 6]))
    assert next(a) == 1
    assert next(a) == 2
    assert next(a) == 3
    assert next(a) == 4
    assert next(a) == 5
    assert next(a) == 6
    assert prev(a) == 5
    assert prev(a) == 4
    assert prev(a) == 3
    assert prev(a) == 2
    assert prev(a) == 1


if __name__ == "__main__":
    test_bidirectional_iterator()
