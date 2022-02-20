#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-05-2021
           """

from typing import Any

import numpy
import pandas

__all__ = ["add_index_level"]


def add_index_level(
    old_index: pandas.Index, value: Any, name: str = None, loc: int = 0
) -> pandas.MultiIndex:
    """
    Expand a (multi)index by adding a level to it.

    :param old_index: The index to expand
    :param name: The name of the new index level
    :param value: Scalar or list-like, the values of the new index level
    :param loc: Where to insert the level in the index, 0 is at the front, negative values count back from the rear end
    :return: A new multi-index with the new level added
    """

    def _handle_insert_loc(loc: int, n: int) -> int:
        """
        Computes the insert index from the right if loc is negative for a given size of n.
        """
        return n + loc + 1 if loc < 0 else loc

    loc = _handle_insert_loc(loc, len(old_index.names))
    old_index_df = old_index.to_frame()
    old_index_df.insert(loc, name, value)
    new_index_names = list(
        old_index.names
    )  # sometimes new index level names are invented when converting to a df,
    new_index_names.insert(loc, name)  # here the original names are reconstructed
    new_index = pandas.MultiIndex.from_frame(old_index_df, names=new_index_names)
    return new_index


if __name__ == "__main__":

    def stest_add_index_level() -> None:
        """
        :rtype: None
        """
        df = pandas.DataFrame(data=numpy.random.normal(size=(6, 3)))
        i1 = add_index_level(df.index, "foo")

        # it does not invent new index names where there are missing
        assert [None, None] == i1.names

        # the new level values are added
        assert numpy.all(i1.get_level_values(0) == "foo")
        assert numpy.all(i1.get_level_values(1) == df.index)

        # it does not invent new index names where there are missing
        i2 = add_index_level(i1, ["x", "y"] * 3, name="xy", loc=2)
        i3 = add_index_level(i2, ["a", "b", "c"] * 2, name="abc", loc=-1)
        assert [None, None, "xy", "abc"] == i3.names

        # the new level values are added
        assert numpy.all(i3.get_level_values(0) == "foo")
        assert numpy.all(i3.get_level_values(1) == df.index)
        assert numpy.all(i3.get_level_values(2) == ["x", "y"] * 3)
        assert numpy.all(i3.get_level_values(3) == ["a", "b", "c"] * 2)

        # df.index = i3
        # print()
        # print(df)

    stest_add_index_level()
