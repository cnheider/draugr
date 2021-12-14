#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-05-2021
           """

import pandas

__all__ = [
    "nested_dict_to_two_level_column_df",
    "nested_dict_to_four_level_column_df",
    "nested_dict_to_three_level_column_df",
]


def nested_dict_to_two_level_column_df(df):  # TODO: MAKE ANY LEVEL!
    """

    :param df:
    :return:
    """
    cols, data = [], []
    index = None
    for k, v in df.items():
        if not isinstance(v, dict):
            cols.append((k, 0))
            data.append(v)
        else:
            for k2, v2 in v.items():
                cols.append((k, k2))
                data.append(v2)
                if index is None:
                    index = v2.index

    return pandas.DataFrame(
        list(zip(*data)), columns=pandas.MultiIndex.from_tuples(cols), index=index
    )


def nested_dict_to_three_level_column_df(df):  # TODO: MAKE ANY LEVEL!
    """

    :param df:
    :return:
    """
    cols, data = [], []
    index = None
    for k, v in df.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                cols.append((k, k2, k3))
                data.append(v3)
                if index is None:
                    index = v3.index

    return pandas.DataFrame(
        list(zip(*data)), columns=pandas.MultiIndex.from_tuples(cols), index=index
    )


def nested_dict_to_four_level_column_df(df):  # TODO: MAKE ANY LEVEL!
    """

    :param df:
    :return:
    """
    cols, data = [], []
    index = None
    for k, v in df.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                for k4, v4 in v3.items():
                    cols.append((k, k2, k3, k4))
                    data.append(v4)
                    if index is None:
                        index = v4.index

    return pandas.DataFrame(
        list(zip(*data)), columns=pandas.MultiIndex.from_tuples(cols), index=index
    )
