#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-01-2021
           """

__all__ = ["duplicate_columns"]

from typing import List

import pandas
from pandas.core.dtypes.missing import array_equivalent


def duplicate_columns(frame: pandas.DataFrame) -> List[str]:
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                ja = vs.iloc[:, j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups
