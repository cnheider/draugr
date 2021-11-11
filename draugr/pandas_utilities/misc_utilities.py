#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-01-2021
           """

__all__ = ["duplicate_columns", "ExportMethodEnum", "ChainedAssignmentOptionEnum"]

import enum
from typing import List

import pandas
from pandas.core.dtypes.missing import array_equivalent
from sorcery import assigned_names


class ExportMethodEnum(enum.Enum):
    """
    Available Pandas Dataframe Export methods"""

    (
        parquet,
        pickle,  # also 'dataframe'
        csv,
        hdf,
        sql,
        dict,
        excel,
        json,
        html,
        feather,
        latex,
        stata,
        gbq,
        records,
        string,
        clipboard,
        markdown,
        xarray,
    ) = assigned_names()


class ChainedAssignmentOptionEnum(enum.Enum):
    """
    from contextlib import suppress
    from pandas.core.common import SettingWithCopyWarning
    """

    (
        warn,  # the default, means a SettingWithCopyWarning is printed.
        raises,  # means pandas will raise a SettingWithCopyException you have to deal with.
    ) = assigned_names()
    none = None  # will suppress the warnings entirely.


def duplicate_columns(frame: pandas.DataFrame) -> List[str]:
    """ """
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    duplicates = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                ja = vs.iloc[:, j].values
                if array_equivalent(ia, ja):
                    duplicates.append(cs[i])
                    break

    return duplicates


if __name__ == "__main__":
    for e in ExportMethodEnum:
        print(e.value)
