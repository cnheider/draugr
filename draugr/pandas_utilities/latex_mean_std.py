#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-05-2021
           """

import random
from functools import partial

import numpy
import pandas

from draugr import indent_lines
from draugr.pandas_utilities.formatting import pandas_mean_std_bold_formatter
from warg import Number, drop_unused_kws, passes_kws_to

__all__ = [
    "pandas_mean_std",
    "pandas_mean_std_to_str",
    "pandas_format_bold_max_row_latex",
    "pandas_to_latex_clean",
]


def pandas_mean_std(df: pandas.DataFrame, group_by: str) -> pandas.DataFrame:
    """

    :param df:
    :param group_by:
    :return:
    """
    return df.groupby(group_by).agg([numpy.mean, numpy.std])


def pandas_mean_std_to_str(
    df: pandas.DataFrame,
    *,
    mean_col: str = "mean",
    std_col: str = "std",
    precision: int = 3,
    level=1,
    axis=1,
) -> pandas.DataFrame:
    """
    latex \pm plus minus
    :param level:
    :param axis:
    :param std_col:
    :param mean_col:
    :param df:
    :param precision:
    :return:
    """
    return (
        df.xs(mean_col, axis=axis, level=level).round(precision).astype(str)
        + " \pm "
        + df.xs(std_col, axis=axis, level=level).round(precision).astype(str)
    )


def pandas_mean_std_latex_tabular(
    df: pandas.DataFrame,
    group_by: str,
    *,
    precision: int = 3,
    header_rotation: Number = 0,
) -> str:
    """

    :param df:
    :param group_by:
    :param precision:
    :param header_rotation:
    :return:
    """
    mean_std = pandas_mean_std(df, group_by)
    mean_std_str = pandas_mean_std_to_str(mean_std, precision=precision)

    return pandas_format_bold_max_column_latex(
        mean_std, mean_std_str, precision=precision, header_rotation=header_rotation
    )


def pandas_format_bold_max_column_latex(
    max_provider_df: pandas.DataFrame,
    entry_provider_df: pandas.DataFrame,
    *,
    precision: int = 3,
    header_rotation: Number = 0,
    max_colwidth: int = 1000,
):
    """

    :param max_provider_df:
    :param entry_provider_df:
    :param precision:
    :param header_rotation:
    :param max_colwidth:
    :return:
    """
    formatters = {
        column[0]: partial(
            pandas_mean_std_bold_formatter,
            value=max_provider_df[column].max(),
            precision=precision,
        )
        for column in max_provider_df.columns
        if "mean" in column
    }
    return pandas_to_latex_clean(
        entry_provider_df,
        header_rotation=header_rotation,
        max_colwidth=max_colwidth,
        formatters=formatters,
        precision=precision,
    )


def pandas_format_bold_max_row_latex(
    max_provider_df: pandas.DataFrame,
    entry_provider_df: pandas.DataFrame,
    *,
    precision: int = 3,
    header_rotation: Number = 0,
    max_colwidth=1000,
):
    """

    :param max_provider_df:
    :param entry_provider_df:
    :param precision:
    :param header_rotation:
    :param max_colwidth:
    :return:
    """
    formatters = {
        column[0]: partial(
            pandas_mean_std_bold_formatter,
            value=max_provider_df[column].max(),
            precision=precision,
        )
        for column in max_provider_df.columns
        if "mean" in column
    }

    return pandas_to_latex_clean(
        entry_provider_df,
        header_rotation=header_rotation,
        max_colwidth=max_colwidth,
        formatters=formatters,
    )


def pandas_to_latex_clean(
    entry_provider_df: pandas.DataFrame,
    *,
    header_rotation: Number = 0,
    precision=3,
    max_colwidth=1000,
    formatters=None,
) -> str:
    """

    :param entry_provider_df:
    :param header_rotation:
    :param precision:
    :param max_colwidth:
    :param formatters:
    :return:
    """
    if True:
        if not isinstance(entry_provider_df.columns, pandas.MultiIndex):
            entry_provider_df.columns = entry_provider_df.columns.str.replace("_", "\_")
        else:
            entry_provider_df.columns = [
                _name.replace("_", "\_") if _name else _name
                for _name in entry_provider_df.columns
            ]
            """
entry_provider_df.columns = pandas.MultiIndex(levels=[
[col.replace('_', '\_') for col in lvl]
for lvl in entry_provider_df.columns.levels
],
codes=entry_provider_df.columns.codes,
names=entry_provider_df.columns.names
)
"""

        if not isinstance(entry_provider_df.index, pandas.MultiIndex):
            entry_provider_df.index = entry_provider_df.index.str.replace("_", "\_")
        else:
            entry_provider_df.index = pandas.MultiIndex(
                levels=[
                    [col.replace("_", "\_") for col in lvl]
                    for lvl in entry_provider_df.index.levels
                ],
                codes=entry_provider_df.index.codes,
                names=entry_provider_df.index.names,
            )

        entry_provider_df.columns.names = [
            _name.replace("_", "\_") if _name else _name
            for _name in entry_provider_df.columns.names
        ]
        entry_provider_df.index.names = [
            _name.replace("_", "\_") if _name else _name
            for _name in entry_provider_df.index.names
        ]

    if formatters is None:
        formatters = {
            column: partial(
                pandas_mean_std_bold_formatter, value=None, precision=precision
            )
            for column in entry_provider_df.columns
        }

    if header_rotation:
        header = [
            f"\\rotatebox{{{header_rotation}}}" + "{" + "\_".join(c.split("_")) + "}"
            for c in entry_provider_df.columns
        ]
    else:
        header = True

    with pandas.option_context("max_colwidth", max_colwidth):
        return entry_provider_df.to_latex(
            index=True,
            index_names=[*entry_provider_df.index.names],
            escape=False,
            formatters=formatters,
            header=header,
        )  # .replace('textbackslash ', '').replace('\$', '$')


@drop_unused_kws
@passes_kws_to(pandas_mean_std_latex_tabular)
def pandas_mean_std_latex_table(
    df: pandas.DataFrame,
    group_by: str,
    tab_label: str,
    *,
    truncate_n_tabular_symbols: int = 2,
    **kwargs,
) -> str:
    """

    :param df:
    :param group_by:
    :param tab_label:
    :param truncate_n_tabular_symbols:
    :param kwargs:
    :return:
    """
    return (
        f"""\\begin{{table}}
  \caption{{{tab_label.replace('_', ' ')}}}
  \label{{tab:{tab_label}}}
  \centering
"""
        + indent_lines(pandas_mean_std_latex_tabular(df, group_by, **kwargs))[
            :-truncate_n_tabular_symbols
        ]
        + """\end{table}"""
    )


'''
def pandas_mean_std_latex_table8(df: pandas.DataFrame,
                                group_by: str,
                                precision: int = 3,
                                header_rotation: Number = 0) -> str:
  out = df.groupby(group_by).agg([numpy.mean, numpy.std])

  def bold_formatter(x, value):
    """Format a number in bold when (almost) identical to a given value.

    Args:
        x: Input number.

        value: Value to compare x with.

        num_decimals: Number of decimals to use for output format.

    Returns:
        String converted output.

    """
    # Consider values equal, when rounded results are equal
    # otherwise, it may look surprising in the table where they seem identical
    a = float(x.split('\pm')[0])

    if round(a, precision) == round(value, precision):
      return f"$\\mathbf{{{x}}}$"
    else:
      return f"${x}$"

  asds = {column[0]:partial(bold_formatter,
                            value=out[column].max()) for column in out.columns if 'mean' in column}

  out = (out.xs('mean', axis=1, level=1).round(precision).astype(str) + ' \pm ' + out.xs('std', axis=1, level=1).round(precision).astype(str))

  return out.to_latex(index=True,
                      escape=False,
                      formatters=asds,
                      header=[f'\\rotatebox{{{header_rotation}}}' + '{' + "\_".join(c.split("_")) + '}' for c in out.columns]
                      )  # .replace('textbackslash ', '').replace('\$', '$')
'''

if __name__ == "__main__":

    def asuhda() -> None:
        """
        :rtype: None
        """
        isjda = "dx_sijdai_iahjdaw-_sdioja_sakodwada_soakd_aoskdiojwd_s"
        df = pandas.DataFrame(
            numpy.random.randint(0, 100, size=(15, 3)),
            columns=("roc_asdoj", "au_c", "la"),
        )
        df["c_asd"] = [
            isjda[random.randint(0, 15) : random.randint(5, 30)] for _ in range(len(df))
        ]

        def asodjiasj():
            """ """
            print(pandas_mean_std(df, "c_asd"))

        def asidj():
            """ """
            print(pandas_mean_std_latex_table(df, "c_asd", "some_table"))

        asodjiasj()
        asidj()

    asuhda()
