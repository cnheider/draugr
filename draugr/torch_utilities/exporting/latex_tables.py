# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-06-2021
           """

from draugr.misc_utilities import latextable

table_row_name = "tablerow"

rows = [
    ["Rocket", "Organisation", "LEO Payload (Tonnes)", "Maiden Flight"],
    ["Saturn V", "NASA", "140", "1967"],
    ["Space Shuttle", "NASA", "24.4", "1981"],
    ["Falcon 9 FT-Expended", "SpaceX", "22.8", "2017"],
    ["Ariane 5 ECA", "ESA", "21", "2002"],
]


def asijd() -> None:
    """
    :rtype: None
    """
    from tabulate import tabulate

    print("\nTabulate Latex:")
    print(tabulate(rows, headers="firstrow", tablefmt="latex"))


def adasdassijd() -> None:
    """
    :rtype: None
    """
    from texttable import Texttable

    table = Texttable()
    table.set_cols_align(["c"] * 4)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    print("\nTexttable Table:")
    print(table.draw())

    print("\nTexttable Latex:")
    print(latextable.draw_latex(table, caption="A comparison of rocket features."))


def iasjduh() -> None:
    """
    :rtype: None
    """
    from texttable import Texttable

    table = Texttable()
    table.set_cols_align(["l", "r", "c"])
    table.set_cols_valign(["t", "m", "b"])
    table.add_rows(
        [
            ["Name", "Age", "Nickname"],
            ["Mr\nXavier\nHuon", 32, "Xav'"],
            ["Mr\nBaptiste\nClement", 1, "Baby"],
            ["Mme\nLouise\nBourgeau", 28, "Lou\n \nLoue"],
        ]
    )
    print(table.draw() + "\n")
    print(latextable.draw_latex(table, caption="An example table.") + "\n")

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(
        [
            "t",  # text
            "f",  # float (decimal)
            "e",  # float (exponent)
            "i",  # integer
            "a",
        ]
    )  # automatic
    table.set_cols_align(["l", "r", "r", "r", "l"])
    table.add_rows(
        [
            ["text", "float", "exp", "int", "auto"],
            ["abcd", "67", 654, 89, 128.001],
            ["efghijk", 67.5434, 0.654, 89.6, 12800000000000000000000.00023],
            ["lmn", 5e-78, 5e-78, 89.4, 0.000000000000128],
            ["opqrstu", 0.023, 5e78, 92.0, 12800000000000000000000],
        ]
    )
    print(table.draw() + "\n")
    print(
        latextable.draw_latex(
            table, caption="Another table.", label="table:another_table"
        )
        + "\n"
    )
    print(
        latextable.draw_latex(
            table,
            caption="A table with dropped columns.",
            label="table:dropped_column_table",
            drop_columns=["exp", "int"],
        )
    )


adasdassijd()
