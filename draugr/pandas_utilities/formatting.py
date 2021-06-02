#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-05-2021
           """

__all__ = ["pandas_mean_std_bold_formatter"]


def pandas_mean_std_bold_formatter(entry, value, *, precision: int = 3):
    """Format a number in bold when (almost) identical to a given value.

    Args:
        entry: Input number.

        value: Value to compare x with.

        num_decimals: Number of decimals to use for output format.

    Returns:
        String converted output.

    """
    # Consider values equal, when rounded results are equal
    # otherwise, it may look surprising in the table where they seem identical
    mean_entry = float(entry.split("\pm")[0])

    if value is not None and round(mean_entry, precision) == round(value, precision):
        return f"$\\mathbf{{{entry}}}$"
    else:
        return f"${entry}$"
