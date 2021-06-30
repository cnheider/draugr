#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.python_utilities.styling import (
    COLORS,
    PrintStyle,
    generate_style,
    get_terminal_size,
    scale,
)

__author__ = "Christian Heider Nielsen"

from typing import Sequence, Dict

import numpy

# sys.stdout.write(generate_style(u'Draugr Ûnicöde Probe\n', underline=True, italic=True))

__all__ = ["terminal_plot", "styled_terminal_plot_stats_shared_x"]


def terminal_plot(
    y: Sequence,
    *,
    x: Sequence = None,
    title: str = "Values",
    rows=None,
    columns=None,
    percent_size=(0.80, 0.80),
    x_offsets=(1, 1),
    y_offsets=(1, 1),
    printer=print,
    print_summary=True,
    plot_character="\u2981",
    print_style: PrintStyle = None,
    border_size=1,
):
    """
    x, y list of values on x- and y-axis
    plot those values within canvas size (rows and columns)"""

    num_y = len(y)
    if num_y == 0:
        return

    if x:
        if len(x) != num_y:
            raise ValueError(
                f"x argument must match the length of y, got x:{len(x)} and y:{num_y}"
            )
    else:
        x = range(num_y)

    if not rows or not columns:
        terminal_size = get_terminal_size()
        columns = terminal_size.columns
        rows = terminal_size.rows

    if percent_size:
        columns, rows = int(columns * percent_size[0]), int(rows * percent_size[1])

    # Scale points such that they fit on canvas
    drawable_columns = columns - sum(x_offsets) - border_size * 2
    drawable_rows = rows - sum(y_offsets) - border_size * 2

    x_scaled = scale(x, drawable_columns)
    y_scaled = scale(y, drawable_rows)

    # Create empty canvas
    canvas = [[" " for _ in range(columns)] for _ in range(rows)]

    # Create borders
    for iy in range(1, rows - 1):
        canvas[iy][0] = "\u2502"
        canvas[iy][columns - 1] = "\u2502"
    for ix in range(1, columns - 1):
        canvas[0][ix] = "\u2500"
        canvas[rows - 1][ix] = "\u2500"

    canvas[0][0] = "\u250c"
    canvas[0][columns - 1] = "\u2510"
    canvas[rows - 1][0] = "\u2514"
    canvas[rows - 1][columns - 1] = "\u2518"

    # Add scaled points to canvas
    y_offsets_start = border_size + y_offsets[0]
    x_offsets_start = border_size + x_offsets[0]
    for ix, iy in zip(x_scaled, y_scaled):
        y_ = y_offsets_start + (drawable_rows - iy)
        x_ = x_offsets_start + ix
        canvas[y_][x_] = plot_character

    print("\n")
    # Print rows of canvas
    for row in ["".join(row) for row in canvas]:
        if print_style:
            printer(print_style(row))
        else:
            printer(row)

    # Print scale
    if print_summary:
        summary = (
            f"{title} - (min, max): x({min(x)}, {max(x)}), y({min(y)}, {max(y)})\n"
        )
        if print_style:
            printer(print_style(summary))
        else:
            printer(summary)


def styled_terminal_plot_stats_shared_x(stats, *, styles=None, **kwargs):
    """

    :param stats:
    :type stats:
    :param styles:
    :type styles:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:"""
    if styles is None:
        styles = [
            generate_style(color=color, highlight=True)
            for color, _ in zip(COLORS.keys(), range(len(stats)))
        ]
    return terminal_plot_stats_shared_x(stats, styles=styles, **kwargs)


def terminal_plot_stats_shared_x(
    stats: Dict,
    *,
    x: Sequence = None,
    styles=None,
    printer=print,
    margin=0.25,
    summary=True,
):
    """

    :param stats:
    :type stats:
    :param x:
    :type x:
    :param styles:
    :type styles:
    :param printer:
    :type printer:
    :param margin:
    :type margin:
    :param summary:
    :type summary:"""
    num_stats = len(stats)

    y_size = (1 - margin) / num_stats

    if styles:
        if len(styles) != num_stats:
            raise ValueError(
                f"styles argument must match the length of stats, got styles:{len(styles)} and "
                f"stats:{num_stats}"
            )
    else:
        styles = [None for _ in range(num_stats)]

    for (key, stat), sty in zip(stats.items(), styles):
        terminal_plot(
            stat.running_value,
            title=key,
            x=x,
            printer=printer,
            print_style=sty,
            percent_size=(1, y_size),
            print_summary=summary,
        )


if __name__ == "__main__":
    terminal_plot(numpy.tile(range(9), 4), plot_character="o")
