#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27-05-2021
           """

from enum import Enum
from pathlib import Path
from typing import List, Union

from pandas.core.generic import NDFrame
from sorcery import assigned_names

from apppath import ensure_existence
from warg import Number

__all__ = [
    "color_highlight_extreme",
    "ColorEnum",
    "NDFrameExtremeEnum",
    "AttrEnum",
    "color_negative_red",
]


class ColorEnum(Enum):
    """ """

    (
        purple,
        cyan,
        yellow,
        red,
        green,
        blue,
        magenta,
        pink,
        orange,
        black,
        white,
    ) = assigned_names()


class NDFrameExtremeEnum(Enum):
    """ """

    min = NDFrame.min
    max = NDFrame.max


class AttrEnum(Enum):
    """ """

    color = "color: {}"
    bg = "background-color: {}"


def color_highlight_extreme(
    s: Union[NDFrame, Number],
    color: Union[str, ColorEnum] = ColorEnum.yellow,
    attr_template: Union[str, AttrEnum] = AttrEnum.bg,
    extreme_func: Union[NDFrameExtremeEnum, callable] = NDFrameExtremeEnum.min,
) -> Union[List[str], str, NDFrame]:
    """
    highlight the maximum in a Series yellow.
    """
    if isinstance(color, ColorEnum):
        color = (
            color.value.lower()
        )  # TODO: LOWER IN THE CURRENT CASE no matter assigned names
    if isinstance(attr_template, AttrEnum):
        attr_template = attr_template.value

    attr = attr_template.format(color)
    if isinstance(s, NDFrame):
        mask = s == extreme_func(s)
        if s.ndim == 1:  # Series from .apply(axis=0) or axis=1
            return [attr if v else "" for v in mask]
        return pandas.DataFrame(
            numpy.where(mask, attr, ""), index=s.index, columns=s.columns
        )  # from .apply(axis=None)
    return attr


def color_negative_red(value):
    """
    Colors elements in a dateframe
    green if positive and red if
    negative. Does not color NaN
    values.
    """

    if value < 0:
        color = "red"
    elif value > 0:
        color = "green"
    else:
        color = "black"

    return f"color: {color}"


if __name__ == "__main__":
    import numpy
    import pandas
    import seaborn

    numpy.random.seed(24)
    df = pandas.DataFrame({"A": numpy.linspace(1, 10, 10)})
    df = pandas.concat(
        [df, pandas.DataFrame(numpy.random.randn(10, 4), columns=list("BCDE"))], axis=1
    )
    df.iloc[3, 3] = numpy.nan
    df.iloc[0, 2] = numpy.nan

    cm = seaborn.light_palette("green", as_cmap=True)
    df = df.style.background_gradient(cmap=cm).highlight_null(
        null_color="red"
    )  # element wise
    # df.style.bar(subset=['A', 'B'], color='#d65f5f')
    # df.style.bar(subset=['A', 'B'], align='mid', color=['#d65f5f', '#5fba7d'])
    df = df.applymap(color_highlight_extreme)  # .format(None, na_rep="-")
    df = df.apply(color_highlight_extreme, color="darkorange")
    df = df.apply(
        color_highlight_extreme,
        extreme_func=NDFrameExtremeEnum.max,
        color="green",
        axis=None,
    )

    html_ = df.render()
    # html_ = df.to_html()
    # from IPython.display import display, HTML
    # display((HTML(),))
    # print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    with open(ensure_existence(Path("exclude")) / "style_test.html", "w") as f:
        f.write(html_)

    """
Styler.applymap(func) for elementwise styles

Styler.apply(func, axis=0) for columnwise styles

Styler.apply(func, axis=1) for rowwise styles

Styler.apply(func, axis=None) for tablewise styles

And crucially the input and output shapes of func must match. If x is the input then func(x).shape == x.shape.
"""
