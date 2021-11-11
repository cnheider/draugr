#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path
from random import randint
from time import time
from typing import Any, List, Mapping, Sequence, Union

import numpy

from warg import NOD, passes_kws_to

__author__ = "Christian Heider Nielsen"

import six

__all__ = [
    "COLORS",
    "DECORATIONS",
    "generate_style",
    "sprint",
    "PrintStyle",
    "scale",
    "get_terminal_size",
    "hyperlink_path",
    "hyperlink_url",
]

COLORS = NOD(
    red="31",
    green="32",
    yellow="33",
    # gray='30', #Black,
    blue="34",
    magenta="35",
    cyan="36",
    white="37",
    crimson="38",
)

DECORATIONS = NOD(
    end="0",
    bold="1",
    dim="2",
    italic="3",
    underline="4",
    underline_end="24",  # '4:0',
    double_underline="21",  # '4:2'
    # double_underline_end='24',  # '4:0'
    curly_underline="4:3",
    blink="5",
    reverse_colors="7",
    invisible="8",  # still copyable
    strikethrough="9",
    overline="53",
    hyperlink="8;;",
)


def hyperlink_url(
    link: str, link_id: str = None, style: bool = False, prefix: str = "https://"
) -> str:
    """ """
    if not link.startswith(prefix):
        if "://" in link:
            link = f'{prefix}{link.split("://")[1]}'
        else:
            link = f"{prefix}{link}"

    if not style:
        return link
    if not link_id:
        link_id = f"{time()}-{randint(0, 999999)}" if link else ""
    return f"\x1b]8;id={link_id};{link}\x1b\\{link}\x1b]8;;\x1b\\"


def hyperlink_path(
    link: Path, link_id: str = None, style: bool = False, linux: bool = True
) -> str:
    """ """
    if linux:
        link = f"file://{link.resolve()}"
    if not style:
        return link
    if not link_id:
        link_id = f"{time()}-{randint(0, 999999)}" if link else ""
    return f"\x1b]8;id={link_id};{link}\x1b\\{link}\x1b]8;;\x1b\\"


class PrintStyle(object):
    """ """

    def __init__(self, attributes_joined, end):
        self._attributes_joined = attributes_joined
        self._end = end

    def __call__(self, obj, *args, **kwargs) -> str:
        intermediate_repr = f"\x1b[{self._attributes_joined}m{obj}\x1b[{self._end}m"
        string = six.u(intermediate_repr)
        return string


def generate_style(
    obj: Any = None,
    *,
    color: Union[str, int] = "random",
    bold: bool = False,
    highlight: bool = False,
    underline: bool = False,
    italic: bool = False,
) -> Union[str, PrintStyle]:
    """

    :param obj:
    :type obj:
    :param color:
    :type color:
    :param bold:
    :type bold:
    :param highlight:
    :type highlight:
    :param underline:
    :type underline:
    :param italic:
    :type italic:
    :return:
    :rtype:"""
    attributes = []

    if color == "random":
        num = int(list(COLORS.values())[randint(0, len(COLORS) - 1)])
    elif color in COLORS:
        num = int(COLORS[color])
    elif color in COLORS.values():
        num = int(color)
    elif isinstance(color, int) and color < 39:
        num = color
    else:
        num = int(COLORS["white"])

    if highlight:
        num += 10

    attributes.append(six.u(f"{num}"))

    if bold:
        attributes.append(six.u(f'{DECORATIONS["bold"]}'))

    if underline:
        attributes.append(six.u(f'{DECORATIONS["underline"]}'))

    if italic:
        attributes.append(six.u(f'{DECORATIONS["italic"]}'))

    end = DECORATIONS["end"]

    attributes_joined = six.u(";").join(attributes)

    print_style = PrintStyle(attributes_joined, end)

    if obj:
        return print_style(obj)
    else:
        return print_style


@passes_kws_to(generate_style)
def sprint(obj: Any, print_kws: Mapping = {}, **kwargs) -> None:
    """
    Stylised print. Defaults to stdout
    Valid colors: gray, red, green, yellow, blue, magenta, cyan, white, crimson"""
    print(generate_style(obj, **kwargs), **print_kws)


def scale(x: Sequence, length: float) -> List[int]:
    """
    Scale points in 'x', such that distance between
    max(x) and min(x) equals to 'length'. min(x)
    will be moved to 0."""
    if type(x) is list:
        s = float(length) / (max(x) - min(x)) if x and max(x) - min(x) != 0 else length
    # elif type(x) is range:
    #  s = length
    else:
        s = (
            float(length) / (numpy.max(x) - numpy.min(x))
            if len(x) and numpy.max(x) - numpy.min(x) != 0
            else length
        )

    return [int((i - min(x)) * s) for i in x]


def get_terminal_size() -> NOD:
    """

    :return:
    :rtype:"""
    try:
        size = shutil.get_terminal_size()
        columns, rows = size.columns, size.lines
    except:
        rows, columns = (os.getenv("LINES", 25), os.getenv("COLUMNS", 80))

    rows, columns = int(rows), int(columns)

    return NOD(rows=rows, columns=columns)


if __name__ == "__main__":
    sprint(get_terminal_size())
    sprint(hyperlink_path(Path.home()))
    print(hyperlink_path(Path.home()))
    sprint(hyperlink_path(Path.home() / ".xprofile"))
    sprint(hyperlink_url("http://dr.dk"))
    sprint(hyperlink_url("dr.dk"))
