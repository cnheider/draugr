#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/03/2020
           """

from itertools import cycle
from typing import Iterable


def busy_indicator(
    *,
    stream: callable = print,
    indicator_interval: int = 1,
    phases: Iterable[str] = ("◑", "◒", "◐", "◓"),
) -> Iterable:
    """
    You can choose arbitrary phases like ['|','/','-','\\']

    :param stream:
    :type stream:
    :param indicator_interval:
    :type indicator_interval:
    :param phases:
    :type phases:
    :return:
    :rtype:"""

    phases = cycle(phases)
    i = 0
    while True:
        if i % indicator_interval == 0:
            stream(f"{next(phases)}", end="\r", flush=True)
        yield i
        i += 1


if __name__ == "__main__":
    import time

    for i in busy_indicator(indicator_interval=10):
        time.sleep(0.1)
        if i > 100:
            break
