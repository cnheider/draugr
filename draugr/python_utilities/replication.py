#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

__all__ = ["replicate", "BroadcastNone"]

from typing import Sequence, Union

import sorcery
from sorcery.core import FrameInfo

from warg import Number


def replicate(x: Union[Sequence, Number], times: int = 2) -> Sequence:
    """
    if not tuple

    :param times:
    :type times:
    :param x:
    :type x:
    :return:
    :rtype:"""
    if isinstance(x, Sequence):
        if len(x) == times:
            return x
    return (x,) * times


class BroadcastNone(type(None).__class__):
    @sorcery.spell
    def __new__(frame_info: FrameInfo, cls):
        num_assignments = len(frame_info.assigned_names(allow_one=True)[0])
        if num_assignments > 1:
            return (None,) * num_assignments
        return None


if __name__ == "__main__":

    def asdaa() -> None:
        """
        :rtype: None
        """
        print(replicate(2))
        print(replicate(2, 4))

        print(replicate((2, 3)))
        print(replicate((2, 3), times=4))

    def asd213aa() -> None:
        """
        :rtype: None
        """

        assd = BroadcastNone()
        ajis, sad, asd = BroadcastNone()

        print(assd)
        print(ajis, sad, asd)

    asd213aa()
