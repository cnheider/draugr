#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
NOT VERY USEFUL! as an equals chain does the same thing as this function 

           Created on 12-05-2021
           """

__all__ = ["BroadcastNone"]

import sorcery
from sorcery.core import FrameInfo


class BroadcastNone(type(None).__class__):
    """description"""

    @sorcery.spell
    def __new__(frame_info: FrameInfo, cls):
        num_assignments = len(frame_info.assigned_names(allow_one=True)[0])
        if num_assignments > 1:
            return (None,) * num_assignments
        return None


if __name__ == "__main__":

    def asd213aa() -> None:
        """
        :rtype: None
        """

        assd = BroadcastNone()
        ajis, sad, asd = BroadcastNone()

        print(assd)
        print(ajis, sad, asd)

    def asd213a2a() -> None:
        """
        :rtype: None
        """

        # ajis = sad = asd = BroadcastNone() # DOES NOT WORK!
        ajis = sad = asd = None

        print(ajis, sad, asd)

    asd213aa()
    asd213a2a()
