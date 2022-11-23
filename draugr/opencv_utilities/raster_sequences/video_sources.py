#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 11/30/22
           """

__all__ = ["get_video_sources"]

from pathlib import Path
from typing import Set


def get_video_sources() -> Set[int]:
    """
    video_source_indices

    :return:
    :rtype:
    """
    char_devices = [f for f in Path("/dev").iterdir() if f.is_char_device()]
    return set(
        sorted(
            [int(dev.name[-1]) for dev in char_devices if dev.name.startswith("video")]
        )
    )
