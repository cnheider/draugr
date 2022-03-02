#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14/01/2020
           """

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()
try:
    from .bounding_boxes import *
    from .color_space import *
    from .cv2_transforms import *
    from .raster_sequences import *
    from .opencv_draw import *
    from .opencv_drawing_utilities import *
    from .resize import *
    from .async_video_stream import *
    from .enums import *
    from .windows import *

    # from .color_conversion_enum import *
except ImportError as ix:
    this_package_name = Path(__file__).parent.name
    this_package_reqs = (
        Path(__file__).parent.parent.parent
        / "requirements"
        / f"requirements_{this_package_name}.txt"
    )
    print(
        f"Make sure requirements is installed for {this_package_name}, see {this_package_reqs}"
    )  # TODO: PARSE WHAT is missing and print
    raise ix
