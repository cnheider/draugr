#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-02-2021
           """

__all__ = [
    "monochrome_hatch_cycler",
    "simple_hatch_cycler",
    "monochrome_line_no_marker_cycler",
    "monochrome_line_cycler",
]

from draugr.visualisation.matplotlib_utilities.styles.hatching import (
    four_times_denser_hatch,
)
from draugr.visualisation.matplotlib_utilities.styles.lines import (
    line_styles,
    marker_styles,
)
from matplotlib import cycler

simple_hatch_cycler = cycler("hatch", four_times_denser_hatch)
monochrome_hatch_cycler = (
    cycler("color", "w")
    * cycler("facecolor", "w")
    * cycler("edgecolor", "k")
    * simple_hatch_cycler
)

monochrome_line_no_marker_cycler = cycler("color", ["k"]) * cycler(
    "linestyle", line_styles
)

monochrome_line_cycler = (
    cycler("color", ["k"])
    * cycler("linestyle", line_styles)
    * cycler("marker", marker_styles)
)


if __name__ == "__main__":
    print([a for _, a in zip(range(10), monochrome_line_cycler)])
