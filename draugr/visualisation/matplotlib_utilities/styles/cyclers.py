#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-02-2021
           """

__all__ = ["monochrome_hatch_cycler", "simple_hatch_cycler"]

from matplotlib import cycler

increasing_density_hatch = (
    b * a for a in range(2, 3 + 2) for b in ("\\", "/", "x", ".",)
)
simple_hatch_cycler = cycler("hatch", increasing_density_hatch)
monochrome_hatch_cycler = (
    cycler("color", "w") * cycler("edgecolor", "k") * simple_hatch_cycler
)

line_styles = ("-", "--", ":", "-.")
marker_styles = (",", "^", "o", "x", "s")
monochrome_line_cycler = (
    cycler("color", ["k"])
    * cycler("linestyle", line_styles)
    * cycler("marker", marker_styles)
)
