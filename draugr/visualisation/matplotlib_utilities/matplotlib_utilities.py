#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

from pathlib import Path

import numpy
from matplotlib.legend_handler import HandlerErrorbar

from draugr.visualisation.matplotlib_utilities.quirks import auto_post_hatch
from draugr.visualisation.matplotlib_utilities.styles.cyclers import (
    simple_hatch_cycler,
    monochrome_hatch_cycler,
    monochrome_line_cycler,
)
from matplotlib import patches, pyplot, rcParams
from cycler import cycler
from typing import Sequence, Union

__all__ = [
    "denormalise_minusoneone",
    "matplotlib_bounding_box",
    "remove_decoration",
    "use_monochrome_style",
    "decolorise_plot",
    "save_pdf_embed_fig",
    "latex_clean_label",
    "make_errorbar_legend",
]

from matplotlib.patches import Rectangle

from matplotlib.pyplot import legend

from warg import Number, passes_kws_to

# Cycler
##Addition
###Equal length Cycler s with different keys can be added to get the ‘inner’ product of two cycles
##Multiplication
###which gives the ‘outer product’ of the two cycles (same as itertools.prod() )

from matplotlib.axes import Axes, ErrorbarContainer


def latex_clean_label(s: str) -> str:
    """ Clean label for troublesome symbols """
    return s.replace("_", " ")  # .replace('\\',' ').replace('/',' ')


def make_errorbar_legend() -> None:
    """ size adjusted legend for error bars, default does not work well with different linestyles """
    pyplot.legend(
        handlelength=5,
        handleheight=3,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.9, yerr_size=0.9)},
    )


@passes_kws_to(pyplot.savefig)
def save_pdf_embed_fig(
    path: Union[Path, str] = "foo.pdf", bbox_inches="tight", transparent=True, **kwargs
) -> None:
    """Save fig for latex pdf embedding"""
    if not isinstance(path, Path):
        path = Path(path)
    pyplot.savefig(
        str(path.with_suffix(".pdf")),
        bbox_inches=bbox_inches,
        transparent=transparent,
        **kwargs,
    )


def remove_decoration(ax: Axes) -> None:
    """
"""
    transparent = (1.0, 1.0, 1.0, 0.0)

    ax.w_xaxis.set_pane_color(transparent)
    ax.w_yaxis.set_pane_color(transparent)
    ax.w_zaxis.set_pane_color(transparent)

    ax.w_xaxis.line.set_color(transparent)
    ax.w_yaxis.line.set_color(transparent)
    ax.w_zaxis.line.set_color(transparent)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def denormalise_minusoneone(
    t: Union[numpy.ndarray, Number], coordinates: Union[numpy.ndarray, Number]
) -> Union[numpy.ndarray, Number]:
    """
"""
    return 0.5 * ((coordinates + 1.0) * t)


def decolorise_plot(ax_: Axes) -> None:
    """
set black and white edge colors and face colors respectively.

"""
    pyplot.setp(ax_.artists, edgecolor="k", facecolor="w")
    pyplot.setp(ax_.lines, color="k")


def matplotlib_bounding_box(
    x: Number, y: Number, size: Number, color: str = "w"
) -> Rectangle:
    """
"""
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def use_monochrome_style(
    prop_cycler=monochrome_line_cycler,  # ONLY COLOR AND LINESTYLE MAKES SENSE FOR NOW, matplotlib seems very undone in this api atleast for bars
) -> None:
    # from matplotlib.pyplot import axes, grid

    # from matplotlib import lines, markers
    # print(lines.lineStyles.keys(),markers.MarkerStyle.markers.keys())

    # print(rcParams.keys())

    # pyplot.style.use('default')
    # print(pyplot.style.available)
    # print(matplotlib.matplotlib_fname())

    pyplot.style.use(Path(__file__).parent / "styles" / "monochrome.mplstyle")
    rcParams.update({"axes.prop_cycle": prop_cycler})
    # auto_post_print_dpi()


if __name__ == "__main__":

    def asiuhda():
        use_monochrome_style()
        bar_styles = monochrome_hatch_cycler()
        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.bar(x, numpy.random.randint(2, 10), **next(bar_styles), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        pyplot.show()

    def asiuh214da():
        pyplot.style.use(Path(__file__).parent / "styles" / "monochrome.mplstyle")
        line_styles = monochrome_line_cycler()
        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.plot(numpy.random.rand(10), **next(line_styles), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        pyplot.show()

    def asiuasdashda():
        use_monochrome_style(prop_cycler=monochrome_hatch_cycler)

        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.bar(x, numpy.random.randint(2, 10), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        auto_post_hatch(ax, simple_hatch_cycler)
        pyplot.show()

    def asiuhdsada_non():

        fig, ax = pyplot.subplots(1, 1)
        for x in range(3):
            ax.plot(numpy.random.rand(10), label=f"{x}")

        legend()
        pyplot.show()

    def asiuhd21412sada():
        use_monochrome_style()
        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.plot(numpy.random.rand(10), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        pyplot.show()

    # asiuhda()
    # asiuasdashda()
    asiuhd21412sada()
