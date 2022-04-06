#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Union

import numpy
from cycler import Cycler
from matplotlib import patches, pyplot, rcParams
from matplotlib.legend_handler import HandlerErrorbar
from sorcery import assigned_names

from draugr.visualisation.matplotlib_utilities.quirks import auto_post_hatch
from draugr.visualisation.matplotlib_utilities.styles.annotation import (
    rt_ann_transform,
    semi_opaque_round_tight_bbox,
)
from draugr.visualisation.matplotlib_utilities.styles.cyclers import (
    monochrome_hatch_cycler,
    monochrome_line_cycler,
    simple_hatch_cycler,
)

__all__ = [
    "denormalise_minusoneone",
    "matplotlib_bounding_box",
    "remove_decoration",
    "use_monochrome_style",
    "decolorise_plot",
    "save_embed_fig",
    "latex_clean_label",
    "make_errorbar_legend",
    "annotate_point",
    "MatplotlibHorizontalAlignment",
    "MatplotlibVerticalAlignment",
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


class MatplotlibHorizontalAlignment(Enum):
    """ """

    (center, right, left) = assigned_names()


class MatplotlibVerticalAlignment(Enum):
    """ """

    (center, top, bottom, baseline, center_baseline) = assigned_names()


def latex_clean_label(s: str) -> str:
    """Clean label for troublesome symbols"""
    return s.replace("_", " ")  # .replace('\\',' ').replace('/',' ')


def make_errorbar_legend(ax: Axes = None) -> None:
    """size adjusted legend for error bars, default does not work well with different linestyles"""

    if ax is None:
        ax = pyplot.gca()
    ax.legend(
        handlelength=5,
        handleheight=3,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.9, yerr_size=0.9)},
    )


def annotate_point(ax: Axes, x: Sequence, y: Sequence, t: Any) -> None:
    """

    :param ax:
    :param x:
    :param y:
    :param t:
    :return:
    """
    if ax is None:
        ax = pyplot.gca()
    ax.annotate(
        f"{t:.2f}",
        (x, y),
        textcoords="offset points",
        fontsize="xx-small",
        bbox=semi_opaque_round_tight_bbox,
        annotation_clip=True,
        # see details https://github.com/matplotlib/matplotlib/issues/14354#issuecomment-523630316
        clip_on=True,
        **rt_ann_transform,
    )


@passes_kws_to(pyplot.savefig)
def save_embed_fig(
    path: Union[Path, str],
    bbox_inches: Union[Number, str] = "tight",
    transparent: bool = True,
    attempt_fix_empty_white_space: bool = False,
    post_process_crop: bool = False,
    ax: Axes = None,
    suffix: str = ".pdf",
    **kwargs,
) -> None:
    """Save fig for latex pdf embedding"""

    if attempt_fix_empty_white_space:  # remove it
        # pyplot.gca().set_axis_off()
        # pyplot.axis('off') # this rows the rectangular frame
        # ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
        # ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
        if ax is None:
            ax = pyplot.gca()
        pyplot.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        pyplot.margins(0, 0)
        ax.xaxis.set_major_locator(pyplot.NullLocator())
        ax.yaxis.set_major_locator(pyplot.NullLocator())

    """
clip_box = Bbox(((0,0),(300,300)))
for o in pyplot.findobj():
o.set_clip_on(True)
o.set_clip_box(clip_box)

"""

    if not isinstance(path, Path):
        path = Path(path)

    path_str = str(path.with_suffix(suffix))
    pyplot.savefig(
        path_str,
        bbox_inches=bbox_inches,
        transparent=transparent,
        **kwargs,
    )

    if (
        post_process_crop and suffix == ".pdf"
    ):  # Generally a good idea since matplotlib does not exclude invisible parts(eg. data points or anchors) of the plot.
        from pdfCropMargins import crop  # pip install pdfCropMargins

        crop(
            [
                "-p",
                "0",  # remove all percentage of old margin
                "-a",
                "-5",  # add 5bp margin around all
                path_str,
            ]
        )


def remove_decoration(ax: Axes) -> None:
    """ """
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
    """ """
    return 0.5 * ((coordinates + 1.0) * t)


def decolorise_plot(ax_: Axes, inverted: bool = False) -> None:
    """

    set black and white edge colors and face colors respectively. Converse if inverted.
    """
    edge_color = "k"
    face_color = "w"
    if inverted:
        face_color, edge_color = edge_color, face_color
    pyplot.setp(ax_.artists, edgecolor=edge_color, facecolor=face_color)
    pyplot.setp(ax_.lines, color=edge_color)


def matplotlib_bounding_box(
    x: Number, y: Number, size: Number, color: str = "w"
) -> Rectangle:
    """ """
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def use_monochrome_style(
    prop_cycler: Cycler = monochrome_line_cycler,
    # ONLY COLOR AND LINESTYLE MAKES SENSE FOR NOW, matplotlib seems very undone in this api atleast for bars
) -> None:
    """

    :param prop_cycler:
    """
    # from matplotlib.pyplot import axes, grid

    # from matplotlib import lines, markers
    # print(lines.lineStyles.keys(),markers.MarkerStyle.markers.keys())

    # print(rcParams.keys())

    # pyplot.style.use('default')
    # print(pyplot.style.available)
    # print(matplotlib.matplotlib_fname())

    pyplot.style.use(Path(__file__).parent / "styles" / "monochrome.mplstyle")
    if pyplot.rcParams["text.usetex"]:
        try:
            report = subprocess.check_output("latex -v", stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            msg = f'No tex: {"latex"}'
            # raise RuntimeError(msg)
            print(f"{msg}, disabling")
            pyplot.rcParams.update({"text.usetex": False})
    rcParams.update({"axes.prop_cycle": prop_cycler})
    # auto_post_print_dpi()


if __name__ == "__main__":

    def asiuhda() -> None:
        """
        :rtype: None
        """
        use_monochrome_style()
        bar_styles = monochrome_hatch_cycler()
        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.bar(x, numpy.random.randint(2, 10), **next(bar_styles), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        pyplot.show()

    def asiuh214da() -> None:
        """
        :rtype: None
        """
        pyplot.style.use(Path(__file__).parent / "styles" / "monochrome.mplstyle")
        line_styles = monochrome_line_cycler()
        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.plot(numpy.random.rand(10), **next(line_styles), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        pyplot.show()

    def asiuasdashda() -> None:
        """
        :rtype: None
        """
        use_monochrome_style(prop_cycler=monochrome_hatch_cycler)

        fig, ax = pyplot.subplots(1, 1)

        for x in range(3):
            ax.bar(x, numpy.random.randint(2, 10), label=f"{x}")

        legend()
        from draugr.visualisation.matplotlib_utilities.quirks import fix_edge_gridlines

        fix_edge_gridlines(ax)
        auto_post_hatch(ax, simple_hatch_cycler)
        pyplot.show()

    def asiuhdsada_non() -> None:
        """
        :rtype: None
        """
        fig, ax = pyplot.subplots(1, 1)
        for x in range(3):
            ax.plot(numpy.random.rand(10), label=f"{x}")

        legend()
        pyplot.show()

    def asiuhd21412sada() -> None:
        """
        :rtype: None
        """
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
