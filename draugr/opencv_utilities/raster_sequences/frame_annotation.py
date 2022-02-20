#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-03-2021
           """

import cv2
import numpy

from warg import DoubleNumber, Number, TripleNumber

__all__ = ["blit_numbering_raster_sequence", "blit_fps"]


def blit_numbering_raster_sequence(
    seq: numpy.ndarray,
    *,
    placement: DoubleNumber = (0, 30),
    color: TripleNumber = (
        0,
        0,
        1,
    ),  # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
    thickness: Number = 2,
    font_scale: Number = 1,
    font: int = cv2.FONT_HERSHEY_COMPLEX,
) -> numpy.ndarray:
    """

    :param seq:
    :param placement:
    :param color:
    :param thickness:
    :param font_scale:
    :param font:
    :return:
    """
    result = numpy.empty_like(seq)
    n = len(seq)
    for i in range(n):
        result[i] = cv2.putText(
            seq[i], f"{i}/{n}", placement, font, font_scale, color, thickness
        )
    return result


def blit_fps(
    seq: numpy.ndarray,
    fps: Number,
    *,
    placement: DoubleNumber = (-140, -10),  # bottom-right corner
    format_str: str = "{0} fps",
    color: TripleNumber = (
        0,
        0,
        1,
    ),  # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
    thickness: Number = 2,
    font_scale: Number = 1,
    font: int = cv2.FONT_HERSHEY_COMPLEX,
) -> numpy.ndarray:
    """

    :param seq:
    :param fps:
    :param placement:
    :param format_str:
    :param color:
    :param thickness:
    :param font_scale:
    :param font:
    :return:
    """
    result = numpy.empty_like(seq)
    n = len(seq)
    if placement[0] < 0:
        placement = (result.shape[-2] + placement[0], placement[1])
    if placement[1] < 0:
        placement = (placement[0], result.shape[-3] + placement[1])

    for i in range(n):
        result[i] = cv2.putText(
            seq[i],
            format_str.format(fps),
            placement,
            font,
            font_scale,
            color,
            thickness,
        )
    return result


if __name__ == "__main__":

    def asd7ad() -> None:
        """
        :rtype: None
        """
        from pathlib import Path
        from apppath import ensure_existence
        from matplotlib import pyplot
        import numpy
        import imageio

        n = 200
        n_frames = 25
        x = numpy.linspace(-numpy.pi * 4, numpy.pi * 4, n)
        base = ensure_existence(Path("exclude"))

        def gen():
            """ """
            for i, t in enumerate(numpy.linspace(0, numpy.pi, n_frames)):
                pyplot.plot(x, numpy.cos(x + t))
                pyplot.plot(x, numpy.sin(2 * x - t))
                pyplot.plot(x, numpy.cos(x + t) + numpy.sin(2 * x - t))
                pyplot.ylim(-2.5, 2.5)
                pyplot.savefig(base / f"frame{i}.png", bbox_inches="tight", dpi=300)
                pyplot.clf()

        def asijsd():
            """ """
            files = [base / f"frame{yu}.png" for yu in range(n_frames)]
            frames = [imageio.imread(f) for f in files]
            frames = blit_numbering_raster_sequence(frames)
            imageio.mimsave(base / "output.gif", frames, fps=(n_frames / 2.0))

        def sadasf():
            """ """
            files = [base / f"frame{yu}.png" for yu in range(n_frames)]
            a = [imageio.imread(f) for f in files]
            frames = numpy.array([a, a])  # copy of itself, just for test
            fps = n_frames / 2.0
            frames = numpy.array(
                [blit_fps(blit_numbering_raster_sequence(f), fps) for f in frames]
            )
            [
                imageio.mimsave(base / f"output{i}.gif", f, fps=fps)
                for i, f in enumerate(frames)
            ]

        gen()
        sadasf()

    asd7ad()
