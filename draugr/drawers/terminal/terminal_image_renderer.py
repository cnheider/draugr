#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""
           Render images on the command line

           Created on 7/5/22
           """


import argparse
from pathlib import Path
from typing import Any, List, Tuple

import numpy
from PIL import Image

from warg import color, Number

__all__ = ["render_file", "terminalise_image", "terminal_render_image"]


def get_pixel(col: Tuple) -> Any:
    """
    Get the pixel value for a given colour.

    :param col:"""
    if isinstance(col, (float, int)) or len(col) == 1:
        col = (col, col, col)
    return color("  ", bg=f"rgb({int(col[0])}, {int(col[1])}, {int(col[2])})")


def terminal_render_image(
    pixels: numpy.ndarray, scale: Tuple, max_val: Number = None
) -> List[List]:
    """
    Render an image on the command line."""
    # first of all scale the image to the scale 'tuple'
    if len(pixels.shape) < 3:
        pixels = numpy.array([[pixel] for pixel in pixels])
        num_channels = 1
    else:
        num_channels = pixels.shape[2]

    if max_val is None:
        max_val = numpy.max(pixels)
        if max_val <= 1:
            pixels = pixels * 255
        else:
            pixels = pixels * (255 / max_val)
    else:
        pixels = pixels * (255 / max_val)

    image_size = pixels.shape[:2]
    block_size = (image_size[0] / scale[0], image_size[1] / scale[1])

    blocks = []
    y = 0
    while y < image_size[0]:
        x = 0
        block_col = []
        while x < image_size[1]:
            # get a block, reshape in into an Nx3 matrix and then get average of each column
            block_col.append(
                pixels[int(y) : int(y + block_size[0]), int(x) : int(x + block_size[1])]
                .reshape(-1, num_channels)
                .mean(axis=0)
            )
            x += block_size[1]
        blocks.append(block_col)
        y += block_size[0]
    output = [[get_pixel(block) for block in row] for row in blocks]
    return output


def terminalise_image(output):
    """

    joins nested str lists with newlines

    :return:"""
    return "\n".join(["".join(row) for row in output])


def get_image(path: Path):
    """
    Get an image from a path."""
    img = numpy.asarray(Image.open(path))
    if img.shape[2] > 3:
        return numpy.array([[pixel[:3] for pixel in row] for row in img])
    return img


def render_file(path: Path, scale=(60, 60)):
    """
    Render an image file on the command line."""
    image = get_image(path)
    output = terminal_render_image(image, scale)
    print(terminalise_image(output))


def entry_point() -> None:
    """
    Entry point for the command line interface.

    :rtype: None
    """
    parser = argparse.ArgumentParser(description="Render images on the command line")
    parser.add_argument("path", metavar="path", type=str, help="the image path")
    parser.add_argument(
        "--width",
        dest="width",
        default=60,
        type=int,
        help="width of the rendered image (default 60 pixels)",
    )
    parser.add_argument(
        "--height",
        dest="height",
        default=60,
        type=int,
        help="height of the rendered image (default 60 pixels)",
    )
    args = parser.parse_args()
    render_file(args.path, (args.height, args.width))


if __name__ == "__main__":
    output = terminal_render_image(numpy.random.random((60, 60, 3)) * 5, (60, 60))
    print(terminalise_image(output))
    # render_file(Path.home() / "OneDrive" / "Billeder" / "pompey.jpg", scale=(20, 20))
