#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/03/2020
           """

from typing import List, Tuple

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy
from PIL import Image

from draugr.opencv_utilities.opencv_draw import draw_masks
from draugr.python_utilities.functions import RGB
from .colors import compute_color_for_labels

__all__ = ["draw_bouding_boxes"]


def draw_single_box(
    image: Image.Image,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: Tuple = RGB(0, 255, 0),
    display_str: str = None,
    font: str = None,
    width: int = 2,
    alpha: float = 0.5,
    fill: bool = False,
) -> Image.Image:
    if font is None:
        try:
            FONT = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            FONT = ImageFont.load_default()
        font = FONT

    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.rectangle(
        [(left, top), (right, bottom)],
        outline=color,
        fill=alpha_color if fill else None,
        width=width,
    )

    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = numpy.ceil(0.05 * text_height)
        draw.rectangle(
            xy=[
                (left + width, text_bottom - text_height - 2 * margin - width),
                (left + text_width + width, text_bottom - width),
            ],
            fill=alpha_color,
        )
        draw.text(
            (left + margin + width, text_bottom - text_height - margin - width),
            display_str,
            fill="black",
            font=font,
        )

    return image


def draw_bouding_boxes(
    image: numpy.ndarray,
    boxes: numpy.ndarray,
    labels: numpy.ndarray = None,
    scores: numpy.ndarray = None,
    class_name_map: List = None,
    width: int = 2,
    alpha: float = 0.5,
    fill: bool = False,
    font: str = None,
    score_format: str = ":{:.2f}",
) -> numpy.ndarray:
    """Draw bboxes(labels, scores) on image
Args:
    image: numpy array image, shape should be (height, width, channel)
    boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax)
    labels: labels, shape: (N, )
    scores: label scores, shape: (N, )
    class_name_map: list or dict, map class id to class name for visualization.
    width: box width
    alpha: text background alpha
    fill: fill box or not
    font: text font
    score_format: score format
Returns:
    An image with information drawn on it.
"""
    boxes = numpy.array(boxes)
    num_boxes = boxes.shape[0]
    if isinstance(image, Image.Image):
        draw_image = image
    elif isinstance(image, numpy.ndarray):
        draw_image = Image.fromarray(image)
    else:
        raise AttributeError(f"Unsupported images type {type(image)}")

    for i in range(num_boxes):
        display_str = ""
        color = (0, 255, 0)
        if labels is not None:
            this_class = labels[i]
            color = compute_color_for_labels(this_class)
            class_name = (
                class_name_map[this_class]
                if class_name_map is not None
                else str(this_class)
            )
            display_str = class_name

        if scores is not None:
            prob = scores[i]
            if display_str:
                display_str += score_format.format(prob)
            else:
                display_str += f"score{score_format.format(prob)}"

        draw_image = draw_single_box(
            image=draw_image,
            xmin=boxes[i, 0],
            ymin=boxes[i, 1],
            xmax=boxes[i, 2],
            ymax=boxes[i, 3],
            color=color,
            display_str=display_str,
            font=font,
            width=width,
            alpha=alpha,
            fill=fill,
        )

    return numpy.array(draw_image, dtype=numpy.uint8)


if __name__ == "__main__":
    from matplotlib import pyplot
    import pickle
    from neodroidvision.data import COCODataset

    name = "000000308476"
    with open(f"data/{name}.pickle", "rb") as f:
        data = pickle.load(f)
    img = Image.open(f"data/{name}.jpg")
    img = draw_masks(img, data["masks"], data["labels"])
    img = draw_bouding_boxes(
        img,
        boxes=data["boxes"],
        labels=data["labels"],
        scores=data["scores"],
        class_name_map=COCODataset.categories,
        score_format=":{:.4f}",
    )
    pyplot.imshow(img)
    pyplot.show()
