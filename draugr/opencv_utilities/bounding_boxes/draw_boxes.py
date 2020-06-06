#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/03/2020
           """

from pathlib import Path
from typing import Sequence, Tuple

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy
from PIL import Image

from draugr.opencv_utilities.bounding_boxes.colors import compute_color_for_labels
from draugr.opencv_utilities.opencv_draw import draw_masks
from draugr.python_utilities.colors import RGB

__all__ = ["draw_bouding_boxes"]


def draw_single_box(
    image: Image.Image,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    score_color: Tuple = RGB(0, 0, 0),
    display_str: str = None,
    font_type: ImageFont = ImageFont.load_default(),
    outline_color: Tuple = RGB(0, 255, 0),
    outline_width: int = 2,
    outline_alpha: float = 0.5,
    color_fill_score: bool = False,
) -> Image.Image:
    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = outline_color + (int(255 * outline_alpha),)
    draw.rectangle(
        [(left, top), (right, bottom)],
        outline=outline_color,
        fill=alpha_color if color_fill_score else None,
        width=outline_width,
    )
    if display_str:
        if font_type is None or not isinstance(
            font_type,
            (ImageFont.FreeTypeFont, ImageFont.ImageFont, ImageFont.TransposedFont),
        ):
            font_type = ImageFont.load_default()
            print("Loading default (Better Than Nothing) font")

        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font_type.getsize(display_str)
        margin = numpy.ceil(0.05 * text_height)
        draw.rectangle(
            xy=[
                (
                    left + outline_width,
                    text_bottom - text_height - 2 * margin - outline_width,
                ),
                (left + text_width + outline_width, text_bottom - outline_width),
            ],
            fill=alpha_color,
        )
        draw.text(
            (
                left + margin + outline_width,
                text_bottom - text_height - margin - outline_width,
            ),
            display_str,
            fill=score_color,
            font=font_type,
        )

    return image


def draw_bouding_boxes(
    image: numpy.ndarray,
    boxes: numpy.ndarray,
    *,
    labels: numpy.ndarray = None,
    scores: numpy.ndarray = None,
    categories: Sequence[str] = None,
    outline_width: int = 2,
    outline_alpha: float = 0.5,
    score_color_fill: bool = False,
    score_font: ImageFont = ImageFont.load_default(),
    score_format: str = ": {:.2f}",
) -> numpy.ndarray:
    """Draw bounding boxes(labels, scores) on image
Args:
image: numpy array image, shape should be (height, width, channel)
boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax), NOT NORMALISED!
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
            ith_cat = int(labels[i])
            color = compute_color_for_labels(ith_cat)
            class_name = categories[ith_cat] if categories is not None else str(ith_cat)
            display_str += class_name

        if scores is not None:
            display_str += f"{score_format.format(scores[i])}"

        draw_image = draw_single_box(
            image=draw_image,
            xmin=boxes[i, 0],
            ymin=boxes[i, 1],
            xmax=boxes[i, 2],
            ymax=boxes[i, 3],
            outline_color=color,
            outline_width=outline_width,
            outline_alpha=outline_alpha,
            display_str=display_str,
            font_type=score_font,
            color_fill_score=score_color_fill,
        )

    return numpy.array(draw_image, dtype=numpy.uint8)


if __name__ == "__main__":

    def a():
        from matplotlib import pyplot
        import pickle
        from neodroidvision.data.datasets.supervised.detection.coco import COCODataset

        name = "000000308476"
        data_root = Path.home() / "Data" / "Coco"
        with open(str(data_root / f"{name}.pickle"), "rb") as f:
            data = pickle.load(f)

        img = Image.open(str(data_root / f"{name}.jpg"))
        img = draw_masks(img, data.masks, data.labels)
        img = draw_bouding_boxes(
            img,
            boxes=data.boxes,
            labels=data.labels,
            scores=data.scores,
            categories=COCODataset.categories,
            score_format=": {:.2f}",
        )
        pyplot.imshow(img)
        pyplot.show()

    def b():
        from matplotlib import pyplot

        data_root = Path.home() / "Pictures"

        img = Image.open(str(data_root / f"0.jpeg"))
        img_width, _ = img.size
        img = draw_bouding_boxes(
            img,
            boxes=(
                (0 * img_width, 0 * img_width, 0.2 * img_width, 0.2 * img_width),
                (0.3 * img_width, 0.3 * img_width, 0.5 * img_width, 0.5 * img_width),
                (0.1 * img_width, 0.1 * img_width, 0.7 * img_width, 0.7 * img_width),
            ),
            labels=(0, 55, 78),
            scores=(0.2, 1.0, 0.4),
            categories=(*[f"cat{i}" for i in range(99)],),
            score_format=": {:.2f}",
            outline_width=5,
            score_color_fill=False,
        )
        pyplot.imshow(img)
        pyplot.show()

    b()
