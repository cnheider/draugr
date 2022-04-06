import argparse
from pathlib import Path
from typing import Any, List, Tuple

import numpy
from PIL import Image

from draugr.python_utilities.colors import color


def get_pixel(col: Tuple) -> Any:
    """ """
    if isinstance(col, (float, int)) or len(col) == 1:
        col = (col, col, col)
    return color("  ", bg=f"rgb({int(col[0])}, {int(col[1])}, {int(col[2])})")


def render_image(pixels: numpy.ndarray, scale: Tuple) -> List[List]:
    """ """
    # first of all scale the image to the scale 'tuple'
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
                .reshape(-1, 3)
                .mean(axis=0)
            )
            x += block_size[1]
        blocks.append(block_col)
        y += block_size[0]
    output = [[get_pixel(block) for block in row] for row in blocks]
    return output


def terminalise_image(output):
    """

    joins lists

    :return:"""
    return "\n".join(["".join(row) for row in output])


def get_image(path: Path):
    """ """
    img = numpy.asarray(Image.open(path))
    if img.shape[2] > 3:
        return numpy.array([[pixel[:3] for pixel in row] for row in img])
    return img


def render_file(path: Path, scale=(60, 60)):
    """ """
    image = get_image(path)
    output = render_image(image, scale)
    print(terminalise_image(output))


def entry_point() -> None:
    """
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
    render_file(Path.home() / "OneDrive" / "Billeder" / "pompey.jpg", scale=(20, 20))
