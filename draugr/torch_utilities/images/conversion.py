#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-11-2020
           """
__all__ = ["quick_to_pil_image"]

import numpy
import torch
from PIL import Image

from draugr.opencv_utilities import to_rgb
from draugr.opencv_utilities.windows.image import show_image


def quick_to_pil_image(tensor: torch.Tensor, mode: str = "RGB") -> Image.Image:
    """

    LOTS OF ASSUMPTIONS!

    :param tensor:
    :param mode:
    :return:"""

    return Image.fromarray(tensor.cpu().numpy().astype(numpy.uint8), mode)


if __name__ == "__main__":

    def asd2() -> None:
        """
        :rtype: None
        """
        import cv2
        import torch
        from PIL import Image
        from tqdm import tqdm

        from draugr.opencv_utilities import frame_generator
        from draugr.torch_utilities import global_torch_device, to_tensor_generator

        with torch.no_grad():
            for image in tqdm(
                to_tensor_generator(
                    frame_generator(cv2.VideoCapture(0), coder=to_rgb),
                    device=global_torch_device(),
                )
            ):
                if show_image(numpy.array(quick_to_pil_image(image)), wait=1):
                    break  # esc to quit

    asd2()
