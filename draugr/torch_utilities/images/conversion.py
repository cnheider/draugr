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


def quick_to_pil_image(inp: torch.Tensor, mode="RGB") -> Image.Image:
    """

    LOTS OF ASSUMPTIONS!

    :param inp:
    :return:"""

    return Image.fromarray(inp.cpu().numpy().astype(numpy.uint8), mode)


if __name__ == "__main__":

    def asd2():

        import cv2
        import torch
        from PIL import Image
        from tqdm import tqdm

        from draugr.opencv_utilities import frame_generator
        from draugr.torch_utilities import global_torch_device, to_tensor_generator

        with torch.no_grad():
            for image in tqdm(
                to_tensor_generator(
                    frame_generator(cv2.VideoCapture(0)), device=global_torch_device(),
                )
            ):
                cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
                cv2.imshow("window_name", numpy.array(quick_to_pil_image(image)))

                if cv2.waitKey(1) == 27:
                    break  # esc to quit

    asd2()
