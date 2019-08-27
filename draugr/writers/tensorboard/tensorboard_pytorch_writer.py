#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
from typing import Union

import PIL
import numpy
import torch
from PIL import Image

from draugr import PROJECT_APP_PATH
from draugr.writers.writer import Writer

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from torch.utils.tensorboard import SummaryWriter


class TensorBoardPytorchWriter(Writer):
    def __init__(
        self, path=pathlib.Path.home() / "Models", comment: str = "", **kwargs
    ):
        super().__init__(**kwargs)

        self._log_dir = path
        self._comment = comment

    def _scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def _close(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def image(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, PIL.Image.Image],
        step,
        **kwargs
    ):
        self.writer.add_image(tag, data, step)

    def _open(self):
        self.writer = SummaryWriter(str(self._log_dir), self._comment)
        return self


if __name__ == "__main__":

    with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log / "test") as writer:
        writer.scalar("What", 4)

        from torchvision.utils import make_grid

        for n_iter in range(20):
            dummy_img = torch.rand(32, 3, 64, 64)  # output from network
            if n_iter % 10 == 0:
                x = make_grid(dummy_img, normalize=True, scale_each=True)
                writer.image("Image", x, n_iter)
