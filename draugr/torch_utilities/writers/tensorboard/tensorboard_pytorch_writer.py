#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
from contextlib import suppress
from typing import Union

import PIL
import numpy
import torch
from PIL import Image

from draugr import PROJECT_APP_PATH
from draugr.torch_utilities.writers.tensorboard.image_writer import ImageWriter
from warg import passes_kws_to

with suppress(FutureWarning):
    from torch.utils.tensorboard import SummaryWriter

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""
__all__ = ["TensorBoardPytorchWriter"]


class TensorBoardPytorchWriter(ImageWriter):
    """

    """

    @passes_kws_to(ImageWriter.__init__)
    def __init__(
        self,
        path: Union[str, pathlib.Path] = pathlib.Path.home() / "Models",
        comment: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)

        self._log_dir = path
        self._comment = comment

    def _scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def graph(self, model, input_to_model):
        """

        :param model:
        :type model:
        :param input_to_model:
        :type input_to_model:
        """
        self.writer.add_graph(model, input_to_model)

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        if hasattr(self, "_writer"):
            self._writer.close()
            delattr(self, "_writer")

    @passes_kws_to(SummaryWriter.add_image)
    def image(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, PIL.Image.Image],
        step,
        *,
        dataformats="NCHW",
        **kwargs
    ):
        """

        :param tag:
        :type tag:
        :param data:
        :type data:
        :param step:
        :type step:
        :param dataformats:
        :type dataformats:
        :param kwargs:
        :type kwargs:
        """
        self.writer.add_image(tag, data, step, dataformats=dataformats, **kwargs)

    @property
    def writer(self):
        """

        :return:
        :rtype:
        """
        if not hasattr(self, "_writer") or not self._writer:
            self._writer = SummaryWriter(str(self._log_dir), self._comment)
        return self._writer

    def _open(self):

        return self


if __name__ == "__main__":

    with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log / "test") as writer:
        writer.scalar("What", 4)

        from torchvision.utils import make_grid

        for n_iter in range(20):
            dummy_img = torch.rand(32, 3, 64, 64)  # output from network
            if n_iter % 10 == 0:
                x = make_grid(dummy_img, normalize=True, scale_each=True)
                writer.image("ImageGrid", x, n_iter, dataformats="CHW")
                writer.image("Image", dummy_img, n_iter)
