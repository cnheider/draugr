#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC
from typing import Union

import numpy
import torch
from PIL import Image

from draugr.writers.writer import Writer

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["ImageWriter"]


class ImageWriter(Writer, ABC):
    """
    Writer subclass that provides an interface for 'writing' images
    """

    def image(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, Image.Image],
        step,
        *,
        dataformats: str = "NCHW",
        **kwargs
    ) -> None:
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
        raise NotImplementedError
