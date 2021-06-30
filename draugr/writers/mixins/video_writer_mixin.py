#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

import numpy
import torch
from PIL import Image

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["VideoWriterMixin", "VideoInputDimsEnum"]

from warg import Number


class VideoInputDimsEnum(Enum):
    """
    Input mode"""

    thw = "THW"
    tchw = "TCHW"
    thwc = "THWC"
    ntchw = "NTCHW"
    nthwc = "NTHWC"
    tnchw = "TNCHW"
    tnhwc = "TNHWC"


class VideoWriterMixin(ABC):
    """
    Writer subclass that provides an interface for 'writing' video clips"""

    @abstractmethod
    def video(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, Image.Image],
        step,
        frame_rate: Number = 30,
        input_dims: VideoInputDimsEnum = VideoInputDimsEnum.ntchw,
        **kwargs
    ) -> None:
        """

            :param frame_rate:
            :param input_dims:
        :param tag:
        :type tag:
        :param data:
        :type data:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:"""
        raise NotImplementedError
