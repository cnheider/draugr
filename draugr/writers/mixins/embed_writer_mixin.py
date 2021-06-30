#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["EmbedWriterMixin"]

from typing import Any, Sequence


class EmbedWriterMixin(ABC):
    """
    Writer mixin that provides an interface for 'writing' embeds/projections(2d,3d) for interactive visualisation"""

    @abstractmethod
    def embed(
        self,
        tag: str,
        response: Sequence,
        metadata: Any = None,
        label_img: Any = None,  # raster grid / image / numpy.array
        step: int = None,
        **kwargs
    ) -> None:
        """
            eg. visualising for projections in lower dimensional space

            :param response:
            :param metadata:
            :param label_img:
        :param tag:
        :type tag:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:"""
        raise NotImplementedError
