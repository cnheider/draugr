#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["EmbedWriterMixin"]


class EmbedWriterMixin(ABC):
    """
  Writer mixin that provides an interface for 'writing' embeds for interactive visualisation
  """

    @abstractmethod
    def embed(
        self, tag: str, features, metadata, label_img, step: int, **kwargs
    ) -> None:
        """
    eg. visualising for projections in lower dimensional space

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
