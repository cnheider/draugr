#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Mapping, Sequence

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["SpectrogramWriterMixin"]


class SpectrogramWriterMixin(ABC):
    """
  Writer mixin that provides an interface for 'writing' spectrogram charts
  """

    @abstractmethod
    def spectrogram(
        self,
        tag: str,
        values: list,
        sample_rate: int,
        step: int,
        NFFT: int = 512,
        x_labels: Sequence = None,
        y_label: str = "Magnitude",
        x_label: str = "Sequence",
        plot_kws: Mapping = {},  # Seperate as parameters name collisions might occur
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

    '''
  @abstractmethod
  def mfcc_spectrogram(self,
                  tag: str,
                  values: list,
                  sample_rate:int,
                  step: int,
                  NFFT:int=512,
                  x_labels: Sequence = None,
                  y_label: str = "Magnitude",
                  x_label: str = "Sequence",
                  plot_kws: Mapping = {},  # Seperate as parameters name collisions might occur
                  **kwargs) -> None:
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
'''
