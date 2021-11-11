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
    Writer mixin that provides an interface for 'writing' spectrogram charts"""

    @abstractmethod
    def spectrogram(
        self,
        tag: str,
        values: list,
        sample_rate: int,
        step: int,
        num_fft: int = 512,
        x_labels: Sequence = None,
        y_label: str = "Magnitude",
        x_label: str = "Sequence",
        plot_kws: Mapping = None,  # Separate as parameters name collisions might occur
        **kwargs
    ) -> None:
        """

            :param values:
            :param sample_rate:
            :param num_fft:
            :param x_labels:
            :param y_label:
            :param x_label:
            :param plot_kws:
        :param tag:
        :type tag:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:"""
        raise NotImplementedError

    '''
@abstractmethod
def mfcc_spectrogram(self,
tag: str,
values: list,
sample_rate:int,
step: int,
num_fft:int=512,
x_labels: Sequence = None,
y_label: str = "Magnitude",
x_label: str = "Sequence",
plot_kws: Mapping = {},  # Separate as parameters name collisions might occur
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
