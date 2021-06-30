#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Mapping, Sequence

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["LineWriterMixin"]


class LineWriterMixin(ABC):
    """
    Writer mixin that provides an interface for 'writing' line charts"""

    @abstractmethod
    def line(
        self,
        tag: str,
        values: list,
        step: int,
        x_labels: Sequence = None,
        y_label: str = "Magnitude",
        x_label: str = "Sequence",
        plot_kws: Mapping = None,  # Separate as parameters name collisions might occur
        **kwargs
    ) -> None:
        """

            :param values:
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
