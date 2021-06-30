#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["BarWriterMixin"]


class BarWriterMixin(ABC):
    """
    Writer mixin that provides an interface for 'writing' bar charts"""

    @abstractmethod
    def bar(
        self,
        tag: str,
        values: list,
        step: int,
        y_error=None,
        x_labels=None,
        y_label="Probability",
        x_label="Action Categorical Distribution",
        **kwargs
    ) -> None:
        """

            :param values:
            :param y_error:
            :param x_labels:
            :param y_label:
            :param x_label:
        :param tag:
        :type tag:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:"""
        raise NotImplementedError
