#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["GraphWriterMixin"]


class GraphWriterMixin(ABC):
    """
    Writer mixin that provides an interface for 'writing' graphs"""

    @abstractmethod
    def graph(
        self, model: torch.nn.Module, input_to_model: torch.Tensor, **kwargs
    ) -> None:
        """

            :param model:
            :param input_to_model:
        :param kwargs:
        :type kwargs:"""
        raise NotImplementedError
