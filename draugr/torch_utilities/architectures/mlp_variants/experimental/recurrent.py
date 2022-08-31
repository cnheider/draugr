#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/24/22
           """

__all__ = []


from typing import MutableMapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional

from draugr.torch_utilities.architectures.mlp import MLP


class RecurrentCategoricalMLP(MLP):
    """
    Recurrent model base the MLP base model
    """

    def __init__(self, r_hidden_layers=10, **kwargs: MutableMapping):
        super().__init__(**kwargs)
        self._r_hidden_layers = r_hidden_layers
        self._r_input_shape = self._output_shape + r_hidden_layers

        self.hidden = nn.Linear(
            self._r_input_shape, r_hidden_layers, bias=self._use_bias
        )
        self.out = nn.Linear(self._r_input_shape, r_hidden_layers, bias=self._use_bias)

        self._prev_hidden_x = torch.zeros(r_hidden_layers)

    def forward(self, x: Sequence, **kwargs: MutableMapping):
        """

        :param x:
        :type x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        x = super().forward(x, **kwargs)
        combined = torch.cat((x, self._prev_hidden_x), 1)
        out_x = self.out(combined)
        hidden_x = self.hidden(combined)
        self._prev_hidden_x = hidden_x

        return functional.log_softmax(out_x, dim=-1)


class ExposedRecurrentCategoricalMLP(RecurrentCategoricalMLP):
    """
    Exposed Variant of Recurrent model base the MLP base model

    """

    def forward(self, x: Sequence, hidden_x: torch.Tensor, **kwargs: MutableMapping):
        """

        :param x:
        :type x:
        :param hidden_x:
        :type hidden_x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        self._prev_hidden_x = hidden_x
        out_x = super().forward(x, self._prev_hidden_x, **kwargs)

        return functional.log_softmax(out_x, dim=-1), self._prev_hidden_x
