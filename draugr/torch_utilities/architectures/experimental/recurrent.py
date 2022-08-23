#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = ""

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


class RecurrentBase(nn.Module):
    """
    A base class for Recurrent models
    """

    def __init__(self, recurrent: bool, recurrent_input_size: int, hidden_size: int):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def _forward_gru(
        self: torch.Tensor, x, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.reshape(T, N, x.size(1))

            # Same deal with masks
            masks = masks.reshape(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.reshape(T * N, -1)

        return x, hxs
