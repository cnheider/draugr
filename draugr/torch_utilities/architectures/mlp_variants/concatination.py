#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, List, Sequence, MutableMapping

import numpy
import torch

from draugr.torch_utilities.architectures.mlp import MLP
from draugr.torch_utilities.tensors.to_tensor import to_tensor

__author__ = "Christian Heider Nielsen"
__doc__ = "Fusion variant of MLPs"

__all__ = ["PreConcatInputMLP", "LateConcatInputMLP"]

from warg import passes_kws_to


class PreConcatInputMLP(MLP):
    """
    Early fusion
    """

    def __init__(self, input_shape: Sequence = (2,), **kwargs: MutableMapping):
        if isinstance(input_shape, Iterable):
            input_shape = sum(input_shape)

        super().__init__(input_shape=input_shape, **kwargs)

    @passes_kws_to(MLP.forward)
    def forward(self, *x, **kwargs) -> List:
        """

        :param x:
        :type x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return super().forward(torch.cat(x, dim=-1), **kwargs)


class LateConcatInputMLP(MLP):
    """
    Late fusion, quite a botch job, only a single addition block fusion supported for now
    You have been warned! ;)
    """

    def __init__(
        self,
        input_shape: Sequence = (2, 2),
        output_shape: Sequence = (2,),
        fusion_hidden_multiplier: int = 10,
        **kwargs: MutableMapping
    ):

        forward_shape, *res = input_shape
        self._residual_shape = res

        if not isinstance(self._residual_shape, Iterable):
            self._residual_shape = (self._residual_shape,)

        if not isinstance(output_shape, Iterable):
            output_shape = (*self._residual_shape, output_shape)
        assert len(output_shape) == 2

        super().__init__(
            input_shape=(forward_shape,), output_shape=output_shape, **kwargs
        )

        s = sum((*output_shape, *self._residual_shape))
        t = s * fusion_hidden_multiplier  # Hidden
        self.post_concat_layer = torch.nn.Sequential(
            torch.nn.Linear(s, t), torch.nn.ReLU(), torch.nn.Linear(t, output_shape[-1])
        )

    @passes_kws_to(MLP.forward)
    def forward(self, *x: Sequence, **kwargs: MutableMapping) -> torch.tensor:
        """

        :param x:
        :type x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        forward_x, *residual_x = x
        return self.post_concat_layer(
            torch.cat((*(super().forward(forward_x, **kwargs)), *residual_x), dim=-1)
        )


if __name__ == "__main__":

    def stest_normal():
        """description"""
        s = (10,)
        a = (10,)
        model = PreConcatInputMLP(input_shape=s, output_shape=a)

        inp = to_tensor(range(s[0]), device="cpu")
        print(model.forward(inp))

    def stest_multi_dim_normal():
        """description"""
        s = (19,)
        s1 = (4,)
        batch_size = (100,)
        a = (2, 10)
        model = PreConcatInputMLP(input_shape=s + s1, output_shape=a)

        inp = to_tensor(numpy.random.random((*batch_size, *s)), device="cpu")
        late_input = to_tensor(numpy.random.random((*batch_size, *s1)), device="cpu")
        print(model.forward(inp, late_input))

    def stest_multi_dim_normal21():
        """description"""
        s = (19,)
        s1 = (4,)
        batch_size = (100,)
        a = (2, 10)
        model = LateConcatInputMLP(input_shape=s + s1, output_shape=a)

        inp = to_tensor(numpy.random.random((*batch_size, *s)), device="cpu")
        late_input = to_tensor(numpy.random.random((*batch_size, *s1)), device="cpu")
        print(model.forward(inp, late_input))

    def stest_multi_dim_normal23121():
        """description"""
        s = (19,)
        s1 = (4,)
        batch_size = (100,)
        output_shape = (1, 2)
        model = LateConcatInputMLP(input_shape=s + s1, output_shape=output_shape)

        inp = to_tensor(numpy.random.random((*batch_size, *s)), device="cpu")
        late_input = to_tensor(numpy.random.random((*batch_size, *s1)), device="cpu")
        print(model.forward(inp, late_input))

    def stest_multi_dim_normal2321412121():
        """description"""
        s = (19,)
        s1 = (4,)
        batch_size = (100,)
        output_shape = 2
        model = LateConcatInputMLP(input_shape=s + s1, output_shape=output_shape)

        inp = to_tensor(
            numpy.random.random((*batch_size, *s)), device="cpu", dtype=float
        )
        late_input = to_tensor(
            numpy.random.random((*batch_size, *s1)), device="cpu", dtype=float
        )
        print(model.forward(inp, late_input).shape)

    # stest_normal()
    # stest_multi_dim_normal()
    # stest_multi_dim_normal21()
    # stest_multi_dim_normal23121()
    stest_multi_dim_normal2321412121()
