#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy
import torch
from torch.distributions import Categorical
from torch.nn import functional

from draugr.torch_utilities.architectures.mlp import MLP
from draugr.torch_utilities.tensors.to_tensor import to_tensor

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["MultipleCategoricalMLP", "CategoricalMLP"]


class MultipleCategoricalMLP(MLP):
    @staticmethod
    def sample(distributions) -> Tuple:
        """

        :param distributions:
        :type distributions:
        :return:
        :rtype:
        """
        actions = [d.sample() for d in distributions][0]

        log_prob = [d.log_prob(action) for d, action in zip(distributions, actions)][0]

        actions = [a.to("cpu").numpy().tolist() for a in actions]
        return actions, log_prob

    @staticmethod
    def entropy(distributions) -> torch.tensor:
        """

        :param distributions:
        :type distributions:
        :return:
        :rtype:
        """
        return torch.mean(to_tensor([d.entropy() for d in distributions]))

    def forward(self, *x, **kwargs) -> List:
        """

        :param x:
        :type x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        out = super().forward(*x, **kwargs)
        outs = []
        for o in out:
            outs.append(Categorical(logits=functional.log_softmax(o, dim=-1)))

        return outs


class CategoricalMLP(MLP):
    def forward(self, *x, **kwargs) -> Categorical:
        """

        :param x:
        :type x:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return Categorical(
            logits=functional.log_softmax(super().forward(*x, **kwargs), dim=-1)
        )


if __name__ == "__main__":

    def multi_cat():
        """description"""
        s = (2, 2)
        a = (2, 2)
        model = MultipleCategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, s[0]), device="cpu")
        print(model.sample(model(inp, inp)))

    def single_cat():
        """description"""
        s = (1, 2)
        a = (2,)
        model = CategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, s[0]), device="cpu")
        inp2 = to_tensor(numpy.random.rand(64, s[1]), device="cpu")
        print(model(inp, inp2).sample())

    def single_cat2():
        """description"""
        s = (4,)
        a = (2,)
        model = CategoricalMLP(input_shape=s, output_shape=a)

        inp = to_tensor(numpy.random.rand(64, s[0]), device="cpu")
        print(model(inp).sample())

    multi_cat()
    single_cat()
    single_cat2()
