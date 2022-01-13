#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

from collections import OrderedDict
from itertools import tee

import torch

from draugr.torch_utilities.optimisation.parameters.freezing import freeze_parameters
from warg import AlsoDecorator

__all__ = [
    "TorchEvalSession",
    "TorchTrainSession",
    "TorchFrozenModelSession",
    "TorchTrainingSession",
]


class TorchEvalSession(AlsoDecorator):
    """
    # speed up evaluating after training finished"""

    def __init__(self, model: torch.nn.Module, no_side_effect: bool = True):
        self.model = model
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = model.training

    def __enter__(self):
        # self.model.eval()
        self.model.train(False)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            self.model.train(self.prev_state)
        else:
            self.model.train(True)


class TorchTrainSession(AlsoDecorator):
    """
    # speed up evaluating after training finished"""

    def __init__(self, model: torch.nn.Module, no_side_effect: bool = True):
        self.model = model
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = model.training

    def __enter__(self):
        self.model.train(True)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            self.model.train(self.prev_state)
        else:
            self.model.train(False)


TorchTrainingSession = TorchTrainSession


class TorchFrozenModelSession(AlsoDecorator):
    """ """

    def __init__(self, model: torch.nn.Module, no_side_effect: bool = True):
        self.model = model
        self._no_side_effect = no_side_effect
        self.params_1, self.params_2, self.params_3 = tee(model.parameters(True), 3)
        if no_side_effect:
            self.previous_states = [a.requires_grad for a in self.params_3]

    def __enter__(self):
        freeze_parameters(self.params_1, True)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            [p.requires_grad_(rg) for p, rg in zip(self.params_2, self.previous_states)]
        else:
            freeze_parameters(self.params_2, False)


if __name__ == "__main__":

    def main() -> None:
        """
        :rtype: None
        """
        a = torch.nn.Sequential(
            OrderedDict(l1=torch.nn.Linear(3, 5), l2=torch.nn.Linear(5, 2))
        )
        p_iter = iter(a.parameters(True))
        l1_w = next(p_iter)
        l1_bias = next(p_iter)
        l1_bias.requires_grad_(False)

        def initial():
            """ """
            for p in a.parameters(True):
                print(p.requires_grad)

        @TorchFrozenModelSession(a)
        def frozen():
            """ """
            for p in a.parameters(True):
                print(p.requires_grad)

        def frozen_session():
            """ """
            with TorchFrozenModelSession(a):
                for p in a.parameters(True):
                    print(p.requires_grad)

        initial()
        print()
        frozen()
        print()
        initial()
        print()
        frozen_session()
        print()
        initial()

    main()
