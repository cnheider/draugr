#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

import torch

from warg.decorators.kw_passing import AlsoDecorator

__all__ = ["TorchEvalSession", "TorchTrainSession"]


class TorchEvalSession(AlsoDecorator):
    """
# speed up evaluating after training finished

"""

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
# speed up evaluating after training finished

"""

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
