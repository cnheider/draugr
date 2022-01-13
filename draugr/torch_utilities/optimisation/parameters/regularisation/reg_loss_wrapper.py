#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/07/2020
           """

__all__ = ["orthogonal_reg", "RegLossWrapper"]

import torch
from torch import nn


class RegLossWrapper(torch.nn.Module):
    """ """

    def __init__(self, loss, model: torch.nn.Module, factor: float = 0.0005):
        super().__init__()
        self.loss = loss
        self.l1_crit = torch.nn.L1Loss()
        self.a = torch.zeros(1)
        self.factor = factor
        self.params = []
        for name, param in model.named_parameters():
            if "bias" not in name:
                self.params.append(param)

    def forward(self, *loss, **kwargs) -> torch.Tensor:
        """ """
        return self.loss(*loss) + self.factor * sum(
            [self.l1_crit(p, self.a) for p in self.params]
        )


def orthogonal_reg(model, reg=1e-6):
    """ """
    with torch.enable_grad():
        orth_loss = torch.zeros(1)
        for name, param in model.named_parameters():
            if "bias" not in name:
                param_flat = param.reshape(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0])
                orth_loss += reg * sym.abs().sum()


if __name__ == "__main__":

    def bb() -> None:
        """
        :rtype: None
        """
        from draugr.torch_utilities.optimisation.parameters.initialisation import (
            normal_init_weights,
        )

        input = torch.randn(3, 5, requires_grad=True)
        model = torch.nn.Linear(5, 5)
        normal_init_weights(model)
        target = torch.empty(3, dtype=torch.long).random_(5)
        loss_fn = RegLossWrapper(nn.CrossEntropyLoss(), model)

        def a(m):
            """ """
            loss = loss_fn(m(input), target)
            print(loss)
            loss.backward()

        a(model)
        a(model)
        normal_init_weights(model, 1.0)
        a(model)

    bb()
