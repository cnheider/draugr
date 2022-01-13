#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/07/2020
           """

__all__ = ["plot_grad_flow"]

import numpy
import torch
from matplotlib import pyplot
from matplotlib.lines import Line2D

from draugr.torch_utilities.optimisation.parameters import normal_init_weights


def plot_grad_flow(
    model: torch.nn.Module,
    lines: bool = True,
    alpha: float = 0.5,
    line_width: float = 1.0,
) -> None:
    """

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: After loss.backwards(), use plot_grad_flow(model) to visualize the gradient flow of model

    :param model:
    :type model:
    :param lines:
    :type lines:
    :param alpha:
    :type alpha:
    :param line_width:
    :type line_width:"""
    assert 0.0 < alpha <= 1.0
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in model.named_parameters():
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            grad_abs = p.grad.abs()
            ave_grads.append(grad_abs.mean())
            max_grads.append(grad_abs.max())

    if lines:
        pyplot.plot(max_grads, alpha=alpha, linewidth=line_width, color="r")
        pyplot.plot(ave_grads, alpha=alpha, linewidth=line_width, color="g")
    else:
        pyplot.bar(
            numpy.arange(len(max_grads)),
            max_grads,
            alpha=alpha,
            linewidth=line_width,
            color="r",
        )
        pyplot.bar(
            numpy.arange(len(max_grads)),
            ave_grads,
            alpha=alpha,
            linewidth=line_width,
            color="g",
        )

    pyplot.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    pyplot.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    pyplot.xlim(left=0, right=len(ave_grads))
    max_g = max(max_grads)
    margin = max_g * 1.1
    pyplot.ylim(
        bottom=max_g - margin, top=margin
    )  # zoom in on the lower gradient regions
    pyplot.xlabel("Layers")
    pyplot.ylabel("Gradient Magnitude")
    pyplot.title("Gradient Flow")
    pyplot.grid(True)
    pyplot.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        input = torch.randn(10, 50, requires_grad=True)
        target = torch.empty(10, dtype=torch.long).random_(2)
        model = torch.nn.Sequential(
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 2),
        )
        normal_init_weights(model, std=1.2)
        criterion = torch.nn.CrossEntropyLoss()
        outputs = model(input)
        loss = criterion(outputs, target)
        loss.backward()
        plot_grad_flow(model)
        pyplot.show()

    a()
