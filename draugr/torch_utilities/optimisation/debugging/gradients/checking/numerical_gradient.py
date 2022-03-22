#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/07/2020
           """

__all__ = []

import copy
import math

import torch

from draugr.torch_utilities.optimisation.parameters import (
    named_trainable_parameters,
    normal_init_weights,
    trainable_parameters,
)
from draugr.torch_utilities.sessions import TorchEvalSession
from warg import ContextWrapper


def loss_grad_check(
    model: torch.nn.Module,
    loss_fn: callable,
    input: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
    error_tolerance: float = 1e-5,
) -> None:
    """
    two sided gradient numerical approximation
    DOES not work, please refer to torch/autograd/gradcheck.py

    :param input:
    :type input:
    :param target:
    :type target:
    :param error_tolerance:
    :type error_tolerance:
    :param model:
    :type model:
    :param loss_fn:
    :type loss_fn:
    :param epsilon:
    :type epsilon:
    :return:
    :rtype:"""
    assert epsilon > 0.0
    c_model = copy.deepcopy(model)

    loss = loss_fn(model(input), target)
    loss.backward()
    compute_gradients = False
    with ContextWrapper(torch.no_grad, not compute_gradients):
        with TorchEvalSession(model):
            for (n, c_p), p in zip(
                named_trainable_parameters(c_model).items(), trainable_parameters(model)
            ):
                for i, c_p_o in enumerate(c_p):
                    a = c_p_o.size()
                    if len(a) > 0:
                        for j in range(a[0]):
                            cp_orig = c_p.data.clone()

                            c_p[i][j] += epsilon  # positive
                            loss_p = loss_fn(
                                c_model(input.clone()), target.clone()
                            ).clone()

                            c_p.data = cp_orig

                            c_p[i][j] -= epsilon  # negative
                            loss_n = loss_fn(
                                c_model(input.clone()), target.clone()
                            ).clone()

                            c_p.data = cp_orig

                            if (
                                True
                            ):  # TODO: make check based on the entire set of parameters at once
                                grad_approx = (loss_p - loss_n) / (2 * epsilon)

                                denom = math.sqrt(grad_approx**2) + math.sqrt(
                                    p.grad[i][j] ** 2
                                )
                                if denom > 0:
                                    deviance = (
                                        math.sqrt((grad_approx - p.grad[i][j]) ** 2)
                                        / denom
                                    )
                                    # assert torch.sign(grad_approx) == torch.sign(p.grad[i][j]), f'apprx: {grad_approx}, analytical {p.grad[i][j]}'
                                    assert (
                                        deviance <= error_tolerance
                                    ), f"Numerical gradient approximation of parameter {n} deviates larger than tolerance {error_tolerance}, deviance: {deviance}, approx:{grad_approx, loss_p, loss_n}, p.grad[i][j]:{p.grad[i][j]}"
                                else:
                                    pass
                                    # print(grad_approx,denom)


if __name__ == "__main__":

    def stest_return_duplicate() -> None:
        """
        :rtype: None
        """
        from torch.autograd import Function, gradcheck, gradgradcheck

        class DoubleDuplicate(Function):
            @staticmethod
            def forward(ctx, x):
                """ """
                output = x * 2
                return output, output

            @staticmethod
            def backward(ctx, grad1, grad2):
                """ """
                return grad1 * 2 + grad2 * 2

        def fn(x):
            """ """
            a, b = DoubleDuplicate.apply(x)
            return a + b

        x = torch.randn(5, 5, requires_grad=True, dtype=torch.double)
        gradcheck(fn, [x], eps=1e-6)
        gradgradcheck(fn, [x])

    def a() -> None:
        """
        :rtype: None
        """
        #    from torch.testing import _get_default_tolerance

        input = torch.randn(5, 5, requires_grad=True, dtype=torch.double)
        target = torch.randn(5, 5, requires_grad=False, dtype=torch.double)
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.Linear(50, 5),
        ).double()
        normal_init_weights(model)
        criterion = torch.nn.MSELoss()
        # _get_default_tolerance(input)
        loss_grad_check(model, criterion, input, target)

    # a()
    stest_return_duplicate()
