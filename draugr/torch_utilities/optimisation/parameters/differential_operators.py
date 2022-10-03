#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/30/22
           """

__all__ = ["laplace", "divergence", "gradient", "hessian", "jacobian"]

from typing import Tuple

import torch
from torch.autograd import grad


def laplace(y: torch.Tensor, x: torch.Tensor) -> float:
    """
    laplacian of y wrt x

    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)

    :param y:
    :type y:
    :param x:
    :type x:
    :return:
    :rtype:
    """
    return divergence(gradient(y, x), x)


def divergence(y: torch.Tensor, x: torch.Tensor) -> float:
    """
    divergence of y wrt x

    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)

    :param y:
    :type y:
    :param x:
    :type x:
    :return:
    :rtype:
    """
    div = 0.0
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(
            y[..., i], x, torch.ones_like(y[..., i]), create_graph=True
        )[0][..., i : i + 1]
    return div


def gradient(
    y: torch.Tensor, x: torch.Tensor, grad_outputs: torch.Tensor = None
) -> torch.Tensor:
    """
    gradient of y wrt x

    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)

    :param y:
    :type y:
    :param x:
    :type x:
    :param grad_outputs:
    :type grad_outputs:
    :return:
    :rtype:
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    return torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]


def hessian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(
        meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]
    ).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][
                ..., :
            ]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def jacobian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """jacobian of y wrt x"""
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(
        y.device
    )  # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status
