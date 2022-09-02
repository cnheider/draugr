#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/10/2019
           """

__all__ = ["set_lr", "exponential_lr_decay"]

import torch


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """

    :param optimizer:
    :type optimizer:
    :param lr:
    :type lr:"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def exponential_lr_decay(
    optim: torch.optim.Adam,
    *,
    initial_learning_rate: float,
    step: int,
    decay_rate: float,
    decay_steps: float,
) -> None:
    """Keras-style per-step learning rate decay.

    Description:
        This method will decay the learning rate continuously for all
        parameter groups in the optimizer by the function:

        lr = lr_0 * decay^(step / decay_steps)

    Args:
        optim (torch.optim.Adam): The optimizer to modify
        initial_learning_rate (float): The initial learning rate (lr_0)
        step (int): The current step in training
        decay_rate (float): The rate at which to decay the learning rate
        decay_steps (float): The number of steps before the learning rate is
                             applied in full.
    """
    decay_rate = decay_rate ** (step / decay_steps)
    lr = initial_learning_rate * decay_rate
    for group in optim.param_groups:
        group["lr"] = lr
