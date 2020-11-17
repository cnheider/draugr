#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/03/2020
           """

import torch

__all__ = ["warmup_lr_scheduler"]


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """

    :param optimizer:
    :type optimizer:
    :param warmup_iters:
    :type warmup_iters:
    :param warmup_factor:
    :type warmup_factor:
    :return:
    :rtype:"""

    def f(x):
        """

        :param x:
        :type x:
        :return:
        :rtype:"""
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
