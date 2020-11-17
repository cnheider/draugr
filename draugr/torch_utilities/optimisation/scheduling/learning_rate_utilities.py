#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/10/2019
           """

__all__ = ["set_lr"]


def set_lr(optimizer, lr):
    """

    :param optimizer:
    :type optimizer:
    :param lr:
    :type lr:"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
