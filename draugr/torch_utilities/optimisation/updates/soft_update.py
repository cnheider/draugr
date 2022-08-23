#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import torch

__all__ = ["soft_update"]


def soft_update(
    target: torch.nn.Module, source: torch.nn.Module, tau: float
) -> torch.nn.Module:
    """
    Soft Update a target module

    :param target:
    :type target:
    :param source:
    :type source:
    :param tau:
    :type tau:
    :return:
    :rtype:
    """
    assert 0 <= tau <= 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    return target
