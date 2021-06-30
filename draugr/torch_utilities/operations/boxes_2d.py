#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

import torch

__all__ = ["minmax_to_xywh_torch"]


def minmax_to_xywh_torch(boxes: torch.Tensor) -> torch.tensor:
    """

    :param boxes:
    :type boxes:
    :return:
    :rtype:"""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
