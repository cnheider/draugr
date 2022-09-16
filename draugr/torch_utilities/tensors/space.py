#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/2/22
           """

__all__ = ["linspace"]

import torch


def linspace(start: torch.Tensor, stop: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Generalization of linspace to arbitrary tensors.

    Args:
        start (torch.Tensor): Minimum 1D tensor. Same length as stop.
        stop (torch.Tensor): Maximum 1D tensor. Same length as start.
        num_samples (int): The number of samples to take from the linear range.

    Returns:
        torch.Tensor: (D, num_samples) tensor of linearly interpolated
                      samples.
    """
    return start.unsqueeze(-1) + (
        torch.linspace(0, 1, num_samples).unsqueeze(0)  # samples
        * (stop - start).unsqueeze(-1)  # diff
    )
