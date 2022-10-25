#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 10/3/22
           """

__all__ = ["Sine", "Cosine"]

import torch
from torch import nn


class Sine(nn.Module):
    """ """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        :param input:
        :type input:
        :return:
        :rtype:
        """
        return torch.sin(
            30 * input
        )  # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30


class Cosine(nn.Module):
    """ """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        :param input:
        :type input:
        :return:
        :rtype:
        """
        return torch.cos(
            30 * input
        )  # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30


if __name__ == "__main__":

    def _main():
        from draugr.torch_utilities import to_tensor

        print(Sine()(to_tensor(1.2)))

    _main()
