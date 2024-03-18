#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 10/3/22
           """

__all__ = ["RBFLayer"]

import numpy
import torch
from torch import nn


class RBFLayer(nn.Module):
    """Transforms incoming data using a given radial basis function.
    - Input: (1, N, in_features) where N is an arbitrary batch size
    - Output: (1, N, out_features) where N is an arbitrary batch size"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(numpy.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        """ """
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        """

        :param input:
        :type input:
        :return:
        :rtype:
        """
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        """

        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


if __name__ == "__main__":

    def _main(): ...

    _main()
