#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/02/2020
           """

import torch


def normalise(x: torch.tensor, eps=1e-6) -> torch.tensor:
    return


def standardise(x: torch.tensor, eps=1e-6) -> torch.tensor:
    """

  :param x:
  :return:
  """
    x -= x.mean()
    x /= x.std() + eps
    return x


if __name__ == "__main__":
    print(standardise(torch.ones(10)))
    print(standardise(torch.ones((10, 1))))
    print(standardise(torch.ones((1, 10))))

    print(standardise(torch.diag(torch.ones(3))))

    print(standardise(torch.ones((1, 10)) * torch.rand((1, 10))))

    print(standardise(torch.rand((1, 10))))
