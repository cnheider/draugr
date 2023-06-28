#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/22/22
           """

__all__ = ["VariationalAutoEncoder"]

from abc import abstractmethod

import torch
from draugr.torch_utilities.tensors.to_tensor import to_tensor


class VariationalAutoEncoder(torch.nn.Module):
    """
    General purpose variational auto encoder
    """

    def __init__(self, latent_size: int = 10):
        super().__init__()
        self._latent_size = latent_size

    @abstractmethod
    def encode(self, *x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          *x:
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, *x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          *x:
        """
        raise NotImplementedError

    def sample(self, *x, num=1) -> torch.Tensor:
        """

        :param x:
        :type x:
        :param num:
        :type num:
        :return:
        :rtype:"""
        return self.decode(
            # z
            torch.randn(num, self._latent_size).to(
                device=next(self.parameters()).device
            ),
            *x,
        ).to("cpu")

    @staticmethod
    def reparameterise(mean, log_var) -> torch.Tensor:
        """
        TODO: MOVE TO SUB ANOTHER CLASS, this is for a guassian specialisation of VAE

                reparameterisation trick

                :param mean:
                :type mean:
                :param log_var:
                :type log_var:
                :return:
                :rtype:"""
        std = log_var.div(2).exp()  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mean)  # Reparameterise distribution

    def sample_from(self, *encoding) -> torch.Tensor:
        """

        Args:
          *encoding:

        Returns:

        """
        sample = to_tensor(*encoding).to(device=next(self.parameters()).device)
        assert sample.shape[-1] == self._latent_size, (
            f"sample.shape[-1]:{sample.shape[-1]} !="
            f" self._encoding_size:{self._latent_size}"
        )
        return self.decode(*sample).to("cpu")
