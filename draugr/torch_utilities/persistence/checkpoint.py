#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 06-04-2021
           """

from pathlib import Path

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_optimiser(
    *,
    optimiser: Optimizer,
    optimiser_save_path: Path,
    raise_on_existing: bool = False,
) -> None:
    """

    :param optimiser:
    :param optimiser_save_path:
    :param raise_on_existing:"""
    if raise_on_existing and optimiser_save_path.exists():
        raise FileExistsError(f"{optimiser_save_path} exists!")
    torch.save(optimiser, str(optimiser_save_path))


def save_checkpoint(PATH: Path, epoch, model, optimiser, loss):
    """

    :param PATH:
    :param epoch:
    :param model:
    :param optimiser:
    :param loss:
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "loss": loss,
        },
        PATH,
    )

    PATH.with_suffix(".tar")


def load_checkpoint(PATH: Path, model, optimizer):
    """

    :param PATH:
    :param model:
    :param optimizer:
    :return:
    """
    checkpoint = torch.load(PATH)

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimiser_state_dict"])
    loss = checkpoint["loss"]

    return epoch, model, optimizer, loss


if __name__ == "__main__":

    def main() -> None:
        """
        :rtype: None
        """
        pass
        # model = TheModelClass(args, **kwargs)
        # optimizer = TheOptimizerClass(args, **kwargs)

    def multi() -> None:
        """
        :rtype: None
        """
        pass

    # checkpoint = torch.load(PATH)
    # modelA.load_state_dict(checkpoint['modelA_state_dict'])
    # modelB.load_state_dict(checkpoint['modelB_state_dict'])
    # optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
    # optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
