#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02-12-2020
           """

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from draugr.torch_utilities.sessions.model_sessions import TorchEvalSession
from draugr.torch_utilities.system.device import global_torch_device
from warg import kws_sink

__all__ = ["find_n_misclassified"]


def find_n_misclassified(
    model: torch.nn.Module,
    evaluation_loader: DataLoader,
    *,
    mapper: callable = kws_sink,
    n: int = 10,
    device: torch.device = global_torch_device(),
) -> None:
    """ """
    j = 0
    num_samples = len(evaluation_loader)
    with TorchEvalSession(model):
        for i, (waveform, target) in tqdm(enumerate(evaluation_loader), total=n):
            output = mapper(model(waveform.to(device)).argmax(dim=-1).squeeze())
            truth = mapper(target)
            if output != truth:
                print(
                    f"Data point #{i}/{num_samples}. Expected: {truth}. Predicted: {output}."
                )
                j += 1
                if j >= n:
                    break
        else:
            print("All examples in this dataset were correctly classified!")
            print("In this case, let's just look at the last data point")
            print(f"Data point #{i}. Expected: {truth}. Predicted: {output}.")
