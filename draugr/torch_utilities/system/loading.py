#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """

__all__ = ["default_worker_init_fn", "global_pin_memory"]

import random
from typing import Union

import numpy
import torch

from draugr.torch_utilities.system.device import global_torch_device


def default_worker_init_fn() -> None:
    """ """
    worker_seed = torch.initial_seed()
    torch.random.seed(worker_seed)
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)


def global_pin_memory(
    num_workers: int,
    preference: Union[torch.device, bool, str] = True,
    update_num_thread_for_pinned_loader: bool = True,
) -> bool:
    """

    #Some weird behaviour of when copying to pinned memory with more workers observed
      :param num_workers:
      :param preference:
      :param update_num_thread_for_pinned_loader:
      :return:
    """
    if isinstance(preference, (torch.device, str)):
        if isinstance(preference, torch.device):
            res = True if "cuda" in torch.device.type else False
        else:
            if "cuda" in preference:
                res = True
            elif preference == "cpu":
                res = False
            else:
                raise NotImplemented
    else:
        res = preference if "cuda" in global_torch_device().type else False

    if num_workers > 1:
        res = False

    if update_num_thread_for_pinned_loader:
        if res and torch.get_num_threads() != 1:
            torch.set_num_threads(1)

    return res


if __name__ == "__main__":

    def main() -> None:
        """
        :rtype: None
        """
        print(global_pin_memory(0))
        print(global_pin_memory(1))
        print(global_pin_memory(2))
        print()
        print(global_pin_memory(0, "cpu"))
        print(global_pin_memory(1, "cpu"))
        print(global_pin_memory(1, "cuda"))
        print()
        print(global_pin_memory(0, "cuda:0"))
        print(global_pin_memory(1, "cuda:0"))

    main()
