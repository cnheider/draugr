#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random

import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["torch_seed"]


def torch_seed(s: int):
    """seeding for reproducibility"""
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    numpy.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
