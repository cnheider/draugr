#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02-12-2020
           """

__all__ = ["get_model_hash"]

import hashlib

import torch


def get_model_hash(model: torch.nn.Module) -> str:
    """ """
    model_repr = "".join([str(a) for a in model.named_children()])
    # print(model_repr)
    model_hash = hashlib.md5(model_repr.encode("utf-8")).hexdigest()
    return model_hash
