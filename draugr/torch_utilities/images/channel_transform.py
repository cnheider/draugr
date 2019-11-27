#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """


def reverse_rgb_channel_transform(inp: numpy.ndarray):
    inp = inp.transpose((1, 2, 0))
    inp = inp * 255.0
    inp = numpy.clip(inp, 0, 255).astype(numpy.uint8)
    return inp


def rgb_channel_transform(inp: numpy.ndarray):
    inp = inp / 255.0
    inp = numpy.clip(inp, 0, 1)
    inp = inp.transpose((2, 0, 1))
    return inp


def rgb_channel_transform_batch(inp: torch.Tensor):
    inp = inp[:, :, :, :3]
    inp = inp / 255.0
    inp = inp.clamp(0, 1)
    inp = inp.transpose(-1, 1)
    return inp


def reverse_rgb_channel_transform_batch(inp: torch.Tensor):
    inp = inp.transpose(-1, 1)
    inp = inp * 255.0
    inp = inp.clamp(0, 255)
    return inp.to(dtype=torch.uint8)


def torch_vision_normalize_batch(inp):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp[:, 0] = (inp[:, 0] - mean[0]) / std[0]
    inp[:, 1] = (inp[:, 1] - mean[1]) / std[1]
    inp[:, 2] = (inp[:, 2] - mean[2]) / std[2]

    return inp


def reverse_torch_vision_normalize_batch(inp):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp[:, 0] = inp[:, 0] * std[0] + mean[0]
    inp[:, 1] = inp[:, 1] * std[1] + mean[1]
    inp[:, 2] = inp[:, 2] * std[2] + mean[2]

    return inp
