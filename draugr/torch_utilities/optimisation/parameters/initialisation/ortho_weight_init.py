#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

import numpy
import torch

__all__ = ["orthogonal_weights"]


def orthogonal_weights(shape, scale: float = 1.0) -> torch.tensor:
    r"""PyTorch port of ortho_init from baselines.a2c.utils"""
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (numpy.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = numpy.random.normal(0.0, 1.0, flat_shape)
    u, _, v = numpy.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(numpy.float32))
    if len(shape) == 4:
        return torch.from_numpy(
            (scale * q[:, : shape[1], : shape[2]]).astype(numpy.float32)
        )
