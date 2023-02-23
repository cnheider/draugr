#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/06/2020
           """

import torch
from torch.nn import functional

__all__ = ["get_num_samples", "get_archetypes", "archetypal_loss"]


def get_num_samples(
    targets: torch.Tensor, num_classes: torch.Tensor, dtype: torch.dtype = None
) -> torch.Tensor:
    """

    :param targets:
    :type targets:
    :param num_classes:
    :type num_classes:
    :param dtype:
    :type dtype:
    :return:
    :rtype:
    """
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets.type(torch.int64), ones)
    return num_samples


def get_archetypes(
    embeddings: torch.FloatTensor, targets: torch.LongTensor, num_classes: int
) -> torch.FloatTensor:
    """Compute the archetypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    archetypes : `torch.FloatTensor` instance
        A tensor containing the archetypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings).type(torch.int64)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes


def archetypal_loss(
    archetypes: torch.FloatTensor,
    embeddings: torch.FloatTensor,
    targets: torch.LongTensor,
    **kwargs
) -> torch.FloatTensor:
    """Compute the loss (i.e. negative log-likelihood) for the archetypal
    network, on the test/query points.

    Parameters
    ----------
    archetypes : `torch.FloatTensor` instance
        A tensor containing the archetypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.

    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    squared_distances = torch.sum(
        (archetypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
    )
    return functional.cross_entropy(-squared_distances, targets, **kwargs)
