#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """


import time

import jax.numpy as jnp
from jax import (
    jit,
    grad,
    # pip install --upgrade "jax[cpu]"
    # pip install jaxlib
)  # pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
from jax.scipy.special import logsumexp
from numpy import random

from datasets import mnist


def init_random_params(scale, layer_sizes, rng=random.RandomState(0)):
    """

    :param scale:
    :type scale:
    :param layer_sizes:
    :type layer_sizes:
    :param rng:
    :type rng:
    :return:
    :rtype:
    """
    return [
        (scale * rng.randn(m, n), scale * rng.randn(n))
        for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def predict(params, inputs):
    """

    :param params:
    :type params:
    :param inputs:
    :type inputs:
    :return:
    :rtype:
    """
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(params, batch):
    """

    :param params:
    :type params:
    :param batch:
    :type batch:
    :return:
    :rtype:
    """
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    """

    :param params:
    :type params:
    :param batch:
    :type batch:
    :return:
    :rtype:
    """
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
    layer_sizes = [784, 1024, 1024, 10]
    param_scale = 0.1
    step_size = 0.001
    num_epochs = 10
    batch_size = 128

    train_images, train_labels, test_images, test_labels = mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        """ """
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    @jit
    def update(params, batch):
        """

        :param params:
        :type params:
        :param batch:
        :type batch:
        :return:
        :rtype:
        """
        grads = grad(loss)(params, batch)
        return [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)
        ]

    params = init_random_params(param_scale, layer_sizes)
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            params = update(params, next(batches))
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")
