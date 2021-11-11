#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

          Checklist items for ensuring optimisation is performed as expected.


verify loss @ init. Verify that your loss starts at the correct loss value. E.g. if you initialize your final layer correctly you should measure -log(1/n_classes) on a softmax at initialization. The same default values can be derived for L2 regression, Huber losses, etc.

init well. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.

overfit one batch. Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage.

verify decreasing training loss. At this stage you will hopefully be underfitting on your dataset because you’re working with a toy model. Try to increase its capacity just a bit. Did your training loss go down as it should?

visualize just before the net. The unambiguously correct place to visualize your data is immediately before your y_hat = model(x) (or sess.run in tf). That is - you want to visualize exactly what goes into your network, decoding that raw tensor of data and labels into visualizations. This is the only “source of truth”. I can’t count the number of times this has saved me and revealed problems in data preprocessing and augmentation.

use backprop to chart dependencies. Your deep learning code will often contain complicated, vectorized, and broadcasted operations. A relatively common bug I’ve come across a few times is that people get this wrong (e.g. they use view instead of transpose/permute somewhere) and inadvertently mix information across the batch dimension. It is a depressing fact that your network will typically still train okay because it will learn to ignore data from the other examples. One way to debug this (and other related problems) is to set the loss to be something trivial like the sum of all outputs of example i, run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the i-th input. The same strategy can be used to e.g. ensure that your autoregressive model at time t only depends on 1..t-1. More generally, gradients give you information about what depends on what in your network, which can be useful for debugging.

#TODO: NOT DONE, FINISH!

           Created on 07/07/2020
           """

from random import random

import torch

from draugr.torch_utilities.tensors import to_tensor

__all__ = ["overfit_single_batch"]


# __all__ = ['init_softmax_loss','overfit_single_batch']


def init_softmax_loss():
    """
    #TODO: NOT DONE, FINISH!"""
    batch_size = 16
    input_f = 4
    n_classes = 10

    model = torch.nn.Sequential(
        torch.nn.Linear(input_f, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, n_classes),
        torch.nn.LogSoftmax(-1),
    )

    for p in model.parameters():
        torch.nn.init.constant_(p, 1)

    input = to_tensor([range(input_f) for _ in range(batch_size)])

    print(input)  # Visualise input just before forward
    out = model(input)
    print(out)

    target = to_tensor(
        [int(n_classes * random()) for _ in range(batch_size)], dtype=torch.long
    )
    # loss = torch.nn.MSELoss()(out, target)
    loss = torch.nn.NLLLoss()(out, target)

    expected_loss = -torch.log(to_tensor(1 / n_classes))
    assert expected_loss - loss < 0.1, f"{expected_loss}!={loss}"


def overfit_single_batch():
    """
    #TODO: NOT DONE, FINISH!
    :return:"""
    input_f = 4
    n_classes = 10

    model = torch.nn.Sequential(
        torch.nn.Linear(input_f, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, n_classes),
        torch.nn.LogSoftmax(-1),
    )

    input = to_tensor([range(input_f)])

    print(input)  # Visualise input just before forward
    out = model(input)
    print(out)

    target = torch.zeros(n_classes)
    target[0] = 1
    # loss = torch.nn.MSELoss()(out, target)
    loss = torch.nn.CrossEntropyLoss()(out, target)

    expected_loss = -torch.log(to_tensor(1 / n_classes))
    assert expected_loss == loss, f"{expected_loss}!={loss}"


if __name__ == "__main__":
    init_softmax_loss()
    # overfit_single_batch()
