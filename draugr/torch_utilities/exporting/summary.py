#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-06-2021
           """

__all__ = ["to_latex_table"]

import torch.nn


def to_latex_table(m: torch.nn.Module) -> str:
    return m.__repr__()


if __name__ == "__main__":

    from torch import nn
    from torch.nn import functional as F

    def aiasujd():
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

                self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
                self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

            def forward(self, x):
                h = self.conv0(x)
                h = self.conv1(h)
                return h

        model = Model()

        print(to_latex_table(model))

    def uiahsduhaw():
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        model = Model()

        print(to_latex_table(model))

    uiahsduhaw()
