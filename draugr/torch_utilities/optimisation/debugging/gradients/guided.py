#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-02-2021
           """

import numpy
import torch
from torch.autograd import Function

__all__ = ["GuidedBackPropReLUModel", "GuidedBackPropReLU"]


class GuidedBackPropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        """

        :param self:
        :param input_img:
        :return:
        """
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask
        )
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """

        :param self:
        :param grad_output:
        :return:
        """
        input_img, output = self.saved_tensors

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(input_img.size()).type_as(input_img),
                grad_output,
                positive_mask_1,
            ),
            positive_mask_2,
        )
        return grad_input


class GuidedBackPropReLUModel:
    """ """

    def __init__(self, model, use_cuda):
        self._model = model
        self._model.eval()
        self._use_cuda = use_cuda
        if self._use_cuda:
            self._model = self._model.cuda()

        def recursive_relu_apply(module_top):
            """

            :param module_top:
            """
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackPropReLU.apply

        recursive_relu_apply(self._model)  # replace ReLU with GuidedBackpropReLU

    def forward(self, input_img):
        """

        :param input_img:
        :return:
        """
        return self._model(input_img)

    def __call__(self, input_img, target_category=None):
        if self._use_cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = numpy.argmax(output.cpu().data.numpy())

        one_hot = numpy.zeros((1, output.size()[-1]), dtype=numpy.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self._use_cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output
