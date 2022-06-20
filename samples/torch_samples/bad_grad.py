#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """

if __name__ == "__main__":
    import torch

    from draugr.torch_utilities import register_bad_grad_hooks

    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    get_dot = register_bad_grad_hooks(z)
    z.backward()
    dot = get_dot()
    # dot.save('tmp.dot') # to get .dot
    # dot.render('tmp') # to get SVG
    dot  # in Jupyter, you can just render the variable
