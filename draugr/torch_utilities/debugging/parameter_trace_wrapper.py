#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/07/2020
           """

import functools
from collections import OrderedDict


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
import torch
from torch import nn


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, (obj, *attr.split(".")))


class IntermediateLayerGetter:
    def __init__(
        self, model: torch.nn.Module, return_layers: list = None, keep_output=True
    ):
        """Wraps a Pytorch module to get intermediate values

    Arguments:
        model {nn.module} -- The Pytorch module to call
        return_layers {dict} -- Dictionary with the selected submodules
        to return the output (format: {[current_module_name]: [desired_output_name]},
        current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)

    Keyword Arguments:
        keep_output {bool} -- If True model_output contains the final model's output
        in the other case model_output is None (default: {True})

    Returns:
        (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are
        your desired_output_name (s) and their values are the returned tensors
        of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
        See keep_output argument for model_output description.
        In case a submodule is called more than one time, all it's outputs are
        stored in a list.
    """
        self._model = model
        if return_layers:
            self.return_layers = return_layers
        else:
            self.return_layers = model.parameters(recurse=True)
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)

            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f"Module {name} not found")
            handles.append(h)

        output = self._model(*args, **kwargs)

        for h in handles:
            h.remove()

        return ret, output


if __name__ == "__main__":

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(2, 2)
            self.fc2 = nn.Linear(2, 2)
            self.nested = nn.Sequential(
                nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3)), nn.Linear(3, 1)
            )
            self.interaction_idty = (
                nn.Identity()
            )  # Simple trick for operations not performed as modules

        def forward(self, x):
            x1 = self.fc1(x)
            x2 = self.fc2(x)

            interaction = x1 * x2
            self.interaction_idty(interaction)

            x_out = self.nested(interaction)

            return x_out

    model = Model()
    return_layers = {
        "fc2": "fc2",
        "nested.0.1": "nested",
        "interaction_idty": "interaction",
    }
    mid_getter = IntermediateLayerGetter(
        model, return_layers=return_layers, keep_output=True
    )
    mid_outputs, model_output = mid_getter(torch.randn(1, 2))

    print(model_output)
    print(mid_outputs)
