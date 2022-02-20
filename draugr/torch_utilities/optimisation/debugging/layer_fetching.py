#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/07/2020
           """

import functools
from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn

__all__ = ["IntermediateLayerGetter"]


class IntermediateLayerGetter:
    """ """

    def __init__(
        self,
        model: torch.nn.Module,
        return_layers: dict = None,
    ):
        """
        Wraps a Pytorch module to get intermediate values, eg for getting intermediate activations

        Arguments:
        model {nn.module} -- The Pytorch module to call
        return_layers {dict} -- Dictionary with the selected submodules
        to return the output (format: {[current_module_name]: [desired_output_name]},
        current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)

        Returns:
        (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are
        your desired_output_name (s) and their values are the returned tensors
        of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).

        In case a submodule is called more than one time, all it's outputs are
        stored in a list."""
        self._model = model
        if return_layers:
            self.return_layers = return_layers.items()
        else:
            self.return_layers = {k: k for k, v in model.named_modules()}.items()

    @staticmethod
    def reduce_getattr(obj, attr, *args):
        """
        # using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

        :param obj:
        :type obj:
        :param attr:
        :type attr:
        :param args:
        :type args:
        :return:
        :rtype:"""

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, (obj, *attr.split(".")))

    def __call__(self, *args, **kwargs) -> Tuple:
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers:
            if name == "":
                continue  # TODO: Fail maybe?
            layer = IntermediateLayerGetter.reduce_getattr(self._model, name)

            if isinstance(layer, torch.nn.Module):  # Should be a torch module!

                def hook(
                    module,
                    input,
                    output,
                    *,
                    new_name_=new_name,  # Hack for new func, otherwise func is overriden. # BUG?
                ):
                    """

                              :param new_name_:
                    :param module:
                    :type module:
                    :param input:
                    :type input:
                    :param output:
                    :type output:"""
                    if new_name_ in ret:
                        cur_val = ret[new_name_]
                        if type(cur_val) is list:
                            ret[new_name_].append(output)
                        else:
                            ret[new_name_] = [cur_val, output]
                    else:
                        ret[new_name_] = output

                try:
                    h = layer.register_forward_hook(hook)
                    handles.append(h)
                except AttributeError as e:
                    raise AttributeError(f"Module {name} not found")
            else:
                raise AttributeError(
                    f"Requested module activation with {name} was not a module but {type(layer)}"
                )

        output = self._model(*args, **kwargs)

        for h in handles:
            h.remove()

        return ret, output


if __name__ == "__main__":

    def adsad() -> None:
        """
        :rtype: None
        """

        class Model(nn.Module):
            """ """

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
                """

                :param x:
                :type x:
                :return:
                :rtype:"""
                x1 = self.fc1(x)
                x2 = self.fc2(x)

                interaction = x1 * x2
                self.interaction_idty(interaction)
                return self.nested(interaction)

        model = Model()
        return_layers = {
            "fc2": "fc2",
            "nested.0.1": "nested",
            "interaction_idty": "interaction",
        }
        mid_getter = IntermediateLayerGetter(model, return_layers=return_layers)
        mid_outputs, model_output = mid_getter(torch.randn(1, 2))

        print(model_output)
        print(mid_outputs)

    def adsad2() -> None:
        """
        :rtype: None
        """

        class Model(nn.Module):
            """ """

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
                """

                :param x:
                :type x:
                :return:
                :rtype:"""
                x1 = self.fc1(x)
                x2 = self.fc2(x)

                interaction = x1 * x2
                self.interaction_idty(interaction)
                return self.nested(interaction)

        model = Model()

        mid_getter = IntermediateLayerGetter(model)
        mid_outputs, model_output = mid_getter(torch.randn(1, 2))

        print(model_output)
        print(mid_outputs)

    adsad()
    # adsad2()
