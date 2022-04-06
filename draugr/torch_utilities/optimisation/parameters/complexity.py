#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

import sys
from typing import Sequence, TextIO, Tuple, Union

import numpy
import torch
from torch import nn

from draugr.torch_utilities.optimisation.parameters.counting import get_num_parameters

__all__ = ["get_model_complexity_info", "MODULES_MAPPING"]


def get_model_complexity_info(
    model: torch.nn.Module,
    input_res: Tuple,
    print_per_layer_stat: bool = True,
    as_strings: bool = True,
    input_constructor: callable = None,
    ost: TextIO = sys.stdout,
) -> Union[Tuple[int, int], Tuple[str, str]]:
    """

    :param model:
    :type model:
    :param input_res:
    :type input_res:
    :param print_per_layer_stat:
    :type print_per_layer_stat:
    :param as_strings:
    :type as_strings:
    :param input_constructor:
    :type input_constructor:
    :param ost:
    :type ost:
    :return:
    :rtype:"""
    assert isinstance(input_res, Sequence)
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)

    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        _ = flops_model(**input_constructor(input_res))
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1, *input_res),
                dtype=next(flops_model.parameters()).dtype,
                device=next(flops_model.parameters()).device,
            )
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model, ost=ost)

    flops_count = flops_model.compute_average_flops_cost()
    params_count = get_num_parameters(flops_model, only_trainable=True)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops: int, units: str = "GMac", precision: int = 2) -> str:
    """

    :param flops:
    :type flops:
    :param units:
    :type units:
    :param precision:
    :type precision:
    :return:
    :rtype:"""
    if units is None:
        if flops // 10**9 > 0:
            return f"{str(round(flops / 10. ** 9, precision))} GMac"
        elif flops // 10**6 > 0:
            return f"{str(round(flops / 10. ** 6, precision))} MMac"
        elif flops // 10**3 > 0:
            return f"{str(round(flops / 10. ** 3, precision))} KMac"
        else:
            return f"{str(flops)} Mac"
    else:
        if units == "GMac":
            return f"{str(round(flops / 10. ** 9, precision))} {units}"
        elif units == "MMac":
            return f"{str(round(flops / 10. ** 6, precision))} {units}"
        elif units == "KMac":
            return f"{str(round(flops / 10. ** 3, precision))} {units}"
        else:
            return f"{str(flops)} Mac"


def params_to_string(params_num) -> str:
    """

    :param params_num:
    :type params_num:
    :return:
    :rtype:"""
    if params_num // 10**6 > 0:
        return str(round(params_num / 10**6, 2)) + " M"
    elif params_num // 10**3:
        return str(round(params_num / 10**3, 2)) + " k"
    else:
        return str(params_num)


def print_model_with_flops(
    model, units: str = "GMac", precision: int = 3, ost: TextIO = sys.stdout
) -> None:
    """

    :param model:
    :type model:
    :param units:
    :type units:
    :param precision:
    :type precision:
    :param ost:
    :type ost:
    :return:
    :rtype:"""
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self) -> int:
        """

        :param self:
        :type self:
        :return:
        :rtype:"""
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self) -> str:
        """

        :param self:
        :type self:
        :return:
        :rtype:"""
        accumulated_flops_cost = self.accumulate_flops()
        return ", ".join(
            [
                flops_to_string(
                    accumulated_flops_cost, units=units, precision=precision
                ),
                f"{accumulated_flops_cost / total_flops:.3%} MACs",
                self.original_extra_repr(),
            ]
        )

    def add_extra_repr(m) -> None:
        """

        :param m:
        :type m:"""
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m) -> None:
        """

        :param m:
        :type m:"""
        if hasattr(m, "original_extra_repr"):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, "accumulate_flops"):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost)
    model.apply(del_extra_repr)


def add_flops_counting_methods(net_main_module: torch.nn.Module) -> torch.nn.Module:
    """

    :param net_main_module:
    :type net_main_module:
    :return:
    :rtype:"""
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module
    )

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self) -> float:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image."""

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network."""
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation."""
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self) -> None:
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far."""
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module: torch.nn.Module, mask) -> None:
    """

    :param module:
    :type module:
    :param mask:
    :type mask:"""

    def add_flops_mask_func(module: torch.nn.Module) -> None:
        """

        :param module:
        :type module:"""
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask

    module.apply(add_flops_mask_func)


def remove_flops_mask(module: torch.nn.Module) -> None:
    """

    :param module:
    :type module:"""
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def empty_flops_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    module.__flops__ += 0


def upsample_flops_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    input = input[0]
    batch_size = input.shape[0]
    module.__flops__ += int(batch_size * input.shape[1] * output.shape[1])


def pool_flops_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    input = input[0]
    module.__flops__ += int(numpy.prod(input.shape))


def bn_flops_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    # module.affine
    input = input[0]

    batch_flops = numpy.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module: torch.nn.Module, input, output) -> None:
    """

    :param conv_module:
    :type conv_module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    input_height, input_width = input.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        kernel_height * kernel_width * in_channels * filters_per_channel
    )

    active_elements_count = batch_size * input_height * input_width
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_height, output_width = output.shape[2:]
        bias_flops = out_channels * batch_size * output_height * output_height
    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def conv_flops_counter_hook(conv_module: torch.nn.Module, input, output) -> None:
    """

    :param conv_module:
    :type conv_module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        numpy.prod(kernel_dims) * in_channels * filters_per_channel
    )

    active_elements_count = batch_size * numpy.prod(output_dims)

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        output_height, output_width = output_dims
        flops_mask = conv_module.__mask__.expand(
            batch_size, 1, output_height, output_width
        )
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module: torch.nn.Module, input, output) -> None:
    """

    :param module:
    :type module:
    :param input:
    :type input:
    :param output:
    :type output:"""
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        print(
            "Warning! No positional inputs found for a module, assuming batch size is 1."
        )
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module: torch.nn.Module) -> None:
    """

    :param module:
    :type module:"""
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module: torch.nn.Module) -> None:
    """

    :param module:
    :type module:
    :return:
    :rtype:"""
    if hasattr(module, "__batch_counter_handle__"):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module: torch.nn.Module) -> None:
    """

    :param module:
    :type module:"""
    if hasattr(module, "__batch_counter_handle__"):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module: torch.nn.Module) -> None:
    """

    :param module:
    :type module:"""
    if is_supported_instance(module):
        module.__flops__ = 0


def is_supported_instance(module: torch.nn.Module) -> bool:
    """

    :param module:
    :type module:
    :return:
    :rtype:"""
    if type(module) in MODULES_MAPPING:
        return True
    return False


def add_flops_counter_hook_function(module: torch.nn.Module) -> None:
    """

    :param module:
    :type module:
    :return:
    :rtype:"""
    if is_supported_instance(module):
        if hasattr(module, "__flops_handle__"):
            return
        handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module) -> None:
    """

    :param module:
    :type module:"""
    if is_supported_instance(module):
        if hasattr(module, "__flops_handle__"):
            module.__flops_handle__.remove()
            del module.__flops_handle__


# --- Masked flops counting

# Also being run in the initialization
def add_flops_mask_variable_or_reset(module) -> None:
    """

    :param module:
    :type module:"""
    if is_supported_instance(module):
        module.__mask__ = None


MODULES_MAPPING = {
    # convolutions
    torch.nn.Conv1d: conv_flops_counter_hook,
    torch.nn.Conv2d: conv_flops_counter_hook,
    torch.nn.Conv3d: conv_flops_counter_hook,
    # activations
    torch.nn.ReLU: relu_flops_counter_hook,
    torch.nn.PReLU: relu_flops_counter_hook,
    torch.nn.ELU: relu_flops_counter_hook,
    torch.nn.LeakyReLU: relu_flops_counter_hook,
    torch.nn.ReLU6: relu_flops_counter_hook,
    # poolings
    torch.nn.MaxPool1d: pool_flops_counter_hook,
    torch.nn.AvgPool1d: pool_flops_counter_hook,
    torch.nn.AvgPool2d: pool_flops_counter_hook,
    torch.nn.MaxPool2d: pool_flops_counter_hook,
    torch.nn.MaxPool3d: pool_flops_counter_hook,
    torch.nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    torch.nn.BatchNorm1d: bn_flops_counter_hook,
    torch.nn.BatchNorm2d: bn_flops_counter_hook,
    torch.nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    torch.nn.Linear: linear_flops_counter_hook,
    # Upscale
    torch.nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    torch.nn.ConvTranspose2d: deconv_flops_counter_hook,
}

if __name__ == "__main__":
    import torchvision.models as models
    import torch

    with torch.cuda.device(0):
        net = models.densenet161()
        macs, params = get_model_complexity_info(
            net, (3, 224, 224), as_strings=True, print_per_layer_stat=True
        )
        print(f'{"Computational complexity: ":<30}  {macs:<8}')
        print(f'{"Number of parameters: ":<30}  {params:<8}')
