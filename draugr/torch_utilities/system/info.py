import multiprocessing
import platform
import typing

import torch

__all__ = ["system_info", "cuda_info"]


def system_info() -> str:
    """

    :return:
    :rtype:"""
    return "\n".join(
        [
            f"Python version: {platform.python_version()}",
            f"Python implementation: {platform.python_implementation()}",
            f"Python compiler: {platform.python_compiler()}",
            f"PyTorch version: {torch.__version__}",
            f"System: {platform.system() or 'Unable to determine'}",
            f"System version: {platform.release() or 'Unable to determine'}",
            f"Processor: {platform.processor() or 'Unable to determine'}",
            f"Number of CPUs: {multiprocessing.cpu_count()}",
        ]
    )


def cuda_info() -> str:
    """

    :return:
    :rtype:"""

    def _cuda_devices_formatting(
        info_function: typing.Callable,
        formatting_function: typing.Callable = None,
        mapping_function: typing.Callable = None,
    ):
        def _setup_default(function):
            return (lambda arg: arg) if function is None else function

        formatting_function = _setup_default(formatting_function)
        mapping_function = _setup_default(mapping_function)

        return " | ".join(
            mapping_function(
                [
                    formatting_function(info_function(i))
                    for i in range(torch.cuda.device_count())
                ]
            )
        )

    def _device_properties(attribute):
        return _cuda_devices_formatting(
            lambda i: getattr(torch.cuda.get_device_properties(i), attribute),
            mapping_function=lambda in_bytes: map(str, in_bytes),
        )

    cuda_cap = _cuda_devices_formatting(
        torch.cuda.get_device_capability,
        formatting_function=lambda capabilities: ".".join(map(str, capabilities)),
    )
    return "\n".join(
        [
            f"Available CUDA devices count: {torch.cuda.device_count()}",
            f"CUDA devices names: {_cuda_devices_formatting(torch.cuda.get_device_name)}",
            f"Major.Minor CUDA capabilities of devices: {cuda_cap}",
            f"Device total memory (bytes): {_device_properties('total_memory')}",
            f"Device multiprocessor count: {_device_properties('multi_processor_count')}",
        ]
    )


if __name__ == "__main__":
    print(system_info())
    print(cuda_info())
