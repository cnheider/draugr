#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/11/2019
           """

GLOBAL_DTYPE = None

__all__ = ["global_torch_dtype", "set_global_torch_dtype"]


def global_torch_dtype(
    override: torch.dtype = None, verbose: bool = False
) -> torch.dtype:
    """

    first time call stores to dtype for global reference, later call must manually override

    :param verbose:
    :type verbose:
    :param override:
    :type override:
    :return:
    :rtype:"""
    global GLOBAL_DTYPE

    if override is not None:
        GLOBAL_DTYPE = override
        set_global_torch_dtype(GLOBAL_DTYPE)
        if verbose:
            print(f"Overriding global torch device to {override}")
    elif GLOBAL_DTYPE is None:
        GLOBAL_DTYPE = torch.get_default_dtype()

    return GLOBAL_DTYPE


def set_global_torch_dtype(dtype: torch.dtype) -> None:
    """ """
    global GLOBAL_DTYPE
    GLOBAL_DTYPE = dtype
    torch.set_default_dtype(GLOBAL_DTYPE)


if __name__ == "__main__":

    def stest_override() -> None:
        """
        :rtype: None
        """
        print(global_torch_dtype(verbose=True))
        print(global_torch_dtype(override=torch.double, verbose=True))
        print(global_torch_dtype(verbose=True))
        print(global_torch_dtype())
        print(global_torch_dtype())
        print(global_torch_dtype(override=torch.half))
        print(global_torch_dtype())

    stest_override()
