#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/24/22
           """

__all__ = ["in_ipynb"]

import sys


def in_ipynb(verbose: bool = False) -> bool:
    """

    :return:
    :rtype:
    """
    try:
        from IPython import get_ipython
        import jupyter

        shell = get_ipython().__class__.__name__
        if verbose:
            print(f"found shell: {shell}")
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        if verbose:
            print(f"Probably standard Python interpreter")
        return False  # Probably standard Python interpreter
    except ModuleNotFoundError:
        if "ipykernel" in sys.modules:
            if verbose:
                print(f"Found ipykernel in sys.modules")
            return True

        if verbose:
            print(f"Did not find Ipython")
        return False  # Did not find Ipython


if __name__ == "__main__":
    in_ipynb()
