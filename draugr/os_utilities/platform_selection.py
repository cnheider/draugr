#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-01-2021
           """

import sys

__all__ = ["get_backend_module"]

from types import ModuleType


def get_backend_module(
    project_name: str, backend_name: str = sys.platform
) -> ModuleType:
    """Returns the backend module.

    :param project_name:
    :type project_name:
    :param backend_name:
    :type backend_name:
    :return:
    :rtype:

    """
    import importlib

    try:
        importlib.import_module(f"{project_name}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"{project_name} not found, please install it")

    modules = []
    if backend_name is not None:
        modules += [backend_name]
    elif sys.platform == "darwin":
        modules += ["darwin"]
    elif sys.platform == "win32":
        modules += ["win10"]
    else:
        modules += [
            "appindicator",
            "gtk",
            "xorg",
            "gtk_dbus"
            # "unity", "kde", "gnome", "fallback",
        ]

    errors = []
    for module in modules:
        try:
            return importlib.import_module(f"{project_name}.{module}")
        except ImportError as e:
            errors.append(e)

    # Did not find any backend, raise error
    raise ImportError(
        f'{sys.platform} platform is not supported: {"; ".join(str(e) for e in errors)}'
    )


if __name__ == "__main__":
    print(get_backend_module("draugr", "python_utilities"))
