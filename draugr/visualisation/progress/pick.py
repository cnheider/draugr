#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/2/22
           """

__all__ = ["choose_progress_bar"]

from draugr.python_utilities.platform_context import in_ipynb

from draugr.visualisation.progress.eta_bar import ETABar
from draugr.visualisation.progress.progress_bar import progress_bar
from warg import LambdaContext


def choose_progress_bar(*args, total, **kwargs) -> callable:
    if in_ipynb():
        return lambda total: ETABar(*args, max=total, **kwargs)
    return LambdaContext(progress_bar(*args, total=total, **kwargs))
