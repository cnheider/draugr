#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/2/22
           """

__all__ = ["choose_progress_bar"]

from itertools import count
from typing import Iterator

from draugr.python_utilities.platform_context import in_ipynb

from draugr.visualisation.progress.eta_bar import ETABar
from draugr.visualisation.progress.progress_bar import progress_bar
from warg import LambdaContext


class IteratorWrapper(Iterator):
    """
    #TO be removed
    """

    def __next__(self):
        return getattr(self.callable_, self.entered)()

    def __init__(self, iter_, cal_):
        self.iter_ = iter_
        self.callable_ = cal_

    def __iter__(self):
        a = iter(self.entered)
        for c in count():
            yield getattr(self.callable_, a)()

    def __enter__(self):
        self.entered = self.iter_.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.entered.__exit__(exc_type, exc_val, exc_tb)


def choose_progress_bar(*args, total=None, verbose: bool = False, **kwargs) -> callable:
    """
    #TODO: REwrite this!

    :param args:
    :type args:
    :param total:
    :type total:
    :param verbose:
    :type verbose:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    if in_ipynb(verbose=verbose):
        print("*args where dropped")  # TODO: POOPY, dont do this
        return ETABar(max=total, verbose=verbose, **kwargs)
    return LambdaContext(progress_bar(*args, total=total, verbose=verbose, **kwargs))
