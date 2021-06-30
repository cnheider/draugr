#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["PrecisionRecallCurveWriterMixin"]

from warg import drop_unused_kws


class PrecisionRecallCurveWriterMixin(ABC):
    """
    Writer mixin that provides an interface for 'writing' instantiation"""

    @drop_unused_kws
    @abstractmethod
    def precision_recall_curve(
        self,
        tag: str,
        predictions: list,
        truths: list,
        step: int,
        num_thresholds: int = 11,
    ) -> None:
        """ """
        raise NotImplementedError
