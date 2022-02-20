#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/06/2020
           """

__all__ = ["Meter", "AverageMeter"]


class Meter:
    """Stores current value"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """ """
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        """

        :param val:
        :type val:
        :param n:
        :type n:"""
        self.val = val
        self.sum += val * n
        self.count += n


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self):
        super().__init__()
        self.avg = 0

    def reset(self):
        """ """
        super().reset()

    def update(self, val, n: int = 1):
        """

        :param val:
        :type val:
        :param n:
        :type n:"""
        super().update(val, n)
        self.avg = self.sum / self.count
