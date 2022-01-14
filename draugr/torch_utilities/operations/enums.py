#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "{user}"
__doc__ = r"""

           Created on {date}
           """

from enum import Enum

from sorcery import assigned_names


class ReductionMethodEnum(Enum):
    """ """

    none, mean, sum = assigned_names()
