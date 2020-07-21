#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18/03/2020
           """

import datetime

__all__ = ["default_datetime_repr"]


def default_datetime_repr(date: datetime, str_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """

  :param date:
  :type date:
  :param str_format:
  :type str_format:
  :return:
  :rtype:
  """
    return date.strftime(str_format)
