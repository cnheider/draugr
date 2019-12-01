#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
import cv2


def add_indent(s_, numSpaces):
    s = s_.split("\n")
    if len(s) == 1:  # don't do anything for single-line stuff
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def resize_image_cv(x, target_size: tuple):
    """

  :param x:
  :param target_size: proper (width, height) shape, no cv craziness
  :return:
  """
    if x.shape != target_size:
        x = cv2.resize(x, target_size[::-1], interpolation=cv2.INTER_LINEAR)
    return x
