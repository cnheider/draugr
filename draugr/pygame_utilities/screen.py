#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-09-2020
           """
__all__ = ["get_screen_resolution"]

import pygame


def get_screen_resolution():
    """

    :return:
    :rtype:
    """
    info = pygame.display.Info()
    width = info.current_w
    height = info.current_h
    return width, height
