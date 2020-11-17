#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09-11-2020
           """

import emoji

print(emoji.emojize("Python is :thumbs_up:"))
print(emoji.emojize("Python is :thumbsup:", use_aliases=True))
print(emoji.demojize("Python is 👍"))
print(emoji.emojize("Python is fun :red_heart:"))
print(emoji.emojize("Python is fun :red_heart:", variant="emoji_type"))
