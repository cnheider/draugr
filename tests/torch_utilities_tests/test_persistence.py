#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/09/2019
           """

from pathlib import Path


def test_resolve_expand_path():
    # file_path = input("Enter another file path: ")
    file_path = Path.home()
    model_save_path = Path(file_path).expanduser().resolve()
    print(model_save_path)


if __name__ == "__main__":
    test_resolve_expand_path()
