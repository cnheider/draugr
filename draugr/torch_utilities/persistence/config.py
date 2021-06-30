#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/07/2020
           """

__all__ = ["save_config", "ensure_directory_exist"]

import shutil

from pathlib import Path


def save_config(config_save_path: Path, config_file_path: Path) -> None:
    """

    :param config_save_path:
    :param config_file_path:
    :return:"""
    shutil.copyfile(str(config_file_path), str(config_save_path))


def ensure_directory_exist(model_path: Path) -> None:
    """

    :param model_path:
    :return:"""
    if not model_path.exists():
        model_path.mkdir(parents=True)
