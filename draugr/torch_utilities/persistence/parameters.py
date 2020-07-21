#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/07/2020
           """

import datetime
import os
import pathlib
from typing import Union

import torch
from torch.nn.modules.module import Module

from draugr.torch_utilities.persistence.config import (
    ensure_directory_exist,
    save_config,
)
from warg import passes_kws_to
from warg.decorators.kw_passing import drop_unused_kws

model_file_ending = ".parameters"
config_file_ending = ".py"

__all__ = [
    "load_model_parameters",
    "load_latest_model_parameters",
    "save_parameters_and_configuration",
    "save_model_parameters",
]


@drop_unused_kws
def load_latest_model_parameters(
    model, *, model_name: str, model_directory: pathlib.Path
) -> Union[torch.nn.Module, None]:
    """

:param model_directory:
:param model_name:
:return:
"""
    list_of_files = list(model_directory.glob(f"{model_name}/*{model_file_ending}"))
    if len(list_of_files) == 0:
        print(f"Found no previous model in subtrees of: {model_directory}")
        return None
    latest_model = max(list_of_files, key=os.path.getctime)
    print(f"loading previous model: {latest_model}")

    model.load_state_dict(torch.load(str(latest_model)))
    return model


load_model_parameters = load_latest_model_parameters


@passes_kws_to(save_config)
def save_parameters_and_configuration(
    *,
    model: Module,
    model_save_path: pathlib.Path,
    config_save_path: pathlib.Path = None,
    loaded_config_file_path: pathlib.Path = None,
) -> None:
    """

:param model:
:param model_save_path:
:param config_save_path:
:param loaded_config_file_path:
:return:
"""
    torch.save(model.state_dict(), str(model_save_path))
    if loaded_config_file_path:
        save_config(config_save_path, loaded_config_file_path)


@drop_unused_kws
def save_model_parameters(
    model: Module,
    *,
    model_name: str,
    save_directory: pathlib.Path,
    config_file_path: pathlib.Path = None,
) -> None:
    """

:param model:
:param save_directory:
:param config_file_path:
:param model_name:
:return:
"""
    model_date = datetime.datetime.now()
    # config_name = config_name.replace(".", "_")

    model_time_rep = model_date.strftime("%Y%m%d%H%M%S")
    model_save_path = (
        save_directory / model_name / f"{model_time_rep}{model_file_ending}"
    )
    config_save_path = (
        save_directory / model_name / f"{model_time_rep}{config_file_ending}"
    )
    ensure_directory_exist(model_save_path.parent)

    saved = False
    try:
        save_parameters_and_configuration(
            model=model,
            model_save_path=model_save_path,
            loaded_config_file_path=config_file_path,
            config_save_path=config_save_path,
        )
        saved = True
    except FileNotFoundError as e:
        print(e)
        while not saved:
            file_path = input("Enter another file path: ")
            model_save_path = pathlib.Path(file_path).expanduser().resolve()
            parent = model_save_path.parent
            ensure_directory_exist(parent)
            config_save_path = parent / f"{model_save_path.name}{config_file_ending}"
            try:
                save_parameters_and_configuration(
                    model=model,
                    model_save_path=model_save_path,
                    loaded_config_file_path=config_file_path,
                    config_save_path=config_save_path,
                )
                saved = True
            except FileNotFoundError as e:
                print(e)
                saved = False

    if saved:
        print(
            f"Successfully saved model and configuration respectively at {model_save_path} and {config_save_path}"
        )
    else:
        print(f"Was unsuccesful at saving model or configuration")
