#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/07/2020
           """

import datetime
import os
from typing import Optional, Tuple, Union

import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer

from draugr.torch_utilities.persistence.config import (
    ensure_directory_exist,
    save_config,
)
from warg.decorators.kw_passing import drop_unused_kws

parameter_extension = ".parameters"
config_extension = ".py"
optimiser_extension = ".optimiser"

__all__ = [
    "load_model_parameters",
    "load_latest_model_parameters",
    "save_parameters_and_configuration",
    "save_model_parameters",
]

from pathlib import Path


@drop_unused_kws
def load_latest_model_parameters(
    model: torch.nn.Module,
    *,
    optimiser: Optimizer = None,
    model_name: str,
    model_directory: Path,
) -> Tuple[Union[torch.nn.Module, Tuple[torch.nn.Module, Optimizer]], bool]:
    """

    inplace but returns model

    :param optimiser:
    :param model:
    :type model:
    :param model_directory:
    :param model_name:
    :return:"""
    if model:
        model_path = model_directory / model_name
        list_of_files = list(model_path.glob(f"*{parameter_extension}"))
        if len(list_of_files) == 0:
            print(
                f"Found no previous models with extension {parameter_extension} in {model_path}"
            )
        else:
            latest_model_parameter_file = max(list_of_files, key=os.path.getctime)
            print(f"loading previous model parameters: {latest_model_parameter_file}")

            model.load_state_dict(torch.load(str(latest_model_parameter_file)))

            if optimiser:
                opt_st_d_file = latest_model_parameter_file.with_suffix(
                    optimiser_extension
                )
                if opt_st_d_file.exists():
                    optimiser.load_state_dict(torch.load(str(opt_st_d_file)))
                    print(f"loading previous optimiser state: {opt_st_d_file}")
                return (model, optimiser), True
            else:
                return model, True
    if optimiser:
        return (model, optimiser), False
    return model, False


load_model_parameters = load_latest_model_parameters


# @passes_kws_to(save_config)
def save_parameters_and_configuration(
    *,
    model: Module,
    model_save_path: Path,
    optimiser: Optional[Optimizer] = None,
    optimiser_save_path: Optional[Path] = None,
    config_save_path: Optional[Path] = None,
    loaded_config_file_path: Optional[Path] = None,
) -> None:
    """

    :param optimiser:
    :type optimiser:
    :param optimiser_save_path:
    :type optimiser_save_path:
    :param model:
    :param model_save_path:
    :param config_save_path:
    :param loaded_config_file_path:
    :return:"""
    torch.save(model.state_dict(), str(model_save_path))
    if optimiser:
        torch.save(optimiser.state_dict(), str(optimiser_save_path))
    if loaded_config_file_path:
        save_config(config_save_path, loaded_config_file_path)


@drop_unused_kws
def save_model_parameters(
    model: Module,
    *,
    model_name: str,
    save_directory: Path,
    optimiser: Optional[Optimizer] = None,
    config_file_path: Optional[Path] = None,
) -> None:
    """

    :param optimiser:
    :param model:
    :param save_directory:
    :param config_file_path:
    :param model_name:
    :return:"""
    model_date = datetime.datetime.now()

    model_time_rep = model_date.strftime("%Y%m%d%H%M%S")
    model_save_path = save_directory / model_name / f"{model_time_rep}"
    ensure_directory_exist(model_save_path.parent)

    saved = False
    try:
        save_parameters_and_configuration(
            model=model,
            model_save_path=model_save_path.with_suffix(parameter_extension),
            optimiser=optimiser,
            optimiser_save_path=(
                model_save_path.parent / f"{model_time_rep}"
            ).with_suffix(optimiser_extension),
            loaded_config_file_path=config_file_path,
            config_save_path=(model_save_path.parent / f"{model_time_rep}").with_suffix(
                config_extension
            ),
        )
        saved = True
    except FileNotFoundError as e:
        print(e)
        while not saved:
            model_save_path = (
                Path(input("Enter another file path: ")).expanduser().resolve()
            )
            ensure_directory_exist(model_save_path.parent)
            try:
                save_parameters_and_configuration(
                    model=model,
                    model_save_path=model_save_path.endswith(parameter_extension),
                    optimiser=optimiser,
                    optimiser_save_path=(
                        model_save_path.parent / f"{model_time_rep}"
                    ).with_suffix(optimiser_extension),
                    loaded_config_file_path=config_file_path,
                    config_save_path=(
                        model_save_path.parent / f"{model_time_rep}"
                    ).with_suffix(config_extension),
                )
                saved = True
            except FileNotFoundError as e:
                print(e)
                saved = False

    if saved:
        print(
            f"Successfully saved model parameters, optimiser state and configuration at names {[model_save_path.with_suffix(a) for a in (parameter_extension, optimiser_extension, config_extension)]}"
        )
    else:
        print(f"Was unsuccessful at saving model or configuration")
