#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/07/2020
           """

__all__ = [
    "load_model_parameters",
    "load_latest_model_parameters",
    "save_parameters_and_configuration",
    "save_model_parameters",
]

from pathlib import Path
import datetime
import os
from typing import Optional, Tuple, Union
from torch import nn
import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer

from draugr.torch_utilities.persistence.config import (
    ensure_directory_exist,
    save_config,
)
from warg.decorators.kw_passing import drop_unused_kws

PARAMETER_EXTENSION = ".parameters"
CONFIG_EXTENSION = ".py"
OPTIMISER_EXTENSION = ".optimiser"


@drop_unused_kws
def load_latest_model_parameters(
    model: Union[torch.nn.Module, nn.Parameter],
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
    model_loaded = False
    optimiser_loaded = False
    model_path = model_directory / model_name

    list_of_files = list(model_path.glob(f"*{PARAMETER_EXTENSION}"))
    if len(list_of_files) == 0:
        print(
            f"Found no previous models with extension {PARAMETER_EXTENSION} in {model_path}"
        )
    else:
        latest_model_parameter_file = max(list_of_files, key=os.path.getctime)
        print(f"loading previous model parameters: {latest_model_parameter_file}")

    if isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(str(latest_model_parameter_file)))
        model_loaded = True
    elif isinstance(model, nn.Parameter):
        model.data = torch.load(str(latest_model_parameter_file))
        model_loaded = True
    else:
        # print(f"model must be a torch.nn.Module or nn.Parameter")
        raise TypeError(f"model must be a torch.nn.Module or nn.Parameter")

    if optimiser:
        opt_st_d_file = latest_model_parameter_file.with_suffix(OPTIMISER_EXTENSION)
        if opt_st_d_file.exists():
            optimiser.load_state_dict(torch.load(str(opt_st_d_file)))
            print(f"loading previous optimiser state: {opt_st_d_file}")
            optimiser_loaded = True

        return (model, optimiser), (model_loaded, optimiser_loaded)
    return model, model_loaded


load_model_parameters = load_latest_model_parameters


# @passes_kws_to(save_config)
def save_parameters_and_configuration(
    *,
    model: Union[torch.nn.Module, nn.Parameter],
    model_save_path: Path,
    optimiser: Optional[Optimizer] = None,
    optimiser_save_path: Optional[Path] = None,
    config_save_path: Optional[Path] = None,
    loaded_config_file_path: Optional[Path] = None,
) -> None:
    """

    save model parameters and configuration to disk

    :param optimiser:
    :type optimiser:
    :param optimiser_save_path:
    :type optimiser_save_path:
    :param model:
    :param model_save_path:
    :param config_save_path:
    :param loaded_config_file_path:
    :return:"""
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), str(model_save_path))
    elif isinstance(model, nn.Parameter):
        torch.save(model.data, str(model_save_path))
    else:
        raise TypeError(f"model must be a torch.nn.Module or nn.Parameter")
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
    verbose: bool = False,
) -> None:
    """

    save model parameters and optionally configuration file and optimiser state dict

    :param verbose:
    :type verbose:
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
            model_save_path=model_save_path.with_suffix(PARAMETER_EXTENSION),
            optimiser=optimiser,
            optimiser_save_path=(
                model_save_path.parent / f"{model_time_rep}"
            ).with_suffix(OPTIMISER_EXTENSION),
            loaded_config_file_path=config_file_path,
            config_save_path=(model_save_path.parent / f"{model_time_rep}").with_suffix(
                CONFIG_EXTENSION
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
                    model_save_path=model_save_path.endswith(PARAMETER_EXTENSION),
                    optimiser=optimiser,
                    optimiser_save_path=(
                        model_save_path.parent / f"{model_time_rep}"
                    ).with_suffix(OPTIMISER_EXTENSION),
                    loaded_config_file_path=config_file_path,
                    config_save_path=(
                        model_save_path.parent / f"{model_time_rep}"
                    ).with_suffix(CONFIG_EXTENSION),
                )
                saved = True
            except FileNotFoundError as e:
                print(e)
                saved = False
    if verbose:
        if saved:
            print(
                f"Successfully saved model parameters, optimiser state and configuration at names {[model_save_path.with_suffix(a) for a in (PARAMETER_EXTENSION, OPTIMISER_EXTENSION, CONFIG_EXTENSION)]}"
            )
        else:
            print(f"Was unsuccessful at saving model or configuration")
