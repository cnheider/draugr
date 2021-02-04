#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import pathlib
import sys
from typing import Union

import torch
from draugr.torch_utilities.persistence.config import (
    ensure_directory_exist,
    save_config,
)
from torch.nn.modules.module import Module
from warg import passes_kws_to
from warg.decorators.kw_passing import drop_unused_kws

__author__ = "Christian Heider Nielsen"

model_extension = ".model"
config_extension = ".py"

__all__ = [
    "load_model",
    "load_latest_model",
    "save_model_and_configuration",
    "save_model",
    "convert_saved_model_to_cpu",
]


@drop_unused_kws
def load_latest_model(
    *, model_name: str, model_directory: pathlib.Path, raise_on_failure: bool = True
) -> Union[torch.nn.Module, None]:
    """

    load model with the lastest time appendix or in this case creation time

    :param raise_on_failure:
    :param model_directory:
    :param model_name:
    :return:"""
    model_path = model_directory / model_name
    list_of_files = list(model_path.glob(f"*{model_extension}"))
    if len(list_of_files) == 0:
        msg = (
            f"Found no previous models with extension {model_extension} in {model_path}"
        )
        if raise_on_failure:
            raise FileNotFoundError(msg)
        print(msg)
        return None
    latest_model = max(list_of_files, key=os.path.getctime)  # USES CREATION TIME
    print(f"loading previous model: {latest_model}")

    return torch.load(str(latest_model))


load_model = load_latest_model


@passes_kws_to(save_config)
def save_model_and_configuration(
    *,
    model: Module,
    model_save_path: pathlib.Path,
    config_save_path: pathlib.Path = None,
    loaded_config_file_path: pathlib.Path = None,
    raise_on_existing: bool = False,
) -> None:
    """

    :param raise_on_existing:
    :param model:
    :param model_save_path:
    :param config_save_path:
    :param loaded_config_file_path:
    :return:"""
    if raise_on_existing and model_save_path.exists():
        raise FileExistsError(f"{model_save_path} exists!")
    torch.save(model, str(model_save_path))
    if loaded_config_file_path:
        save_config(config_save_path, loaded_config_file_path)


@drop_unused_kws
@passes_kws_to(save_model_and_configuration)
def save_model(
    model: Module,
    *,
    model_name: str,
    save_directory: pathlib.Path,  # TODO: RENAME to model directory for consistency
    config_file_path: pathlib.Path = None,
    prompt_on_failure: bool = True,
    verbose: bool = False,
) -> None:
    """

    save a model with a timestamp appendix to later to loaded

    :param prompt_on_failure:
    :param verbose:
    :param model:
    :param save_directory:
    :param config_file_path:
    :param model_name:
    :return:"""
    model_date = datetime.datetime.now()
    # config_name = config_name.replace(".", "_")

    model_time_rep = model_date.strftime("%Y%m%d%H%M%S")
    model_save_path = save_directory / model_name / f"{model_time_rep}{model_extension}"
    config_save_path = (
        save_directory / model_name / f"{model_time_rep}{config_extension}"
    )
    ensure_directory_exist(model_save_path.parent)

    saved = False
    try:
        save_model_and_configuration(
            model=model,
            model_save_path=model_save_path,
            loaded_config_file_path=config_file_path,
            config_save_path=config_save_path,
        )
        saved = True
    except FileNotFoundError as e:
        if prompt_on_failure:
            print(e)
            while not saved:
                file_path = input("Enter another file path: ")
                model_save_path = pathlib.Path(file_path).expanduser().resolve()
                parent = model_save_path.parent
                ensure_directory_exist(parent)
                config_save_path = parent / f"{model_save_path.name}{config_extension}"
                try:
                    save_model_and_configuration(
                        model=model,
                        model_save_path=model_save_path,
                        loaded_config_file_path=config_file_path,
                        config_save_path=config_save_path,
                    )
                    saved = True
                except FileNotFoundError as e:
                    print(e)
                    saved = False
        else:
            raise e

    if verbose:
        if saved:
            print(
                f"Successfully saved model and configuration respectively at {model_save_path} and {config_save_path}"
            )
        else:
            print(f"Was unsuccesful at saving model or configuration")


def convert_saved_model_to_cpu(path: pathlib.Path) -> None:
    """

    :param path:
    :return:"""
    model = torch.load(path, map_location=lambda storage, loc: storage)
    torch.save(model, f"{path}.cpu{model_extension}")


if __name__ == "__main__":
    convert_saved_model_to_cpu(pathlib.Path(sys.argv[1]))
