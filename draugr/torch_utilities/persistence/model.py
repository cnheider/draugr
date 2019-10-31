#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
import datetime
import os
import shutil
import sys
import torch
from torch.nn.modules.module import Module

from warg import passes_kws_to

__author__ = "Christian Heider Nielsen"

model_file_ending = ".model"
config_file_ending = ".py"


def load_latest_model(model_directory, **kwargs):
    list_of_files = list(model_directory.glob(f"**/*{model_file_ending}"))
    if len(list_of_files) == 0:
        return
    latest_model = max(list_of_files, key=os.path.getctime)
    print(f"loading previous model: {latest_model}")

    return torch.load(str(latest_model))


def ensure_directory_exist(model_path: pathlib.Path):
    if not model_path.exists():
        model_path.mkdir(parents=True)


def save_config(config_save_path: pathlib.Path, config_file_path: pathlib.Path):
    shutil.copyfile(str(config_file_path), str(config_save_path))


@passes_kws_to(save_config)
def save_model_and_configuration(
    *,
    model: Module,
    model_save_path: pathlib.Path,
    config_save_path: pathlib.Path,
    config_file_path: pathlib.Path,
):
    torch.save(model.state_dict(), str(model_save_path))
    save_config(config_save_path, config_file_path)


@passes_kws_to(save_model_and_configuration)
def save_model(
    model: Module,
    *,
    save_directory: pathlib.Path,
    project_name,
    config_name,
    config_file_path,
    model_name: str = "",
    **kwargs,
):
    model_date = datetime.datetime.now()
    prepend = ""
    if len(model_name) > 0:
        prepend = f"{model_name}-"
    model_name = f'{prepend}{project_name}-{config_name.replace(".", "_")}'

    model_time_rep = model_date.strftime("%y%m%d%H%M")
    model_save_path = (
        save_directory / model_name / f"{model_time_rep}{model_file_ending}"
    )
    config_save_path = (
        save_directory / model_name / f"{model_time_rep}{config_file_ending}"
    )
    ensure_directory_exist(model_save_path.parent)

    saved = False
    try:
        save_model_and_configuration(
            model=model,
            model_save_path=model_save_path,
            config_file_path=config_file_path,
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
                save_model_and_configuration(
                    model=model,
                    model_save_path=model_save_path,
                    config_file_path=config_file_path,
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


def convert_to_cpu(path=""):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    torch.save(model, f"{path}.cpu{model_file_ending}")


if __name__ == "__main__":
    convert_to_cpu(sys.argv[1])
