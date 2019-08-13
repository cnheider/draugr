#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib

__author__ = "cnheider"

import csv
import datetime


def save_metric(
    metric, *, metric_name, project_name, config_name, directory="logs"
) -> bool:
    if metric:
        _file_date = datetime.datetime.now()
        _file_name = (
            f'{project_name}-{config_name.replace(".", "_")}-'
            f'{_file_date.strftime("%y%m%d%H%M")}.{metric_name}.csv'
        )
        _file_path = pathlib.Path.joinpath(directory, _file_name)

        stat = [[s] for s in metric]
        with open(_file_path, "w") as f:
            w = csv.writer(f)
            w.writerows(stat)
        print(f"Saved metric at {_file_path}")

        return True
    return False
