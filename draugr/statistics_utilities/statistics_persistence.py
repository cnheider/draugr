#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "cnheider"

import csv
import datetime
import os


def save_statistic(
    statistic, *, stat_name, project_name, config_name, directory="logs"
) -> bool:
    if statistic:
        _file_date = datetime.datetime.now()
        _file_name = (
            f'{project_name}-{config_name.replace(".", "_")}-'
            f'{_file_date.strftime("%y%m%d%H%M")}.{stat_name}.csv'
        )
        _file_path = pathlib.Path.joinpath(directory, _file_name)

        stat = [[s] for s in statistic]
        with open(_file_path, "w") as f:
            w = csv.writer(f)
            w.writerows(stat)
        print("Saved statistics_utilities at {}".format(_file_path))

        return True
    return False
