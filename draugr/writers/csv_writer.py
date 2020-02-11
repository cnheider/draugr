#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import pathlib

from draugr import PROJECT_APP_PATH
from draugr.writers.writer import Writer
from draugr.writers.writer_utilities import create_folders_if_necessary

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["CSVWriter"]


class CSVWriter(Writer):
    @staticmethod
    def get_csv_writer(path=pathlib.Path.home() / "Models"):
        csv_path = path / "log.csv"
        create_folders_if_necessary(csv_path)
        csv_file = open(str(csv_path), "a")
        return csv_file, csv.writer(csv_file)

    def _scalar(self, tag: str, value: float, step: int):
        self._write(step, tag, value)

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.f = None
        self.writer = None

    def _open(self):
        self.f, self.writer = self.get_csv_writer(self.path)
        return self

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        self.f.close()

    def _write(self, *d):
        self.writer.writerow(d)


if __name__ == "__main__":

    with CSVWriter(PROJECT_APP_PATH.user_log / "test") as p:
        p.scalar("s", 2)
