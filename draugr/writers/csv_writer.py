#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
from typing import Any, TextIO, Tuple

from apppath import ensure_existence
from draugr import PROJECT_APP_PATH
from draugr.writers.writer import Writer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["CSVWriter"]

from pathlib import Path


class CSVWriter(Writer):
    """ """

    @staticmethod
    def get_csv_writer(path: Path = Path.home() / "Models") -> Tuple[TextIO, Any]:
        """

        :param path:
        :type path:
        :return:
        :rtype:"""
        if path.is_dir() or path.suffix != ".csv":
            path /= "log.csv"
        csv_file = open(
            str(
                ensure_existence(path, overwrite_on_wrong_type=True, declare_file=True)
            ),
            mode="a",
        )
        return csv_file, csv.writer(csv_file)

    def _scalar(self, tag: str, value: float, step: int) -> None:
        self._write(step, tag, value)

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self._path = path
        self._file = None
        self._writer = None

    def _open(self):
        self._file, self._writer = self.get_csv_writer(self._path)
        return self

    def _close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        self._file.close()

    def _write(self, *d) -> None:
        self._writer.writerow(d)


if __name__ == "__main__":
    with CSVWriter(PROJECT_APP_PATH.user_log / "test") as p:
        p.scalar("s", 2)
