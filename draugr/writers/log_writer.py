#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pathlib
import sys

from draugr import PROJECT_APP_PATH
from draugr.writers.writer import Writer
from draugr.writers.writer_utilities import create_folders_if_necessary

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""
__all__ = ["LogWriter"]


class LogWriter(Writer):
    def _scalar(self, tag: str, value: float, step: int):
        self.writer.info(f"{step} [{tag}] {value}")

    @staticmethod
    def get_logger(path=pathlib.Path.home() / "Models"):
        path = path / "log.txt"
        create_folders_if_necessary(path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(filename=path),
                logging.StreamHandler(sys.stdout),
            ],
        )

        return logging.getLogger()

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.f = None
        self.writer = None

    def _open(self):
        self.writer = self.get_logger(self.path)
        return self

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        del self.writer

    def __getattr__(self, item):
        return getattr(self.writer, item)


if __name__ == "__main__":

    with LogWriter(PROJECT_APP_PATH.user_log / "test") as w:
        w.scalar("What", 4)
