#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
from typing import Any

from apppath import ensure_existence
from draugr import PROJECT_APP_PATH
from draugr.writers.writer import Writer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""
__all__ = ["LogWriter"]

from pathlib import Path


class LogWriter(Writer):
    """ """

    def _scalar(self, tag: str, value: float, step: int) -> None:
        self.logger.info(f"{step} [{tag}] {value}")

    @staticmethod
    def get_logger(
        path: Path = Path.cwd() / "0.log",
        write_to_std_out: bool = False,
    ) -> logging.Logger:
        """

        :param path:
        :type path:
        :param write_to_std_out:
        :type write_to_std_out:
        :return:
        :rtype:"""
        ensure_existence(path, declare_file=True, overwrite_on_wrong_type=True)

        handlers = [logging.FileHandler(filename=str(path))]

        if write_to_std_out:
            handlers.append(logging.StreamHandler(sys.stdout))

        logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)

        return logging.getLogger()

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.log_path = path
        self.logger: logging.Logger = None

    def _open(self) -> Any:
        self.logger = self.get_logger(self.log_path)
        return self

    def _close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        del self.logger

    def __getattr__(self, item) -> Any:
        return getattr(self.logger, item)

    def __call__(self, msg: str) -> None:
        self.logger.info(msg)

    def text(self, msg: str) -> None:
        """

        :param msg:
        """
        self(msg)

    def log(self, msg: str) -> None:
        """

        :param msg:
        """
        self(msg)


if __name__ == "__main__":
    with LogWriter(PROJECT_APP_PATH.user_log / "test") as w:
        w.scalar("What", 4)
