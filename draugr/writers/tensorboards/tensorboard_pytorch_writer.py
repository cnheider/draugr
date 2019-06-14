#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib

from draugr.writers.writer import Writer

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from torch.utils.tensorboard import SummaryWriter


class TensorBoardPytorchWriter(Writer):
    def __init__(
        self,
        log_dir=pathlib.Path.home() / "Models",
        comment: str = "",
        interval: int = 1,
    ):
        super().__init__(interval)

        self._log_dir = log_dir
        self._comment = comment

    def _scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def _graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def _close(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def _open(self):
        self.writer = SummaryWriter(str(self._log_dir), self._comment)
        return self


if __name__ == "__main__":

    with TensorBoardPytorchWriter(pathlib.Path.home() / "Models") as w:
        w.scalar("What", 4)
