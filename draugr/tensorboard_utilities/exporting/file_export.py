#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

            TODO: Finish this off, only a placeholder for now

           Created on 17-03-2021
           """

from draugr.tensorboard_utilities.exporting.event_export import TensorboardEventExporter
from typing import Sequence, MutableMapping


class TensorboardEventExporterFile(TensorboardEventExporter):
    def __init__(self, *args: Sequence, **kwargs: MutableMapping):
        super().__init__(*args, **kwargs)
        self.file = kwargs.get("file")
        if self.file is not None:
            self.file.write("")
        else:
            raise ValueError("No file provided")

    def export(self):

        self.file.write("")
        super().export()

    def close(self):
        self.file.close()
