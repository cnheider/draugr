#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

            TODO: Finish this off, only a placeholder for now

           Created on 17-03-2021
           """

from draugr.tensorboard_utilities.exporting.event_export import TensorboardEventExporter
from typing import Sequence, MutableMapping


class TensorboardEventExporterDatabase(TensorboardEventExporter):
    def __init__(self, *args: Sequence, **kwargs: MutableMapping):
        super().__init__(*args, **kwargs)
        postgres_db = kwargs.get("postgres_db")
        if postgres_db is not None:
            self.database = postgres_db
        else:
            raise ValueError("No postgres_db provided")

    def export(self, database):
        self.database = database
        super().export(database)

    def export_event(self, event):
        self.database.add_event(event)
        super().export_event(event)
