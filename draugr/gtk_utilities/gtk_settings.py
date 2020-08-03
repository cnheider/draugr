#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

__all__ = ["GtkSettings"]

import gi  # PyGObject lib

gi.require_version("Gtk", "3.0")

from warg import SingletonMeta


class GtkSettings(metaclass=SingletonMeta):
    """

  """

    def __init__(self):
        from gi.repository import Gtk

        self.settings = Gtk.Settings.get_default()

    def __getitem__(self, item):
        return self.settings.get_property(item)

    def __setitem__(self, key, value):
        self.settings.set_property(key, value)

    @property
    def all_settings(self):
        """

    """
        for i in self.settings.list_properties():  # getting all existing properties
            yield i


if __name__ == "__main__":

    def abvdfj():
        """

    """
        a = GtkSettings()

        def ofgof():
            """

      """
            for b in a.all_settings:
                print(b)

        ofgof()

        print(a["gtk-dialogs-use-header"])
        # print(a['gtk-dialogs-use-header1'])

    abvdfj()
