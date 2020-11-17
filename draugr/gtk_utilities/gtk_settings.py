#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

__all__ = ["GtkSettings"]

import gi  # PyGObject lib

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")

from warg import SingletonMeta


class GtkSettings(metaclass=SingletonMeta):
    """"""

    def __init__(self):
        from gi.repository import Gtk
        from gi.repository import Gdk

        # self.settings = Gtk.Settings.get_default()
        self.settings = Gtk.Settings.get_for_screen(
            Gdk.get_default_root_window().get_screen()
        )

    def __getitem__(self, item):
        return self.settings.get_property(item)

    def __setitem__(self, key, value):
        self.settings.set_property(key, value)

    @property
    def all_settings(self):
        """"""
        for i in self.settings.list_properties():  # getting all existing properties
            yield i

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return 0  # self.settings


if __name__ == "__main__":

    def abvdfj():
        """"""
        a = GtkSettings()

        def ofgof():
            """"""
            for b in a.all_settings:
                print(b)

        ofgof()

        print(a["gtk-dialogs-use-header"])
        # print(a['gtk-dialogs-use-header1'])

    abvdfj()
