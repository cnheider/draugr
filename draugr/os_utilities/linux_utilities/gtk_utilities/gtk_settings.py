#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

__all__ = ["GtkSettings"]

from sys import stderr

import gi  # PyGObject lib

#  Ubuntu / Debian
## system
### sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0

## conda
### conda install -c conda-forge pygobject
### conda install -c conda-forge gtk3

## pip
### sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
### pip3 install pycairo
### pip3 install PyGObject

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")

from warg import SingletonMeta


class GtkSettings(metaclass=SingletonMeta):
    """ """

    def __init__(self):
        try:
            from gi.repository import Gtk, Gdk  # , GObject, Gio, GLib
        except (ImportError, ImportWarning):
            stderr.write("Could not import GTK. Please install it.")

        # a = Gio.Settings("org.gnome.gedit.preferences.editor")
        # print(a.get_string("wrap-mode"))
        # settings = Gio.Settings('org.my.font')
        # face = settings.get_string('default-face')
        # size = settings.get_double('default-size')

        self.settings = Gtk.Settings.get_default()
        # self.settings = Gtk.Settings.get_for_screen(Gdk.get_default_root_window().get_screen())

    def __getitem__(self, item):
        return self.settings.get_property(item)

    def __setitem__(self, key, value):
        self.settings.set_property(key, value)

    @property
    def all_settings(self):
        """ """
        for i in self.settings.list_properties():  # getting all existing properties
            yield i

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return 0  # self.settings


if __name__ == "__main__":

    def abvdfj() -> None:
        """
        :rtype: None
        """
        a = GtkSettings()

        def ofgof():
            """ """
            for b in a.all_settings:
                print(b)

        ofgof()

        print(a["gtk-dialogs-use-header"])
        # print(a['gtk-dialogs-use-header1'])

    abvdfj()
