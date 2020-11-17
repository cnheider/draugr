#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

__all__ = ["GtkThemePreferences"]

from draugr.gtk_utilities.gtk_settings import GtkSettings


class GtkThemePreferences(GtkSettings):
    @property
    def theme(self):
        """"""
        return self.settings.get_property("gtk-theme-name")

    @theme.setter
    def theme(self, theme_name: str):
        self.settings.set_property("gtk-theme-name", theme_name)

    @property
    def prefer_dark_mode(self) -> bool:
        """

        :return:
        :rtype:"""
        return self.settings.get_property("gtk-application-prefer-dark-theme")

    @prefer_dark_mode.setter
    def prefer_dark_mode(self, enabled: bool):
        self.settings.set_property("gtk-application-prefer-dark-theme", enabled)


if __name__ == "__main__":

    def asdad():
        """"""
        a = GtkThemePreferences()

        print(a.prefer_dark_mode)
        a.prefer_dark_mode = not a.prefer_dark_mode
        print(a.prefer_dark_mode)
        a.prefer_dark_mode = not a.prefer_dark_mode
        print(a.prefer_dark_mode)

    def asdad2312():
        """"""
        with GtkThemePreferences() as a:
            print(a.prefer_dark_mode)

        with GtkThemePreferences() as a:
            print(a.prefer_dark_mode)
            a.prefer_dark_mode = not a.prefer_dark_mode
            print(a.prefer_dark_mode)

        with GtkThemePreferences() as a:
            print(a.prefer_dark_mode)

    def asda213sad2312d():
        """"""
        a = GtkThemePreferences()

        print(a.prefer_dark_mode)
        a.prefer_dark_mode = not a.prefer_dark_mode  # TODO: DOES NOT WORK!
        print(a.prefer_dark_mode)

    asda213sad2312d()
    # asdad2312()
    # asdad()
