#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

__all__ = ["GtkThemePreferences"]

from typing import Union

from draugr.os_utilities.linux_utilities.gtk_utilities.gtk_settings import GtkSettings


class GtkThemePreferences(GtkSettings):
    """
    Presents a slim series of properties for manipulation of GTK settings
    """

    @property
    def theme(self) -> Union[str, bytes]:
        """

        :return:
        """
        return self.settings.get_property("gtk-theme-name")

    @theme.setter
    def theme(self, theme_name: str) -> None:
        self.settings.set_property("gtk-theme-name", theme_name)

    @property
    def prefer_dark_mode(self) -> bool:
        """

        :return:
        """
        return self.settings.get_property("gtk-application-prefer-dark-theme")

    @prefer_dark_mode.setter
    def prefer_dark_mode(self, enabled: bool) -> None:
        self.settings.set_property("gtk-application-prefer-dark-theme", enabled)


if __name__ == "__main__":

    def asdad() -> None:
        """
        :rtype: None
        """
        a = GtkThemePreferences()

        print(a.prefer_dark_mode)
        a.prefer_dark_mode = not a.prefer_dark_mode
        print(a.prefer_dark_mode)
        a.prefer_dark_mode = not a.prefer_dark_mode
        print(a.prefer_dark_mode)

    def asdad2312() -> None:
        """
        :rtype: None
        """
        with GtkThemePreferences() as a:
            print(a.prefer_dark_mode)

        with GtkThemePreferences() as a:
            print(a.prefer_dark_mode)
            a.prefer_dark_mode = not a.prefer_dark_mode
            print(a.prefer_dark_mode)

        with GtkThemePreferences() as a:
            print(a.prefer_dark_mode)

    def asda213sad2312d() -> None:
        """
        :rtype: None
        """
        a = GtkThemePreferences()

        print(a.prefer_dark_mode)
        a.prefer_dark_mode = not a.prefer_dark_mode  # TODO: DOES NOT WORK!
        print(a.prefer_dark_mode)

    asda213sad2312d()
    # asdad2312()
    # asdad()
