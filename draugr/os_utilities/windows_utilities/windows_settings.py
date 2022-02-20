#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30-08-2020
           """

try:
    import winreg
except ModuleNotFoundError as e:
    raise ModuleNotFoundError((e, "missing winreg"))

from warg import SingletonMeta


class WindowsSettings(metaclass=SingletonMeta):
    """ """

    def __init__(self):
        self.registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
        self.changes = set()

    def get_dark_mode(self) -> bool:
        """
        True if

        :return:"""
        key = winreg.OpenKey(
            self.registry,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        return False if winreg.QueryValueEx(key, "AppsUseLightTheme")[0] else True

    def set_dark_mode(self, value: bool) -> None:
        """ """
        key = winreg.OpenKey(
            self.registry,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            0,
            winreg.KEY_ALL_ACCESS | winreg.KEY_WOW64_64KEY,
        )
        self.changes.add(key)

        # noinspection PyTypeChecker
        winreg.SetValueEx(
            key,
            "AppsUseLightTheme",
            None,
            winreg.REG_DWORD,
            int(not value),  # SHOULD BE OF TYPE INT!
        )

    def __enter__(self):
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
        # FlushKey
        # for c in self.changes:
        #  pass
        # self.registry

    @property
    def all_settings(self):
        """ """
        return None
        # for i in self.settings.list_properties():  # getting all existing properties
        #  yield i


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        ws = WindowsSettings()
        dark_mode = ws.get_dark_mode()
        ws.set_dark_mode(not dark_mode)

    a()
