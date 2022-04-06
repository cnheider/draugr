#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "heider"
__doc__ = r"""

           Created on 03/02/2022
           """

__all__ = []

import subprocess


def is_dark_mode_active():
    """
    Bad
    :return:
    :rtype:
    """
    try:
        out = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
            capture_output=True,
        )
        # Gtk.Settings.get_default().set_property("gtk-theme-name", "MS-Windows-XP")
        stdout = out.stdout.decode()
    except Exception:
        return False

    if stdout.lower().strip()[1:-1].endswith("-dark"):
        return True
    return False


if __name__ == "__main__":
    print(is_dark_mode_active())
