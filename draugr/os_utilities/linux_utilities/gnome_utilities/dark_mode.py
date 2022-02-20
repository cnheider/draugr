#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

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
