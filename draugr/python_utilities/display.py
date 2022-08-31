#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/30/22
           """

__all__ = []

from warg.os.os_platform import has_x_server


def windows_display_test():
    import subprocess, os

    if os.name == "nt":  # Windows
        import wmi

        try:
            wmi.WMI().computer.Win32_VideoController()[0]  # Tested on Windows 10
            return 1
        except:
            pass

    elif os.name == "posix":  # Linux
        out = subprocess.getoutput(
            "sudo lshw -c video | grep configuration"
        )  # Tested on CENTOS 7 and Ubuntu
        if out:
            return 1


if __name__ == "__main__":
    print(has_x_server())
