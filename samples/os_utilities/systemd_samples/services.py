#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """

if __name__ == "__main__":
    # print()

    # print(sh.ls("/home/heider"))
    # print(sys.executable)

    # print(sh.systemctl('status', 'lightdm.service'))

    from draugr.os_utilities.linux_utilities import (
        RunAsEnum,
        remove_service,
    )

    remove_service("busy_script", run_as=RunAsEnum.root)
    # install_service(Path(busy_script.__file__), "busy_script",run_as=RunAsEnum.root)
