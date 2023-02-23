#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """

from warg import IgnoreInterruptSignal, busy_indicator


def asd() -> None:
    """
    :rtype: None
    """
    import time

    next_reading = time.time()
    with IgnoreInterruptSignal():
        print("Publisher started")

        for _ in busy_indicator():
            next_reading += 3
            sleep_time = next_reading - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    if False:
        from pathlib import Path

        from draugr.os_utilities.linux_utilities.systemd_utilities.service_management import (
            install_service,
        )

        install_service(Path(__file__), "busy_script")
    else:
        asd()
