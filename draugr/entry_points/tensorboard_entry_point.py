#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from shutil import rmtree

__author__ = "Christian Heider Nielsen"
__doc__ = ""


def main(keep_alive: bool = True) -> str:
    """

    :param keep_alive:
    :type keep_alive:
    :return:
    :rtype:
    """
    from draugr.torch_utilities.writers.tensorboard.launcher import launch_tensorboard
    from contextlib import suppress
    from time import sleep

    from apppath import AppPath

    import argparse

    parser = argparse.ArgumentParser(description="Option for launching tensorboard")
    parser.add_argument(
        "APP_NAME", metavar="Name", type=str, help="App name to open AppPath for"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Adding --clean argument will wipe tensorboard logs",
    )
    parser.add_argument(
        "--port", default=6006, help="Which port should tensorboard be served on"
    )
    args = parser.parse_args()

    PROJECT_APP_PATH = AppPath(args.APP_NAME)

    log_dir = PROJECT_APP_PATH.user_log

    if args.clean:
        print(f"Wiping {log_dir}")
        if log_dir.exists():
            rmtree(str(log_dir))
        else:
            log_dir.mkdir()

    address = launch_tensorboard(log_dir, args.port)

    if keep_alive:
        print(f"tensorboard address: {address} for log_dir {log_dir}")
        with suppress(KeyboardInterrupt):
            while True:
                sleep(10)

    return address


if __name__ == "__main__":
    main()
