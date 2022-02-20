#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from shutil import rmtree

__author__ = "Christian Heider Nielsen"
__doc__ = ""


def main(keep_alive: bool = True, use_asyncio: bool = True) -> str:
    """

      :param use_asyncio:
    :param keep_alive:
    :type keep_alive:
    :return:
    :rtype:"""
    from draugr.torch_utilities import launch_tensorboard

    # from draugr import IgnoreInterruptSignal
    # from contextlib import suppress
    from time import sleep

    from apppath import AppPath

    import argparse

    parser = argparse.ArgumentParser(description="Option for launching tensorboard")
    parser.add_argument("NAME", type=str, help="App name to open AppPath for")
    parser.add_argument(
        "--author", type=str, help="App author to open AppPath for", default=None
    )
    parser.add_argument(
        "--version", type=str, help="App version to open AppPath for", default=None
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

    project_app_path = AppPath(args.NAME, args.author, args.version)
    log_dir = project_app_path.user_log

    if args.clean:
        print(f"Wiping {log_dir}")
        if log_dir.exists():
            rmtree(str(log_dir))
        else:
            log_dir.mkdir()

    address = launch_tensorboard(log_dir, args.port)

    if keep_alive:
        print(f"tensorboard address: {address} for log_dir {log_dir}")

        if use_asyncio:
            import asyncio

            async def work():
                """ """
                while True:
                    await asyncio.sleep(1)
                    # print("Task Executed")

            loop = asyncio.get_event_loop()
            try:
                asyncio.ensure_future(work())
                loop.run_forever()
            except KeyboardInterrupt:
                pass
            finally:
                print("Closing Loop")
                loop.close()
        else:
            # with IgnoreInterruptSignal(): # Do not block
            while True:
                sleep(10)

    return address


if __name__ == "__main__":
    import webbrowser

    """
Type Name

Class Name

Notes

'mozilla'

Mozilla('mozilla')

'firefox'

Mozilla('mozilla')

'netscape'

Mozilla('netscape')

'galeon'

Galeon('galeon')

'epiphany'

Galeon('epiphany')

'skipstone'

BackgroundBrowser('skipstone')

'kfmclient'

Konqueror()

(1)

'konqueror'

Konqueror()

(1)

'kfm'

Konqueror()

(1)

'mosaic'

BackgroundBrowser('mosaic')

'opera'

Opera()

'grail'

Grail()

'links'

GenericBrowser('links')

'elinks'

Elinks('elinks')

'lynx'

GenericBrowser('lynx')

'w3m'

GenericBrowser('w3m')

'windows-default'

WindowsDefault

(2)

'macosx'

MacOSX('default')

(3)

'safari'

MacOSX('safari')

(3)

'google-chrome'

Chrome('google-chrome')

'chrome'

Chrome('chrome')

'chromium'

Chromium('chromium')

'chromium-browser'

Chromium('chromium-browser')



"""

    # Open URL in a new tab, if a browser window is already open.
    webbrowser.open_new_tab(main())

    ## Open URL in new window, raising the window if possible.
    # webbrowser.open_new(url)
