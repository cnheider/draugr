#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import contextlib
from time import sleep
from typing import Callable, Iterable

from pynput.keyboard import KeyCode

from draugr.python_utilities.styling import sprint

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from warg import GDKC, drop_unused_kws, passes_kws_to

# import keyboard

__all__ = ["add_early_stopping_key_combination", "CaptureEarlyStop"]


@drop_unused_kws
def add_early_stopping_key_combination(
    *callbacks: Iterable[Callable], has_x_server: bool = True, verbose: bool = False
):  # -> keyboard.Listener:

    """

    :param callbacks:
    :param has_x_server:
    :param verbose:
    :return:"""
    if not has_x_server:
        return

    from pynput import keyboard

    combinations = [
        {keyboard.Key.ctrl, keyboard.KeyCode(char="c")},
        {keyboard.Key.ctrl, keyboard.KeyCode(char="d")},
        {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="s")},
        {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="S")},
        {
            keyboard.Key.ctrl_l,
            keyboard.KeyCode(char="c"),
        },  # windows is annoying, does something weird translation....
        {keyboard.Key.ctrl_l, keyboard.KeyCode(char="d")},
        {KeyCode.from_char("\x04")},  # ctrl+d on windows
        {KeyCode.from_char("\x03")},  # ctrl+d on windows
    ]

    # The currently active modifiers
    current = set()

    # keyboard.add_hotkey(key, callback)
    assert all([isinstance(cb, Callable) for cb in callbacks])
    if verbose:
        sprint(
            f"\n\nPress any of:\n{combinations}\n for early stopping\n",
            color="red",
            bold=True,
            highlight=True,
        )

    def on_press(key):
        """ """
        if any([key in COMBO for COMBO in combinations]):
            if verbose:
                print(f"Adding key {key}")
            current.add(key)
            if any(all(k in current for k in COMBO) for COMBO in combinations):
                for clbck in callbacks:
                    if verbose:
                        print(f"Calling {clbck}")
                    clbck()

    def on_release(key):
        """ """
        if any([key in combo for combo in combinations]):
            if key in current:
                if verbose:
                    print(f"Removing key {key}")
                current.remove(key)

    return keyboard.Listener(on_press=on_press, on_release=on_release)


class CaptureEarlyStop(contextlib.AbstractContextManager):
    """
    Context for early stopping a loop"""

    @passes_kws_to(add_early_stopping_key_combination)
    def __init__(self, *args, **kwargs):
        self.listener = add_early_stopping_key_combination(*args, **kwargs)

    def __enter__(self):
        if self.listener:
            self.listener.start()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.listener:
            self.listener.stop()
        return False


if __name__ == "__main__":

    def c() -> None:
        """
        :rtype: None
        """
        print("start")
        RUN = True

        def stop_loop():
            """ """
            global RUN
            RUN = False

        with CaptureEarlyStop(stop_loop) as _:
            while RUN:
                sleep(0.1)
        print("done")

    def b():  # DOES NOT WORK!
        """ """
        print("start2")
        with CaptureEarlyStop(GDKC(exit, code=0)) as _:
            while True:
                sleep(0.1)
        print("done2")

    c()
    # b()
