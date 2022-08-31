#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = ""

__all__ = ["add_early_stopping_key_combination", "CaptureEarlyStop"]

import contextlib
from time import sleep
from typing import Callable, Iterable, Sequence, MutableMapping

from pynput.keyboard import KeyCode

from warg import GDKC, drop_unused_kws, passes_kws_to, sprint

# import keyboard

try:
    from pynput import keyboard

    default_combinations = [
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
except Exception as e:
    default_combinations = []
    print("pynput not installed, no early stopping, error:", e)


@drop_unused_kws
def add_early_stopping_key_combination(
    *callbacks: Callable,
    has_x_server: bool = True,
    verbose: bool = False,
    combinations: Iterable = default_combinations,
):  # -> keyboard.Listener:

    """

    :param combinations:
    :type combinations:
    :param callbacks:
    :param has_x_server:
    :param verbose:
    :return:"""
    if not has_x_server:
        return

    if combinations is None:
        combinations = default_combinations

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
        """description"""
        if any([key in COMBO for COMBO in combinations]):
            if verbose:
                print(f"Adding key {key}")
            current.add(key)
            if any(all(k in current for k in COMBO) for COMBO in combinations):
                for clbck in callbacks:
                    if verbose:
                        print(f"Calling {clbck}")
                    clbck("User pressed a early stopping key")

    def on_release(key):
        """description"""
        if any([key in combo for combo in combinations]):
            if key in current:
                if verbose:
                    print(f"Removing key {key}")
                current.remove(key)

    try:
        return keyboard.Listener(on_press=on_press, on_release=on_release)
    except Exception as e1:
        print("pynput not installed, no early stopping, error:", e1)
        return


class CaptureEarlyStop(contextlib.AbstractContextManager):
    """
    Context for early stopping a loop"""

    @passes_kws_to(add_early_stopping_key_combination)
    def __init__(self, *args: Sequence, **kwargs: MutableMapping):
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
            """description"""
            global RUN
            RUN = False

        with CaptureEarlyStop(stop_loop) as _:
            while RUN:
                sleep(0.1)
        print("done")

    def b():  # DOES NOT WORK!
        """description"""
        print("start2")
        with CaptureEarlyStop(GDKC(exit, code=0)) as _:
            while True:
                sleep(0.1)
        print("done2")

    c()
    # b()
