#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import contextlib
from time import sleep
from typing import Iterable, MutableSequence

from draugr.python_utilities.styling import sprint

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from warg import GDKC, drop_unused_kws, passes_kws_to

__all__ = ["add_early_stopping_key_combination", "CaptureEarlyStop"]


@drop_unused_kws
def add_early_stopping_key_combination(
    callbacks: callable = None, has_x_server: bool = True, verbose: bool = False
):
    """

:param callbacks:
:param has_x_server:
:param verbose:
:return:
"""
    if not has_x_server:
        return

    from pynput import keyboard

    # import keyboard

    COMBINATIONS = [
        {keyboard.Key.ctrl, keyboard.KeyCode(char="c")},
        {keyboard.Key.ctrl, keyboard.KeyCode(char="d")},
        {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="s")},
        {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="S")},
    ]

    CALLBACKS: MutableSequence = []
    # The currently active modifiers
    current = set()

    # keyboard.add_hotkey(key, callback)
    if callbacks is not None:
        if isinstance(callbacks, Iterable):
            CALLBACKS.extend(callbacks)
        else:
            CALLBACKS.append(callbacks)
    if verbose:
        sprint(
            f"\n\nPress any of:\n{COMBINATIONS}\n for early stopping\n",
            color="red",
            bold=True,
            highlight=True,
        )

    def on_press(key):
        if any([key in COMBO for COMBO in COMBINATIONS]):
            if verbose:
                print(f"Adding key {key}")
            current.add(key)
            if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
                for clbck in CALLBACKS:
                    if verbose:
                        print(f"Calling {clbck}")
                    clbck()

    def on_release(key):
        if any([key in combo for combo in COMBINATIONS]):
            if key in current:
                if verbose:
                    print(f"Removing key {key}")
                current.remove(key)

    return keyboard.Listener(on_press=on_press, on_release=on_release)


class CaptureEarlyStop(contextlib.AbstractContextManager):
    """
  Context for early stopping a loop
  """

    @passes_kws_to(add_early_stopping_key_combination)
    def __init__(self, *args, **kwargs):
        self.listener = add_early_stopping_key_combination(*args, **kwargs)

    def __enter__(self):
        self.listener.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.listener.stop()
        return False


if __name__ == "__main__":

    def c():
        print("start")
        RUN = True

        def stop_loop():
            global RUN
            RUN = False

        with CaptureEarlyStop(stop_loop) as _:
            while RUN:
                sleep(0.1)
        print("done")

    def b():  # DOES NOT WORK!
        print("start2")
        with CaptureEarlyStop(GDKC(exit, code=0)) as _:
            while True:
                sleep(0.1)
        print("done2")

    c()
    # b()
