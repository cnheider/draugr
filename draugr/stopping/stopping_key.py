#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import MutableSequence

from draugr.writers.terminal import sprint

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from warg import drop_unused_kws

__all__ = ["add_early_stopping_key_combination"]


@drop_unused_kws
def add_early_stopping_key_combination(
    callback: callable, has_x_server: bool = True, verbose: bool = False
):
    """

:param callback:
:param has_x_server:
:param verbose:
:return:
"""
    if not has_x_server:
        return

    from pynput import keyboard

    # import keyboard

    COMBINATIONS = [
        {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="s")},
        {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="S")},
    ]

    CALLBACKS: MutableSequence = []
    # The currently active modifiers
    current = set()

    # keyboard.add_hotkey(key, callback)
    CALLBACKS.append(callback)
    if verbose:
        sprint(
            f"\n\nPress any of:\n{COMBINATIONS}\n for early stopping\n",
            color="red",
            bold=True,
            highlight=True,
        )

    def on_press(key):
        if any([key in COMBO for COMBO in COMBINATIONS]):
            current.add(key)
            if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
                for clbck in CALLBACKS:
                    clbck()

    def on_release(key):
        if any([key in COMBO for COMBO in COMBINATIONS]):
            current.remove(key)

    return keyboard.Listener(on_press=on_press, on_release=on_release)
