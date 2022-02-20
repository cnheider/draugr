#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """

from itertools import tee
from typing import Any, Generator, Iterable

import tqdm

from notus.notification import JobNotificationSession
from warg import drop_unused_kws, passes_kws_to

__all__ = ["progress_bar"]


@drop_unused_kws
@passes_kws_to(tqdm.tqdm)
def progress_bar(
    iterable: Iterable,
    description: str = None,
    *,
    leave: bool = False,
    notifications: bool = False,
    total: int = None,
    auto_total_generator: bool = False,
    auto_describe_iterator: bool = True,  # DOES NOT WORK IS THIS FUNCTION IS ALIAS does not match!
    alias="progress_bar",
    disable: bool = False,
    **kwargs,
) -> Any:
    """
    hint:
    use next first and then later use send instead of next to set a new tqdm description if desired.
    """
    if not disable:
        if description is None and auto_describe_iterator:
            from warg import get_first_arg_name

            description = get_first_arg_name(alias)

        if total is None and isinstance(iterable, Generator) and auto_total_generator:
            iterable, ic = tee(iterable, 2)
            total = len(list(ic))
            if total == 0:
                print(f"WARNING zero length iterable - {description}:{iterable}")

        generator = tqdm.tqdm(
            iterable,
            description,
            leave=leave,
            total=total,
            disable=disable,  # redundant
            **kwargs,
        )
        if notifications:
            with JobNotificationSession(description):
                for val in generator:
                    a = yield val
                    if a:
                        generator.set_description(a)
            return
        for val in generator:
            a = yield val
            if a:
                generator.set_description(a)
    else:
        yield from iterable


if __name__ == "__main__":

    def dsad3123() -> None:
        """
        :rtype: None
        """
        from time import sleep

        for a in progress_bar([2.13, 8921.9123, 923], notifications=False):
            sleep(1)

    def asd21sa() -> None:
        """
        :rtype: None
        """
        from time import sleep

        pb = progress_bar  # Aliased!

        for a in pb([2.13, 8921.9123, 923], notifications=False):
            sleep(1)

    def dict_items() -> None:
        """
        :rtype: None
        """
        from time import sleep

        class exp_v:
            Test_Sets = {v: v for v in range(9)}

        for a in progress_bar(exp_v.Test_Sets.items()):
            sleep(1)

    def send_example() -> None:
        """
        :rtype: None
        """
        from itertools import count

        pb = progress_bar(count())
        next(pb)
        for a in range(100000):
            pb.send(f"step_{a}")

    # dsad3123()
    # asd21sa()
    # dict_items()
    send_example()
