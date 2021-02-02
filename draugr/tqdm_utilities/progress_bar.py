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
    auto_total_generator: bool = True,
    auto_describe_iterator: bool = True,  # DOES NOT WORK IS THIS FUNCTION IS ALIAS does not match!
    alias="progress_bar",
    **kwargs
) -> Any:
    if description is None and auto_describe_iterator:
        from warg import get_first_arg_name

        description = get_first_arg_name(alias)

    if total is None and isinstance(iterable, Generator) and auto_total_generator:
        iterable, ic = tee(iterable, 2)
        total = len(list(ic))
    if notifications:
        with JobNotificationSession(description):
            yield from tqdm.tqdm(
                iterable, description, leave=leave, total=total, **kwargs
            )
        return
    yield from tqdm.tqdm(iterable, description, leave=leave, total=total, **kwargs)


if __name__ == "__main__":

    def dsad3123():
        from time import sleep

        for a in progress_bar([2.13, 8921.9123, 923], notifications=False):
            sleep(1)

    def asd21sa():
        from time import sleep

        pb = progress_bar  # Aliased!

        for a in pb([2.13, 8921.9123, 923], notifications=False):
            sleep(1)

    def dict_items():
        from time import sleep

        class exp_v:
            Test_Sets = {v: v for v in range(9)}

        for a in progress_bar(exp_v.Test_Sets.items()):
            sleep(1)

    # dsad3123()
    # asd21sa()
    dict_items()
