#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """

import time
from enum import Enum
from itertools import tee
from typing import Any, Generator, Iterable

import tqdm
from sorcery import assigned_names

from notus.notification import JobNotificationSession
from warg import drop_unused_kws, passes_kws_to, empty_str

__all__ = ["progress_bar"]


class TimestampModeEnum(Enum):
    none, prefix, postfix = assigned_names()


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
    verbose: bool = False,
    timestamp_mode: TimestampModeEnum = TimestampModeEnum.none,
    **kwargs,
) -> Any:
    """
    hint:
    use next first and then later use send instead of next to set a new tqdm description if desired.
    """
    if not disable:
        if description is None and auto_describe_iterator:
            from warg import get_first_arg_name

            description = get_first_arg_name(alias, verbose=verbose)

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
        prefix = empty_str
        postfix = empty_str
        update_timestamp = True
        if timestamp_mode == TimestampModeEnum.prefix:
            prefix = time.time
        elif timestamp_mode == TimestampModeEnum.postfix:
            postfix = time.time
        else:
            update_timestamp = False

        if notifications:
            with JobNotificationSession(description):
                for val in generator:
                    a = yield val
                    if update_timestamp or a:
                        if a:
                            description = str(a)
                        generator.set_description(
                            " ".join(
                                (str(prefix()), description, str(postfix()))
                            ).strip()
                        )
            return
        for val in generator:
            a = yield val
            if update_timestamp or a:
                if a:
                    description = str(a)
                generator.set_description(
                    " ".join((str(prefix()), description, str(postfix()))).strip()
                )
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

        pb = progress_bar  # Aliased! does not find description at the moment

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

    def named() -> None:
        """
        :rtype: None
        """
        from time import sleep

        list_items = [2.13, 8921.9123, 923]

        for a in progress_bar(list_items):
            sleep(1)

    def named_time() -> None:
        """
        :rtype: None
        """
        from time import sleep

        list_items = [2.13, 8921.9123, 923]

        for a in progress_bar(list_items, timestamp_mode=TimestampModeEnum.prefix):
            sleep(1)

    def send_example() -> None:
        """
        :rtype: None
        """
        from itertools import count

        pb = progress_bar(count(), timestamp_mode=TimestampModeEnum.none)
        next(pb)
        for a in range(100000):
            pb.send(f"step_{a}")

    def send_example_time() -> None:
        """
        :rtype: None
        """
        from itertools import count

        pb = progress_bar(count(), timestamp_mode=TimestampModeEnum.postfix)
        next(pb)
        for a in range(100000):
            pb.send(f"step_{a}")

    # dsad3123()
    # asd21sa()
    # dict_items()
    # send_example()
    # named()
    named_time()
    # send_example_time()
