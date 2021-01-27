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
    desc: str = None,
    *,
    leave: bool = False,
    notifications: bool = False,
    total: int = None,
    auto_total_generator: bool = True,
    auto_desc: bool = True,
    **kwargs
) -> Any:
    if desc is None and auto_desc:  # TODO: MAY break!
        # Local imports
        import inspect
        import textwrap
        import ast
        from warg import FirstArgIdentifier

        caller_frame = inspect.currentframe().f_back
        # caller_src_code_snippet = inspect.getsource(caller_frame) # Only gets scope
        caller_src_code_lines = inspect.getsourcelines(caller_frame)
        caller_src_code_valid = textwrap.dedent(
            "".join(caller_src_code_lines[0])
        )  # TODO: maybe there is a nicer way?
        call_nodes = ast.parse(
            caller_src_code_valid
        )  # parse code to get nodes of abstract syntax tree of the call
        fai = FirstArgIdentifier("progress_bar")
        fai.visit(call_nodes)
        snippet_offset = caller_src_code_lines[1] - 1
        desc = fai.result["progress_bar"][caller_frame.f_lineno - snippet_offset]

    if isinstance(iterable, Generator) and auto_total_generator:
        iterable, ic = tee(iterable, 2)
        total = len(list(ic))
    if notifications:
        with JobNotificationSession(desc):
            yield from tqdm.tqdm(iterable, desc, leave=leave, total=total, **kwargs)
        return
    yield from tqdm.tqdm(iterable, desc, leave=leave, total=total, **kwargs)
