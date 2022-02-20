#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-05-2021
           """

import psutil

PAGING_BUFFER_SIZE = 0.2 * psutil.virtual_memory().total

__all__ = ["num_instance_no_paging"]


def num_instance_no_paging(expected_size_mb: int = 1024) -> int:
    """

    :param expected_size_mb:
    :return:
    """
    return int(
        (psutil.virtual_memory().available - PAGING_BUFFER_SIZE)
        / expected_size_mb
        * 1e-6
    )


if __name__ == "__main__":
    print(num_instance_no_paging())
