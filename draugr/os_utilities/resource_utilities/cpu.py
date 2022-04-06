#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

            Small utilities to keep track of cores dedicated to workers threads

           Created on 07-12-2020
           """

import os

from warg import AlsoDecorator, passes_kws_to

CORE_COUNT: int = os.cpu_count()
IN_USE_BY_THIS_PROCESS: int = 0

__all__ = [
    "request_worker_cores",
    "reset_worker_tracker",
    "release_worker_cores",
    "worker_cores_available",
    "worker_cores_in_use",
    "WorkerSession",
]


def request_worker_cores(
    percentage: float, *, of_remaining: bool = False, verbose: bool = False
) -> int:
    """
    global_pin_memory
    :param percentage:
    :param of_remaining:
    :param verbose:
    :return:"""
    global IN_USE_BY_THIS_PROCESS

    if IN_USE_BY_THIS_PROCESS >= CORE_COUNT:
        print(
            f"WARNING! (IN_USE_BY_THIS_PROCESS/CORES_AVAILABLE) {IN_USE_BY_THIS_PROCESS}/{CORE_COUNT}"
        )
        return 1

    if of_remaining:
        cores = round((CORE_COUNT - IN_USE_BY_THIS_PROCESS) * percentage)
    else:
        cores = round(CORE_COUNT * percentage)

    if verbose:
        print(f"reserving {cores} workers")

    IN_USE_BY_THIS_PROCESS += cores

    return cores


def release_worker_cores(num: int) -> int:
    """

    :param num:
    :return:"""
    global IN_USE_BY_THIS_PROCESS
    res = max(IN_USE_BY_THIS_PROCESS - num, 0)
    IN_USE_BY_THIS_PROCESS = res
    return res


def core_count() -> int:
    """

    :return:"""
    return CORE_COUNT


def worker_cores_available() -> int:
    """
    maybe negative if over allocated
    :return:"""
    return CORE_COUNT - IN_USE_BY_THIS_PROCESS


def worker_cores_in_use() -> int:
    """

    :return:"""
    return IN_USE_BY_THIS_PROCESS


def reset_worker_tracker() -> None:
    """

    :return:"""
    global IN_USE_BY_THIS_PROCESS
    IN_USE_BY_THIS_PROCESS = 0


class WorkerSession(AlsoDecorator):
    """
    request cores
    """

    @passes_kws_to(request_worker_cores)
    def __init__(self, percentage, **kwargs):
        self.percentage = percentage
        self.kws = kwargs

    def __enter__(self):
        self.num = request_worker_cores(self.percentage, **self.kws)
        return self.num

    def __exit__(self, exc_type, exc_val, exc_tb):
        release_worker_cores(self.num)


if __name__ == "__main__":
    print(worker_cores_available())
    print(worker_cores_in_use())
    print(request_worker_cores(0.5))
    print(worker_cores_available())
    print(worker_cores_in_use())
    release_worker_cores(round(core_count() * 0.5))
    print(worker_cores_in_use())
    print()
    print(worker_cores_available())
    print()
    with WorkerSession(0.33) as num_cores:
        print(num_cores)
        print(worker_cores_in_use())
        print(worker_cores_available())
    print(worker_cores_in_use())
    print()
    with WorkerSession(0.25) as num_cores:
        with WorkerSession(0.25) as num_cores_inner:
            print(num_cores)
            print(num_cores_inner)
            print(worker_cores_in_use())
            print(worker_cores_available())
    print(worker_cores_in_use())
