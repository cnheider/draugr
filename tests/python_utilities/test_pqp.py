#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import pytest

from draugr.multiprocessing_utilities.pooled_queue_processor import (
    PooledQueueProcessor,
    PooledQueueTask,
)
from warg.functions import identity

__author__ = "Christian Heider Nielsen"


class Square(PooledQueueTask):
    def call(self, i, *args, **kwargs):
        """ """
        return i * 2


class Exc(PooledQueueTask):
    def call(self, *args, **kwargs):
        """ """
        raise NotImplementedError


@pytest.mark.skip
def test_integration_success():
    task = Square()

    with PooledQueueProcessor(
        task, [2], fill_at_construction=True, max_queue_size=10
    ) as processor:
        for a, _ in zip(processor, range(30)):
            pass
            # print(a)


@pytest.mark.skip
def test_integration_func():
    task = identity

    with PooledQueueProcessor(task, [2], max_queue_size=10) as processor:
        for a, _ in zip(processor, range(30)):
            pass
            # print(a)


@pytest.mark.skip
def test_lambda_func():
    task = lambda x: x

    with PooledQueueProcessor(task, [2], max_queue_size=10) as processor:
        for a, _ in zip(processor, range(30)):
            pass
            # print(a)


@pytest.mark.skip
def test_integration_except():
    task = Exc()

    with pytest.raises(NotImplementedError) as exc_info:
        task()  # TODO: MP does not work in pytest
        processor = PooledQueueProcessor(task, [2], max_queue_size=10, blocking=True)
        for a, _ in zip(processor, range(30)):
            pass
            # print(a)

    assert exc_info.type is NotImplementedError


@pytest.mark.skip
#
def test_integration_except_ctx():
    task = Exc()

    with pytest.raises(NotImplementedError) as exc_info:
        task()  # TODO: MP does not work in pytest
        with PooledQueueProcessor(task, [2], max_queue_size=10) as processor:
            for a, _ in zip(processor, range(30)):
                pass
                # print(a)

    assert exc_info.type is NotImplementedError


if __name__ == "__main__":
    test_lambda_func()
