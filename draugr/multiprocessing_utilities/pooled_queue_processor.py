#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable

from cloudpickle import cloudpickle

__author__ = "Christian Heider Nielsen"

import multiprocessing
import queue

__all__ = ["CloudPickleBase", "PooledQueueTask", "PooledQueueProcessor"]


class CloudPickleBase(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    :param x: (Any) the variable you wish to wrap for pickling with cloudpickle"""

    def __init__(self, x: Any):
        self._x = x

    def __getstate__(self):
        return cloudpickle.dumps(self._x)

    def __setstate__(self, x):
        self._x = pickle.loads(x)

    def __call__(self, *args, **kwargs):
        return self._x(*args, **kwargs)


class PooledQueueTask(ABC):
    """
    Pooled queue task interface"""

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs) -> Any:
        """

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:"""
        raise NotImplemented


class PooledQueueProcessor(object):
    """
    This is a workaround of Pythons extremely slow interprocess communication pipes.
    The ideal solution would be to use a multiprocessing.queue, but it apparently communication is band
    limited.
    This solution has processes complete tasks (batches) and a thread add the results to a queue.queue."""

    def __init__(
        self,
        func,
        args: Iterable = (),
        kwargs=None,
        max_queue_size=100,
        n_proc=None,
        max_tasks_per_child=None,
        fill_at_construction=True,
        blocking=True,
    ):

        if kwargs is None:
            kwargs = {}

        self._max_queue_size = max_queue_size
        if isinstance(func, type):
            func = func()
        self._func = CloudPickleBase(func)
        self.args = args
        self.kwargs = kwargs
        self.blocking = blocking
        if max_tasks_per_child is None:
            max_tasks_per_child = max_queue_size // 4

        if n_proc is None:
            n_proc = multiprocessing.cpu_count()

        self._queue = queue.Queue(maxsize=max_queue_size)
        self._pool = multiprocessing.Pool(n_proc, maxtasksperchild=max_tasks_per_child)

        if fill_at_construction:
            self.fill()

    def fill(self) -> None:
        """
        fill queue"""
        for i in range(self._max_queue_size):
            self.maybe_fill()

    def close(self) -> None:
        """
        close pool"""
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        """
        terminate pool"""
        self._pool.terminate()
        self._pool.join()

    def maybe_fill(self) -> None:
        """
        fill queue if not full"""
        if self.queue_size < self._max_queue_size:  # and not self._queue.full():
            self._pool.apply_async(
                self._func, self.args, self.kwargs, self.put, self.raise_error
            )

    @property
    def queue_size(self) -> int:
        """

        :return:
        :rtype:"""
        return self._queue.qsize()

    def put(self, res) -> None:
        """

        :param res:
        :type res:"""
        self._queue.put(res)

    def raise_error(self, excptn) -> None:
        """

        :param excptn:
        :type excptn:"""
        self._pool.terminate()
        self._pool.close()
        # print(excptn.__cause__)
        # sys.exit(1)
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        raise excptn

    def get(self) -> Any:
        """

        :return:"""
        if self.queue_size < 1:  # self._queue.empty():
            if len(multiprocessing.active_children()) == 0:
                if self.blocking:
                    self.maybe_fill()
                else:
                    raise StopIteration

        res = self._queue.get(self.blocking)
        self.maybe_fill()
        return res

    def __len__(self) -> int:
        return self.queue_size

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        return self.get()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.terminate()
        self._pool.close()
        if exc_type:
            # print(exc_type, exc_val, exc_tb) # trace_back
            raise exc_type(exc_val)
            # sys.exit()


if __name__ == "__main__":

    class Square(PooledQueueTask):
        def call(self, i, *args, **kwargs):
            """

            :param i:
            :type i:
            :param args:
            :type args:
            :param kwargs:
            :type kwargs:
            :return:
            :rtype:"""
            return i * 2

    class Exc(PooledQueueTask):
        def call(self, *args, **kwargs):
            """

            :param args:
            :type args:
            :param kwargs:
            :type kwargs:"""
            raise NotImplementedError

    task = Square()

    processor = PooledQueueProcessor(
        task, [2], fill_at_construction=True, max_queue_size=100
    )
    for GPU_STATS, _ in zip(processor, range(30)):
        print(GPU_STATS)

    processor.blocking = True
    processor.args = [4]
    time.sleep(3)
    for GPU_STATS in processor:
        print(GPU_STATS)
        if GPU_STATS == 8:
            break
