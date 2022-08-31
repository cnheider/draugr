from abc import ABC, abstractmethod
from itertools import cycle
from typing import MutableMapping
from warg import Number, passes_kws_to, drop_unused_kws
from draugr.python_utilities import CounterFilter


__all__ = ["ScalarWriterMixin"]


class ScalarWriterMixin(CounterFilter, ABC):
    """description"""

    @abstractmethod
    def _scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    @passes_kws_to(CounterFilter.__init__)
    @drop_unused_kws
    def __init__(self, **kwargs: MutableMapping):
        super().__init__(**kwargs)
        self._blip_iterators = {}

    def scalar(self, tag: str, value: Number, step_i: int = None) -> None:
        """

        :param tag:
        :type tag:
        :param value:
        :type value:
        :param step_i:
        :type step_i:"""
        if step_i:
            self._counter[tag] = step_i
        else:
            self._counter[tag] += 1

        if self.filter(tag):
            self._scalar(tag, value, self._counter[tag])

    def blip(self, tag: str, step_i: int = None) -> None:
        """

        :param tag:
        :type tag:
        :param step_i:
        :type step_i:"""
        if tag not in self._blip_iterators:
            self._blip_iterators[tag] = iter(cycle(range(2)))
        self.scalar(tag, next(self._blip_iterators[tag]), step_i=step_i)
