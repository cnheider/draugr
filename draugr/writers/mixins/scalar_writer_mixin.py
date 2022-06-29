from abc import ABC, abstractmethod
from itertools import cycle

from warg import Number

__all__ = ["ScalarWriterMixin"]


class ScalarWriterMixin(ABC):
    """description"""

    @abstractmethod
    def _scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    def __init__(self):
        self._blip_values = iter(cycle(range(2)))

    def scalar(self, tag: str, value: Number, step_i: int = None) -> None:
        """

        :param tag:
        :type tag:
        :param value:
        :type value:
        :param step_i:
        :type step_i:"""
        if step_i:
            if self.filter(tag):
                self._scalar(tag, value, self._counter[tag])
            self._counter[tag] = step_i
        else:
            if self.filter(tag):
                self._scalar(tag, value, self._counter[tag])
            self._counter[tag] += 1

    def blip(self, tag: str, step_i: int = None) -> None:
        """

        :param tag:
        :type tag:
        :param step_i:
        :type step_i:"""
        if step_i:
            self.scalar(tag, next(self._blip_values), step_i)
            self.scalar(tag, next(self._blip_values), step_i)
        else:
            self.scalar(tag, next(self._blip_values))
            self.scalar(tag, next(self._blip_values), self._counter[tag])
