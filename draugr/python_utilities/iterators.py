#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/10/22
           """

__all__ = ["BidirectionalIterator", "prev"]


from typing import Iterator, Any


class BidirectionalIterator(Iterator):
    """
    Enable iteration forward and backward

    """

    def __next__(self) -> Any:
        self.i += 1
        if self.i < len(self.history):
            return self.history[self.i]
        else:
            elem = next(self.iterator)
            self.history.append(elem)
            return elem

    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.history = [
            None,
        ]
        self.i = 0

    def next(self):
        """

        :return:
        :rtype:
        """
        return self.__next__()

    def prev(self):
        """

        :return:
        :rtype:
        """
        self.i -= 1
        if self.i == 0:
            raise StopIteration
        else:
            return self.history[self.i]


def prev(iterator: BidirectionalIterator) -> Any:
    """

    :param iterator:
    :type iterator:
    :return:
    :rtype:
    """
    return iterator.prev()


if __name__ == "__main__":

    def asdasf() -> None:
        a = BidirectionalIterator(iter([1, 2, 3, 4, 5, 6]))
        print(next(a))
        print(next(a))
        print(prev(a))
        print(next(a))
        print(a.next())
        print(a.prev())
        print(a.prev())
        print(a.next())
        print(a.next())

    asdasf()
