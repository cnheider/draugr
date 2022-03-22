#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 16-12-2020
           """

from typing import Iterable, Sequence

__all__ = ["last_dim_truncate", "min_length_truncate_batch", "truncate_to_power_2"]

from warg.math.powers import prev_pow_2


def min_length_truncate_batch(batch: Iterable[Sequence]) -> Iterable:
    """

    :param batch:
    :return:"""
    min_seq_len = min([s.shape[-1] for s in batch])
    return [last_dim_truncate(s, min_seq_len) for s in batch]


def min_length_truncate_batch_2d(batch: Iterable[Sequence]) -> Iterable:
    """

    :param batch:
    :return:"""
    min_seq_len = min([len(s) for s in batch])
    return [last_dim_truncate(s, min_seq_len) for s in batch]


def last_dim_truncate(sequence: Sequence, min_length: int) -> Sequence:
    """

    :param sequence:
    :param min_length:
    :return:"""
    return sequence[..., :min_length]


def truncate_to_power_2(signal: Sequence) -> Sequence:
    """ """
    return last_dim_truncate(signal, prev_pow_2(len(signal)))


if __name__ == "__main__":

    def gasdasa() -> None:
        """
        :rtype: None
        """
        from draugr.torch_utilities import to_tensor

        base = 5
        stair_length = 9
        stair = [to_tensor(range(i + base)) for i in range(stair_length)]

        trunc = [last_dim_truncate(s, base) for s in stair]
        print(to_tensor(trunc))

    def basdiuj() -> None:
        """
        :rtype: None
        """
        import numpy

        asda = numpy.arange(2**5 - 1)
        print(asda, len(asda))
        trunc = truncate_to_power_2(asda)
        print(trunc, len(trunc))

    # gasdasa()
    basdiuj()
