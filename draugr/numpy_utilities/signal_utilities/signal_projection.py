#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-12-2020
           """

import numpy

__all__ = [
    "normalise_signal",
    "normalise_signal_minmax",
    "normalise_signal_max_abs",
    "heaviside",
]

from draugr.numpy_utilities.signal_utilities.signal_statistics import mean_square


def normalise_signal_minmax(signal: numpy.ndarray) -> numpy.ndarray:
    """ """
    return numpy.interp(signal, (signal.min(), signal.max()), (-1, 1))


def normalise_signal_max_abs(signal: numpy.ndarray) -> numpy.ndarray:
    """ """
    return signal / numpy.abs(signal).max()


def normalise_signal(
    signal: numpy.ndarray, variance: numpy.ndarray = None
) -> numpy.ndarray:
    """Normalise power in y to a (standard normal) white noise signal.
    Optionally normalise to power in signal `x`.
    The mean power of a Gaussian with `mu=0` and `sigma=1` is 1."""
    if variance is not None:
        variance = mean_square(variance)
    else:
        variance = 1.0

    return signal * numpy.sqrt(variance / mean_square(signal))


def heaviside(signal: numpy.ndarray) -> numpy.ndarray:
    """Heaviside.

    Returns the value 0 for `x < 0`, 1 for `x > 0`, and 1/2 for `x = 0`."""
    return 0.5 * (numpy.sign(signal) + 1)


if __name__ == "__main__":

    def asiudha() -> None:
        """
        :rtype: None
        """
        a = numpy.random.random(9) - 0.5
        print(
            normalise_signal_minmax(a),
            "\n",
            normalise_signal_max_abs(a),
            "\n",
            normalise_signal(a),
        )
        print(heaviside(a))

    asiudha()
