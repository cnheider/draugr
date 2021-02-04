#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27-01-2021
           """

__all__ = ["max_decimation_subsample", "fir_subsample", "fft_subsample"]

from typing import Tuple

import numpy
from scipy.signal import decimate, resample


def max_decimation_subsample(
    signal: numpy.ndarray, decimation_factor: int = 10
) -> numpy.ndarray:
    return numpy.max(signal.reshape(-1, decimation_factor), axis=1)


def fir_subsample(
    signal: numpy.ndarray, max_resolution: int, sampling_rate: int
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if signal.shape[-1] > max_resolution:
        sub_signal = decimate(signal, (signal.shape[-1] // max_resolution) + 1, axis=-1)
    else:
        sub_signal = signal
    sub_time = numpy.linspace(
        0,
        signal.shape[-1] // sampling_rate,  # Get time from indices
        num=sub_signal.shape[-1],
    )
    return sub_time, sub_signal


def fft_subsample(
    signal: numpy.ndarray, max_resolution: int, sampling_rate: int
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if signal.shape[-1] > max_resolution:
        sub_signal = resample(signal, max_resolution, axis=-1)
    else:
        sub_signal = signal
    sub_time = numpy.linspace(
        0,
        signal.shape[-1] // sampling_rate,  # Get time from indices
        num=sub_signal.shape[-1],
    )
    return sub_time, sub_signal


if __name__ == "__main__":

    def aisjd():
        a = numpy.arange(0, 100)
        print(max_decimation_subsample(a))

    aisjd()
