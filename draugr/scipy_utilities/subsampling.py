#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27-01-2021
           """

__all__ = [
    "max_decimation_subsample",
    "min_decimation_subsample",
    "mag_decimation_subsample",
    "fir_subsample",
    "fft_subsample",
]

from typing import Iterable, Tuple, Union

import numpy
from scipy.signal import decimate, resample


def max_decimation_subsample(
    signal: Union[Iterable, numpy.ndarray],
    decimation_factor: int = 10,
    return_indices: bool = False,
    truncate_last_indivisible: bool = True,
) -> numpy.ndarray:
    """ """
    signal = numpy.array(signal)
    if truncate_last_indivisible:
        div = len(signal) // decimation_factor
    signal = signal[: div * decimation_factor]

    s = signal.reshape(-1, decimation_factor)
    if return_indices:
        a = numpy.argmax(s, axis=1)
        return numpy.array(
            [ao * decimation_factor + am for ao, am in zip(range(len(a)), a)]
        )
    return numpy.max(s, axis=1)


def min_decimation_subsample(
    signal: Union[Iterable, numpy.ndarray],
    decimation_factor: int = 10,
    return_indices: bool = False,
    truncate_last_indivisible: bool = True,
) -> numpy.ndarray:
    """ """
    signal = numpy.array(signal)
    if truncate_last_indivisible:
        div = len(signal) // decimation_factor
    signal = signal[: div * decimation_factor]

    s = signal.reshape(-1, decimation_factor)
    if return_indices:
        a = numpy.argmin(s, axis=1)
        return numpy.array(
            [ao * decimation_factor + am for ao, am in zip(range(len(a)), a)]
        )
    return numpy.min(s, axis=1)


def mag_decimation_subsample(
    signal: Union[Iterable, numpy.ndarray],
    decimation_factor: int = 10,
    return_indices: bool = False,
    truncate_last_indivisible: bool = True,
) -> numpy.ndarray:
    """

    truncate_last_undivisible is false, signal be divisible by the decimation_factor
    """
    signal = numpy.array(signal)
    if truncate_last_indivisible:
        div = len(signal) // decimation_factor
        signal = signal[: div * decimation_factor]

    s = signal.reshape(-1, decimation_factor)
    s_min = numpy.argmin(s, axis=1)
    s_mi = [ao * decimation_factor + am for ao, am in zip(range(len(s_min)), s_min)]
    s_max = numpy.argmax(s, axis=1)
    s_ma = [ao * decimation_factor + am for ao, am in zip(range(len(s_max)), s_max)]
    s_mag = [
        smax if (numpy.abs(signal[smin]) < numpy.abs(signal[smax])) else smin
        for smin, smax in zip(s_mi, s_ma)
    ]
    if return_indices:
        return numpy.array(s_mag)
    return signal[s_mag]


def grad_decimation_subsample() -> None:
    """
    :rtype: None
    """
    # Gradient based windowed subsampling
    raise NotImplemented()


def fir_subsample(
    signal: numpy.ndarray, max_resolution: int, sampling_rate: int
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """ """
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
    """ """
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

    def aisjd() -> None:
        """
        :rtype: None
        """
        t = numpy.linspace(-1, 1, 200)
        a = (
            numpy.sin(2 * numpy.pi * 0.75 * t * (1 - t) + 2.1)
            + 0.1 * numpy.sin(2 * numpy.pi * 1.25 * t + 1)
            + 0.18 * numpy.cos(2 * numpy.pi * 3.85 * t)
        )
        print(a)
        print(max_decimation_subsample(a, decimation_factor=10, return_indices=True))
        print(min_decimation_subsample(a, decimation_factor=10, return_indices=True))
        print(mag_decimation_subsample(a, decimation_factor=10, return_indices=True))

    aisjd()
