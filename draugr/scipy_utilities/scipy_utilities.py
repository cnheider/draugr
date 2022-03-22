#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-02-2021
           """

from pathlib import Path
from typing import Tuple, Union

import numpy
from scipy.io import wavfile

__all__ = ["read_normalised_wave", "write_normalised_wave"]


def read_normalised_wave(wav_file_name: Union[str, Path]) -> Tuple[int, numpy.ndarray]:
    """
    [-1..1] normalised
    """
    sampling_rate, signal = wavfile.read(str(wav_file_name))
    if signal.dtype == numpy.int16:
        num_bits = 16 - 1  # -> 16-bit wav files, -1 for sign
    elif signal.dtype == numpy.int32:
        num_bits = 32 - 1  # -> 32-bit wav files, -1 for sign
    elif signal.dtype == numpy.uint8:
        num_bits = 8
    elif signal.dtype == numpy.float32:
        return sampling_rate, signal
        # num_bits = 0
    else:
        raise NotImplementedError(f"{signal.dtype} is not supported")
    return (
        sampling_rate,
        (signal / (2**num_bits)).astype(numpy.float),
    )  # normalise by max possible val of dtype


def write_normalised_wave(
    wav_file_name: Union[str, Path],
    sampling_rate: int,
    signal: numpy.ndarray,
    dtype=numpy.float32,
) -> None:
    """
    [-1..1] normalised
    """
    assert signal.dtype == numpy.float

    if dtype == numpy.int16:
        num_bits = 16 - 1  # -> 16-bit wav files, -1 for sign
    elif dtype == numpy.int32:
        num_bits = 32 - 1  # -> 32-bit wav files, -1 for sign
    elif dtype == numpy.uint8:
        num_bits = 8
    elif dtype == numpy.float32:
        # num_bits = 0
        wavfile.write(wav_file_name, sampling_rate, signal)
        return
    else:
        raise NotImplementedError(f"{signal.dtype} is not supported")
    wavfile.write(
        str(wav_file_name), sampling_rate, signal * (2**num_bits)
    )  # unnormalise by max possible val of dtype
