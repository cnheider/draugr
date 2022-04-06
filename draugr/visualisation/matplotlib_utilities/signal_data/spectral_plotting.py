#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 06-01-2021
           """

from typing import Sequence

import numpy
from matplotlib import pyplot
from scipy.signal import spectrogram, welch

from warg import Number, drop_unused_kws, passes_kws_to, next_pow_2

__all__ = ["spectral_plot", "ltas_plot", "spectrum_plot", "fft_plot"]


def spectral_plot(
    time: numpy.ndarray,
    frequencies: numpy.ndarray,
    fxt: numpy.ndarray,
    fig_size: Sequence = (4.8, 2.4),
) -> pyplot.Figure:
    """
    return new figure"""
    assert fxt.shape == (*frequencies.shape, *time.shape)
    f, ax = pyplot.subplots(figsize=fig_size)
    ax.pcolormesh(time, frequencies / 1000, 10 * numpy.log10(fxt), cmap="viridis")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_xlabel("Time [s]")
    return f


@drop_unused_kws
@passes_kws_to(pyplot.semilogy)
def ltas_plot(
    signal: Sequence, sampling_rate: int, label: str = "#", **kwargs
) -> pyplot.Figure:
    """
    Long-term average spectrum

    :param signal:
    :param sampling_rate:
    :param label:
    :return:
    """
    n_per_seg = next_pow_2(
        sampling_rate * (20 / 1000)
    )  # 20 ms, next_pow_2 per seg == n_fft

    f, spectrum = welch(
        signal, sampling_rate, window="hanning", nperseg=n_per_seg, scaling="spectrum"
    )  # Average Power spectrum of signal.
    pyplot.semilogy(f, numpy.sqrt(spectrum), label=label, **kwargs)
    pyplot.xlabel("frequency [Hz]")
    pyplot.ylabel("Amplitude")
    return pyplot.gcf()


def spectrum_plot(
    signal: Sequence, sampling_rate: int, window_length_ms: Number = (20 / 1000)
) -> pyplot.Figure:
    """

    :param signal:
    :param sampling_rate:
    :param window_length_ms:
    :return:
    """
    n_per_seg = next_pow_2(
        sampling_rate * window_length_ms
    )  # 20 ms, next_pow_2 per seg == n_fft
    f, t, fxt = spectrogram(
        signal,
        fs=sampling_rate,
        window="hanning",
        nperseg=n_per_seg,
        scaling="spectrum",
    )

    return spectral_plot(t, f, fxt)


def fft_plot(signal: Sequence, *, line_width: float = 0.2) -> pyplot.Figure:
    """ """
    n_per_seg = next_pow_2(len(signal))
    spectrum = numpy.fft.fft(signal, n_per_seg)[: n_per_seg // 2]
    frequencies = numpy.fft.fftfreq(n_per_seg)[: n_per_seg // 2]
    pyplot.plot(frequencies, spectrum.real, label="Amplitude", linewidth=line_width)
    pyplot.plot(frequencies, spectrum.imag, label="Phase", linewidth=line_width)
    pyplot.xlabel("Frequency [kHz]")
    pyplot.legend()
    return pyplot.gcf()


if __name__ == "__main__":

    def main() -> None:
        """
        :rtype: None
        """
        sr = 1000
        t = numpy.arange(sr * 4) / sr
        # noise = numpy.random.rand(sr * 2) * 0.001
        signal = numpy.sin(200 * 2 * numpy.pi * t)  # + noise

        fft_plot(signal[:256])
        pyplot.show()

        spectrum_plot(signal, sr)
        pyplot.show()

        ltas_plot(signal, sr)
        pyplot.show()

    main()
