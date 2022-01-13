#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-01-2021
           """

from typing import Sequence

import numpy
from matplotlib import pyplot

__all__ = ["dissected_channel_plot", "overlay_channel_plot"]

from draugr.scipy_utilities.subsampling import (
    fft_subsample,
    fir_subsample,
)


def overlay_channel_plot(
    signal: numpy.ndarray,
    title: str = "Channels",
    channel_names: Sequence[str] = None,
    sampling_rate: int = 16000,
    line_width: float = 0.2,
    max_resolution: int = 20000,
    color_func=pyplot.cm.rainbow,
) -> None:
    """ """
    n_channels = len(signal)
    sub_time, sub_signal = fft_subsample(signal, max_resolution, sampling_rate)

    pyplot.figure(1)
    pyplot.suptitle(title)
    alpha = 1 / len(sub_signal)
    if channel_names:
        labels = channel_names
    else:
        labels = [f"C{i}" for i in range(n_channels)]

    colors = iter(color_func(numpy.linspace(0, 1, n_channels)))
    for i, channel in enumerate(sub_signal):
        pyplot.plot(
            sub_time,
            channel,
            alpha=alpha,
            label=f"{labels[i]}",
            color=next(colors),
            linewidth=line_width,
        )
    pyplot.legend()


def dissected_channel_plot(
    signal: numpy.ndarray,
    *,
    title: str = "Channels",
    channel_names: Sequence[str] = None,
    sampling_rate: int = 16000,
    line_width: float = 0.2,
    col_size=4,
    max_resolution: int = 20000,
    color_func=pyplot.cm.rainbow,
) -> None:
    """ """
    n_channels = len(signal)
    f, axs = pyplot.subplots(
        n_channels, 1, sharex="all", sharey="all", figsize=(n_channels, col_size)
    )
    if not n_channels > 1:
        axs = [axs]

    sub_time, sub_signal = fft_subsample(signal, max_resolution, sampling_rate)

    colors = iter(color_func(numpy.linspace(0, 1, n_channels)))
    pyplot.subplots_adjust(wspace=0.2, hspace=0.5)
    pyplot.suptitle(title)
    for i in range(len(sub_signal)):
        axs[i].plot(sub_time, sub_signal[i], color=next(colors), linewidth=line_width)
        if channel_names:
            axs[i].set_title(f"{channel_names[i]}")
        else:
            axs[i].set_title(f"C{i}")


def orthogonal_stereo_channel_3d_plot(
    signal: numpy.ndarray,
    *,
    max_resolution: int = 20000,
    sampling_rate: int = 16000,
) -> None:
    """ """
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")

    sub_time, sub_signal = fir_subsample(signal, max_resolution, sampling_rate)

    ax.plot(sub_time, sub_signal[0])
    ax.plot(sub_time, numpy.zeros(len(sub_time)), zs=sub_signal[1])


def deinterleaved_channel_plot_file(wav_file):
    """ """
    import wave

    with wave.open(wav_file, "r") as wav_file:
        signal = numpy.fromstring(wav_file.readframes(-1), "Int16")

        length_signal = len(signal)
        num_channels = wav_file.getnchannels()

        channels = [
            [] for channel in range(num_channels)
        ]  # Split the data into channels
        for index, datum in enumerate(signal):
            channels[index % num_channels].append(datum)

        deinterleaved_length = length_signal // num_channels
        t = numpy.linspace(
            0,
            deinterleaved_length // wav_file.getframerate(),  # Get time from indices
            num=deinterleaved_length,
        )

        pyplot.figure(1)
        pyplot.title("Signal")
        alpha = 1 / num_channels
        for channel in channels:
            pyplot.plot(t, channel, alpha=alpha)


if __name__ == "__main__":

    def iushjaqdfu() -> None:
        """
        :rtype: None
        """
        sr = 1000
        max_res = sr * 4
        t = numpy.arange(sr * 4) / sr
        # noise = numpy.random.rand(sr * 2) * 0.001
        signal = numpy.sin(200 * 2 * numpy.pi * t)  # + noise
        dissected_channel_plot(signal, sampling_rate=sr, max_resolution=max_res)
        pyplot.show()
        overlay_channel_plot(signal, sampling_rate=sr, max_resolution=max_res)
        pyplot.show()
        orthogonal_stereo_channel_3d_plot(
            signal, sampling_rate=sr, max_resolution=max_res
        )
        pyplot.show()

    def decimate_stest() -> None:
        """
        :rtype: None
        """
        sr = 1000
        max_res = sr * 4
        t = numpy.arange(sr * 4) / sr
        # noise = numpy.random.rand(sr * 2) * 0.001
        signal = numpy.sin(200 * 2 * numpy.pi * t)  # + noise
        sub_time, sub_signal = fir_subsample(signal, max_res, signal)

        print(signal.shape, sub_signal.shape)
        print(signal[:10])
        print(sub_signal[:10])
        assert numpy.equal(signal, sub_signal).all()

    def iushjaqsfaddfu() -> None:
        """
        :rtype: None
        """
        sr = 1000
        max_res = sr * 4
        t = numpy.arange(sr * 4) / sr
        # noise = numpy.random.rand(sr * 2) * 0.001
        signal = numpy.sin(200 * 2 * numpy.pi * t)  # + noise
        deinterleaved_channel_plot_file(signal)
        pyplot.show()

    # iushjaqdfu()
    iushjaqsfaddfu()
    # decimate_stest()
