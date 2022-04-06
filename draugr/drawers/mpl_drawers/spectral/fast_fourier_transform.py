#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy
from matplotlib.gridspec import GridSpec

from draugr.drawers.mpl_drawers.mpldrawer import MplDrawer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["FastFourierTransformPlot"]

from matplotlib import pyplot

from draugr.tqdm_utilities import progress_bar

FLOAT_EPS = numpy.finfo(float).eps


class FastFourierTransformPlot(MplDrawer):
    """
    Plots last computed fft of data"""

    def __init__(
        self,
        n_fft: int = 64,
        sampling_rate=int(1.0 / 0.0005),
        title: str = "",
        figure_size=(9, 9),
        render: bool = True,
        **kwargs
    ):
        """

        :param n_fft:
        :type n_fft:
        :param sampling_rate:
        :type sampling_rate:
        :param title:
        :type title:
        :param placement:
        :type placement:
        :param fig_size:
        :type fig_size:
        :param render:
        :type render:"""
        super().__init__(render=render, figure_size=figure_size, **kwargs)

        if not render:
            return

        self.fig = pyplot.figure(figsize=figure_size)

        self.n_fft = n_fft
        self.abs_n_fft = (self.n_fft + 1) // 2
        self.sampling_rate = sampling_rate

        freq_bins = numpy.arange(self.n_fft)
        raw_array = numpy.zeros(self.n_fft)

        self.zeroes_padding = numpy.zeros((self.abs_n_fft, n_fft))

        gs = GridSpec(3, 1)
        (self.raw_ax, self.angle_ax, self.mag_ax) = [
            pyplot.subplot(gs[i]) for i in range(3)
        ]

        freqs = numpy.fft.fftfreq(self.n_fft, 1 / sampling_rate)

        (self.dft_raw_img,) = self.raw_ax.plot(freq_bins, raw_array)
        self.raw_ax.set_xlabel("Time (Sec)")
        self.raw_ax.set_ylabel("Amplitude")

        (self.dft_angle_img,) = self.angle_ax.plot(freq_bins, raw_array)
        self.angle_ax.set_xlabel("Phase [Hz]")
        self.angle_ax.set_ylabel("Angle (Radians)")
        self.angle_ax.set_ylim([-math.pi, math.pi])
        # self.angle_ax.set_xticks(freqs)

        (self.dft_mag_img,) = self.mag_ax.plot(freq_bins, raw_array)
        self.mag_ax.set_xlabel("Frequency [Hz]")
        self.mag_ax.set_xlabel("Magnitude (dB)")
        # self.mag_ax.set_xticks(freqs)

        pyplot.title(title)
        pyplot.tight_layout()

    def _draw(self, signal_sample: float, delta: float = 1 / 120) -> None:
        """

        :param signal_sample:
        :param delta: 1 / 60 for 60fps
        :return:"""
        raw_array = self.dft_raw_img.get_ydata()
        raw_array = numpy.hstack((signal_sample, raw_array[:-1]))
        self.dft_raw_img.set_ydata(raw_array)
        cur_lim = self.raw_ax.get_ylim()
        self.raw_ax.set_ylim(
            [min(cur_lim[0], signal_sample), max(cur_lim[1], signal_sample)]
        )

        f_coef = numpy.fft.fft(raw_array, n=self.n_fft)

        self.dft_angle_img.set_ydata(numpy.angle(f_coef))

        mag = 10 * numpy.log10((numpy.abs(f_coef) ** 2) + FLOAT_EPS)

        self.dft_mag_img.set_ydata(mag)
        self.mag_ax.set_ylim([min(mag), max(mag)])


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        duration_sec = 4
        mul = 1000
        sampling_Hz = 44
        sampling_rate = sampling_Hz * mul  # Hz
        delta = 1 / sampling_rate
        n_fft = 64
        s = FastFourierTransformPlot(n_fft=n_fft, sampling_rate=sampling_rate)
        for t in progress_bar(numpy.arange(0, duration_sec, delta)):
            ts = 2 * numpy.pi * t
            s1 = numpy.sin(ts * 1 * sampling_Hz / 2**4 * mul)
            s2 = numpy.sin(ts * 3 * sampling_Hz / 2**3 * mul + 0.33 * numpy.pi)
            s3 = numpy.sin(ts * 5 * sampling_Hz / 2**2 * mul + 0.66 * numpy.pi)
            signal = s1 + s2 + s3
            signal /= 3
            # signal += (numpy.random.random() - 0.5) * 2 * 1 / 2  # Noise
            s.draw(signal, delta=delta)

    a()
