#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Tuple

import matplotlib
import matplotlib.pyplot
import numpy
from matplotlib.gridspec import GridSpec

from draugr.drawers.drawer import Drawer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["FastFourierTransformPlot"]

from matplotlib import pyplot


class FastFourierTransformPlot(Drawer):
    """
    Plots last computed fft of data
"""

    def __init__(
        self,
        n_fft: int = 64,
        sampling_rate=int(1.0 / 0.0005),
        title: str = "",
        placement: Tuple = (0, 0),
        fig_size=(9, 9),
        render: bool = True,
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
    :type render:
    """
        self.fig = None
        if not render:
            return

        self.n_fft = n_fft
        self.abs_n_fft = (self.n_fft + 1) // 2
        self.sampling_rate = sampling_rate

        freq_bins = numpy.arange(self.n_fft)
        raw_array = numpy.zeros(self.n_fft, dtype="complex")

        self.zeroes_padding = numpy.zeros((self.abs_n_fft, n_fft))

        self.fig = pyplot.figure(figsize=fig_size)
        gs = GridSpec(3, 1)
        (self.raw_ax, self.angle_ax, self.mag_ax) = [
            pyplot.subplot(gs[i]) for i in range(3)
        ]

        freqs = numpy.fft.fftfreq(self.n_fft, 1 / sampling_rate)

        self.dft_raw_img, = self.raw_ax.plot(freq_bins, raw_array)
        self.raw_ax.set_xlabel("Time (Sec)")
        self.raw_ax.set_ylabel("Amplitude")

        self.dft_angle_img, = self.angle_ax.plot(freq_bins, raw_array)
        self.angle_ax.set_xlabel("Phase [Hz]")
        self.angle_ax.set_ylabel("Angle (Radians)")
        self.angle_ax.set_ylim([-math.pi, math.pi])
        # self.angle_ax.set_xticks(freqs)

        self.dft_mag_img, = self.mag_ax.plot(freq_bins, raw_array)
        self.mag_ax.set_xlabel("Frequency [Hz]")
        self.mag_ax.set_xlabel("Magnitude (Linear)")
        # self.mag_ax.set_xticks(freqs)

        self.placement = placement
        self.n = 0

        pyplot.title(title)
        pyplot.tight_layout()

    @staticmethod
    def move_figure(figure: pyplot.Figure, x: int = 0, y: int = 0) -> None:
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if hasattr(figure.canvas.manager, "window"):
            window = figure.canvas.manager.window
            if backend == "TkAgg":
                window.wm_geometry(f"+{x:d}+{y:d}")
            elif backend == "WXAgg":
                window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                window.move(x, y)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fig:
            pyplot.close(self.fig)

    def draw(self, signal_sample: float, delta: float = 1 / 120) -> None:
        """

:param signal_sample:
:param delta: 1 / 60 for 60fps
:return:
"""
        raw_array = self.dft_raw_img.get_ydata()
        raw_array = numpy.hstack((signal_sample, raw_array[:-1]))
        self.dft_raw_img.set_ydata(raw_array)
        cur_lim = self.raw_ax.get_ylim()
        self.raw_ax.set_ylim(
            [min(cur_lim[0], signal_sample), max(cur_lim[1], signal_sample)]
        )

        f_coef = numpy.fft.fft(raw_array, n=self.n_fft)

        # f_coef = f_coef[: self.abs_n_fft]
        # print(f_coef)

        self.dft_angle_img.set_ydata(numpy.angle(f_coef))

        mag = numpy.abs(f_coef) ** 2
        self.dft_mag_img.set_ydata(mag)
        self.mag_ax.set_ylim([min(mag), max(mag)])

        pyplot.draw()
        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1
        if delta:
            pyplot.pause(delta)


if __name__ == "__main__":

    def a():
        duration_sec = 4
        mul = 1000
        sampling_Hz = 44
        sampling_rate = sampling_Hz * mul  # Hz
        delta = 1 / sampling_rate
        n_fft = 64
        s = FastFourierTransformPlot(n_fft=n_fft, sampling_rate=sampling_rate)
        for t in numpy.arange(0, duration_sec, delta):
            ts = 2 * numpy.pi * t
            s1 = numpy.sin(ts * 1 * sampling_Hz / 2 ** 4 * mul)
            s2 = numpy.sin(ts * 3 * sampling_Hz / 2 ** 3 * mul + 0.33 * numpy.pi)
            s3 = numpy.sin(ts * 5 * sampling_Hz / 2 ** 2 * mul + 0.66 * numpy.pi)
            signal = s1 + s2 + s3
            signal /= 3
            # signal += (numpy.random.random() - 0.5) * 2 * 1 / 2  # Noise
            s.draw(signal, delta=delta)

    a()
