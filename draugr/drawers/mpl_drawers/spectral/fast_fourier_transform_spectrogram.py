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

__all__ = ["FastFourierTransformSpectrogramPlot"]

from matplotlib import pyplot

from draugr.tqdm_utilities import progress_bar

FLOAT_EPS = numpy.finfo(float).eps


class FastFourierTransformSpectrogramPlot(MplDrawer):
    """
    TODO: CENTER Align fft maybe, to mimick librosa stft
    Short Time Fourier Transform (STFT), with step size of 1 and window lenght of n_fft, and no window function ( TODO: Hanning Smoothing)"""

    def __init__(
        self,
        n_fft: int = 64,
        sampling_rate=int(1.0 / 0.0005),
        buffer_size_sec: float = 1.0,
        title: str = "",
        vertical: bool = True,
        reverse: bool = False,
        figure_size=(9, 9),
        cmap="viridis",
        render: bool = True,
        **kwargs
    ):
        """

        :param n_fft:
        :type n_fft:
        :param sampling_rate:
        :type sampling_rate:
        :param buffer_size_sec:
        :type buffer_size_sec:
        :param title:
        :type title:
        :param vertical:
        :type vertical:
        :param reverse:
        :type reverse:
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
        self.n_positive_fft = (self.n_fft + 1) // 2
        self.sampling_rate = sampling_rate
        if buffer_size_sec is not None:
            self.buffer_size_sec = buffer_size_sec
        else:
            self.buffer_size_sec = self.n_fft / sampling_rate * 2

        # self.window_function = window_function(self.n_fft)
        self.buffer_array_size = int(sampling_rate * self.buffer_size_sec)
        assert self.buffer_array_size >= self.n_fft
        time_s = numpy.linspace(
            0, self.buffer_size_sec, self.buffer_array_size, endpoint=False
        )
        raw_array = numpy.zeros(self.buffer_array_size)
        zeroes_img = numpy.zeros((self.n_positive_fft, self.buffer_array_size - n_fft))
        self.zeroes_padding = numpy.zeros((self.n_positive_fft, n_fft))

        gs = GridSpec(3, 2, width_ratios=[100, 2])
        (
            self.raw_ax,
            _,
            self.angle_ax,
            self.angle_cbar_ax,
            self.spec_ax,
            self.spec_cbar_ax,
        ) = [pyplot.subplot(gs[i]) for i in range(6)]

        (self.raw_line2d,) = self.raw_ax.plot(time_s, raw_array)
        self.raw_ax.set_xlim([time_s[0], time_s[-1]])
        self.raw_ax.set_ylabel("Signal [Magnitude]")

        max_freq = numpy.max(numpy.fft.fftfreq(self.n_fft, 1 / sampling_rate))
        self.dft_angle_img = self.angle_ax.imshow(
            zeroes_img,
            vmin=-math.pi,
            vmax=math.pi,
            interpolation="hanning",
            aspect="auto",
            extent=[time_s[0], time_s[-1], max_freq, 0],
            cmap=cmap,
        )
        self.angle_ax.set_ylabel("Phase [Hz]")
        _ = self.fig.colorbar(self.dft_angle_img, cax=self.angle_cbar_ax)
        self.angle_cbar_ax.set_ylabel("Angle (Radians)", rotation=90)

        self.dft_mag_img = self.spec_ax.imshow(
            zeroes_img,
            vmin=0,
            vmax=1,
            interpolation="hanning",
            aspect="auto",
            extent=[time_s[0], time_s[-1], max_freq, 0],
            cmap=cmap,
        )
        self.spec_ax.set_ylabel("Frequency [Hz]")
        self.spec_ax.set_xlabel("Time [Sec]")
        _ = self.fig.colorbar(self.dft_mag_img, cax=self.spec_cbar_ax)
        self.spec_cbar_ax.set_ylabel("Magnitude (dB)", rotation=90)

        self.vertical = vertical
        self.reverse = reverse

        pyplot.xlim(time_s[0], time_s[-1])

        pyplot.title(title)
        pyplot.tight_layout()

    def _draw(self, signal_sample: float, delta: float = 1 / 120) -> None:
        """

        :param signal_sample:
        :param delta: 1 / 60 for 60fps
        :return:"""
        y_data = self.raw_line2d.get_ydata()

        if not self.reverse:
            y_data = numpy.hstack((y_data[1:], signal_sample))
            y_data_view = y_data[-self.n_fft :]
        else:
            y_data = numpy.hstack((signal_sample, y_data[:-1]))
            y_data_view = y_data[: self.n_fft]

        self.raw_line2d.set_ydata(y_data)
        cur_lim = self.raw_ax.get_ylim()
        self.raw_ax.set_ylim(
            [min(cur_lim[0], signal_sample), max(cur_lim[1], signal_sample)]
        )

        # if self.window_function is not None:
        #   y_data_view *= self.window_function

        f_coef = numpy.fft.fft(y_data_view, n=self.n_fft)[
            : self.n_positive_fft
        ].reshape(
            -1, 1
        )  # Only select the positive frequencies

        phase_data = self.dft_angle_img.get_array()
        new_phase = numpy.angle(f_coef)
        # new_phase = f_coef.imag

        if not self.reverse:
            phase_data = numpy.concatenate(
                (phase_data[:, 1 : -self.n_fft], new_phase, self.zeroes_padding),
                axis=-1,
            )
        else:
            phase_data = numpy.concatenate(
                (self.zeroes_padding, new_phase, phase_data[:, self.n_fft : -1]),
                axis=-1,
            )

        self.dft_angle_img.set_array(phase_data)

        magnitude_data = self.dft_mag_img.get_array()
        new_mag = 10 * numpy.log10(
            (
                numpy.abs(f_coef)
                # f_coef.real
                ** 2
            )
            + FLOAT_EPS
        )  # db
        if not self.reverse:
            magnitude_data = numpy.concatenate(
                (magnitude_data[:, 1 : -self.n_fft], new_mag, self.zeroes_padding),
                axis=-1,
            )
        else:
            magnitude_data = numpy.concatenate(
                (self.zeroes_padding, new_mag, magnitude_data[:, self.n_fft : -1]),
                axis=-1,
            )

        self.dft_mag_img.set_clim(
            vmin=numpy.min(magnitude_data), vmax=numpy.max(magnitude_data)
        )
        self.dft_mag_img.set_array(magnitude_data)
        # self.spec_cbar_ax.set_ylabel("Magnitude (Linear)", rotation=90)
        self.spec_cbar_ax.set_ylabel("Magnitude (dB)", rotation=90)


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        duration_sec = 4
        mul = 1000
        sampling_Hz = 44.1
        sampling_rate = int(sampling_Hz * mul)  # Hz
        delta = 1 / sampling_rate
        n_fft = 128  # 1024
        s = FastFourierTransformSpectrogramPlot(
            n_fft=n_fft, sampling_rate=sampling_rate, buffer_size_sec=delta * n_fft * 4
        )
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
