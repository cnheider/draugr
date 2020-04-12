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

__all__ = ["FastFourierTransformSpectrogramPlot"]

from matplotlib import pyplot


class FastFourierTransformSpectrogramPlot(Drawer):
    """
TODO: CENTER Align fft maybe, to mimick librosa stft

"""

    def __init__(
        self,
        n_fft: int = 64,
        sampling_rate=int(1.0 / 0.0005),
        buffer_size_sec: float = 1.0,
        title: str = "",
        vertical: bool = True,
        reverse: bool = False,
        placement: Tuple = (0, 0),
        fig_size=(9, 9),
        cmap="viridis",
        # window_function:callable=numpy.hanning, # NOT supported yet because fft is not calculated from center.
        # log_scale:bool = True,
        render: bool = True,
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
    :type render:
    """
        self.fig = None
        if not render:
            return

        self.n_fft = n_fft
        self.abs_n_fft = (self.n_fft + 1) // 2
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
        raw_array = numpy.zeros(self.buffer_array_size, dtype="complex")
        zeroes_img = numpy.zeros((self.abs_n_fft, self.buffer_array_size - n_fft))
        self.zeroes_padding = numpy.zeros((self.abs_n_fft, n_fft))

        self.fig = pyplot.figure(figsize=fig_size)
        gs = GridSpec(3, 2, width_ratios=[100, 2])
        (
            self.raw_ax,
            _,
            self.angle_ax,
            self.angle_cbar_ax,
            self.spec_ax,
            self.spec_cbar_ax,
        ) = [pyplot.subplot(gs[i]) for i in range(6)]

        self.raw_line2d, = self.raw_ax.plot(time_s, raw_array)
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
        self.placement = placement
        self.n = 0

        pyplot.xlim(time_s[0], time_s[-1])

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

        f_coef = numpy.fft.fft(y_data_view, n=self.n_fft)[: self.abs_n_fft].reshape(
            -1, 1
        )  # Only select the positive frequencies

        phase_data = self.dft_angle_img.get_array()
        # sin_, cos = f_coef.imag,f_coef.real
        new_phase = numpy.angle(f_coef)

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
        new_mag_lin = numpy.abs(f_coef) ** 2
        new_mag_db = 10 * numpy.log10(new_mag_lin + numpy.finfo(float).eps)
        new_mag = new_mag_db
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
        sampling_Hz = 44.1
        sampling_rate = int(sampling_Hz * mul)  # Hz
        delta = 1 / sampling_rate
        n_fft = 128  # 1024
        s = FastFourierTransformSpectrogramPlot(
            n_fft=n_fft, sampling_rate=sampling_rate, buffer_size_sec=delta * n_fft * 4
        )
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
