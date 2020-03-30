#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import matplotlib.pyplot
import numpy


import matplotlib


from draugr.drawers.drawer import Drawer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["SpectrumPlot"]

from matplotlib import pyplot, mlab


class SpectrumPlot(Drawer):
    """
Waterfall plot

"""

    def __init__(
        self,
        x_w_l: int = 1024,  # the length of the windowing segments
        sampling_frequency=int(1.0 / 0.0005),  # the sampling frequency
        window_length=20,
        noverlap: int = 900,
        title: str = "",
        time_label: str = "Time [sec]",
        data_label: str = "Frequency [Hz]",
        vertical: bool = True,
        reverse: bool = True,
        overwrite: bool = False,
        placement: Tuple = (0, 0),
        render: bool = True,
    ):
        self.fig = None
        if not render:
            return

        if not window_length:
            window_length = x_w_l

        array = numpy.zeros(window_length)

        self.vertical = vertical
        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length
        self.n = 0

        self.fig = pyplot.figure(figsize=(2, window_length / 10))
        self.x_w_l = x_w_l
        self.sampling_frequency = sampling_frequency
        self.noverlap = noverlap
        self.xextent = [0, -window_length]
        self.yextent = [0, 2 * numpy.sqrt(2)]

        fig, (self.raw_ax, self.spec_ax) = pyplot.subplots(nrows=2)
        self.raw_i, = self.raw_ax.plot(numpy.arange(0, len(array)), array)
        # self.raw_ax.set_xlim(self.xextent)
        # self.raw_ax.set_ylim(self.yextent)

        self.draw_spec_fig(array)

        self.placement = placement

        if vertical:
            pyplot.xlabel(time_label)
            pyplot.ylabel(data_label)
        else:
            pyplot.xlabel(data_label)
            pyplot.ylabel(time_label)

        pyplot.title(title)
        pyplot.tight_layout()

    def draw_spec_fig(self, array):
        """
      # The `specgram` method returns 4 objects. They are:
      # - Pxx: the periodogram
      # - freqs: the frequency vector
      # - bins: the centers of the time bins
      # - im: the matplotlib.image.AxesImage instance representing the data in the plot

    :param array:
    :type array:
    :param x_w_l:
    :type x_w_l:
    :param sampling_frequency:
    :type sampling_frequency:
    :param noverlap:
    :type noverlap:
    :param extent:
    :type extent:
    :return:
    :rtype:
    """
        self.spec_ax.cla()
        """
    spec, freqs, t = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
                                   detrend=detrend, window=window,
                                   noverlap=noverlap, pad_to=pad_to,
                                   sides=sides,
                                   scale_by_freq=scale_by_freq,
                                   mode=mode)

    if scale == 'linear':
        Z = spec
    elif scale == 'dB':
        if mode is None or mode == 'default' or mode == 'psd':
            Z = 10. * np.log10(spec)
        else:
            Z = 20. * np.log10(spec)
    else:
        raise ValueError('Unknown scale %s', scale)

    Z = np.flipud(Z)

    if xextent is None:
        # padding is needed for first and last segment:
        pad_xextent = (NFFT-noverlap) / Fs / 2
        xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    im = self.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax,
                     **kwargs)
    self.axis('auto')
    """

        Pxx, freqs, bins, im = self.spec_ax.specgram(
            array,
            NFFT=self.x_w_l,
            Fs=self.sampling_frequency,
            noverlap=self.noverlap,
            xextent=self.xextent,
        )

    @staticmethod
    def move_figure(figure: pyplot.Figure, x: int = 0, y: int = 0):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if hasattr(figure.canvas.manager, "window"):
            window = figure.canvas.manager.window
            if backend == "TkAgg":
                window.wm_geometry("+%d+%d" % (x, y))
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

    def draw(self, signal_sample: float, delta: float = 1 / 120):
        """

:param signal_sample:
:param delta: 1 / 60 for 60fps
:return:
"""

        array = self.raw_i.get_ydata()

        if not self.overwrite:
            if not self.reverse:
                striped = numpy.delete(array, 0)
                array = numpy.hstack((striped, signal_sample))
            else:
                striped = numpy.delete(array, -1)
                array = numpy.hstack((signal_sample, striped))
        else:
            array[self.n % self.window_length] = signal_sample

        self.raw_i.set_ydata(array)
        cur_lim = self.raw_ax.get_ylim()
        self.raw_ax.set_ylim(
            [min(cur_lim[0], signal_sample), max(cur_lim[1], signal_sample)]
        )
        self.draw_spec_fig(array)

        pyplot.draw()
        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1
        if delta:
            pyplot.pause(delta)


if __name__ == "__main__":

    def a():

        delta = 0.0005
        sample_freq = int(1.0 / delta)
        s = SpectrumPlot(
            sampling_frequency=sample_freq, window_length=sample_freq // 100
        )
        for t in numpy.arange(0, 100, delta):
            s1 = numpy.sin(2 * numpy.pi * 100 * t)
            s2 = 2 * numpy.sin(2 * numpy.pi * 400 * t)

            signal = s1 + s2
            noise = 0.01 * numpy.random.random()
            s.draw(signal + noise)

    def b():
        from scipy import signal

        fs = 10e3
        N = 1e5
        amp = 2 * numpy.sqrt(2)
        noise_power = 0.01 * fs / 2
        time = numpy.arange(N) / float(fs)
        mod = 500 * numpy.cos(2 * numpy.pi * 0.25 * time)
        carrier = amp * numpy.sin(2 * numpy.pi * 3e3 * time + mod)
        noise = numpy.random.normal(scale=numpy.sqrt(noise_power), size=time.shape)
        noise *= numpy.exp(-time / 5)
        x = carrier + noise
        f, t, Sxx = signal.spectrogram(x, fs)
        pyplot.pcolormesh(t, f, Sxx)
        pyplot.ylabel("Frequency [Hz]")
        pyplot.xlabel("Time [sec]")
        pyplot.show()

    a()
    # b()
