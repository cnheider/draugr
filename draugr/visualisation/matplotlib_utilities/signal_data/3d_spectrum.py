# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 06-01-2021
           """

from typing import Sequence

import mpl_toolkits.mplot3d.axes3d as p3
import numpy
from matplotlib import animation, cm, pyplot
from mpl_toolkits.mplot3d import axes3d
from scipy.signal import chirp, spectrogram

from warg import next_pow_2

__all__ = ["spectral_plot3d", "spectrum_plot3d"]


# TODO: ANIMATED VARIANT, maybe as a drawer!


def spectral_plot3d(
    time: numpy.ndarray, frequencies: numpy.ndarray, fxt: numpy.ndarray
) -> pyplot.Figure:
    """
    return new figure"""
    assert fxt.shape == (*frequencies.shape, *time.shape)
    assert fxt.dtype == numpy.complex

    fig = pyplot.figure()
    ax = p3.Axes3D(fig)

    x, y = numpy.meshgrid(time, frequencies)

    # colors=cm.jet(norm(colorfunction))
    colors = numpy.empty(x.shape, dtype=numpy.float)
    z = numpy.empty(x.shape, dtype=numpy.float)

    for y_i in range(len(time)):
        for x_i in range(len(frequencies)):
            com = fxt[x_i, y_i]
            z[x_i, y_i] = com.real
            colors[x_i, y_i] = com.imag * 0.5 + 0.5

    colors = colors / colors.max()

    surf = ax.plot_surface(
        x,
        y,
        z,
        facecolors=cm.jet(colors),
        # linewidth=0
        # color='0.75',
        # rstride=1,
        # cstride=1
        # rcount=50
        # #ccount=50
    )

    ax.set_ylabel("Frequency [kHz]")
    ax.set_xlabel("Time [s]")
    ax.set_zlabel("Magnitude")
    return fig


def spectrum_plot3d(
    signal: Sequence, sampling_rate: int, window_length_ms=(20 / 1000)
) -> pyplot.Figure:
    """ """
    n_per_seg = next_pow_2(
        sampling_rate * window_length_ms
    )  # 20 ms, next_pow_2 per seg == n_fft
    f, t, fxt = spectrogram(
        signal,
        fs=sampling_rate,
        window="hanning",
        nperseg=n_per_seg,
        scaling="spectrum",
        mode="complex",
    )

    return spectral_plot3d(t, f, fxt)


if __name__ == "__main__":

    def asdijaisd() -> None:
        """
        :rtype: None
        """
        sr = 1000
        t = numpy.arange(sr * 4) / sr
        # noise = numpy.random.rand(sr * 2) * 0.001
        w = chirp(t, f0=100, f1=500, t1=4, method="linear")
        signal = numpy.sin(200 * 2 * numpy.pi * t) + w  # + noise

        spectrum_plot3d(signal, sr)
        pyplot.show()

    def aisjd() -> None:
        """
        :rtype: None
        """
        fig = pyplot.figure()
        ax = axes3d.Axes3D(fig)

        def gen(n):
            """ """
            phi = 0
            while phi < 2 * numpy.pi:
                yield numpy.array([numpy.cos(phi), numpy.sin(phi), phi])
                phi += 2 * numpy.pi / n

        def update(num, data, line):
            """ """
            line.set_data(data[:2, :num])
            line.set_3d_properties(data[2, :num])

        N = 100
        data = numpy.array(list(gen(N))).T
        (line,) = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

        # Setting the axes properties
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_xlabel("X")

        ax.set_ylim3d([-1.0, 1.0])
        ax.set_ylabel("Y")

        ax.set_zlim3d([0.0, 10.0])
        ax.set_zlabel("Z")

        ani = animation.FuncAnimation(
            fig, update, N, fargs=(data, line), interval=10000 / N, blit=False
        )
        # ani.save('matplot003.gif', writer='imagemagick')
        pyplot.show()

    # aisjd()
    asdijaisd()
