#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 06/04/2020
           """

from pathlib import Path

import numpy
from matplotlib import pyplot
from scipy.io import wavfile

from draugr.drawers.spectral.fast_fourier_transform import FastFourierTransformPlot
from draugr.drawers.spectral.fast_fourier_transform_spectrogram import (
    FastFourierTransformSpectrogramPlot,
)

if __name__ == "__main__":

    def main():
        sampling_rate, audio = wavfile.read(
            str(
                Path.home()
                / "Data"
                / "Audio"
                / "/home/heider/Data/Audio/Nightingale-sound.wav"
            )
        )
        audio = numpy.mean(audio, axis=1)
        num_samples = audio.shape[0]
        length_sec = num_samples / sampling_rate

        print(f"Audio length: {length_sec:.2f} seconds")

        def a():
            n_fft = 32
            s = FastFourierTransformPlot(n_fft=n_fft, sampling_rate=sampling_rate)
            for sample in audio:
                s.draw(sample)

        def b():
            n_fft = 32
            delta = 1 / sampling_rate
            s = FastFourierTransformSpectrogramPlot(
                n_fft=n_fft,
                sampling_rate=sampling_rate,
                buffer_size_sec=delta * n_fft * 4,
            )
            for sample in audio:
                s.draw(sample)

        def c():
            f, ax = pyplot.subplots()
            ax.plot(numpy.arange(num_samples) / sampling_rate, audio)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Amplitude [unknown]")
            pyplot.show()

        def d():
            from skimage import util

            M = 1024

            slices = util.view_as_windows(audio, window_shape=(M,), step=100)
            print(f"Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}")

            win = numpy.hanning(M + 1)[:-1]
            slices = slices * win

            slices = slices.T
            print("Shape of `slices`:", slices.shape)

            spectrum = numpy.fft.fft(slices, axis=0)[: M // 2 + 1 : -1]
            spectrum = numpy.abs(spectrum)

            f, ax = pyplot.subplots(figsize=(4.8, 2.4))

            S = numpy.abs(spectrum)
            S = 20 * numpy.log10(S / numpy.max(S))

            ax.imshow(
                S,
                origin="lower",
                cmap="viridis",
                extent=(0, length_sec, 0, sampling_rate / 2 / 1000),
            )
            ax.axis("tight")
            ax.set_ylabel("Frequency [kHz]")
            ax.set_xlabel("Time [s]")
            pyplot.show()

        d()

    main()
