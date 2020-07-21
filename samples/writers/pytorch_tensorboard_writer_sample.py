#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07/07/2020
           """

import librosa
import numpy
import torch
from librosa.display import specshow
from matplotlib import pyplot

from draugr import PROJECT_APP_PATH
from draugr.torch_utilities import TensorBoardPytorchWriter
from draugr.torch_utilities.initialisation.fan_in_weight_init import (
    constant_init,
    fan_in_init,
    normal_init,
    xavier_init,
)
from draugr.torch_utilities.writers.torch_module_writer.module_writer_parameters import (
    weight_bias_histograms,
)

if __name__ == "__main__":
    NFFT = 256
    STEP_SIZE = NFFT // 2

    DELTA = 0.001
    time_ = numpy.arange(0, 1, DELTA)
    SAMPLING_RATE = int(1 / DELTA)

    SIGNAL = numpy.sin(2 * numpy.pi * 50 * time_) + numpy.sin(
        2 * numpy.pi * 120 * time_
    )

    def module_param_histograms():

        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            input_f = 4
            n_classes = 10
            num_updates = 20

            model = torch.nn.Sequential(
                torch.nn.Linear(input_f, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, n_classes),
                torch.nn.LogSoftmax(-1),
            )

            for i in range(num_updates):
                normal_init(
                    model, 0.2 * float((i - num_updates * 0.5) ** 2), 1 / (i + 1)
                )
                weight_bias_histograms(writer, model, step=i, prefix="normal")

                xavier_init(model)
                weight_bias_histograms(writer, model, step=i, prefix="xavier")

                constant_init(model, i)
                weight_bias_histograms(writer, model, step=i, prefix="constant")

                fan_in_init(model)
                weight_bias_histograms(writer, model, step=i, prefix="fan_in")

    def signal_plot():
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            writer.line("Signal", SIGNAL, step=0)

    def fft_plot():
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            spectral = numpy.fft.fft(SIGNAL, NFFT)
            writer.line("FFT", spectral, title="Frequency", step=0)

    def spectral_plot():
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            writer.spectrogram(
                "STFT", SIGNAL, int(1 / DELTA), step=0, n_fft=NFFT, step_size=STEP_SIZE
            )

    def spectral_plot_scipy():
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            writer.spectrogram(
                "STFT_Scipy",
                SIGNAL,
                int(1 / DELTA),
                step=0,
                n_fft=NFFT,
                step_size=STEP_SIZE,
            )

    def cepstral_plot():
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            fig = pyplot.figure()
            stft = librosa.core.stft(SIGNAL, n_fft=NFFT, hop_length=STEP_SIZE)
            specshow(stft, sr=SAMPLING_RATE, x_axis="time")
            pyplot.colorbar()
            writer.figure("STFT_Rosa", fig, step=0)

    def mel_cepstral_plot():
        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            fig = pyplot.figure()
            mfccs = librosa.feature.mfcc(
                SIGNAL, sr=SAMPLING_RATE, n_mfcc=20, n_fft=NFFT, hop_length=STEP_SIZE
            )
            specshow(mfccs, sr=SAMPLING_RATE, x_axis="time")
            pyplot.colorbar()
            writer.figure("MFCC_Rosa", fig, step=0)

    module_param_histograms()
    signal_plot()
    fft_plot()
    spectral_plot()
    spectral_plot_scipy()
    cepstral_plot()
    mel_cepstral_plot()
