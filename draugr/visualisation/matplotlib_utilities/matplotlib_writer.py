#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 7/13/22
           """

__all__ = []

from pathlib import Path
from typing import Sequence, Mapping, Union, MutableMapping

import numpy
from PIL import Image
from matplotlib import pyplot
from matplotlib.figure import Figure

from draugr.writers import (
    Writer,
    ImageWriterMixin,
    HistogramWriterMixin,
    BarWriterMixin,
    LineWriterMixin,
    SpectrogramWriterMixin,
    FigureWriterMixin,
)
from warg import passes_kws_to


class MatPlotLibWriter(
    Writer,
    ImageWriterMixin,
    HistogramWriterMixin,
    BarWriterMixin,
    LineWriterMixin,
    SpectrogramWriterMixin,
    FigureWriterMixin,
):
    """
    description
    """

    @passes_kws_to(pyplot.savefig)
    def __init__(self, path: Path, format_: str = "png", **kwargs: MutableMapping):
        super().__init__()
        self._path = path
        self._format = format_
        self.kws = kwargs
        self._open()

    def _close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        del self.scalars

    def _open(self):
        self.scalars = {}
        return self

    def _scalar(self, tag: str, value: float, step: int) -> None:
        if tag not in self.scalars:
            self.scalars[tag] = []
        self.scalars[tag] += value
        fig: Figure = pyplot.imshow(value)
        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()

    @passes_kws_to(pyplot.imshow)
    def image(
        self,
        tag: str,
        data: Union[numpy.ndarray, Image.Image],
        step: int,
        *,
        dataformats: str = "NCHW",
        **kwargs: MutableMapping,
    ) -> None:
        """
        Plot an image in matplotlib.

        :param tag:
        :type tag:
        :param data:
        :type data:
        :param step:
        :type step:
        :param dataformats:
        :type dataformats:
        :param kwargs:
        :type kwargs:
        """
        fig = pyplot.imshow(data, **kwargs)
        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()

    @passes_kws_to(pyplot.hist)
    def histogram(
        self, tag: str, values: list, step: int, **kwargs: MutableMapping
    ) -> None:
        """
        Plot a histogram in matplotlib.

        :param tag:
        :type tag:
        :param values:
        :type values:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:
        """
        fig = pyplot.hist(values, **kwargs)

        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()

    def bar(
        self,
        tag: str,
        values: list,
        step: int,
        y_error=None,
        x_labels=None,
        y_label="Probability",
        x_label="Action Categorical Distribution",
        **kwargs: MutableMapping,
    ) -> None:
        """
        Plot a bar chart in matplotlib.

        :param tag:
        :type tag:
        :param values:
        :type values:
        :param step:
        :type step:
        :param y_error:
        :type y_error:
        :param x_labels:
        :type x_labels:
        :param y_label:
        :type y_label:
        :param x_label:
        :type x_label:
        :param kwargs:
        :type kwargs:
        """
        fig = pyplot.bar(values)
        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()

    def line(
        self,
        tag: str,
        values: list,
        step: int,
        x_labels: Sequence = None,
        y_label: str = "Magnitude",
        x_label: str = "Sequence",
        plot_kws: Mapping = None,
        **kwargs: MutableMapping,
    ) -> None:
        """
        Plot a line chart in matplotlib.

        :param tag:
        :type tag:
        :param values:
        :type values:
        :param step:
        :type step:
        :param x_labels:
        :type x_labels:
        :param y_label:
        :type y_label:
        :param x_label:
        :type x_label:
        :param plot_kws:
        :type plot_kws:
        :param kwargs:
        :type kwargs:
        """
        fig = pyplot.plot(values)
        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()

    def spectrogram(
        self,
        tag: str,
        values: list,
        sample_rate: int,
        step: int,
        num_fft: int = 512,
        x_labels: Sequence = None,
        y_label: str = "Magnitude",
        x_label: str = "Sequence",
        plot_kws: Mapping = None,
        **kwargs: MutableMapping,
    ) -> None:
        """
        Plot a spectrogram in matplotlib.

        :param tag:
        :type tag:
        :param values:
        :type values:
        :param sample_rate:
        :type sample_rate:
        :param step:
        :type step:
        :param num_fft:
        :type num_fft:
        :param x_labels:
        :type x_labels:
        :param y_label:
        :type y_label:
        :param x_label:
        :type x_label:
        :param plot_kws:
        :type plot_kws:
        :param kwargs:
        :type kwargs:
        """
        fig = pyplot.specgram(values, Fs=sample_rate, NFFT=num_fft)
        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()

    def figure(
        self, tag: str, figure: Figure, step: int, **kwargs: MutableMapping
    ) -> None:
        """
        Plot a figure in matplotlib.

        :param tag:
        :type tag:
        :param figure:
        :type figure:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:
        """
        fig = pyplot.figure(figure)
        fig.savefig(
            (self._path / f"{tag}_{step}").with_suffix(self._format),
            format=self._format,
            **self.kws,
        )
        fig.close()


if __name__ == "__main__":
    with MatPlotLibWriter() as writer:
        writer.scalar("test", 1.0, 0)
        writer.scalar("test", 2.0, 1)
        writer.scalar("test", 3.0, 2)

        writer.image("test", numpy.random.rand(3, 3, 3), 0)
        writer.image("test", numpy.random.rand(3, 3, 3), 1)

        writer.histogram("test", [1, 2, 3, 4, 5], 0)
        writer.histogram("test", [1, 2, 3, 4, 5], 1)

        writer.bar("test", [1, 2, 3, 4, 5], 0)
        writer.bar("test", [1, 2, 3, 4, 5], 1)

        writer.line("test", [1, 2, 3, 4, 5], 0)
        writer.line("test", [1, 2, 3, 4, 5], 1)

        writer.spectrogram("test", [1, 2, 3, 4, 5], 1, 0)
        writer.spectrogram("test", [1, 2, 3, 4, 5], 1, 1)

        writer.figure("test", pyplot.figure(), 0)
        writer.figure("test", pyplot.figure(), 1)
