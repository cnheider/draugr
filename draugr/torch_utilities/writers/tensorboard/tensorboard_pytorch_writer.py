#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import suppress
from typing import Any, Iterable, Sequence, Union

import PIL
import numpy
import torch
from PIL import Image
from matplotlib import pyplot
from matplotlib.figure import Figure

from draugr import PROJECT_APP_PATH
from draugr.numpy_utilities.mixing import mix_channels
from draugr.opencv_utilities import draw_masks
from draugr.python_utilities import sprint
from draugr.torch_utilities import to_tensor
from draugr.torch_utilities.tensors.dimension_order import (
    nhwc_to_nchw_tensor,
    nthwc_to_ntchw_tensor,
)
from draugr.torch_utilities.writers.torch_module_writer.module_parameter_writer_mixin import (
    ModuleParameterWriterMixin,
)
from draugr.torch_utilities.writers.torch_module_writer.module_writer_parameters import (
    weight_bias_histograms,
)
from draugr.writers.mixins import (
    BarWriterMixin,
    EmbedWriterMixin,
    GraphWriterMixin,
    HistogramWriterMixin,
    ImageWriterMixin,
    MeshWriterMixin,
    VideoInputDimsEnum,
    VideoWriterMixin,
)
from draugr.writers.mixins.figure_writer_mixin import FigureWriterMixin
from draugr.writers.mixins.instantiation_writer_mixin import InstantiationWriterMixin
from draugr.writers.mixins.line_writer_mixin import LineWriterMixin
from draugr.writers.mixins.precision_recall_writer_mixin import (
    PrecisionRecallCurveWriterMixin,
)
from draugr.writers.mixins.spectrogram_writer_mixin import SpectrogramWriterMixin
from draugr.writers.writer import Writer
from warg import drop_unused_kws, passes_kws_to

with suppress(FutureWarning):
    from torch.utils.tensorboard import SummaryWriter

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""
__all__ = ["TensorBoardPytorchWriter", "PTW"]

from pathlib import Path

GIF_FPS_UPPER_LIMIT = 50  # https://wunkolo.github.io/post/2020/02/buttery-smooth-10fps/ ... Browser-engine image decoders will automatically reset the frame rate to 10fps if not requested fps is not supported


class TensorBoardPytorchWriter(
    Writer,
    ImageWriterMixin,
    GraphWriterMixin,
    HistogramWriterMixin,
    BarWriterMixin,
    LineWriterMixin,
    SpectrogramWriterMixin,
    FigureWriterMixin,
    InstantiationWriterMixin,
    PrecisionRecallCurveWriterMixin,
    EmbedWriterMixin,
    ModuleParameterWriterMixin,
    VideoWriterMixin,
    MeshWriterMixin,
):
    """
    Provides a pytorch-tensorboard-implementation writer interface"""

    def video(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, Image.Image],
        step=None,
        frame_rate=30,
        input_dims=VideoInputDimsEnum.ntchw,
        **kwargs,
    ) -> None:
        """
        Shape:

            fastest expects vid_tensor: (N,T,C,H,W) .
             The values should lie in [0, 255] for type uint8 or [0, 1] for type float.

        """

        data = to_tensor(data)
        if input_dims == VideoInputDimsEnum.thwc:
            data = nhwc_to_nchw_tensor(data).unsqueeze(0)  # batch dim
        elif input_dims == VideoInputDimsEnum.tchw:
            data = data.unsqueeze(0)  # batch dim
        elif input_dims == VideoInputDimsEnum.thw:
            data = data.unsqueeze(1).unsqueeze(0)  # channel then batch dim
        elif input_dims == VideoInputDimsEnum.nthwc:
            data = nthwc_to_ntchw_tensor(data)
        elif input_dims == VideoInputDimsEnum.ntchw:
            pass
        else:
            raise NotImplementedError(
                "Not supported yet, use one of the other combinations"
            )

        assert len(data.shape) == 5

        frame_rate = min(frame_rate, GIF_FPS_UPPER_LIMIT)

        self.writer.add_video(tag, data, fps=frame_rate, global_step=step, **kwargs)

    def mesh(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, Image.Image],
        step=None,
        **kwargs,
    ) -> None:
        """
        Data being vertices here.

        Shape: (B,N,3)
        data: (B,N,3). (batch, number_of_vertices, channels)
        colors: (B,N,3). The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        faces: (B,N,3). The values should lie in [0, number_of_vertices] for type uint8.

        :param tag:
        :param data:
        :param step:
        :param kwargs:
        :return:
        """
        self.writer.add_mesh(tag, data, global_step=step, **kwargs)

    def parameters(
        self, model: torch.nn.Module, step: int, tag: str = "", **kwargs
    ) -> None:
        """

        :param model:
        :param step:
        :param tag:
        :param kwargs:
        """
        weight_bias_histograms(self, model, prefix=tag, step=step, **kwargs)

    @passes_kws_to(SummaryWriter.add_embedding)
    def embed(
        self,
        tag: str,
        response: Sequence,
        metadata: Any = None,
        label_img: Any = None,
        step: int = None,
        **kwargs,
    ) -> None:
        """

        BORKED!

        :param tag:
        :param response:
        :param metadata:
        :param label_img:
        :param step:
        :param kwargs:
        :return:"""
        try:
            self.writer.add_embedding(
                response,
                metadata=metadata,
                label_img=label_img,
                global_step=step,
                tag=tag,
                **kwargs,
            )
        except Exception as e:
            sprint("Try update tensorflow and/or tensorboard")
            raise e

    @passes_kws_to(Writer.__init__)
    def __init__(
        self,
        path: Union[str, Path] = Path.cwd() / "Logs",
        summary_writer_kws=None,
        **kwargs,
    ):
        """

        :param path:
        :param summary_writer_kws:
        :param kwargs:"""
        super().__init__(**kwargs)

        if summary_writer_kws is None:
            summary_writer_kws = {}

        self._log_dir = path
        self._summary_writer_kws = summary_writer_kws

    # @passes_kws_to(SummaryWriter.add_hparams)
    def instance(self, instance: dict, metrics: dict, **kwargs) -> None:
        """

        TODO: Not finished!

        :param instance:
        :param metrics:
        :return:"""
        self.writer.add_hparams(instance, metrics, **kwargs)

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_pr_curve)
    def precision_recall_curve(
        self,
        tag: str,
        predictions: Iterable,
        truths: Iterable,
        step: int = None,
        **kwargs,
    ) -> None:
        """

        :param tag:
        :param predictions:
        :param truths:
        :param step:
        :param kwargs:"""
        self.writer.add_pr_curve(
            tag,
            to_tensor(truths, device="cpu"),
            to_tensor(predictions, device="cpu"),
            global_step=step,
            **kwargs,
        )

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_figure)
    def figure(self, tag: str, figure: Figure, step: int, **kwargs) -> None:
        """

        :param tag:
        :type tag:
        :param figure:
        :type figure:
        :param step:
        :type step:
        :param kwargs:
        :type kwargs:"""
        self.writer.add_figure(tag, figure, global_step=step, **kwargs)

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_figure)
    def spectrogram(
        self,
        tag: str,
        values: list,
        sample_rate: int,
        step: int,
        n_fft: int = 512,
        step_size=128,
        x_labels: Sequence = None,
        y_label: str = "Frequency [Hz]",
        x_label: str = "Time [sec]",
        plot_kws=None,
        **kwargs,
    ) -> None:
        """

        :param sample_rate:
        :param n_fft:
        :param step_size:
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
        :type kwargs:"""
        if plot_kws is None:
            plot_kws = {}
        fig = pyplot.figure()

        spec = pyplot.specgram(
            values, NFFT=n_fft, Fs=sample_rate, noverlap=step_size, **plot_kws
        )

        """
ind = numpy.arange(len(values))
if x_labels:
pyplot.xticks(ind, labels=x_labels)
else:
pyplot.xticks(ind)

pyplot.xlabel(x_label)
pyplot.ylabel(y_label)
pyplot.title(tag)
"""

        pyplot.colorbar()
        self.writer.add_figure(
            tag, fig, global_step=step, close=True, **kwargs
        )  # TODO: Pull out hardcoded kws to argument list for this method, duplicate kwarg issue possible

    # def embed(self, tag: str, features, metadata, label_img, step: int, **kwargs) -> None:

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_figure)
    def bar(
        self,
        tag: str,
        values: list,
        step: int,
        value_error: float = None,
        x_labels: Sequence = None,
        y_label: str = "Probabilities",
        x_label: str = "Distribution",
        **kwargs,
    ) -> None:
        """

        :param x_label:
        :type x_label:
        :param tag:
        :type tag:
        :param values:
        :type values:
        :param step:
        :type step:
        :param value_error:
        :type value_error:
        :param x_labels:
        :type x_labels:
        :param y_label:
        :type y_label:
        :param kwargs:
        :type kwargs:"""
        fig = pyplot.figure()
        ind = numpy.arange(len(values))
        im = pyplot.bar(ind, values, yerr=value_error)
        if x_labels:
            pyplot.xticks(ind, labels=x_labels)
        else:
            pyplot.xticks(ind)

        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        pyplot.title(tag)

        self.writer.add_figure(
            tag, fig, global_step=step, close=True, **kwargs
        )  # TODO: Pull out hardcoded kws to argument list for this method, duplicate kwarg issue possible

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_figure)
    def line(
        self,
        tag: str,
        values: list,
        step: int,
        x_labels: Sequence = None,
        y_label: str = "Magnitude",
        x_label: str = "Sequence",
        plot_kws=None,  # Separate as parameters name collisions might occur
        **kwargs,
    ) -> None:
        """

        :param x_label:
        :type x_label:
        :param plot_kws:
        :type plot_kws:
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
        :param kwargs:
        :type kwargs:"""
        if plot_kws is None:
            plot_kws = {}
        fig = pyplot.figure()
        ind = numpy.arange(len(values))
        im = pyplot.plot(values, **plot_kws)
        if x_labels:
            pyplot.xticks(ind, labels=x_labels)
        else:
            pyplot.xticks(ind)

        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        pyplot.title(tag)

        self.writer.add_figure(
            tag, fig, global_step=step, close=True, **kwargs
        )  # TODO: Pull out hardcoded kws to argument list for this method, duplicate kwarg issue possible

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_histogram)
    def histogram(
        self, tag: str, values: list, step: int, bins: Any = "auto", **kwargs
    ) -> None:
        """

        :param tag:
        :type tag:
        :param values:
        :type values:
        :param step:
        :type step:
        :param bins:
        :type bins:
        :param kwargs:
        :type kwargs:"""
        self.writer.add_histogram(tag, values, global_step=step, bins=bins, **kwargs)

    def _scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_graph)
    def graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor,
        verbose: bool = False,
    ) -> None:
        """

        :param verbose:
        :param model:
        :type model:
        :param input_to_model:
        :type input_to_model:"""
        self.writer.add_graph(model, input_to_model, verbose)

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        if hasattr(self, "_writer"):
            self._writer.close()
            delattr(self, "_writer")

    @drop_unused_kws
    @passes_kws_to(SummaryWriter.add_image)
    def image(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, PIL.Image.Image],
        step,
        *,
        data_formats: str = "NCHW",
        multi_channel_method: ImageWriterMixin.MultiChannelMethodEnum = ImageWriterMixin.MultiChannelMethodEnum.seperate,
        seperate_channel_postfix: str = "channel",
        seperate_image_postfix: str = "image",
        **kwargs,
    ) -> None:
        """

        :param tag:
        :type tag:
        :param data:
        :type data:
        :param step:
        :type step:
        :param data_formats:
        :type data_formats:
        :param kwargs:
        :type kwargs:"""
        if data_formats == "NCHW":
            num_channels = data.shape[-3]
            if num_channels == 2 or num_channels > 3:
                if (
                    multi_channel_method
                    == ImageWriterMixin.MultiChannelMethodEnum.seperate
                ):
                    for i in range(num_channels):
                        self.writer.add_image(
                            f"{tag}_{seperate_channel_postfix}_{i}",
                            data[:, i].unsqueeze(-3),
                            step,
                            dataformats=data_formats,
                            **kwargs,
                        )
                elif (
                    multi_channel_method == ImageWriterMixin.MultiChannelMethodEnum.mix
                ):
                    # TODO: Merge channels into a single channel of overlapping values'
                    mixed = mix_channels(data)
                    self.writer.add_image(
                        tag,
                        mixed.unsqueeze(-3),
                        step,
                        dataformats=data_formats,
                        **kwargs,
                    )
                elif (
                    multi_channel_method
                    == ImageWriterMixin.MultiChannelMethodEnum.project
                ):
                    # TODO: Project channels into RGB space,NOT DONE
                    for i in range(data.shape[0]):
                        img = numpy.zeros(3, *data.shape[-2:])
                        img = draw_masks(img, data[i])
                        self.writer.add_image(
                            f"{tag}_{seperate_image_postfix}_{i}",
                            img,
                            step,
                            dataformats="CHW",
                            **kwargs,
                        )
                else:

                    raise NotImplementedError(
                        f"{multi_channel_method} is not implemented"
                    )
            else:
                self.writer.add_image(
                    tag, data, step, dataformats=data_formats, **kwargs
                )
        else:
            self.writer.add_image(tag, data, step, dataformats=data_formats, **kwargs)

    @property
    def writer(self) -> SummaryWriter:
        """

        :return:
        :rtype:"""
        if not hasattr(self, "_writer") or not self._writer:
            self._writer = SummaryWriter(
                str(self._log_dir), **self._summary_writer_kws
            )  # DB MODEL    --db sqlite:~/.tensorboard.db ON HOLD..
            if self._verbose:
                print(f"Logging at {self._log_dir}")
        return self._writer

    def _open(self):
        return self


# NAMING AND CASING, SO MANY IMPLEMENTATION OF TENSORBOARD WRITERS, COVER BASES!
TensorboardPytorchWriter = TensorBoardPytorchWriter
TensorBoardPyTorchWriter = TensorBoardPytorchWriter
TensorboardPyTorchWriter = TensorBoardPytorchWriter
PytorchTensorboardWriter = TensorBoardPytorchWriter
PyTorchTensorBoardWriter = TensorBoardPytorchWriter
PyTorchTensorboardWriter = TensorBoardPytorchWriter
TensorboardTorchWriter = TensorBoardPytorchWriter
TensorBoardTorchWriter = TensorBoardPytorchWriter
TorchTensorboardWriter = TensorBoardPytorchWriter
TorchTensorBoardWriter = TensorBoardPytorchWriter
PTW = TensorBoardPytorchWriter

if __name__ == "__main__":

    with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log / "test") as writer:
        writer.scalar("What", 4)

        from torchvision.utils import make_grid

        for n_iter in range(20):
            dummy_img = torch.rand(32, 3, 64, 64)  # output from network
            if n_iter % 10 == 0:
                x = make_grid(dummy_img, normalize=True, scale_each=True)
                writer.image("ImageGrid", x, n_iter, data_formats="CHW")
                writer.image("Image", dummy_img, n_iter)
