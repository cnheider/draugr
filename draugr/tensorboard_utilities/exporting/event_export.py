import enum
from enum import Enum
from itertools import zip_longest
from pathlib import Path
from pickle import dump
from typing import Iterable, Mapping, Tuple, TypeVar, Union

import numpy
import pandas
import tensorflow
from PIL.Image import Image
from matplotlib import pyplot
from tensorboard.backend.event_processing import event_accumulator

from apppath import AppPath, ensure_existence

__all__ = ["TensorboardEventExporter"]

from warg import passes_kws_to

# TODO: implement export options using ExportMethodEnum
# TODO: MAJOR REFACTOR INCOMING

TagTypeEnum = TypeVar("TagTypeEnum")


class TensorboardEventExporter:
    """
    Reads event files and exports the requested tags."""

    # class TagTypeEnum(Enum): #Static version, does not adapt to plugins!
    #    images = 'images'
    #    scalars = 'scalars'
    #    tensors = 'tensors'
    #    audio = 'audio'
    #    distributions= 'distributions'
    #    graph = 'graph'
    #    meta_graph = 'meta_graph'
    #    histograms = 'histograms'
    #    run_metadata = 'run_metadata'

    def __init__(
        self,
        path_to_events_file_s: Path,
        size_guidance: Mapping = None,
        *,
        save_to_disk: bool = False,
    ):
        """

        :param path_to_events_file_s:
        :param size_guidance:
        :param save_to_disk:"""
        if size_guidance is None:
            size_guidance = 0

        if isinstance(size_guidance, Mapping):
            pass
        elif isinstance(
            size_guidance, int
        ):  # if only an integer was provided override all size_guidance entries to that integer
            size_guidance_map = (
                event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
            )  # Get entries with store everything
            if (
                size_guidance > 0
            ):  # the provided integer was above 0 (store_everything) then limit store to that integer (limited)
                size_guidance_map = {k: size_guidance for k in size_guidance_map.keys()}
            size_guidance = size_guidance_map
        else:
            raise TypeError(f"Invalid type of size guidance {type(size_guidance)}")

        self.path_to_events_file = str(path_to_events_file_s)

        self.event_acc = event_accumulator.EventAccumulator(
            self.path_to_events_file, size_guidance=size_guidance
        )

        self.event_acc.Reload()
        self.tags_available = self.event_acc.Tags()
        self.save_to_disk = save_to_disk

        tags_dict = {}
        for (
            t
        ) in self.tags_available:  # TODO: Automatic but not nice for code completion
            setattr(self, f"available_{t}", self.tags_available[t])
            tags_dict[str(t)] = str(t)

        TensorboardEventExporter.TagTypeEnum = enum.Enum(
            "TagTypeEnum", tags_dict
        )  # dynamic version

    def tag_test(self, *tags, type_str: Union[str, TagTypeEnum]) -> bool:
        """

        :param tags:
        :param type_str:
        :return:"""
        if not len(tags):
            print("No tags requested")
            # raise Exception #TODO: maybe

        if isinstance(type_str, Enum):
            type_str = type_str.value

        if (
            len(tags) == 1
            and isinstance(tags[0], Iterable)
            and not isinstance(tags[0], str)
        ):
            tags = tags[0]
        tags_available = self.tags_available[type_str]
        assert all(
            [tags_available.__contains__(t) for t in tags]
        ), f"{type_str} tags available: {tags_available}, tags requested {tags}"
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def export_line_plot(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Tuple[pyplot.Figure]:
        """

        :param tags:
        :param out_dir:
        :return:"""
        self.tag_test(*tags, type_str="scalars")
        out = []
        for t in tags:
            w_times, step_nums, vals = zip(*self.event_acc.Scalars(t))
            fig, ax = pyplot.subplots(nrows=1, ncols=1)
            ax.plot(step_nums, vals)
            if self.save_to_disk:
                fig.savefig(str(out_dir / f"{t}_line_plot.png"))
            out.append(ax)
        return (*out,)

    def export_image(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Tuple[Image]:
        """

        :param tags:
        :param out_dir:
        :return:"""
        self.tag_test(*tags, type_str="images")
        out = []
        for t in tags:
            img = self.event_acc.Images(t)  # TODO: CHECK VIDEO compatibility(gifs)
            if self.save_to_disk:
                with open(str(out_dir / f"{t}_img_{img.step}.png"), "wb") as f:
                    f.write(img.encoded_image_string)
            out.append(Image.fromstring(img.encoded_image_string))
        return (*out,)

    def export_scalar(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Iterable:
        """
        if save to files it pickles tags values with file ending .pkl

        :param tags:
        :param out_dir:
        :return:"""
        self.tag_test(*tags, type_str="scalars")
        out = []
        for t in tags:
            w_times, step_nums, vals = zip(*self.event_acc.Scalars(t))
            if self.save_to_disk:
                with open(str(out_dir / f"{t}.pkl"), "wb") as f:
                    dump(vals, f)
            out.append(vals)
        return (*out,)

    def export_distribution(self, *tags: Iterable[str], out_dir: Path = Path.cwd()):
        """

        :param tags:
        :param out_dir:"""
        self.tag_test(*tags, type_str="distributions")
        raise NotImplemented("not implemented yet!")
        out = []
        for t in tags:
            vals = None  # t
            out.append(vals)
        return (*out,)

    def export_tensor(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Iterable:
        """

        :param tags:
        :param out_dir:
        :return:"""
        self.tag_test(*tags, type_str="tensors")
        out = []
        for t in tags:
            w_times, step_nums, vals = zip(*self.event_acc.Tensors(t))
            if self.save_to_disk:
                with open(str(out_dir / f"{t}.pkl"), "wb") as f:
                    dump(vals, f)
            out.append(vals)
        return (*out,)

    def export_graph(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Iterable:
        """

        :param tags:
        :param out_dir:
        :return:"""
        out = []
        w_times, step_nums, vals = zip(*self.event_acc.Graph())
        if self.save_to_disk:
            with open(str(out_dir / f"graph.pkl"), "wb") as f:
                dump(vals, f)
        out.append(vals)
        return (*out,)

    def export_audio(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Iterable:
        """

        :param tags:
        :param out_dir:
        :return:"""
        self.tag_test(*tags, type_str="audio")
        out = []
        for t in tags:
            w_times, step_nums, vals = zip(*self.event_acc.Audio(t))
            if self.save_to_disk:
                with open(str(out_dir / f"{t}.pkl"), "wb") as f:
                    dump(vals, f)
            out.append(vals)
        return (*out,)

    def export_histogram(
        self, *tags: Iterable[str], out_dir: Path = Path.cwd()
    ) -> Iterable:
        """

        https://www.tensorflow.org/api_docs/python/tf/summary/histogram

        :param tags:
        :param out_dir:
        :return:"""
        self.tag_test(*tags, type_str="histograms")

        out = []
        for t in tags:
            w_times, step_nums, vals = zip(*self.event_acc.Histograms(t))
            if self.save_to_disk:
                with open(str(out_dir / f"{t}.pkl"), "wb") as f:
                    dump(vals, f)
            out.append(vals)
        return (*out,)

    @passes_kws_to(pandas.DataFrame.to_csv)
    def scalar_export_csv(
        self,
        *tags: Iterable[str],
        out_dir: Path = Path.cwd(),
        index_label: str = "epoch",
        **kwargs,
    ) -> Tuple[pandas.DataFrame]:
        """
        size_guidance = 0 means all events, no aggregation or dropping

            :param index_label:
        :return:
        :param tags:
        :param out_dir:"""
        if not len(tags):
            print("No tags requested")  # TODO: maybe just return
            # return tuple()

        self.tag_test(*tags, type_str="scalars")

        out = []

        df = pandas.DataFrame(
            list(
                zip_longest(
                    *[
                        list(zip_longest(*self.event_acc.Scalars(t), fillvalue=None))[
                            -1
                        ]
                        for t in tags
                    ],
                    fillvalue=None,
                )
            ),
            columns=tags,
        )
        if self.save_to_disk:
            df.to_csv(
                str(out_dir / f'scalars_{"_".join(tags) if len(tags) else "none"}.csv'),
                columns=tags,
                index_label=index_label,
                **kwargs,
            )
        out.append(df)
        return (*out,)

    @passes_kws_to(pandas.DataFrame.to_csv)
    def pr_curve_export_csv(
        self,
        *tags: Iterable[str],
        out_dir: Path = Path.cwd(),
        index_label: str = "epoch",
        **kwargs,
    ) -> Tuple[pandas.DataFrame]:
        """
        #TODO only supports a single step and tag for now

        size_guidance = 0 means all events, no aggregation or dropping

            :param index_label:
        :return:
        :param tags:
        :param out_dir:"""
        if not len(tags):
            print("No tags requested")  # TODO: maybe just return
            # return tuple()

        self.tag_test(*tags, type_str="tensors")

        out = []

        numpy_rep = numpy.array(
            [
                tensorflow.make_ndarray(b)
                for a in list(
                    zip_longest(
                        *[
                            list(
                                zip_longest(*self.event_acc.Tensors(t), fillvalue=None)
                            )[-1]
                            for t in tags
                        ],
                        fillvalue=None,
                    )
                )
                for b in a
            ]
        )
        labels = (
            "true_positive_counts",
            "false_positive_counts",
            "true_negative_counts",
            "false_negative_counts",
            "precision",
            "recall",
        )
        df = pandas.DataFrame(
            numpy_rep.tolist(), columns=[f"{t}_{l}" for t in tags for l in labels]
        )
        if self.save_to_disk:
            df.to_csv(
                str(out_dir / f'tensors_{"_".join(tags) if len(tags) else "none"}.csv'),
                # columns=tags,
                index_label=index_label,
                **kwargs,
            )
        out.append(df)
        return (*out,)

    @passes_kws_to(pandas.DataFrame.to_csv)
    def tensor_export_csv(
        self,
        *tags: Iterable[str],
        out_dir: Path = Path.cwd(),
        index_label: str = "epoch",
        **kwargs,
    ) -> Tuple[pandas.DataFrame]:
        """

        size_guidance = 0 means all events, no aggregation or dropping

            :param index_label:
        :return:
        :param tags:
        :param out_dir:"""
        if not len(tags):
            print("No tags requested")  # TODO: maybe just return
            # return tuple()

        self.tag_test(*tags, type_str="tensors")

        out = []

        numpy_rep = numpy.array(
            [
                tensorflow.make_ndarray(b)
                for a in list(
                    zip_longest(
                        *[
                            list(
                                zip_longest(*self.event_acc.Tensors(t), fillvalue=None)
                            )[-1]
                            for t in tags
                        ],
                        fillvalue=None,
                    )
                )
                for b in a
            ]
        )

        df = pandas.DataFrame([[numpy_rep.tolist()]], columns=tags)
        if self.save_to_disk:
            df.to_csv(
                str(out_dir / f'tensors_{"_".join(tags) if len(tags) else "none"}.csv'),
                # columns=tags,
                index_label=index_label,
                **kwargs,
            )
        out.append(df)
        return (*out,)


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        _path_to_events_file = next(
            AppPath("Draugr", "Christian Heider Nielsen").user_log.rglob(
                "events.out.tfevents.*"
            )
        )
        print(_path_to_events_file)
        tee = TensorboardEventExporter(_path_to_events_file.parent, save_to_disk=True)
        print(tee.tags_available)
        # tee.export_csv('train_loss')
        # tee.export_line_plot('train_loss')
        # pyplot.show()
        print(tee.export_histogram())
        print(tee.available_scalars)
        print(
            tee.pr_curve_export_csv(
                *tee.tags_available["tensors"],
                out_dir=ensure_existence(Path.cwd() / "exclude"),
            )
        )
        print(list(iter(tee.TagTypeEnum)))

    a()
