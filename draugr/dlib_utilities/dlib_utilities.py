#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-05-2021
           """

from enum import Enum
from typing import Sequence, Tuple, Union

import numpy
from scipy.spatial import distance

__all__ = [
    "mouth_aspect_ratio",
    "Dlib68faciallandmarksindices",
    "Dlib5faciallandmarksindices",
    "rect_to_bounding_box",
    "shape_to_ndarray",
    "eye_aspect_ratio",
]


class ExtendTuple(Tuple):
    def __or__(self, *other) -> "ExtendTuple":
        return ExtendTuple((*self, *other))

    def __and__(self, *other) -> "ExtendTuple":
        return ExtendTuple((*self, *other))


class Dlib68faciallandmarksindices(Enum):
    mouth = (48, 67 + 1)
    inner_mouth = (60, 67 + 1)
    right_eyebrow = (17, 21 + 1)
    left_eyebrow = (22, 26 + 1)
    right_eye = (36, 41 + 1)
    left_eye = (42, 47 + 1)
    nose = (27, 35 + 1)
    jaw = (0, 16 + 1)

    @staticmethod
    def slice(
        seq: Sequence,
        ind: Union[
            "Dlib68faciallandmarksindices", Tuple["Dlib68faciallandmarksindices"]
        ],
    ):
        """

        :param seq:
        :param ind:
        :return:
        """
        if isinstance(ind, Tuple):
            agg = []
            for (
                i
            ) in (
                ind
            ):  # Some flag implementation would probably be faster and more valid.
                agg.extend(Dlib68faciallandmarksindices.slice(seq, i))
            return agg
        start, end = ind.value
        return seq[start:end]

    def __or__(
        self, other: "Dlib68faciallandmarksindices"
    ) -> Tuple["Dlib68faciallandmarksindices"]:
        return ExtendTuple().__or__(self, other)

    def __and__(
        self, other: "Dlib68faciallandmarksindices"
    ) -> Tuple["Dlib68faciallandmarksindices"]:
        return ExtendTuple().__and__(self, other)


class Dlib5faciallandmarksindices(Enum):
    """
    dlib_utilities’s 5-point facial landmark detector
    """

    right_eye = (2, 3 + 1)
    left_eye = (0, 1 + 1)
    nose = (4, 4 + 1)

    @staticmethod
    def slice(
        seq: Sequence,
        ind: Union["Dlib5faciallandmarksindices", Tuple["Dlib5faciallandmarksindices"]],
    ):
        """

        :param seq:
        :param ind:
        :return:
        """
        if isinstance(ind, Tuple):
            agg = []
            for (
                i
            ) in (
                ind
            ):  # Some flag implementation would probably be faster and more valid.
                agg.extend(Dlib5faciallandmarksindices.slice(seq, i))
            return agg
        start, end = ind.value
        return seq[start:end]

    def __or__(
        self, other: "Dlib5faciallandmarksindices"
    ) -> Tuple["Dlib5faciallandmarksindices"]:
        return ExtendTuple().__or__(self, other)

    def __and__(
        self, other: "Dlib5faciallandmarksindices"
    ) -> Tuple["Dlib5faciallandmarksindices"]:
        return ExtendTuple().__and__(self, other)


def rect_to_bounding_box(rect) -> Tuple[float, float, float, float]:
    """
    # take a bounding predicted by dlib_utilities and convert it
    # to the format (x, y, w, h)"""
    x = rect.left()
    y = rect.top()

    return x, y, rect.right() - x, rect.bottom() - y  # return a tuple of (x, y, w, h)


def shape_to_ndarray(shape, dtype: str = "int"):
    """

    :param shape:
    :param dtype:
    :return:
    """
    coordinates = numpy.zeros(
        (shape.num_parts, 2), dtype=dtype
    )  # initialize the list of (x, y)-coordinates

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates  # return the list of (x, y)-coordinates


def mouth_aspect_ratio(coordinates: Sequence[Sequence]) -> float:
    """

    :param coordinates:
    :return:"""
    average = (
        distance.euclidean(coordinates[3], coordinates[9])
        + distance.euclidean(coordinates[2], coordinates[10])
        + distance.euclidean(coordinates[4], coordinates[8])
    ) / 3
    return average / distance.euclidean(coordinates[0], coordinates[6])


def eye_aspect_ratio(coordinates: Sequence[Sequence]) -> float:
    """

    :param coordinates:
    :return:
    """
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(coordinates[1], coordinates[5])
    B = distance.euclidean(coordinates[2], coordinates[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(coordinates[0], coordinates[3])

    # compute the eye aspect ratio
    return (A + B) / (2.0 * C)


# To improve our blink detector, Soukupová and Čech recommend constructing a 13-dim feature vector of eye aspect ratios (N-th frame, N – 6 frames, and N + 6 frames), followed by feeding this feature vector into a Linear SVM for classification.

if __name__ == "__main__":

    def asud():
        a = list(range(99))
        slices = (
            Dlib68faciallandmarksindices.left_eye
            | Dlib68faciallandmarksindices.right_eye
            | Dlib68faciallandmarksindices.nose & Dlib68faciallandmarksindices.mouth
        )
        print(Dlib68faciallandmarksindices.slice(a, slices))

    def as34ud():
        a = list(range(99))
        print(
            Dlib68faciallandmarksindices.slice(a, Dlib68faciallandmarksindices.left_eye)
        )

    asud()
    as34ud()
