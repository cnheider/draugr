#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/04/2020
           """

from typing import Sequence, Tuple

import numpy

__all__ = ["intersect_numpy", "jaccard_overlap_numpy", "remove_null_boxes"]


def intersect_numpy(box_a: Sequence, box_b: Sequence) -> numpy.ndarray:
    """

    :param box_a:
    :type box_a:
    :param box_b:
    :type box_b:
    :return:
    :rtype:"""
    max_xy = numpy.minimum(box_a[:, 2:], box_b[2:])
    min_xy = numpy.maximum(box_a[:, :2], box_b[:2])
    inter = numpy.clip((max_xy - min_xy), a_min=0, a_max=numpy.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_overlap_numpy(box_a: numpy.ndarray, box_b: numpy.ndarray) -> numpy.ndarray:
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
    box_a: Multiple bounding boxes, Shape: [num_boxes,4]
    box_b: Single bounding box, Shape: [4]
    Return:
    jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]"""
    inter = intersect_numpy(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def remove_null_boxes(
    boxes: numpy.ndarray, labels: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Removes bounding boxes of W or H equal to 0 and its labels

    Args:
    boxes   (ndarray): NP Array with bounding boxes as lines
                   * BBOX[x1, y1, x2, y2]
    labels  (labels): Corresponding labels with boxes

    Returns:
    ndarray: Valid bounding boxes
    ndarray: Corresponding labels"""
    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)

    return numpy.delete(boxes, del_boxes, 0), numpy.delete(labels, del_boxes)
