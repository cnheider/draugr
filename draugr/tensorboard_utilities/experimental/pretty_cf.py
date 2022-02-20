#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20-04-2021
           """

import itertools
from typing import Sequence

import numpy
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

__all__ = ["pretty_print_conf_matrix"]


def pretty_print_conf_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    classes: Sequence,
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap=pyplot.cm.Blues,
) -> None:
    """
    Mostly stolen from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Normalization changed, classification_report stats added below plot
    """

    cm = confusion_matrix(y_true, y_pred)

    # Configure Confusion Matrix Plot Aesthetics (no text yet)
    pyplot.imshow(cm, interpolation="nearest", cmap=cmap)
    pyplot.title(title, fontsize=14)
    tick_marks = numpy.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)
    pyplot.ylabel("True label", fontsize=12)
    pyplot.xlabel("Predicted label", fontsize=12)

    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = numpy.round(cm.astype("float") / cm.sum(), 2)  # (axis=1)[:, numpy.newaxis]

    # Place Numbers as Text on Confusion Matrix Plot
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )

    # Add Precision, Recall, F-1 Score as Captions Below Plot
    rpt = classification_report(y_true, y_pred)
    rpt = rpt.replace("avg / total", "      avg")
    rpt = rpt.replace("support", "N Obs")

    pyplot.annotate(
        rpt,
        xy=(0, 0),
        xytext=(-50, -140),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=12,
        ha="left",
    )

    # Plot
    pyplot.tight_layout()


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.svm import SVC

    # get data, make predictions
    (X, y) = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    # Plot Confusion Matrix
    pyplot.style.use("classic")
    pyplot.figure(figsize=(3, 3))
    pretty_print_conf_matrix(
        y_test,
        y_test_pred,
        classes=["0", "1", "2"],
        normalize=True,
        title="Confusion Matrix",
    )
    pyplot.show()
