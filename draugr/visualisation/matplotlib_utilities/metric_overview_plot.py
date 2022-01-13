#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Iterable, Iterator, Sequence, Tuple

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from matplotlib.figure import Figure
from matplotlib.pyplot import matshow, imshow
from itertools import cycle

from numpy import interp

from warg import passes_kws_to
from matplotlib import pyplot
import numpy
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.multiclass import unique_labels

__all__ = [
    "horizontal_imshow",
    "biplot",
    "pca_biplot",
    "correlation_matrix_plot",
    "confusion_matrix_plot",
    "roc_plot",
    "precision_recall_plot",
]


@passes_kws_to(matshow)
def correlation_matrix_plot(
    cor: Sequence, labels: Sequence = None, title: str = "", **kwargs
) -> Figure:
    """

    :param cor:
    :type cor:
    :param labels:
    :type labels:
    :param title:
    :type title:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    if labels is None:
        labels = [f"P{i}" for i in range(len(cor))]

    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)

    cax = ax1.matshow(cor, **kwargs)
    ax1.grid(True)
    pyplot.title(title)

    ax1.set_xticks(range(len(cor)))
    ax1.set_yticks(range(len(cor)))

    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)

    fig.colorbar(cax, ticks=numpy.arange(-1.1, 1.1, 0.1))

    return fig


@passes_kws_to(imshow)
def horizontal_imshow(
    images: Sequence, titles: Sequence = None, num_columns: int = 4, **kwargs
):
    """Small helper function for creating horizontal subplots with pyplot

    :param images:
    :param titles:
    :param num_columns:
    :param kwargs:
    :return:
    """
    num_d = len(images) / num_columns
    num_d_f = math.floor(num_d)
    if num_d_f != num_d:
        num_d_f += 1

    if titles is None:
        titles = [f"fig{a_}" for a_ in range(len(images))]
    fig, axes = pyplot.subplots(
        num_d_f,
        num_columns,
        squeeze=False,
        sharex="all",
        sharey="all",
        constrained_layout=True,
    )
    figure_axes = []
    for a_ in axes:
        figure_axes.extend(a_)
    for i, image in enumerate(images):
        ax = figure_axes[i]
        if titles:
            ax.set_title(titles[i])
        ax.imshow(image, **kwargs)
    return fig


def biplot(
    scores: Sequence,
    coefficients: numpy.ndarray,
    categories: Iterable = None,
    labels: Sequence = None,
    label_multiplier: float = 1.06,
) -> pyplot.Figure:
    """
       Use only the 2 Principal components.

      :rtype: object
    :param label_multiplier:
    :param categories:
    :param scores:
    :param coefficients:
    :param labels:
    :return:"""
    assert label_multiplier >= 1.0

    fig = pyplot.figure()

    ax1 = fig.add_axes([0.12, 0.1, 0.76, 0.8], label="ax1")
    ax2 = fig.add_axes([0.12, 0.1, 0.76, 0.8], label="ax2", frameon=False)

    ax1.yaxis.tick_left()
    ax1.xaxis.tick_bottom()

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_offset_position("right")
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")

    ax1.spines["right"].set_color("red")
    ax1.spines["top"].set_color("red")

    ax2.tick_params(color="red")

    for ylabel, xlabel in zip(ax2.get_yticklabels(), ax2.get_xticklabels()):
        ylabel.set_color("red")
        xlabel.set_color("red")

    ax1.set_xlabel("$Score_{PC1}$")
    ax1.set_ylabel("$Score_{PC2}$")

    ax2.set_xlabel("$Coefficient_{PC1}$", color="red")
    ax2.set_ylabel("$Coefficient_{PC2}$", color="red")

    ax2.set_xlim(-label_multiplier, label_multiplier)
    ax2.set_ylim(-label_multiplier, label_multiplier)

    xs = scores[:, 0]
    ys = scores[:, 1]
    n = coefficients.shape[0]
    scale_x = 1.0 / (xs.max() - xs.min())
    scale_y = 1.0 / (ys.max() - ys.min())
    ax1.scatter(xs * scale_x, ys * scale_y, c=categories)
    # pyplot.colorbar(ax=ax1)

    for i in range(n):
        ax2.arrow(0, 0, coefficients[i, 0], coefficients[i, 1], color="r", alpha=0.5)

        label = f"$x_{i}$"
        if labels is not None:
            label = labels[i]

        ax2.text(
            coefficients[i, 0] * label_multiplier,
            coefficients[i, 1] * label_multiplier,
            label,
            color="g",
            ha="center",
            va="center",
        )

    return fig


def pca_biplot(
    predictor: Iterable, response: Iterable, labels: Iterable[str] = None
) -> pyplot.Figure:
    """
    produces a pca projection and plot the 2 most significant component score and the component coefficients.

    :param predictor:
    :param response:
    :param labels:
    :return:"""

    scaler = StandardScaler()
    scaler.fit(predictor)

    pca = PCA()

    return biplot(
        pca.fit_transform(scaler.transform(predictor))[:, 0:2],
        numpy.transpose(pca.components_[0:2, :]),
        response,
        labels,
    )


def precision_recall_plot(
    truth,
    score,
    num_classes,
    *,
    num_decimals: int = 2,
    include_thresholds: bool = False,
    y_lim: Tuple[float, float] = (0.0, 1.05),
    figure_size: Tuple[int, int] = (7, 9),
    color_cycle: Iterator = cycle(
        ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"]
    ),
) -> pyplot.Figure:
    """

    # A "micro-average": quantifying score on all classes jointly

    :param truth:
    :param score:
    :param num_classes:
    :param num_decimals:
    :param include_thresholds:
    :param y_lim:
    :param figure_size:
    :param color_cycle:
    :return:
    """
    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds = dict()
    for i in range(num_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(
            truth[:, i], score[:, i]
        )
        average_precision[i] = average_precision_score(truth[:, i], score[:, i])

    precision["micro"], recall["micro"], thresholds["micro"] = precision_recall_curve(
        truth.ravel(), score.ravel()
    )
    average_precision["micro"] = average_precision_score(truth, score, average="micro")

    fig = pyplot.figure(figsize=figure_size)
    f_scores = numpy.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    iso_curves = None
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        iso_curves = pyplot.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        pyplot.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

    lines.extend(iso_curves)
    labels.append("iso-f1 curves")

    lines.extend(pyplot.plot(recall["micro"], precision["micro"], color="gold", lw=2))
    if include_thresholds:
        for p, r, t in zip(precision["micro"], recall["micro"], thresholds["micro"]):
            pyplot.annotate(f"{t:.{num_decimals}f}", (r, p))
    labels.append(
        f"micro-average Precision-recall (area = {average_precision['micro']:0.{num_decimals}f})"
    )

    for i, color in zip(range(num_classes), color_cycle):
        lines.extend(pyplot.plot(recall[i], precision[i], color=color, lw=2))
        if include_thresholds:
            for p, r, t in zip(precision[i], recall[i], thresholds[i]):
                pyplot.annotate(f"{t:.{num_decimals}f}", (r, p))
        labels.append(
            f"Precision-recall for class {i} (area = {average_precision[i]:0.{num_decimals}f})"
        )

    pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")  # plot no skill

    fig = pyplot.gcf()
    fig.subplots_adjust(bottom=0.25)
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim(y_lim)
    pyplot.xlabel("Recall")
    pyplot.ylabel("Precision")
    pyplot.title("Multi-class extension of Precision-Recall curve")
    pyplot.legend(lines, labels, loc=(0, -0.38), prop=dict(size=14))

    return fig


def roc_plot(
    truth,
    score,
    num_classes: int,
    *,
    figure_size: Tuple[int, int] = (8, 8),
    num_decimals: int = 2,
) -> Figure:
    """ """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(truth[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(truth.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = numpy.unique(
        numpy.concatenate([fpr[i] for i in range(num_classes)])
    )  # First aggregate all false positive rates
    mean_tpr = numpy.zeros_like(
        all_fpr
    )  # Then interpolate all ROC curves at this points
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes  # Finally average it and compute AUC
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = pyplot.figure(figsize=figure_size)
    lw = 1
    pyplot.plot([0, 1], [0, 1], "k--", lw=lw)

    pyplot.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.{num_decimals}f})',
        color="deeppink",
        linestyle=":",
        linewidth=lw,
    )

    pyplot.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.{num_decimals}f})',
        color="navy",
        linestyle=":",
        linewidth=lw,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "teal"])
    for i, color in zip(range(num_classes), colors):
        pyplot.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.{num_decimals}f})",
        )

    pyplot.xlim([-0.05, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Multi-class Extension Receiver Operating Characteristic")
    pyplot.legend(loc="lower right")

    return fig


def confusion_matrix_plot(
    truth: Iterable,
    prediction: Iterable,
    category_names: Iterable[str],
    *,
    figure_size: Tuple[int, int] = (8, 8),
    decimals: int = 3,
) -> Figure:
    """

    :param truth:
    :type truth:
    :param prediction:
    :type prediction:
    :param category_names:
    :type category_names:
    :param figure_size:
    :type figure_size:
    :param decimals:
    :type decimals:
    :return:
    :rtype:"""

    def confusion_matrix_figure(
        truth_, prediction_, classes, normalize=False, title=None, cmap=pyplot.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`."""
        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        cm = confusion_matrix(truth_, prediction_)
        unique = unique_labels(
            truth_, prediction_
        )  # Only use the labels that appear in the data
        classes = classes[unique]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, numpy.newaxis]

        fig, ax = pyplot.subplots(figsize=figure_size)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        pyplot.colorbar(im, ax=ax)

        ax.set(
            xticks=numpy.arange(cm.shape[1]),
            yticks=numpy.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )  # Show all ticks

        pyplot.setp(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )  # Rotate the tick labels and set their alignment.

        # Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        return fig

    numpy.set_printoptions(precision=decimals)

    # Plot normalized confusion matrix
    return confusion_matrix_figure(
        truth,
        prediction,
        classes=category_names,
        normalize=True,
        title="Normalized confusion matrix",
    )


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        from sklearn import datasets

        iris = datasets.load_iris()
        x = iris.data
        y = iris.target

        num_classes = len(iris.target_names)

        # Add noisy features to make the problem harder
        random_state = numpy.random.RandomState(0)
        n_samples, n_features = x.shape
        # X_noisy = numpy.c_[X, random_state.randn(n_samples, 200 * n_features)*0.01]

        x_noisy = x

        (X_train, X_test, y_train, y_test) = train_test_split(
            x_noisy,
            label_binarize(y, classes=range(len(iris.target_names))),
            test_size=0.3,
            random_state=0,
            shuffle=True,
        )  # shuffle and split training and test sets

        classifier = OneVsRestClassifier(
            svm.SVC(kernel="linear", probability=True, random_state=random_state)
        )  # Learn to predict each class against the other
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        def sasasasgasgssiasjdijasagsaagdi():
            """ """
            confusion_matrix_plot(
                numpy.argmax(y_test, axis=-1),
                numpy.argmax(y_score, axis=-1),
                category_names=iris.target_names,
            )
            pyplot.show()

        def sasasasgasgssagsaagdi():
            """ """
            roc_plot(y_test, y_score, num_classes)
            pyplot.show()

        def sasasgssagsaagdi():
            """ """
            precision_recall_plot(y_test, y_score, num_classes)
            pyplot.show()

        def sasafgsagdi():
            """ """
            pca_biplot(x, y)
            pyplot.show()

        def sadi():
            """ """
            import pandas

            df = pandas.DataFrame(x, columns=iris.feature_names)
            correlation_matrix_plot(cor=df.corr(), labels=df.columns)
            pyplot.show()

        sasasasgasgssiasjdijasagsaagdi()
        sasasasgasgssagsaagdi()
        sasasgssagsaagdi()
        sasafgsagdi()
        sadi()

    a()
