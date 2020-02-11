#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Sequence, Iterable, Tuple

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from matplotlib.pyplot import matshow, imshow
from itertools import cycle

from numpy import interp

from warg import passes_kws_to
from matplotlib import pyplot
import numpy
from sklearn import svm
from sklearn.datasets import make_classification
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.multiclass import unique_labels

__all__ = [
    "correlation_matrix_plot",
    "horizontal_imshow",
    "biplot",
    "plot_confusion_matrix",
    "precision_recall_plt2",
    "pca_biplot",
    "roc_plot",
    "precision_recall_plt",
]


@passes_kws_to(matshow)
def correlation_matrix_plot(cor, labels=None, title="", **kwargs):
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


@passes_kws_to(imshow)
def horizontal_imshow(
    images: Sequence, titles: Sequence = None, columns: int = 4, **kwargs
):
    """Small helper function for creating horizontal subplots with pyplot"""
    sadasf = len(images) / columns
    ssa = math.floor(sadasf)
    if ssa != sadasf:
        ssa += 1

    if titles is None:
        titles = [f"fig{a}" for a in range(len(images))]
    fig, axes = pyplot.subplots(
        ssa, columns, squeeze=False, sharex="all", sharey="all", constrained_layout=True
    )
    faxes = []
    for a in axes:
        faxes.extend(a)
    for i, image in enumerate(images):
        ax = faxes[i]
        if titles:
            ax.set_title(titles[i])
        ax.imshow(image, **kwargs)
    return fig


def biplot(
    scores: Sequence,
    coefficients: numpy.ndarray,
    categories: Sequence = None,
    labels: Sequence = None,
    label_multiplier: float = 1.06,
):
    """
# Call the function. Use only the 2 PCs.

:param label_multiplier:
:param categories:
:param scores:
:param coefficients:
:param labels:
:return:
"""
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


def pca_biplot(X, y, labels=None):
    """
produces a pca projection and plot the 2 most significant component score and the component coefficients.

:param X:
:param y:
:param labels:
:return:
"""

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA()
    x_new = pca.fit_transform(X)

    biplot(x_new[:, 0:2], numpy.transpose(pca.components_[0:2, :]), y, labels)


def precision_recall_plt2():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Use label_binarize to be multi-label like settings
    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]

    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=random_state
    )

    # Run classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
    classifier.fit(X_train, Y_train)
    y_score = classifier.decision_function(X_test)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        Y_test, y_score, average="micro"
    )
    print(
        "Average precision score, micro-averaged over all classes: {0:0.2f}".format(
            average_precision["micro"]
        )
    )

    from itertools import cycle

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    pyplot.figure(figsize=(7, 8))
    f_scores = numpy.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    l = None
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = pyplot.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        pyplot.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    l, = pyplot.plot(recall["micro"], precision["micro"], color="gold", lw=2)
    lines.append(l)
    labels.append(
        "micro-average Precision-recall (area = {0:0.2f})"
        "".format(average_precision["micro"])
    )

    for i, color in zip(range(n_classes), colors):
        l, = pyplot.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class {0} (area = {1:0.2f})"
            "".format(i, average_precision[i])
        )

    fig = pyplot.gcf()
    fig.subplots_adjust(bottom=0.25)
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("Recall")
    pyplot.ylabel("Precision")
    pyplot.title("Extension of Precision-Recall curve to multi-class")
    pyplot.legend(lines, labels, loc=(0, -0.38), prop=dict(size=14))

    pyplot.show()


def precision_recall_plt(y, yhat, n_classes=2):
    # generate 2 class dataset
    X, y = make_classification(
        n_samples=1000, n_classes=2, weights=[1, 1], random_state=1
    )
    # split into train/test sets
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    # fit a model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(trainX, trainy)

    # predict probabilities
    probs = model.predict_proba(testX)

    # keep probabilities for the positive outcome only
    probs = probs[:, 1]

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(testy, probs)

    # calculate precision-recall AUC
    auc_v = auc(recall, precision)

    # plot no skill
    pyplot.plot([0, 1], [0.5, 0.5], linestyle="--")
    # plot the precision-recall curve for the model
    pyplot.plot(recall, precision, marker=".")

    for p, r, t in zip(precision, recall, thresholds):
        pyplot.annotate(f"{t:.2f}", (r, p))

    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.45, 1.05])
    pyplot.xlabel("Precision")
    pyplot.ylabel("Recall")
    pyplot.title(f"Multi-class extension Precision Recall {auc_v} (Varying threshold)")
    pyplot.legend(loc="lower right")


def roc_plot(
    y_test, y_pred, n_classes: int, size: Tuple[int, int] = (8, 8), decimals: int = 3
):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = numpy.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    pyplot.figure(figsize=size)
    lw = 1
    pyplot.plot([0, 1], [0, 1], "k--", lw=lw)

    pyplot.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
        color="deeppink",
        linestyle=":",
        linewidth=lw,
    )

    pyplot.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
        color="navy",
        linestyle=":",
        linewidth=lw,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "teal"])
    for i, color in zip(range(n_classes), colors):
        pyplot.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
        )

    pyplot.xlim([-0.05, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("multi-class extension Receiver operating characteristic")
    pyplot.legend(loc="lower right")


def plot_confusion_matrix(
    y_test, y_pred, class_names, size: Tuple[int, int] = (8, 8), decimals: int = 3
):
    def confusion_matrix_figure(
        y_true, y_pred, classes, normalize=False, title=None, cmap=pyplot.cm.Blues
    ):
        """
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        unique = unique_labels(y_true, y_pred)
        classes = classes[unique]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, numpy.newaxis]

        fig, ax = pyplot.subplots(figsize=size)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.barh.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=numpy.arange(cm.shape[1]),
            yticks=numpy.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        pyplot.setp(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

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
        return ax

    numpy.set_printoptions(precision=decimals)

    # Plot normalized confusion matrix
    confusion_matrix_figure(
        y_test,
        y_pred,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix",
    )


if __name__ == "__main__":

    # df = pd.read_csv(my_csv, index_col=0)
    # cor = df.corr()

    # correlation_matrix_plot(cor, labels=df.columns)
    # pyplot.show()

    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    pca_biplot(X, y)
    pyplot.show()

    # Binarise the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = numpy.random.RandomState(0)
    n_samples, n_features = X.shape
    X = numpy.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=random_state)
    )
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # plot_cf(,y,class_names=df.columns)
    # pyplot.show()

    # roc_plot(numpy.argmax(y_score, axis=-1), y_test, n_classes)
    # pyplot.show()

    precision_recall_plt(y_score, y_test, n_classes)
    pyplot.show()
