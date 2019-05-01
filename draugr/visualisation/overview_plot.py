#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""


def correlation_matrix_plot(cor, labels=None, title=""):
    from matplotlib import pyplot as plt
    import numpy as np

    if labels is None:
        labels = [f"P{i}" for i in range(len(cor))]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    cax = ax1.matshow(cor)
    ax1.grid(True)
    plt.title(title)

    ax1.set_xticks(range(len(cor)))
    ax1.set_yticks(range(len(cor)))

    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)

    fig.colorbar(cax, ticks=np.arange(-1.1, 1.1, 0.1))


def biplot(X, y, labels=None):
    """
produces a pca projection and plot the 2 most significant component score and the component coefficients.

:param X:
:param y:
:param labels:
:return:
"""

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA()
    x_new = pca.fit_transform(X)

    def pca2_plot(scores, coefficients, skew_label=1.08):
        """
# Call the function. Use only the 2 PCs.

:param scores:
:param coefficients:
:param labels:
:return:
"""

        fig = plt.figure()

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

        ax1.set_xlabel("ScorePCA1")
        ax1.set_ylabel("ScorePCA2")

        ax2.set_xlabel("CoefficientPCA1", color="red")
        ax2.set_ylabel("CoefficientPCA2", color="red")

        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)

        xs = scores[:, 0]
        ys = scores[:, 1]
        n = coefficients.shape[0]
        scale_x = 1.0 / (xs.max() - xs.min())
        scale_y = 1.0 / (ys.max() - ys.min())
        ax1.scatter(xs * scale_x, ys * scale_y, c=y)

        for i in range(n):
            ax2.arrow(
                0, 0, coefficients[i, 0], coefficients[i, 1], color="r", alpha=0.5
            )

            label = f"P{i}"
            if labels is not None:
                label = labels[i]

            ax2.text(
                coefficients[i, 0] * skew_label,
                coefficients[i, 1] * skew_label,
                label,
                color="g",
                ha="center",
                va="center",
            )

    pca2_plot(x_new[:, 0:2], np.transpose(pca.components_[0:2, :]))


def a():
    import numpy as np

    x1 = np.arange(10)
    y1 = x1 ** 2
    x2 = np.arange(100, 200)
    y2 = x2


def plot_cf(y_pred, y_test, class_names, size=(8, 8), decimals=3):
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    def plot_confusion_matrix(
        y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
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
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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

    np.set_printoptions(precision=decimals)

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix",
    )


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt

    my_csv = "/home/heider/Data/Datasets/DecisionSupportSystems/Boston.csv"

    df = pd.read_csv(my_csv, index_col=0)
    cor = df.corr()

    correlation_matrix_plot(cor, labels=df.columns)
    plt.show()

    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    biplot(X, y)
    plt.show()

    a()
