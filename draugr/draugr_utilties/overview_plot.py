#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"
__doc__ = ""


def correlation_matrix_plot(
    cor,  # pairwise correlation
    labels=(
        "Sex",
        "Length",
        "Diam",
        "Height",
        "Whole",
        "Shucked",
        "Viscera",
        "Shell",
        "Rings",
    ),
    title="Abalone Feature Correlation",
):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    import numpy as np

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap("jet", 30)
    cax = ax1.imshow(cor, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title(title)

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

    def pca2_plot(scores, coefficients):
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

            label = f"P{i + 1}"
            if labels is not None:
                label = labels[i]

            ax2.text(
                coefficients[i, 0] * 1.05,
                coefficients[i, 1] * 1.05,
                label,
                color="g",
                ha="center",
                va="center",
            )

    pca2_plot(x_new[:, 0:2], np.transpose(pca.components_[0:2, :]))


def a():
    import numpy as np
    import matplotlib.pyplot as plt

    x1 = np.arange(10)
    y1 = x1 ** 2
    x2 = np.arange(100, 200)
    y2 = x2


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt

    my_csv = "/home/heider/Data/Datasets/DecisionSupportSystems/Boston.csv"

    df = pd.read_csv(my_csv, index_col=0)
    cor = df.corr()

    correlation_matrix_plot(cor)
    plt.show()

    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    biplot(X, y)
    plt.show()

    a()
