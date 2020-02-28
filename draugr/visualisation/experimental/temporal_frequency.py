#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 24/02/2020
           """

"""
tx, ty = load_action_data(folder, smooth, bin_size[3])

ax3.set_title('Action Selection Frequency(%) vs Timestep')

if tx is not None or ty is not None:
  ax3.set_ylabel('Action Selection Frequency(%)')
  labels = ['Action {}'.format(i) for i in range(ty.shape[0])]
  p3 = ax3.stackplot(tx, ty, labels=labels)

  base = 0.0
  for percent, index in zip(ty, range(ty.shape[0])):
    offset = base + percent[-1]/3.0
    ax3.annotate(str('{:.2f}'.format(ty[index][-1])), xy=(tx[-1], offset), color=p3[index].get_facecolor().ravel())
    base += percent[-1]

  #ax3.yaxis.label.set_color(p3.get_color())
  #ax3.tick_params(axis='y', colors=p3.get_color())

  ax3.legend(loc=4) #remake g2 legend because we have a new line

plt.tight_layout() # prevent label cutoff
"""


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def a():
        x = [1, 2, 3, 4, 5]
        y1 = [1, 1, 2, 3, 5]
        y2 = [0, 4, 2, 6, 8]
        y3 = [1, 3, 5, 7, 9]

        y = np.vstack([y1, y2, y3])

        labels = ["Fibonacci ", "Evens", "Odds"]

        fig, ax = plt.subplots()
        ax.stackplot(x, y1, y2, y3, labels=labels)
        ax.legend(loc="upper left")
        plt.show()

        fig, ax = plt.subplots()
        ax.stackplot(x, y)
        plt.show()

    def b():
        def layers(n, m):
            """
      Return *n* random Gaussian mixtures, each of length *m*.
      """

            def bump(a):
                x = 1 / (0.1 + np.random.random())
                y = 2 * np.random.random() - 0.5
                z = 10 / (0.1 + np.random.random())
                for i in range(m):
                    w = (i / m - y) * z
                    a[i] += x * np.exp(-w * w)

            a = np.zeros((m, n))
            for i in range(n):
                for j in range(5):
                    bump(a[:, i])
            return a

        d = layers(3, 100)

        fig, ax = plt.subplots()
        ax.stackplot(range(100), d.T, baseline="wiggle")
        plt.show()

    a()
    b()
