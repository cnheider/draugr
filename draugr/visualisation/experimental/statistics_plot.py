#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import matplotlib
import numpy
import torch

from apppath import AppPath
from draugr.torch_utilities import to_tensor
from draugr import MetricAggregator

__author__ = "Christian Heider Nielsen"

import csv

from matplotlib import pyplot

from neodroidagent import utilities as U

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

pyplot.ion()

__all__ = ["ma_plot", "simple_plot", "error_plot", "plot_durations"]


def ma_plot(file_name, name):
    with open(file_name, "r") as f:
        agg = MetricAggregator()
        agg_ma = MetricAggregator()

        reader = csv.reader(f, delimiter=" ", quotechar="|")
        for line in reader:
            if line and line[0] != "":
                agg.append(float(line[0][1:-2]))
                ma = agg.calc_moving_average()
                agg_ma.append(ma)

        pyplot.plot(agg_ma.values)
        pyplot.title(name)


def simple_plot(file_name, name="Statistic Name"):
    with open(file_name, "r") as f:
        agg = MetricAggregator()

        reader = csv.reader(f, delimiter=" ", quotechar="|")
        for line in reader:
            agg.append(float(line[0]))

        # pyplot.annotate('local max', xy=(2, 1), xytext=(3, 1.5),            arrowprops=dict(facecolor='black',
        # shrink=0.05),)

        pyplot.plot(agg.values)
        pyplot.title(name)


def error_plot(results, interval=1, file_name=""):
    # if results is notnumpy.ndarray:
    # results =numpy.ndarray(results)

    y = numpy.mean(results, axis=0)
    error = numpy.std(results, axis=0)

    x = range(0, results.shape[1] * interval, interval)
    fig, ax = pyplot.subplots(1, 1, figsize=(6, 5))
    pyplot.xlabel("Time step")
    pyplot.ylabel("Average Reward")
    ax.errorbar(x, y, yerr=error, fmt="-o")
    # pyplot.savefig(file_name + '.png')


def plot_durations(episode_durations):
    pyplot.figure(2)
    pyplot.clf()
    durations_t = to_tensor(episode_durations)
    pyplot.title("Training...")
    pyplot.xlabel("Episode")
    pyplot.ylabel("Duration")
    pyplot.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        pyplot.plot(means.numpy())

    pyplot.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(pyplot.gcf())


if __name__ == "__main__":
    _list_of_files = list(AppPath("NeodroidAgent").user_log.glob("*.csv"))
    _latest_model = max(_list_of_files, key=os.path.getctime)

    # ma_plot(_file_name_1, 'NoCur')
    # ma_plot(_file_name_2, 'Cur')
    # simple_plot(_latest_model)
    GPU_STATS = [0, 92, 3, 2, 5, 644, 34, 36, 423, 421]
    b = [215, 92, 6, 1, 5, 644, 328, 32, 413, 221]
    c = [62, 68, 8, 25, 7, 611, 29, 38, 421, 425]
    d = numpy.array(zip([GPU_STATS, b, c]))
    error_plot(d)

    pyplot.show()
