#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import csv

from matplotlib import pyplot

from draugr import MetricAggregator
from neodroidagent import utilities as U

# print(pyplot.style.available)
plot_style = "fivethirtyeight"
# plot_style='bmh'
# plot_style='ggplot'
pyplot.style.use("seaborn-poster")
pyplot.style.use(plot_style)
pyplot.rcParams["axes.edgecolor"] = "#ffffff"
pyplot.rcParams["axes.facecolor"] = "#ffffff"
pyplot.rcParams["figure.facecolor"] = "#ffffff"
pyplot.rcParams["patch.edgecolor"] = "#ffffff"
pyplot.rcParams["patch.facecolor"] = "#ffffff"
pyplot.rcParams["savefig.edgecolor"] = "#ffffff"
pyplot.rcParams["savefig.facecolor"] = "#ffffff"
pyplot.rcParams["xtick.labelsize"] = 16
pyplot.rcParams["ytick.labelsize"] = 16

# set up matplotlib
is_ipython = "inline" in pyplot.get_backend()
if is_ipython:
    pass

pyplot.ion()

__all__ = ["simple_plot"]


def simple_plot(file_path, name="Statistic Name"):
    with open(file_path, "r") as f:
        agg = MetricAggregator()

        reader = csv.reader(f, delimiter=" ", quotechar="|")
        for line in reader:
            agg.append(float(line[0]))

        pyplot.plot(agg.values)
        pyplot.title(name)

        pyplot.show()


if __name__ == "__main__":
    #  import configs.base_config as C

    # _list_of_files = list(C.LOG_DIRECTORY.glob('*.csv'))
    # _latest_model = max(_list_of_files, key=os.path.getctime)

    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    # import easygui
    # print easygui.fileopenbox()

    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    file_path = (
        askopenfilename()
    )  # show an "Open" dialog box and return the path to the selected file
    file_name = file_path.split("/")[-1]
    simple_plot(file_path, file_name)
