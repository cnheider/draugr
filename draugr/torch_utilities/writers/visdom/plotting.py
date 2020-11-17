#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

import numpy
import visdom

vis = visdom.Visdom()

__all__ = ["plot_episode_stats"]


def plot_episode_stats(stats):
    """

    :param stats:
    :type stats:
    :return:
    :rtype:"""
    vis.line(
        X=numpy.arange(len(stats.signal_mas)),
        Y=numpy.array(stats.signal_mas),
        win="DDPG MEAN REWARD (100 episodes)",
        opts=dict(
            title=("DDPG MEAN REWARD (100 episodes)"),
            ylabel="MEAN REWARD (100 episodes)",
            xlabel="Episode",
        ),
    )  # Plot the mean of last 100 episode rewards over time.

    vis.line(
        X=numpy.cumsum(stats.episode_lengths),
        Y=numpy.arange(len(stats.episode_lengths)),
        win="DDPG Episode per time step",
        opts=dict(
            title=("DDPG Episode per time step"), ylabel="Episode", xlabel="Time Steps"
        ),
    )  # Plot time steps and episode number.
