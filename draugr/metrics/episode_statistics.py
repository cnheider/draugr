#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
import numpy


class EpisodeStatistics(object):
    """
Episodic Statistics data container
"""

    def __init__(self):
        super().__init__()

    durations = []
    signals = []

    def moving_average(self, window_size=100):
        signal_ma = numpy.mean(self.signals[-window_size:])
        duration_ma = numpy.mean(self.durations[-window_size:])
        return signal_ma, duration_ma

    """
def smoothed(self, stats, smoothing_window=10):  # Plot the episode reward over time
import pandas
# fig2 = pyplot.figure(figsize=(10,5))
rewards_smoothed = pandas.Series(stats.episode_rewards).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
# pyplot.plot(rewards_smoothed)
# pyplot.xlabel("Episode")
# pyplot.ylabel("Episode Reward (Smoothed)")
# pyplot.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
# if noshow:
#    pyplot.close(fig2)
# else:
#    pyplot.show(fig2)
return rewards_smoothed

def per_s(self, stats):  # Plot time steps and episode number
sb = numpy.cumsum(stats.episode_lengths), numpy.arange(len(stats.episode_lengths))
# fig3 = pyplot.figure(figsize=(10,5))
# pyplot.plot(sb)
# pyplot.xlabel("Time Steps")
# pyplot.ylabel("Episode")
# pyplot.title("Episode per time step")
# if noshow:
#    pyplot.close(fig3)
# else:
#    pyplot.show(fig3)
return sb
"""


if __name__ == "__main__":
    EpisodeStatistics()
