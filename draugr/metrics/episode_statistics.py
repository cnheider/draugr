#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"
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
    # fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pandas.Series(stats.episode_rewards).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
    # plt.plot(rewards_smoothed)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Reward (Smoothed)")
    # plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # if noshow:
    #    plt.close(fig2)
    # else:
    #    plt.show(fig2)
    return rewards_smoothed

  def per_s(self, stats):  # Plot time steps and episode number
    sb = numpy.cumsum(stats.episode_lengths), numpy.arange(len(stats.episode_lengths))
    # fig3 = plt.figure(figsize=(10,5))
    # plt.plot(sb)
    # plt.xlabel("Time Steps")
    # plt.ylabel("Episode")
    # plt.title("Episode per time step")
    # if noshow:
    #    plt.close(fig3)
    # else:
    #    plt.show(fig3)
    return sb
  """
