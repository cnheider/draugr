#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"
"""
Description: Visualisation
Author: Christian Heider Nielsen
"""
import numpy

__all__ = ["update_visualiser"]


def update_visualiser(
    visualiser,
    episode,
    loss,
    moving_loss,
    signal,
    moving_reward,
    episode_length,
    moving_length,
    rgb_array,
    windows,
    configuration,
):
    """

    :param moving_loss:
    :param signal:
    :param moving_reward:
    :param moving_length:
    :param configuration:
    :param visualiser:
    :param episode:
    :param loss:
    :param episode_length:
    :param rgb_array:
    :param windows:
    :return:"""
    if "value" in windows:
        loss_window = windows["value"]
        visualiser.line(
            X=numpy.array([episode]),
            Y=numpy.array([moving_loss]),
            win=loss_window,
            env=configuration.CONFIG_NAME,
            update="append",
        )
    else:
        windows["value"] = visualiser.line(
            X=numpy.array([episode]),
            Y=numpy.array([moving_loss]),
            env=configuration.CONFIG_NAME,
            opts={"title": "Average Episode Q Value Loss"},
        )

    if "signal" in windows:
        reward_window = windows["signal"]
        visualiser.line(
            X=numpy.array([episode]),
            Y=numpy.array([moving_reward]),
            win=reward_window,
            env=configuration.CONFIG_NAME,
            update="append",
        )
    else:
        windows["signal"] = visualiser.line(
            X=numpy.array([episode]),
            Y=numpy.array([moving_reward]),
            env=configuration.CONFIG_NAME,
            opts={"title": "Average Episode Reward"},
        )

    if "episode_length" in windows:
        episode_window = windows["episode_length"]
        visualiser.line(
            X=numpy.array([episode]),
            Y=numpy.array([moving_length]),
            win=episode_window,
            env=configuration.CONFIG_NAME,
            update="append",
        )
    else:
        windows["episode_length"] = visualiser.line(
            X=numpy.array([episode]),
            Y=numpy.array([moving_length]),
            env=configuration.CONFIG_NAME,
            opts={"title": "Episode Length"},
        )

    # if 'rgb_array' in windows:
    #  rgb_array_window = windows['rgb_array']
    #  visualiser.image(rgb_array, win=rgb_array_window,
    #                   env=configuration.CONFIG_NAME)
    # else:
    #  windows['rgb_array'] = visualiser.image(rgb_array,
    #                                          env=configuration.CONFIG_NAME,
    #                                          opts={'title'  : 'Episode Ending',
    #                                                'caption': 'Episode
    # Ending'})

    return windows
