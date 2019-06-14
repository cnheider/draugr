#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib

from tqdm import tqdm

import draugr
from draugr.writers.writer import Writer

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""


class DraugrWriter(Writer):
    def __init__(self, interval: int = 1):
        super().__init__(interval)

    def _scalar(self, tag: str, value: float, step: int):
        pass


if __name__ == "__main__":

    with DraugrWriter() as w:
        w.scalar("What", 4)

    def train_episodically_old(
        self,
        env,
        test_env,
        *,
        rollouts=2000,
        render=False,
        render_frequency=100,
        stat_frequency=10,
    ) -> TR:

        E = range(1, rollouts)
        E = tqdm(E, f"Episode: {1}", leave=False, disable=not render)

        stats = draugr.StatisticCollection(stats=("signal", "duration", "entropy"))

        for episode_i in E:
            initial_state = env.reset()

            if episode_i % stat_frequency == 0:
                draugr.styled_terminal_plot_stats_shared_x(stats, printer=E.write)

                E.set_description(
                    f"Epi: {episode_i}, "
                    f"Sig: {stats.signal.running_value[-1]:.3f}, "
                    f"Dur: {stats.duration.running_value[-1]:.1f}"
                )

            if render and episode_i % render_frequency == 0:
                signal, dur, entropy, *extras = self.rollout(
                    initial_state, env, render=render
                )
            else:
                signal, dur, entropy, *extras = self.rollout(initial_state, env)

            stats.duration.append(dur)
            stats.signal.append(signal)
            stats.entropy.append(entropy)

            if self.end_training:
                break

        return NOD(model=self._distribution_parameter_regressor, stats=stats)

    def train_episodically_old(
        self,
        _environment,
        *,
        rollouts=10000,
        render=False,
        render_frequency=100,
        stat_frequency=100,
        **kwargs,
    ) -> TR:
        """
      :param _environment:
      :type _environment:,0
      :param rollouts:
      :type rollouts:
      :param render:
      :type render:
      :param render_frequency:
      :type render_frequency:
      :param stat_frequency:
      :type stat_frequency:
      :return:
      :rtype:
    """

        stats = draugr.StatisticCollection(
            stats=("signal", "duration", "td_error", "epsilon")
        )

        E = range(1, rollouts)
        E = tqdm(E, leave=False, disable=not render)

        for episode_i in E:
            initial_state = _environment.reset()

            if episode_i % stat_frequency == 0:
                draugr.styled_terminal_plot_stats_shared_x(stats, printer=E.write)
                E.set_description(
                    f"Epi: {episode_i}, "
                    f"Sig: {stats.signal.running_value[-1]:.3f}, "
                    f"Dur: {stats.duration.running_value[-1]:.1f}, "
                    f"TD Err: {stats.td_error.running_value[-1]:.3f}, "
                    f"Eps: {stats.epsilon.running_value[-1]:.3f}"
                )

            if render and episode_i % render_frequency == 0:
                signal, dur, td_error, *extras = self.rollout(
                    initial_state, _environment, render=render
                )
            else:
                signal, dur, td_error, *extras = self.rollout(
                    initial_state, _environment
                )

            stats.append(signal, dur, td_error, self._current_eps_threshold)

            if self.end_training:
                break

        return TR(model=self._value_model, stats=stats)
