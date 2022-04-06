#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import numpy
import torch
from PIL import Image
from tqdm import tqdm

from draugr.metrics import MetricCollection
from draugr.writers.mixins import ImageWriterMixin
from draugr.writers.terminal.terminal_image_renderer import (
    render_image,
    terminalise_image,
)
from draugr.writers.terminal.terminal_plot import (
    styled_terminal_plot_stats_shared_x,
    terminal_plot,
)
from draugr.writers.writer import Writer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""
__all__ = ["TerminalPlotWriter"]


class TerminalPlotWriter(Writer, ImageWriterMixin):
    """ """

    def image(
        self,
        tag: str,
        data: Union[numpy.ndarray, torch.Tensor, Image.Image],
        step,
        *,
        dataformats: str = "NCHW",
        **kwargs,
    ) -> None:
        """ """
        self.E.write(terminalise_image(render_image(data, scale=(28, 28))))

    def _open(self):
        self.E = tqdm()
        self.values = MetricCollection()
        return self

    def _close(self, exc_type=None, exc_val=None, exc_tb=None):
        return self.E.close()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._stats_tag = "stats"

    def stats(self, value: MetricCollection, step_i: int = None):
        """

            :param step_i:
        :type value: object"""
        if step_i:
            if self.filter(self._stats_tag):
                self._stats(value, self._counter[self._stats_tag])
            self._counter[self._stats_tag] = step_i
        else:
            if self.filter(self._stats_tag):
                self._stats(value, self._counter[self._stats_tag])
            self._counter[self._stats_tag] += 1

    def _stats(self, stats: MetricCollection, step_i):
        styled_terminal_plot_stats_shared_x(stats, printer=self.E.write)
        self.E.set_description(
            f"Epi: {step_i}, "
            f"Sig: {stats.signal.running_value[-1]:.3f}, "
            f"Dur: {stats.duration.running_value[-1]:.1f}, "
            f"TD Err: {stats.td_error.running_value[-1]:.3f}, "
            f"Eps: {stats.epsilon.running_value[-1]:.3f}"
        )

    def _scalar(self, tag: str, value: float, step: int):
        self.values[tag] = value
        # styled_terminal_plot_stats_shared_x(self.values, printer=self.E.write)
        terminal_plot([value], printer=self.E.write)
        self.E.set_description(f"Tag:{tag} Val:{value} Step:{step}")


if __name__ == "__main__":
    with TerminalPlotWriter() as w:
        w.scalar("What", 4)
        w.image("bro", numpy.random.randint(0, 255, (28, 28, 3)), 0)

    '''
def train_episodically_old(self,
env,
test_env,
*,
rollouts=2000,
render=False,
render_frequency=100,
stat_frequency=10,
):

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


def train_episodically_old(self,
_environment,
*,
rollouts=10000,
render=False,
render_frequency=100,
stat_frequency=100,
**kwargs,
):
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

return NOD(model=self._value_model, stats=stats)
'''
