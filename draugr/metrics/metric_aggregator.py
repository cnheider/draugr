#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List
from warnings import warn

__author__ = "Christian Heider Nielsen"

import statistics

__all__ = ["MetricAggregator", "save_metric"]


class MetricAggregator(object):
    """ """

    def __init__(
        self,
        measures=statistics.__all__[1:],
        keep_measure_history=False,
        use_disk_cache=True,
    ):
        self._values = []
        self._length = 0

        self._running_value = None
        self._running_value_key = "running_value"

        # for key in self._measure_keys:
        #  setattr(self,key,None)

        self._stat_measure_keys = measures
        self._keep_measure_history = keep_measure_history
        if self._keep_measure_history:
            self._measures = {}
            for key in self._stat_measure_keys:
                self._measures[key] = []
            self._measures[self._running_value_key] = []

    @property
    def values(self):
        """

        :return:
        :rtype:"""
        return self._values

    @property
    def max(self):
        """

        :return:
        :rtype:"""
        return max(self._values)

    @property
    def min(self):
        """

        :return:
        :rtype:"""
        return min(self._values)

    @property
    def measures(self):
        """

        :return:
        :rtype:"""
        if self._keep_measure_history:
            return self._measures
        else:
            out = {}
            for key in self._stat_measure_keys:
                try:
                    val = getattr(statistics, key)(self._values)
                except statistics.StatisticsError as e:
                    # TODO: warn(f'{e}')
                    val = None
                out[key] = val
            return out

    def add(self, values):
        """

        :param values:
        :type values:"""
        self.append(values)

    def append(self, values):
        """

        :param values:
        :type values:"""
        self._values.append(values)
        if type is list:
            self._length += len(values)
        else:
            self._length += 1

        self.calc_running_value(values)

        if self._keep_measure_history:
            for key in self._stat_measure_keys:
                if self._length > 1:
                    try:
                        val = getattr(statistics, key)(self._values)
                    except:
                        val = None
                    self._measures[key].append(val)
                else:
                    # warn(f'Length of statistical values are <=1, measure "{key}" maybe ill-defined')
                    try:
                        val = getattr(statistics, key)(self._values)
                    except statistics.StatisticsError as e:
                        # TODO: warn(f'{e}')
                        val = None
                    self._measures[key].append(val)

    # def __setitem__(self, key, value):
    #    if self._keep_measure_history:
    #        self.

    def __getitem__(self, item):
        return self._values[item]

    def __contains__(self, item):
        return self._values[item]

    def __iter__(self):
        return self._values

    def __getattr__(self, item):
        if item in self._stat_measure_keys:
            if self._length > 1:
                if self._keep_measure_history:
                    return self._measures[item]
                else:
                    try:
                        return getattr(statistics, item)(self._values)
                    except statistics.StatisticsError as e:
                        warn(f"{e}")
                        return None
            else:
                warn(
                    f'Length of statistical values are <=1, measure "{item}" maybe ill-defined'
                )
                try:
                    return getattr(statistics, item)(self._values)
                except statistics.StatisticsError as e:
                    warn(f"{e}")
                    return None
        elif item == self._running_value_key:
            return self._measures[item]
        else:
            raise AttributeError

    # def __call__(self, *args, **kwargs):
    #  return self._values

    def __repr__(self):
        return f"<StatisticAggregator> values: {self._values}, measures: {self.measures} </StatisticAggregator>"

    def __str__(self):
        return str(self._values)

    def __len__(self):
        return len(self._values)

    def calc_moving_average(self, window_size=100):
        """

        :param window_size:
        :type window_size:
        :return:
        :rtype:"""
        if self._length >= window_size:
            return statistics.mean(self._values[-window_size:])
        elif self._length > 0:
            return statistics.mean(self._values)
        else:
            return 0

    def calc_running_value(self, new_val=None, *, lambd=0.99):
        """

        :param new_val:
        :type new_val:
        :param lambd:
        :type lambd:
        :return:
        :rtype:"""
        if new_val is None:
            return self._running_value

        if new_val is list:
            for nw in new_val:
                self.calc_running_value(nw, lambd=lambd)

        if self._running_value:
            self._running_value = self._running_value * lambd + new_val * (1 - lambd)
        else:
            self._running_value = new_val

        if self._keep_measure_history:
            self._measures[self._running_value_key].append(self._running_value)

        return self._running_value

    def save(
        self, *, stat_name, project_name="non", config_name="non", directory="logs"
    ):
        """

        :param stat_name:
        :type stat_name:
        :param project_name:
        :type project_name:
        :param config_name:
        :type config_name:
        :param directory:
        :type directory:"""
        save_metric(
            self._values,
            metric_name=stat_name,
            project_name=project_name,
            config_name=config_name,
            directory=directory,
        )


def save_metric(
    metric: List[MetricAggregator],
    *,
    metric_name,
    project_name,
    config_name,
    directory=Path("logs"),
) -> bool:
    """

    :param metric:
    :type metric:
    :param metric_name:
    :type metric_name:
    :param project_name:
    :type project_name:
    :param config_name:
    :type config_name:
    :param directory:
    :type directory:
    :return:
    :rtype:"""
    import csv
    import datetime

    if metric:
        _file_date = datetime.datetime.now()
        _file_name = (
            f'{project_name}-{config_name.replace(".", "_")}-'
            f'{_file_date.strftime("%y%m%d%H%M")}.{metric_name}.csv'
        )
        _file_path = directory / _file_name

        stat = [[s] for s in metric]
        with open(_file_path, "w") as f:
            w = csv.writer(f)
            w.writerows(stat)
        print(f"Saved metric at {_file_path}")

        return True
    return False


if __name__ == "__main__":
    signals = MetricAggregator(keep_measure_history=False)

    for i in range(10):
        signals.append(i)

    print(signals)
    print(signals.measures)
    print(signals.variance)
    print(signals.calc_moving_average())
    print(signals.max)
    print(signals.min)
