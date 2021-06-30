#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import statistics

from draugr.metrics.metric_aggregator import MetricAggregator

MEASURES = statistics.__all__[1:]

__all__ = ["MetricCollection"]


class MetricCollection(dict):
    """ """

    def __init__(
        self,
        metrics=("signal", "length"),
        measures=MEASURES,
        keep_measure_history=True,
        use_disk_cache=True,
    ):
        super().__init__()
        self._metrics = {}
        self._measures = measures
        self._keep_measure_history = keep_measure_history
        self._use_disk_cache = use_disk_cache

        for metric in metrics:
            self._metrics[metric] = MetricAggregator(
                measures=self._measures,
                keep_measure_history=self._keep_measure_history,
                use_disk_cache=self._use_disk_cache,
            )

    def add_metric(self, name):
        """

        :param name:
        :type name:"""
        self._metrics[name] = MetricAggregator(
            measures=self._measures, keep_measure_history=self._keep_measure_history
        )

    def append(self, *args, **kwargs):
        """

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:"""
        for (arg, (k, v)) in zip(args, self._metrics.items()):
            self._metrics[k].append(arg)

        for (k, v) in kwargs:
            self._metrics[k].append(v)

    def remove_metric(self, name):
        """

        :param name:
        :type name:"""
        del self._metrics[name]

    def __len__(self):
        return len(self._metrics)

    @property
    def metrics(self):
        """

        :return:
        :rtype:"""
        return self._metrics

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __repr__(self):
        return f"<StatisticCollection> {self._metrics} </StatisticCollection>"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self.metrics

    def __getitem__(self, name):
        if name in self._metrics:
            return self._metrics[name]
        else:
            # return self.add_metric(name)
            raise AttributeError

    # def __setitem__(self, key, value):
    #     if key in self._metrics:
    #         if self._keep_measure_history:
    #             self._metrics[key].append(value)
    #         else:
    #             self._metrics[key] = value
    #     else:
    #         self.add_metric(key)
    #         self.append({key:value})

    def keys(self):
        """

        :return:
        :rtype:"""
        return self.metrics.keys()

    def __contains__(self, item):
        return item in self.metrics

    def items(self):
        """

        :return:
        :rtype:"""
        return self.metrics.items()

    def save(self, **kwargs):
        """

        :param kwargs:
        :type kwargs:"""
        for key, value in self._metrics.items():
            value.save(stat_name=key, **kwargs)


if __name__ == "__main__":
    stats = MetricCollection(keep_measure_history=False)
    stats2 = MetricCollection(keep_measure_history=True)

    for i in range(10):
        stats.signal.append(i)
        stats2.signal.append(i)

    print(stats)
    print(stats.signal)
    print(stats.length)
    print(stats.length.measures)
    print(stats.signal.measures)
    print(stats.signal.variance)
    print(stats.signal.calc_moving_average())
    print(stats.signal.max)
    print(stats.signal.min)
    print("\n")
    print(stats2)
    print(stats2.signal.min)
