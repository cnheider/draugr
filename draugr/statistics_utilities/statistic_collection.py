#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"
import statistics as S

from .statistic_aggregator import StatisticAggregator

MEASURES = S.__all__[1:]


class StatisticCollection(dict):
    def __init__(
        self,
        stats=("signal", "length"),
        measures=MEASURES,
        keep_measure_history=True,
        use_disk_cache=True,
    ):
        super().__init__()
        self._statistics = {}
        self._measures = measures
        self._keep_measure_history = keep_measure_history
        self._use_disk_cache = use_disk_cache

        for stat in stats:
            self._statistics[stat] = StatisticAggregator(
                measures=self._measures,
                keep_measure_history=self._keep_measure_history,
                use_disk_cache=self._use_disk_cache,
            )

    def add_statistic(self, stat_name):
        self._statistics[stat_name] = StatisticAggregator(
            measures=self._measures, keep_measure_history=self._keep_measure_history
        )

    def append(self, *args, **kwargs):
        for (arg, (k, v)) in zip(args, self._statistics.items()):
            self._statistics[k].append(arg)

        for (k, v) in kwargs:
            self._statistics[k].append(v)

    def remove_statistic(self, stat_name):
        del self._statistics[stat_name]

    def __len__(self):
        return len(self._statistics)

    @property
    def statistics(self):
        return self._statistics

    def __getattr__(self, stat_name):
        if stat_name in self._statistics:
            return self._statistics[stat_name]
        else:
            raise AttributeError

    def __repr__(self):
        return f"<StatisticCollection> {self._statistics} </StatisticCollection>"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self.statistics

    def __getitem__(self, item):
        return self.statistics[item]

    def keys(self):
        return self.statistics.keys()

    def __contains__(self, item):
        return item in self.statistics

    def items(self):
        return self.statistics.items()

    def save(self, **kwargs):
        for key, value in self._statistics.items():
            value.save(stat_name=key, **kwargs)


if __name__ == "__main__":
    stats = StatisticCollection(keep_measure_history=False)
    stats2 = StatisticCollection(keep_measure_history=True)

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
