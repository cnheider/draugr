#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07-02-2021
           """

if __name__ == "__main__":

    import numpy
    import pytest

    from draugr.python_utilities import IgnoreInterruptSignal
    from draugr.drawers import DiscreteScrollPlot
    from draugr.stopping import CaptureEarlyStop

    stopped = False

    def stop() -> None:
        """
        :rtype: None
        """
        global stopped
        stopped = True

    @pytest.mark.skip
    def asidhsa() -> None:
        """
        :rtype: None
        """
        with DiscreteScrollPlot(num_bins=2) as rollout_drawer:
            with IgnoreInterruptSignal():
                with CaptureEarlyStop(stop, verbose=True):
                    while not stopped:
                        rollout_drawer.draw(numpy.random.randn(2))
