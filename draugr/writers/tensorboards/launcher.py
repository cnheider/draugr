#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"
__doc__ = ""


def launch_tensorboard(log_dir):
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir])
    return tb.launch()
