#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["launch_tensorboard"]


def launch_tensorboard(log_dir, port=6006):
    from tensorboard import program

    tb = program.TensorBoard()
    # tb.configure(argv=['', '--logdir', log_dir, '--port', port])
    tb.configure(logdir=log_dir, port=port)
    return tb.launch()
