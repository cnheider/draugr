#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["launch_tensorboard"]

from pathlib import Path


def launch_tensorboard(log_dir: Path, port: int = 6006) -> str:
    """

    :param log_dir:
    :type log_dir:
    :param port:
    :type port:
    :return:
    :rtype:
    """
    from tensorboard import program

    tb = program.TensorBoard()
    # tb.configure(argv=['', '--logdir', log_dir, '--port', port])
    tb.configure(logdir=str(log_dir), port=port)
    return tb.launch()
