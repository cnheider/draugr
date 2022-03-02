#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["launch_tensorboard"]

from pathlib import Path

from tensorboard import program  # IMPORT OUT HERE; POSSIBLE RAISE CONDITIONS

from draugr.python_utilities.sockets import is_port_in_use


def launch_tensorboard(log_dir: Path, port: int = 6006) -> str:
    """

    :param log_dir:
    :type log_dir:
    :param port:
    :type port:
    :return:
    :rtype:"""

    if is_port_in_use(port):
        raise RuntimeError(f"Port {port} is in use")

    tb = program.TensorBoard()
    # tb.configure(argv=['', '--logdir', log_dir, '--port', port])
    tb.configure(logdir=str(log_dir), port=port)
    tb._fix_mime_types()
    return tb.launch()
