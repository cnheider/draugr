#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["launch_tensorboard"]

from pathlib import Path

from tensorboard import program  # IMPORT OUT HERE; POSSIBLE RAISE CONDITIONS


def launch_tensorboard(log_dir: Path, port: int = 6006) -> str:
    """

    :param log_dir:
    :type log_dir:
    :param port:
    :type port:
    :return:
    :rtype:"""

    def port_is_in_use(port_):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port_)) == 0

    if port_is_in_use(port):
        raise RuntimeError(f"Port {port} is in use")

    tb = program.TensorBoard()
    # tb.configure(argv=['', '--logdir', log_dir, '--port', port])
    tb.configure(logdir=str(log_dir), port=port)
    return tb.launch()
