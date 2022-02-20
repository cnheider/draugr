#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18/03/2020
           """
__all__ = ["get_host_ip", "find_unclaimed_port", "is_port_in_use"]


def get_host_ip() -> str:
    """Get host ip.

    Returns:
    str: The obtained ip. UNKNOWN if failed."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "UNKNOWN"
    finally:
        s.close()

    return ip


def find_unclaimed_port() -> int:
    """
    # NOTE: there is still a chance the port could be taken by other processes.
    :return:
    :rtype:"""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(
        ("", 0)
    )  # Binding to port 0 will cause the OS to find an available port for us
    port = sock.getsockname()[1]
    sock.close()
    return port


def is_port_in_use(port: int) -> bool:
    """

    :param port_:
    :return:
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


if __name__ == "__main__":
    HOST_IP = get_host_ip()
