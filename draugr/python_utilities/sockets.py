#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18/03/2020
           """
__all__ = ["get_host_ip"]


def get_host_ip():
    """Get host ip.

Returns:
str: The obtained ip. UNKNOWN if failed.
"""
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


if __name__ == "__main__":
    HOST_IP = get_host_ip()
