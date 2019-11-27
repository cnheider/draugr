#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"


# http://localhost:8097
# python -m visdom.server


def run_visdom_server():
    """
Will start a visdom server
"""
    server.main()


if __name__ == "__main__":
    import visdom.server as server

    run_visdom_server()
