#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """

from pathlib import Path

import torch
from pytorch_memlab import LineProfiler

from apppath import ensure_existence

if __name__ == "__main__":

    def inner():
        torch.nn.Linear(100, 100).cuda()

    def outer():
        linear = torch.nn.Linear(100, 100).cuda()
        linear2 = torch.nn.Linear(100, 100).cuda()
        inner()

    with LineProfiler(outer, inner) as prof:
        outer()

    with open(ensure_existence(Path("exclude")) / "test.html", "w") as f:
        f.write(prof.display()._repr_html_())
