#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import pathlib

with open(pathlib.Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ = this_init_file.read()

from .system import *
from .datasets import *
from .distributions import *
from .generators import *
from .images import *
from .operations import *
from .optimisation import *
from .persistence import *
from .tensors import *
from .writers import *
from .sessions import *
from .evaluation import *

if __name__ == "__main__":
    print(__doc__)
