import os
from pathlib import Path
from typing import Iterable


from enum import Enum
import numpy
from sorcery import assigned_names

__all__ = ["ExtensionEnum", "ret_val_comp"]

ESC_CHAR = chr(27)


def ret_val_comp(ret_val, chars: Iterable[str] = ("q", ESC_CHAR)):
    return any(ret_val & 0xFF == ord(c) for c in chars)


class ExtensionEnum(Enum):
    png, exr = assigned_names()
