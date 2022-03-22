from enum import Enum
from typing import Iterable

from sorcery import assigned_names

__all__ = ["ExtensionEnum", "match_return_code"]

ESC_CHAR = chr(27)


def match_return_code(ret_val, chars: Iterable[str] = ("q", ESC_CHAR)):
    return any(ret_val & 0xFF == ord(c) for c in chars)


class ExtensionEnum(Enum):
    png, exr = assigned_names()
