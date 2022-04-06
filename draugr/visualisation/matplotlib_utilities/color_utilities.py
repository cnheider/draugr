from typing import Any

from matplotlib import pyplot

__all__ = ["array_to_color"]


def array_to_color(array: Any, cmap: str = "Oranges") -> object:
    """
    Args:
      array:
      cmap:

    Returns:

    """
    return pyplot.cm.ScalarMappable(cmap=cmap).to_rgba(array)[:, :-1]
