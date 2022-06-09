from typing import Sequence, List

import numpy


def scale(x: Sequence, length: float) -> List[int]:
    """
    Scale points in 'x', such that distance between
    max(x) and min(x) equals to 'length'. min(x)
    will be moved to 0."""
    if type(x) is list:
        s = float(length) / (max(x) - min(x)) if x and max(x) - min(x) != 0 else length
    # elif type(x) is range:
    #  s = length
    else:
        s = (
            float(length) / (numpy.max(x) - numpy.min(x))
            if len(x) and numpy.max(x) - numpy.min(x) != 0
            else length
        )

    return [int((i - min(x)) * s) for i in x]
