from pathlib import Path
from typing import Union
import cv2
import numpy
from draugr.opencv_utilities.enums import WindowFlagEnum, FontEnum

__all__ = ["show_image"]

from warg import get_first_arg_name


def ret_val_comp(ret_val, char: str = "q"):
    return ret_val & 0xFF == ord(char)


def show_image(
    image,
    title: str = None,
    *,
    flag: WindowFlagEnum = WindowFlagEnum.gui_expanded.value,
    wait: Union[bool, int] = False,
    draw_title: bool = False,
    save_path: Path = None,
    char: str = "q"
) -> None:
    if title is None:
        title = get_first_arg_name("show_image", verbose=True)
    if title is None:
        title = "image"

    if draw_title:
        cv2.putText(
            image.copy(),
            title,
            (25, image.shape[0] - 20),
            FontEnum.hershey_simplex.value,
            2.0,
            (0, 255, 0),
            3,
        )
    cv2.namedWindow(title, flag)
    cv2.imshow(title, image)
    if save_path:
        cv2.imwrite(save_path, image)

    if wait:
        if (
            wait is int and wait >= 0
        ):  # DO NOT REFACTOR TO ISINSTANCE as bool is an instance of int!
            return ret_val_comp(cv2.waitKey(wait), char)
        else:
            return ret_val_comp(cv2.waitKey(), char)
    return False


if __name__ == "__main__":

    def assd():
        aasdsad = numpy.zeros((50, 50))
        print(show_image(aasdsad, wait=True))
        asd_a_as = numpy.zeros((50, 50))
        print(show_image(asd_a_as, wait=True))

    assd()
