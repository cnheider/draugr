from typing import Optional

import cv2

from warg import sink
from draugr.opencv_utilities import ButtonTypeEnum

__all__ = ["add_button"]


def add_button(
    checkbox_name: str, *, callback: callable = sink, new_line: bool = True
) -> None:
    """
    :param checkbox_name:
    :type checkbox_name:
    :param new_line:
    :type new_line:
    :param callback: Callback function to be called when the button is pressed.
    :return: None
    """

    type_ = ButtonTypeEnum.push_button.value

    if new_line:
        type_ |= ButtonTypeEnum.button_bar.value

    cv2.createButton(checkbox_name, callback, None, type_, 0)
