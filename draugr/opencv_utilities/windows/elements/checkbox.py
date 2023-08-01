from typing import Callable

import cv2
from draugr.opencv_utilities import ButtonTypeEnum

from warg import sink

__all__ = ["add_checkbox"]


def add_checkbox(
    checkbox_name: str,
    *,
    initial_button_state: bool = False,
    callback: Callable = sink,
    new_line: bool = True
) -> None:
    """

    :param checkbox_name:
    :type checkbox_name:
    :param initial_button_state:
    :type initial_button_state:
    :param new_line:
    :type new_line:
    :param callback: Callback function to be called when the checkbox is changed.
    :return: None
    """
    checked = 0
    if initial_button_state:
        checked = 1

    type_ = ButtonTypeEnum.check_box.value

    if new_line:
        type_ |= ButtonTypeEnum.button_bar.value

    cv2.createButton(
        checkbox_name,
        # window_name,
        callback,
        None,
        type_,
        checked,
    )
