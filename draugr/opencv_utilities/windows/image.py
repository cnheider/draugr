import os
from pathlib import Path
from typing import Union, Iterable

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy

from apppath import ensure_existence
from draugr.opencv_utilities.namespaces.enums import WindowFlagEnum, FontEnum
from draugr.opencv_utilities.windows.default import (
    ESC_CHAR,
    ExtensionEnum,
    ret_val_comp,
)
from warg import get_first_arg_name

__all__ = ["show_image"]


def show_image(
    image: numpy.ndarray,
    title: str = None,
    *,
    flag: WindowFlagEnum = WindowFlagEnum.gui_expanded.value,
    wait: Union[bool, int] = False,
    draw_title: bool = False,
    save_path: Path = None,
    exit_chars: Iterable[str] = ("q", ESC_CHAR),
    extension: ExtensionEnum = ExtensionEnum.exr,  # 'png'
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

    if save_path is not None:
        extension = ExtensionEnum(extension)
        if extension == ExtensionEnum.exr:
            assert (
                os.environ.get("OPENCV_IO_ENABLE_OPENEXR") == "1"
            ), f'Openexr is support not enabled, must be declared before import of cv2, OPENCV_IO_ENABLE_OPENEXR{os.environ.get("OPENCV_IO_ENABLE_OPENEXR")}'
            image = image.astype(dtype=numpy.float32)
            # cv2.

        ensure_existence(save_path.parent)

        cv2.imwrite(str(save_path.with_suffix(f".{extension.value}")), image)

    if wait is not None:
        """# WEIRDO PYTHON
        if (
            wait is int or wait is float and wait >= 0
        ):  # DO NOT REFACTOR TO ISINSTANCE as bool is an instance of int!
          return ret_val_comp(cv2.waitKey(wait), char)
        else:
          return ret_val_comp(cv2.waitKey(), char)
        """
        if isinstance(wait, bool):
            if wait:
                return ret_val_comp(cv2.waitKey(), exit_chars)
        elif wait >= 0:
            return ret_val_comp(cv2.waitKey(wait), exit_chars)
    return False


if __name__ == "__main__":

    def assd():
        aasdsad = numpy.zeros((50, 50))
        print(show_image(aasdsad, wait=True))
        asd_a_as = numpy.zeros((50, 50))
        print(show_image(asd_a_as, wait=True, save_path=Path("exclude") / "out.exr"))

    assd()
