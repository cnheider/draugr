import cv2

from draugr.opencv_utilities import WindowFlagEnum, FontEnum

__all__ = ["show_image"]


def show_image(
    image,
    title: str = "image",
    *,
    flag: WindowFlagEnum = WindowFlagEnum.gui_expanded.value,
    halt: bool = False,
    draw_title: bool = False
) -> None:
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
    if halt:
        cv2.waitKey()
