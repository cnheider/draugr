from pathlib import Path
from typing import Iterable

import cv2
import numpy

from draugr.opencv_utilities.color_space.threshold import hsv_min_max_clip_mask
from draugr.opencv_utilities.windows.elements import add_trackbar
from draugr.opencv_utilities.windows.image import show_image

__all__ = ["interactive_hsv_color_picker"]


def interactive_hsv_color_picker(
    ps: Iterable, wait_time: int = 33, verbose: bool = False
) -> None:
    """

    :param ps:
    :type ps:
    :param wait_time:
    :type wait_time:
    :param verbose:
    :type verbose:
    """
    window_label = "image"

    show_image(numpy.zeros((600, 600)), window_label)

    hmin_label = "HMin"
    hmax_label = "HMax"
    smin_label = "SMin"
    smax_label = "SMax"
    vmin_label = "VMin"
    vmax_label = "VMax"

    add_trackbar(
        window_label, hmin_label, default=0, max_val=179
    )  # Hue is from 0-179 for Opencv
    add_trackbar(window_label, hmax_label, default=179, max_val=179)

    add_trackbar(window_label, smin_label, default=0)
    add_trackbar(window_label, smax_label, default=255)

    add_trackbar(window_label, vmin_label, default=0)
    add_trackbar(window_label, vmax_label, default=255)

    # Initialize to check if HSV min/max value changes
    h_min = s_min = v_min = h_max = s_max = v_max = 0
    ph_min = ps_min = pv_min = ph_max = ps_max = pv_max = 0

    for p in ps:
        if not p.exists():
            continue

        img = cv2.imread(str(p))
        output = img
        while 1:

            h_min = cv2.getTrackbarPos(hmin_label, window_label)
            h_max = cv2.getTrackbarPos(hmax_label, window_label)

            s_min = cv2.getTrackbarPos(smin_label, window_label)
            s_max = cv2.getTrackbarPos(smax_label, window_label)

            v_min = cv2.getTrackbarPos(vmin_label, window_label)
            v_max = cv2.getTrackbarPos(vmax_label, window_label)

            output = cv2.bitwise_and(
                img,
                img,
                mask=hsv_min_max_clip_mask(
                    img,
                    numpy.array([h_min, s_min, v_min]),
                    numpy.array([h_max, s_max, v_max]),
                ),
            )

            # Print if there is a change in HSV value
            if (
                (ph_min != h_min)
                | (ps_min != s_min)
                | (pv_min != v_min)
                | (ph_max != h_max)
                | (ps_max != s_max)
                | (pv_max != v_max)
            ):
                if verbose:
                    print(
                        f"(hMin = {h_min:d} , sMin = {s_min:d}, vMin = {v_min:d}), (hMax = {h_max:d} , sMax = {s_max:d}, vMax = {v_max:d})"
                    )

                # TODO :look at neater handling of min-maxing values
                th_min = min(h_min, h_max)
                ts_min = min(s_min, s_max)
                tv_min = min(v_min, v_max)
                h_max = max(h_min, h_max)
                s_max = max(s_min, s_max)
                v_max = max(v_min, v_max)
                h_min = th_min
                s_min = ts_min
                v_min = tv_min

                cv2.setTrackbarPos(hmin_label, window_label, h_min)
                cv2.setTrackbarPos(hmax_label, window_label, h_max)
                cv2.setTrackbarPos(smin_label, window_label, s_min)
                cv2.setTrackbarPos(smax_label, window_label, s_max)
                cv2.setTrackbarPos(vmin_label, window_label, v_min)
                cv2.setTrackbarPos(vmax_label, window_label, v_max)

                ph_min = h_min
                ps_min = s_min
                pv_min = v_min
                ph_max = h_max
                ps_max = s_max
                pv_max = v_max

            if show_image(output, window_label, wait=wait_time):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    pss = (
        Path.home()
        / "ProjectsWin/AiBitbucket/Internal/OptikosPrime/exclude/new_images/adam_plus0p5/215asd.jpg",
        Path(r"C:\Users\Christian\OneDrive\Billeder\buh\7BIsT.png"),
        Path(r"C:\Users\Christian\OneDrive\Billeder\Portraits\thomas.jpg"),
    )
    if any(p.exists() for p in pss):
        interactive_hsv_color_picker(iter(pss))
    else:
        print(f"{pss} does not exist")
