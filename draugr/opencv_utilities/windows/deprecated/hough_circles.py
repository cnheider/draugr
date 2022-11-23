from pathlib import Path
from typing import Iterable

import cv2
import numpy

from draugr.opencv_utilities.color_space import to_gray
from draugr.opencv_utilities.windows.elements import add_trackbar
from draugr.opencv_utilities.windows.image import show_image

__all__ = ["interactive_hough_circles"]


def interactive_hough_circles(
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
    # show_image(numpy.zeros((600, 600)), "canny")

    # add_trackbar("canny", "canny_t1", default=30, max_val=1000, min_val=1.0)
    # add_trackbar("canny", "canny_t2", default=100, max_val=1000, min_val=1.0)

    show_image(numpy.zeros((600, 600)), "image")

    add_trackbar("image", "dp", default=10.0, max_val=30.0, min_val=1.0)
    add_trackbar("image", "minDist", default=4, max_val=255, min_val=1.0)
    add_trackbar(
        "image", "param1", default=50, max_val=1000, min_val=1.0
    )  # param1 (passed as argument into cvHoughCircles()) and the lower (second) threshold is set to half of this value.
    add_trackbar(
        "image", "param2", default=30, max_val=255, min_val=1.0
    )  # accumulator threshold. This value is used in the accumulator plane that must be reached so that a line is retrieved.
    add_trackbar("image", "minRadius", default=0, max_val=255, min_val=1.0)
    add_trackbar("image", "maxRadius", default=255, max_val=255, min_val=1.0)

    canny_t1 = canny_t2 = dp = minDist = param1 = param2 = minRadius = maxRadius = 0
    pcanny_t1 = (
        pcanny_t2
    ) = pdp = pminDist = pparam1 = pparam2 = pminRadius = pmaxRadius = 0

    for p in ps:
        if not p.exists():
            continue

        img = cv2.imread(str(p))
        while 1:
            output = img.copy()

            # canny_t1 = cv2.getTrackbarPos("canny_t1", "canny")
            # canny_t2 = cv2.getTrackbarPos("canny_t2", "canny")

            dp = cv2.getTrackbarPos("dp", "image") / 10
            minDist = cv2.getTrackbarPos("minDist", "image")
            param1 = cv2.getTrackbarPos("param1", "image")
            param2 = cv2.getTrackbarPos("param2", "image")
            minRadius = cv2.getTrackbarPos("minRadius", "image")
            maxRadius = cv2.getTrackbarPos("maxRadius", "image")

            # imgray = cv2.Canny(output, canny_t1, canny_t2)
            imgray = to_gray(output)
            circles = cv2.HoughCircles(
                imgray,
                method=cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius,
            )

            if circles is not None:
                circles = numpy.uint16(numpy.around(circles))
                blue = (0, 0, 255)
                green = (0, 255, 0)
                for c in circles[0]:
                    cv2.circle(output, (c[0], c[1]), c[2], blue, 3)

            # Print if there is a change in HSV value
            if (
                (pcanny_t1 != canny_t1)
                | (pcanny_t2 != canny_t2)
                | (pdp != dp)
                | (pminDist != minDist)
                | (pparam1 != param1)
                | (pparam2 != param2)
                | (pminRadius != minRadius)
                | (pmaxRadius != maxRadius)
            ):
                # cv2.setTrackbarPos("canny_t1", "canny", canny_t1)
                # cv2.setTrackbarPos("canny_t2", "canny", canny_t2)

                cv2.setTrackbarPos("dp", "image", int(dp * 10))
                cv2.setTrackbarPos("minDist", "image", minDist)
                cv2.setTrackbarPos("param1", "image", param1)
                cv2.setTrackbarPos("param2", "image", param2)
                cv2.setTrackbarPos("minRadius", "image", minRadius)
                cv2.setTrackbarPos("maxRadius", "image", maxRadius)

                pcanny_t1 = canny_t1
                pcanny_t1 = canny_t1
                pdp = dp
                pminDist = minDist
                pparam1 = param1
                pparam2 = param2
                pminRadius = minRadius
                pmaxRadius = maxRadius

            # show_image(imgray, "canny")

            if show_image(output, "image", wait=wait_time):
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
        interactive_hough_circles(iter(pss))
    else:
        print(f"{pss} does not exist")
