import math

import cv2
import numpy

from draugr.opencv_utilities import LineTypeEnum


# ============================================================================


def ellipse_bbox(h, k, a, b, theta):
    ux = a * math.cos(theta)
    uy = a * math.sin(theta)
    vx = b * math.cos(theta + math.pi / 2)
    vy = b * math.sin(theta + math.pi / 2)
    box_halfwidth = numpy.ceil(math.sqrt(ux**2 + vx**2))
    box_halfheight = numpy.ceil(math.sqrt(uy**2 + vy**2))
    return (
        (int(h - box_halfwidth), int(k - box_halfheight)),
        (int(h + box_halfwidth), int(k + box_halfheight)),
    )


# ----------------------------------------------------------------------------

# Rotated elliptical gradient - slow, Python-only approach
def make_gradient_v1(width, height, h, k, a, b, theta):
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2

    weights = numpy.zeros((height, width), numpy.float64)
    for y in range(height):
        for x in range(width):
            weights[y, x] = (((x - h) * ct + (y - k) * st) ** 2) / aa + (
                ((x - h) * st - (y - k) * ct) ** 2
            ) / bb

    return numpy.clip(1.0 - weights, 0, 1)


# ----------------------------------------------------------------------------

# Rotated elliptical gradient - faster, vectorized numpy approach
def make_gradient_v2(width, height, h, k, a, b, theta):
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2

    # Generate (x,y) coordinate arrays
    y, x = numpy.mgrid[-k : height - k, -h : width - h]
    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)

    return numpy.clip(1.0 - weights, 0, 1)


# ============================================================================
if __name__ == "__main__":

    from pathlib import Path
    from apppath import ensure_existence
    from draugr.opencv_utilities import show_image

    basep = ensure_existence(Path("exclude"))

    def draw_image(a, b, theta, inner_scale, save_intermediate=False):
        # Calculate the image size needed to draw this and center the ellipse
        _, (h, k) = ellipse_bbox(0, 0, a, b, theta)  # Ellipse center
        h += 2  # Add small margin
        k += 2  # Add small margin
        width, height = (h * 2 + 1, k * 2 + 1)  # Canvas size

        # Parameters defining the two ellipses for OpenCV (a RotatedRect structure)
        ellipse_outer = ((h, k), (a * 2, b * 2), math.degrees(theta))
        ellipse_inner = (
            (h, k),
            (a * 2 * inner_scale, b * 2 * inner_scale),
            math.degrees(theta),
        )

        # Generate the transparency layer -- the outer ellipse filled and anti-aliased
        transparency = numpy.zeros((height, width), numpy.uint8)
        cv2.ellipse(
            transparency, ellipse_outer, 255, -1, LineTypeEnum.anti_aliased.value
        )
        if save_intermediate:
            show_image(
                transparency,
                wait=True
                # save_path = basep/"eligrad-t.png"
            )

        # Generate the gradient and scale it to 8bit grayscale range
        intensity = numpy.uint8(
            make_gradient_v1(width, height, h, k, a, b, theta) * 255
        )
        if save_intermediate:
            show_image(
                intensity,
                wait=True
                # save_path =  str(basep / "eligrad-i1.png")
            )

        # Draw the inter ellipse filled and anti-aliased
        cv2.ellipse(intensity, ellipse_inner, 255, -1, LineTypeEnum.anti_aliased.value)
        if save_intermediate:
            show_image(
                intensity,
                # save_path =  str(basep / "eligrad-i2.png")
                wait=True,
            )

        # Turn it into a BGRA image
        result = cv2.merge([intensity, intensity, intensity, transparency])
        return result

    # ============================================================================

    a, b = (360.0, 200.0)  # Semi-major and semi-minor axis
    theta = math.radians(40.0)  # Ellipse rotation (radians)
    inner_scale = 0.6  # Scale of the inner full-white ellipse

    show_image(
        draw_image(a, b, theta, inner_scale, True),
        wait=True
        # save_path = str(basep/"eligrad.png")
    )

    # ============================================================================

    rows = []
    for j in range(0, 4, 1):
        cols = []
        for i in range(0, 90, 10):
            tile = numpy.zeros((170, 170, 4), numpy.uint8)
            image = draw_image(80.0, 50.0, math.radians(i + j * 90), 0.6)
            tile[: image.shape[0], : image.shape[1]] = image
            cols.append(tile)
        rows.append(numpy.hstack(cols))

    show_image(
        numpy.vstack(rows),
        # save_path = str(basep/"eligrad-m.png")
        wait=True,
    )
