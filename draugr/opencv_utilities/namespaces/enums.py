#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 01/02/2022
           """

__all__ = [
    "MarkerTypeEnum",
    "DistanceTypeEnum",
    "HoughModeEnum",
    "WindowFlagEnum",
    "ContourRetrievalModeEnum",
    "ContourApproximationModeEnum",
    "MouseEventEnum",
    "FontEnum",
    "DataTypeEnum",
    "ComparisonEnum",
    "LineTypeEnum",
    "MorphTypeEnum",
    "MorphShapeEnum",
    "DistanceTransformMaskEnum",
    "DistanceTransformLabelTypeEnum",
    "KmeansEnum",
    "CameraPropertyEnum",
    "RectanglesIntersectTypes",
    "BorderTypeEnum",
    "VideoCaptureAPIEnum",
]

from enum import Enum

import cv2


class ComparisonEnum(Enum):
    eq = cv2.CMP_EQ  # equal
    gt = cv2.CMP_GT  # greater than
    ge = cv2.CMP_GE  # # greater equal
    lt = cv2.CMP_LT  # less than
    le = cv2.CMP_LE  # less equal
    ne = cv2.CMP_NE  # unequal


class DataTypeEnum(Enum):
    # Unsigned 8bits # 	uchar	0 ~ 255
    u8 = cv2.CV_8U
    u8c1 = cv2.CV_8UC1
    u8c2 = cv2.CV_8UC2
    u8c3 = cv2.CV_8UC3
    u8c4 = cv2.CV_8UC4
    # u8cN
    # Signed 8bits	char	-128 ~ 127
    s8 = cv2.CV_8S
    s8c1 = cv2.CV_8SC1
    s8c2 = cv2.CV_8SC2
    s8c3 = cv2.CV_8SC3
    s8c4 = cv2.CV_8SC4
    # s8cN
    # Unsigned 16bits	ushort	0 ~ 65535
    u16 = cv2.CV_16U
    u16c1 = cv2.CV_16UC1
    u16c2 = cv2.CV_16UC2
    u16c3 = cv2.CV_16UC3
    u16c4 = cv2.CV_16UC4
    # u16cN
    # Signed 16bits	short	-32768 ~ 32767
    s16 = cv2.CV_16S
    s16c1 = cv2.CV_16SC1
    s16c2 = cv2.CV_16SC2
    s16c3 = cv2.CV_16SC3
    s16c4 = cv2.CV_16SC4
    # s16cN
    # Signed 32bits	int	-2147483648 ~ 2147483647
    s32 = cv2.CV_32S
    s32c1 = cv2.CV_32SC1
    s32c2 = cv2.CV_32SC2
    s32c3 = cv2.CV_32SC3
    s32c4 = cv2.CV_32SC4
    # s32cN
    # Float 32bits	float	-1.18e-38 ~ 3.40e-38
    f32 = cv2.CV_32F
    f32c1 = cv2.CV_32FC1
    f32c2 = cv2.CV_32FC2
    f32c3 = cv2.CV_32FC3
    f32c4 = cv2.CV_32FC4
    # f32cN
    # Double 64bits	double	-1.7e+308 ~ +1.7e+308
    f64 = cv2.CV_64F
    f64c1 = cv2.CV_64FC1
    f64c2 = cv2.CV_64FC2
    f64c3 = cv2.CV_64FC3
    f64c4 = cv2.CV_64FC4
    # f64cN


class VideoCaptureAPIEnum(Enum):
    """



    VideoCapture API backends identifier.
    """

    any = cv2.CAP_ANY
    # Auto detect == 0.

    vfw = cv2.CAP_VFW
    # Video For Windows (platform native)

    v4l = cv2.CAP_V4L
    # V4L/V4L2 capturing support via libv4l.

    v4l2 = cv2.CAP_V4L2
    # Same as CAP_V4L.

    firewire = cv2.CAP_FIREWIRE
    # IEEE 1394 drivers.

    fireware = cv2.CAP_FIREWARE
    # Same as CAP_FIREWIRE.

    ieee1394 = cv2.CAP_IEEE1394
    # Same as CAP_FIREWIRE.

    dc1394 = cv2.CAP_DC1394
    # Same as CAP_FIREWIRE.

    cmu1394 = cv2.CAP_CMU1394
    # Same as CAP_FIREWIRE.

    qt = cv2.CAP_QT
    # QuickTime.

    unicap = cv2.CAP_UNICAP
    # Unicap drivers.

    dshow = cv2.CAP_DSHOW
    # DirectShow (via videoInput)

    pvapi = cv2.CAP_PVAPI
    # PvAPI, Prosilica GigE SDK.

    openni = cv2.CAP_OPENNI
    # OpenNI (for Kinect)

    openni_asus = cv2.CAP_OPENNI_ASUS
    # OpenNI (for Asus Xtion)

    android = cv2.CAP_ANDROID
    # Android - not used.

    xiapi = cv2.CAP_XIAPI
    # XIMEA Camera API.

    avfoundation = cv2.CAP_AVFOUNDATION
    # AVFoundation framework for iOS (OS X Lion will have the same API)

    giganetix = cv2.CAP_GIGANETIX
    # Smartek Giganetix GigEVisionSDK.

    msmf = cv2.CAP_MSMF
    # Microsoft Media Foundation (via videoInput)

    winrt = cv2.CAP_WINRT
    # Microsoft Windows Runtime using Media Foundation.

    intelperc = cv2.CAP_INTELPERC
    # Intel Perceptual Computing SDK.

    openni2 = cv2.CAP_OPENNI2
    # OpenNI2 (for Kinect)

    openni2_asus = cv2.CAP_OPENNI2_ASUS
    # OpenNI2 (for Asus Xtion and Occipital Structure sensors)

    gphoto2 = cv2.CAP_GPHOTO2
    # gPhoto2 connection

    gstreamer = cv2.CAP_GSTREAMER
    # GStreamer.

    ffmpeg = cv2.CAP_FFMPEG
    # Open and record video file or stream using the FFMPEG library.

    images = cv2.CAP_IMAGES
    # OpenCV Image Sequence (e.g. img_%02d.jpg)

    aravis = cv2.CAP_ARAVIS
    # Aravis SDK.

    opencv_mjpeg = cv2.CAP_OPENCV_MJPEG
    # Built-in OpenCV MotionJPEG codec.

    intel_mfx = cv2.CAP_INTEL_MFX
    # Intel MediaSDK.

    xine = cv2.CAP_XINE
    # XINE engine (Linux)


class CameraPropertyEnum(Enum):
    settings = cv2.CAP_PROP_SETTINGS
    auto_focus = cv2.CAP_PROP_AUTOFOCUS
    pos_msec = (
        cv2.CAP_PROP_POS_MSEC
    )  # Current position of the video file in milliseconds.
    pos_frames = (
        cv2.CAP_PROP_POS_FRAMES
    )  # 0-based index of the frame to be decoded/captured next.
    avi_ratio = (
        cv2.CAP_PROP_POS_AVI_RATIO
    )  # Relative position of the video file: 0 - start of the film, 1 - end of the film.
    frame_width = cv2.CAP_PROP_FRAME_WIDTH  # Width of the frames in the video stream.
    frame_height = (
        cv2.CAP_PROP_FRAME_HEIGHT
    )  # Height of the frames in the video stream.
    fps = cv2.CAP_PROP_FPS  # Frame rate.
    fourcc = cv2.CAP_PROP_FOURCC  # 4-character code of codec.
    frame_count = cv2.CAP_PROP_FRAME_COUNT  # Number of frames in the video file.
    format = cv2.CAP_PROP_FORMAT  # Format of the Mat objects returned by retrieve() .
    mode = (
        cv2.CAP_PROP_MODE
    )  # Backend-specific value indicating the current capture mode.
    brightness = cv2.CAP_PROP_BRIGHTNESS  # Brightness of the image (only for cameras).
    contrast = cv2.CAP_PROP_CONTRAST  # Contrast of the image (only for cameras).
    saturation = cv2.CAP_PROP_SATURATION  # Saturation of the image (only for cameras).
    hue = cv2.CAP_PROP_HUE  # Hue of the image (only for cameras).
    gain = cv2.CAP_PROP_GAIN  # Gain of the image (only for cameras).
    exposure = cv2.CAP_PROP_EXPOSURE  # Exposure (only for cameras).
    convert_rgb = (
        cv2.CAP_PROP_CONVERT_RGB
    )  # Boolean flags indicating whether images should be converted to RGB.
    white_balance_blue = cv2.CAP_PROP_WHITE_BALANCE_BLUE_U  # Currently unsupported
    white_balance_red = cv2.CAP_PROP_WHITE_BALANCE_RED_V  # Currently unsupported
    white_balance_ios = cv2.CAP_PROP_IOS_DEVICE_WHITEBALANCE  # Currently unsupported
    rectification = (
        cv2.CAP_PROP_RECTIFICATION
    )  # Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
    monochrome = cv2.CAP_PROP_MONOCHROME
    sharpness = cv2.CAP_PROP_SHARPNESS
    gamma = cv2.CAP_PROP_GAMMA
    temperature = cv2.CAP_PROP_TEMPERATURE
    trigger = cv2.CAP_PROP_TRIGGER
    trigger_delay = cv2.CAP_PROP_TRIGGER_DELAY
    zoom = cv2.CAP_PROP_ZOOM
    focus = cv2.CAP_PROP_FOCUS
    guid = cv2.CAP_PROP_GUID
    iso_speed = cv2.CAP_PROP_ISO_SPEED
    backlight = cv2.CAP_PROP_BACKLIGHT
    pan = cv2.CAP_PROP_PAN
    tilt = cv2.CAP_PROP_TILT
    roll = cv2.CAP_PROP_ROLL
    iris = cv2.CAP_PROP_IRIS
    buffer_size = cv2.CAP_PROP_BUFFERSIZE
    sar_num = cv2.CAP_PROP_SAR_NUM
    sar_den = cv2.CAP_PROP_SAR_DEN
    backend = cv2.CAP_PROP_BACKEND
    channel = cv2.CAP_PROP_CHANNEL
    auto_wb = cv2.CAP_PROP_AUTO_WB
    wb_temperature = cv2.CAP_PROP_WB_TEMPERATURE
    codec_pixel_format = cv2.CAP_PROP_CODEC_PIXEL_FORMAT
    bitrate = cv2.CAP_PROP_BITRATE
    orientation_meta = cv2.CAP_PROP_ORIENTATION_META
    orientation_auto = cv2.CAP_PROP_ORIENTATION_AUTO
    open_timeout_msec = cv2.CAP_PROP_OPEN_TIMEOUT_MSEC
    read_timeout_msec = cv2.CAP_PROP_READ_TIMEOUT_MSEC


class MarkerTypeEnum(Enum):
    cross = cv2.MARKER_CROSS  # A crosshair marker shape.
    tilted_cross = cv2.MARKER_TILTED_CROSS  # A 45 degree tilted crosshair marker shape.
    star = (
        cv2.MARKER_STAR
    )  # A star marker shape, combination of cross and tilted cross.
    diamond = cv2.MARKER_DIAMOND  # A diamond marker shape.
    square = cv2.MARKER_SQUARE  # A square marker shape.
    triangle_up = cv2.MARKER_TRIANGLE_UP  # An upwards pointing triangle marker shape.
    triangle_down = (
        cv2.MARKER_TRIANGLE_DOWN
    )  # A downwards pointing triangle marker shape.


class DistanceTypeEnum(Enum):
    user = cv2.DIST_USER  # User defined distance.
    l1 = cv2.DIST_L1  # distance = |x1-x2| + |y1-y2|
    l2 = cv2.DIST_L2  # the simple euclidean distance
    c = cv2.DIST_C  # distance = max(|x1-x2|,|y1-y2|)
    l12 = cv2.DIST_L12  # L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
    fair = cv2.DIST_FAIR  # distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
    welsch = cv2.DIST_WELSCH  # distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
    huber = cv2.DIST_HUBER  # distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345


class DistanceTransformLabelTypeEnum(Enum):
    """
    distanceTransform algorithm flags
    """

    ccomp = cv2.DIST_LABEL_CCOMP
    """each connected component of zeros in src (as well as all the non-zero pixels closest to the connected component) will be assigned the same label
"""

    pixel = cv2.DIST_LABEL_PIXEL
    """each zero pixel (and all the non-zero pixels closest to it) gets its own label."""


class BorderTypeEnum(Enum):
    """
    Various border types, image boundaries are denoted with |
    """

    constant = cv2.BORDER_CONSTANT  # iiiiii|abcdefgh|iiiiiii with some specified i
    replicate = cv2.BORDER_REPLICATE  # aaaaaa|abcdefgh|hhhhhhh
    reflect = cv2.BORDER_REFLECT  # fedcba|abcdefgh|hgfedcb
    wrap = cv2.BORDER_WRAP  # cdefgh|abcdefgh|abcdefg
    reflect_101 = cv2.BORDER_REFLECT_101  # gfedcb|abcdefgh|gfedcba
    transparent = cv2.BORDER_TRANSPARENT  # uvwxyz|abcdefgh|ijklmno
    reflect101 = cv2.BORDER_REFLECT101  # same as BORDER_REFLECT_101
    default = cv2.BORDER_DEFAULT  # same as BORDER_REFLECT_101
    isolated = cv2.BORDER_ISOLATED  # do not look outside of ROI


class DistanceTransformMaskEnum(Enum):
    """
    Mask size for distance transform.
    """

    mask_3 = cv2.DIST_MASK_3
    mask_5 = cv2.DIST_MASK_5
    mask_precise = cv2.DIST_MASK_PRECISE


class KmeansEnum(Enum):
    random_centers = (
        cv2.KMEANS_RANDOM_CENTERS
    )  # Select random initial centers in each attempt.
    pp_centers = (
        cv2.KMEANS_PP_CENTERS
    )  # Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
    use_initial_labels = (
        cv2.KMEANS_USE_INITIAL_LABELS
    )  # During the first (and possibly the only) attempt, use the user-supplied labels instead of computing them from the initial centers. For the second and further attempts, use the random or semi-random centers. Use one of KMEANS_*_CENTERS flag to specify the exact method.


class HoughModeEnum(Enum):
    standard = (
        cv2.HOUGH_STANDARD
    )  # classical or standard Hough transform. Every line is represented by two floating-point numbers (ρ,θ) , where ρ is a distance between (0,0) point and the line, and θ is the angle between x-axis and the normal to the line. Thus, the matrix must be (the created sequence will be) of CV_32FC2 type
    gradient = cv2.HOUGH_GRADIENT  # basically 21HT
    probabilistic = (
        cv2.HOUGH_PROBABILISTIC
    )  # probabilistic Hough transform (more efficient in case if the picture contains a few long linear segments). It returns line segments rather than the whole line. Each segment is represented by starting and ending points, and the matrix must be (the created sequence will be) of the CV_32SC4 type.
    multi_scale = (
        cv2.HOUGH_MULTI_SCALE
    )  # multi-scale variant of the classical Hough transform. The lines are encoded the same way as HOUGH_STANDARD.
    gradient_alternative = (
        cv2.HOUGH_GRADIENT_ALT
    )  # variation of HOUGH_GRADIENT to get better accuracy


class WindowFlagEnum(Enum):
    normal = (
        cv2.WINDOW_NORMAL
    )  # the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size.
    autosize = (
        cv2.WINDOW_AUTOSIZE
    )  # the user cannot resize the window, the size is constrainted by the image displayed.
    opengl = cv2.WINDOW_OPENGL  # window with opengl support.
    fullscreen = cv2.WINDOW_FULLSCREEN  # change the window to fullscreen.
    free_ratio = (
        cv2.WINDOW_FREERATIO
    )  # the image expends as much as it can (no ratio constraint).
    keep_ratio = cv2.WINDOW_KEEPRATIO  # the ratio of the image is respected.
    gui_expanded = cv2.WINDOW_GUI_EXPANDED  # status bar and tool bar
    gui_normal = cv2.WINDOW_GUI_NORMAL  # old fashious way


class MorphShapeEnum(Enum):
    rect = cv2.MORPH_RECT  # a rectangular structuring element: Eij=1
    cross = (
        cv2.MORPH_CROSS
    )  # a cross-shaped structuring element: Eij={10if i=anchor.y or j=anchor.x otherwise
    ellipse = (
        cv2.MORPH_ELLIPSE
    )  # an elliptic structuring element, that is, a filled ellipse inscribed into the rectangle Rect(0, 0, esize.width, 0.esize.height)


class RectanglesIntersectTypes(Enum):
    """
    types of intersection between rectangles
    """

    none = cv2.INTERSECT_NONE
    # No intersection.

    partial = cv2.INTERSECT_PARTIAL
    # There is a partial intersection.

    full = cv2.INTERSECT_FULL
    # One of the rectangle is fully enclosed in the other.


class MorphTypeEnum(Enum):
    erode = cv2.MORPH_ERODE
    dilate = cv2.MORPH_DILATE
    open = cv2.MORPH_OPEN  # dst=open(src,element)=dilate(erode(src,element))
    close = cv2.MORPH_CLOSE  # dst=close(src,element)=erode(dilate(src,element))
    gradient = (
        cv2.MORPH_GRADIENT
    )  # dst=morph_grad(src,element)=dilate(src,element)−erode(src,element)
    tophat = cv2.MORPH_TOPHAT  # dst=tophat(src,element)=src−open(src,element)
    blackhat = cv2.MORPH_BLACKHAT  # dst=blackhat(src,element)=close(src,element)−src
    hitmiss = cv2.MORPH_HITMISS  # Only supported for CV_8UC1 binary images.


class FontEnum(Enum):
    hershey_simplex = cv2.FONT_HERSHEY_SIMPLEX  # normal size sans-serif font

    hershey_plain = cv2.FONT_HERSHEY_PLAIN  # small size sans-serif font

    hershey_duplex = (
        cv2.FONT_HERSHEY_DUPLEX
    )  # normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)

    hershey_complex = cv2.FONT_HERSHEY_COMPLEX  # normal size serif font

    hershey_triplex = (
        cv2.FONT_HERSHEY_TRIPLEX
    )  # normal size serif font (more complex than FONT_HERSHEY_COMPLEX)

    hershey_complex_small = (
        cv2.FONT_HERSHEY_COMPLEX_SMALL
    )  # smaller version of FONT_HERSHEY_COMPLEX

    hershey_script_simplex = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # hand-writing style font

    hershey_script_complex = (
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    )  # more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX

    italic = cv2.FONT_ITALIC  # flag for italic font


class ContourRetrievalModeEnum(Enum):
    """

      RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
    RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
    RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
    RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours. This full hierarchy is built and shown in the OpenCV contours.c demo.

      :return:
      :rtype:
    """

    external = cv2.RETR_EXTERNAL
    list_all = cv2.RETR_LIST
    ccomp = cv2.RETR_CCOMP
    tree = cv2.RETR_TREE


class ContourApproximationModeEnum(Enum):
    """
    the contour approximation algorithm
    """

    none = cv2.CHAIN_APPROX_NONE
    # stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.

    simple = cv2.CHAIN_APPROX_SIMPLE
    # compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.

    tc89_l1 = cv2.CHAIN_APPROX_TC89_L1
    # applies one of the flavors of the Teh-Chin chain approximation algorithm [244]

    tc89_kcos = cv2.CHAIN_APPROX_TC89_KCOS
    # applies one of the flavors of the Teh-Chin chain approximation algorithm [244]


class MouseEventEnum(Enum):
    move = (
        cv2.EVENT_MOUSEMOVE
    )  # indicates that the mouse pointer has moved over the window.
    left_down = (
        cv2.EVENT_LBUTTONDOWN
    )  # indicates that the left mouse button is pressed.
    left_up = cv2.EVENT_LBUTTONUP  # indicates that left mouse button is released.
    left_double = (
        cv2.EVENT_LBUTTONDBLCLK
    )  # indicates that left mouse button is double clicked.
    right_down = (
        cv2.EVENT_RBUTTONDOWN
    )  # indicates that the right mouse button is pressed.
    right_up = cv2.EVENT_RBUTTONUP  # indicates that right mouse button is released.
    right_double = (
        cv2.EVENT_RBUTTONDBLCLK
    )  # indicates that right mouse button is double clicked.
    middle_down = (
        cv2.EVENT_MBUTTONDOWN
    )  # indicates that the middle mouse button is pressed.
    middle_up = cv2.EVENT_MBUTTONUP  # indicates that middle mouse button is released.
    middle_double = (
        cv2.EVENT_MBUTTONDBLCLK
    )  # indicates that middle mouse button is double clicked.
    wheel_ud = (
        cv2.EVENT_MOUSEWHEEL
    )  # positive and negative values mean forward and backward scrolling, respectively.
    wheel_rl = (
        cv2.EVENT_MOUSEHWHEEL
    )  # positive and negative values mean right and left scrolling, respectively.


class LineTypeEnum(Enum):
    filled = cv2.FILLED
    line4 = cv2.LINE_4  # 4-connected line
    line8 = cv2.LINE_8  # 8-connected line
    anti_aliased = cv2.LINE_AA  # antialiased line
