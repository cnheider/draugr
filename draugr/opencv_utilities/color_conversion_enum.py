from enum import Enum

import cv2

__all__ = ["ColorConversionCodesEnum"]


class ColorConversionCodesEnum(Enum):
    """
    the color conversion codes"""

    bgr2bgra = cv2.COLOR_BGR2BGRA  # add alpha channel to RGB or BGR image

    rgb2rgba = cv2.COLOR_RGB2RGBA

    bgra2bgr = cv2.COLOR_BGRA2BGR  # remove alpha channel from RGB or BGR image

    rgba2rgb = cv2.COLOR_RGBA2RGB

    bgr2rgba = (
        cv2.COLOR_BGR2RGBA
    )  # convert between RGB and BGR color spaces (with or without alpha channel)

    a = cv2.COLOR_RGB2BGRA

    a1 = cv2.COLOR_RGBA2BGR

    a2 = cv2.COLOR_BGRA2RGB

    a3 = cv2.COLOR_BGR2RGB

    a4 = cv2.COLOR_RGB2BGR

    a5 = cv2.COLOR_BGRA2RGBA

    a6 = cv2.COLOR_RGBA2BGRA

    a7 = cv2.COLOR_BGR2GRAY  # convert between RGB/BGR and grayscale, color conversions

    a8 = cv2.COLOR_RGB2GRAY

    a9 = cv2.COLOR_GRAY2BGR

    a11 = cv2.COLOR_GRAY2RGB

    a12 = cv2.COLOR_GRAY2BGRA

    a13 = cv2.COLOR_GRAY2RGBA

    a14 = cv2.COLOR_BGRA2GRAY

    a15 = cv2.COLOR_RGBA2GRAY

    a16 = cv2.COLOR_BGR2BGR565  # convert between RGB/BGR and BGR565 (16-bit images)

    a17 = cv2.COLOR_RGB2BGR565

    a = cv2.COLOR_BGR5652BGR

    a = cv2.COLOR_BGR5652RGB

    a = cv2.COLOR_BGRA2BGR565

    a = cv2.COLOR_RGBA2BGR565

    a = cv2.COLOR_BGR5652BGRA

    a = cv2.COLOR_BGR5652RGBA

    a = cv2.COLOR_GRAY2BGR565  # convert between grayscale to BGR565 (16-bit images)

    a = cv2.COLOR_BGR5652GRAY

    a = cv2.COLOR_BGR2BGR555  # convert between RGB/BGR and BGR555 (16-bit images)

    a = cv2.COLOR_RGB2BGR555

    a = cv2.COLOR_BGR5552BGR

    a = cv2.COLOR_BGR5552RGB

    a = cv2.COLOR_BGRA2BGR555

    a = cv2.COLOR_RGBA2BGR555

    a = cv2.COLOR_BGR5552BGRA

    a = cv2.COLOR_BGR5552RGBA

    a = cv2.COLOR_GRAY2BGR555  # convert between grayscale and BGR555 (16-bit images)

    a = cv2.COLOR_BGR5552GRAY

    a = cv2.COLOR_BGR2XYZ  # convert RGB/BGR to CIE XYZ, color conversions

    a = cv2.COLOR_RGB2XYZ

    a = cv2.COLOR_XYZ2BGR

    a = cv2.COLOR_XYZ2RGB

    a = (
        cv2.COLOR_BGR2YCrCb
    )  # convert RGB/BGR to luma-chroma (aka YCC), color conversions

    a = cv2.COLOR_RGB2YCrCb

    a = cv2.COLOR_YCrCb2BGR

    a = cv2.COLOR_YCrCb2RGB

    a = (
        cv2.COLOR_BGR2HSV
    )  # convert RGB/BGR to HSV (hue saturation value) with H range 0..180 if 8 bit image, color conversions

    a = cv2.COLOR_RGB2HSV

    a = cv2.COLOR_BGR2Lab  # convert RGB/BGR to CIE Lab, color conversions

    a = cv2.COLOR_RGB2Lab

    a = cv2.COLOR_BGR2Luv  # convert RGB/BGR to CIE Luv, color conversions

    a = cv2.COLOR_RGB2Luv
    a = (
        cv2.COLOR_BGR2HLS
    )  # convert RGB/BGR to HLS (hue lightness saturation) with H range 0..180 if 8 bit image, color conversions

    a = cv2.COLOR_RGB2HLS

    a = (
        cv2.COLOR_HSV2BGR
    )  # backward conversions HSV to RGB/BGR with H range 0..180 if 8 bit image

    a = cv2.COLOR_HSV2RGB

    a = cv2.COLOR_Lab2BGR

    a = cv2.COLOR_Lab2RGB

    a = cv2.COLOR_Luv2BGR

    a = cv2.COLOR_Luv2RGB

    a = (
        cv2.COLOR_HLS2BGR
    )  # backward conversions HLS to RGB/BGR with H range 0..180 if 8 bit image

    a = cv2.COLOR_HLS2RGB

    a = (
        cv2.COLOR_BGR2HSV_FULL
    )  # convert RGB/BGR to HSV (hue saturation value) with H range 0..255 if 8 bit image, color conversions

    a = cv2.COLOR_RGB2HSV_FULL

    a = (
        cv2.COLOR_BGR2HLS_FULL
    )  # convert RGB/BGR to HLS (hue lightness saturation) with H range 0..255 if 8 bit image, color conversions

    a = cv2.COLOR_RGB2HLS_FULL

    a = (
        cv2.COLOR_HSV2BGR_FULL
    )  # backward conversions HSV to RGB/BGR with H range 0..255 if 8 bit image

    a = cv2.COLOR_HSV2RGB_FULL

    a = (
        cv2.COLOR_HLS2BGR_FULL
    )  # backward conversions HLS to RGB/BGR with H range 0..255 if 8 bit image

    a = cv2.COLOR_HLS2RGB_FULL

    a = cv2.COLOR_LBGR2Lab

    a = cv2.COLOR_LRGB2Lab

    a = cv2.COLOR_LBGR2Luv

    a = cv2.COLOR_LRGB2Luv

    a = cv2.COLOR_Lab2LBGR

    a = cv2.COLOR_Lab2LRGB

    a = cv2.COLOR_Luv2LBGR

    a = cv2.COLOR_Luv2LRGB

    a = cv2.COLOR_BGR2YUV  # convert between RGB/BGR and YUV

    a = cv2.COLOR_RGB2YUV

    a = cv2.COLOR_YUV2BGR

    a = cv2.COLOR_YUV2RGB

    a = cv2.COLOR_YUV2RGB_NV12  # YUV 4:2:0 family to RGB.

    a = cv2.COLOR_YUV2BGR_NV12

    a = cv2.COLOR_YUV2RGB_NV21

    a = cv2.COLOR_YUV2BGR_NV21

    a = cv2.COLOR_YUV420sp2RGB

    a = cv2.COLOR_YUV420sp2BGR

    a = cv2.COLOR_YUV2RGBA_NV12

    a = cv2.COLOR_YUV2BGRA_NV12

    a = cv2.COLOR_YUV2RGBA_NV21

    a = cv2.COLOR_YUV2BGRA_NV21

    a = cv2.COLOR_YUV420sp2RGBA

    a = cv2.COLOR_YUV420sp2BGRA

    a = cv2.COLOR_YUV2RGB_YV12

    a = cv2.COLOR_YUV2BGR_YV12

    a = cv2.COLOR_YUV2RGB_IYUV

    a = cv2.COLOR_YUV2BGR_IYUV

    a = cv2.COLOR_YUV2RGB_I420

    a = cv2.COLOR_YUV2BGR_I420

    a = cv2.COLOR_YUV420p2RGB

    a = cv2.COLOR_YUV420p2BGR

    a = cv2.COLOR_YUV2RGBA_YV12

    a = cv2.COLOR_YUV2BGRA_YV12

    a = cv2.COLOR_YUV2RGBA_IYUV

    a = cv2.COLOR_YUV2BGRA_IYUV

    a = cv2.COLOR_YUV2RGBA_I420

    a = cv2.COLOR_YUV2BGRA_I420

    a = cv2.COLOR_YUV420p2RGBA

    a = cv2.COLOR_YUV420p2BGRA

    a = cv2.COLOR_YUV2GRAY_420

    a = cv2.COLOR_YUV2GRAY_NV21

    a = cv2.COLOR_YUV2GRAY_NV12

    a = cv2.COLOR_YUV2GRAY_YV12

    a = cv2.COLOR_YUV2GRAY_IYUV

    a = cv2.COLOR_YUV2GRAY_I420

    a = cv2.COLOR_YUV420sp2GRAY

    a = cv2.COLOR_YUV420p2GRAY

    a = cv2.COLOR_YUV2RGB_UYVY  # YUV 4:2:2 family to RGB.

    a = cv2.COLOR_YUV2BGR_UYVY

    a = cv2.COLOR_YUV2RGB_Y422

    a = cv2.COLOR_YUV2BGR_Y422

    a = cv2.COLOR_YUV2RGB_UYNV

    a = cv2.COLOR_YUV2BGR_UYNV

    a = cv2.COLOR_YUV2RGBA_UYVY

    a = cv2.COLOR_YUV2BGRA_UYVY

    a = cv2.COLOR_YUV2RGBA_Y422

    a = cv2.COLOR_YUV2BGRA_Y422

    a = cv2.COLOR_YUV2RGBA_UYNV

    a = cv2.COLOR_YUV2BGRA_UYNV

    a = cv2.COLOR_YUV2RGB_YUY2

    a = cv2.COLOR_YUV2BGR_YUY2

    a = cv2.COLOR_YUV2RGB_YVYU

    a = cv2.COLOR_YUV2BGR_YVYU

    a = cv2.COLOR_YUV2RGB_YUYV

    a = cv2.COLOR_YUV2BGR_YUYV

    a = cv2.COLOR_YUV2RGB_YUNV

    a = cv2.COLOR_YUV2BGR_YUNV

    a = cv2.COLOR_YUV2RGBA_YUY2

    a = cv2.COLOR_YUV2BGRA_YUY2

    a = cv2.COLOR_YUV2RGBA_YVYU

    a = cv2.COLOR_YUV2BGRA_YVYU

    a = cv2.COLOR_YUV2RGBA_YUYV

    a = cv2.COLOR_YUV2BGRA_YUYV

    a = cv2.COLOR_YUV2RGBA_YUNV

    a = cv2.COLOR_YUV2BGRA_YUNV

    a = cv2.COLOR_YUV2GRAY_UYVY

    a = cv2.COLOR_YUV2GRAY_YUY2

    a = cv2.COLOR_YUV2GRAY_Y422
    a = cv2.COLOR_YUV2GRAY_UYNV

    a = cv2.COLOR_YUV2GRAY_YVYU

    a = cv2.COLOR_YUV2GRAY_YUYV

    a = cv2.COLOR_YUV2GRAY_YUNV

    a = cv2.COLOR_RGBA2mRGBA  # alpha premultiplication

    a = cv2.COLOR_mRGBA2RGBA

    a = cv2.COLOR_RGB2YUV_I420  # RGB to YUV 4:2:0 family.

    a = cv2.COLOR_BGR2YUV_I420

    a = cv2.COLOR_RGB2YUV_IYUV

    a = cv2.COLOR_BGR2YUV_IYUV

    a = cv2.COLOR_RGBA2YUV_I420

    a = cv2.COLOR_BGRA2YUV_I420

    a = cv2.COLOR_RGBA2YUV_IYUV

    a = cv2.COLOR_BGRA2YUV_IYUV

    a = cv2.COLOR_RGB2YUV_YV12

    a = cv2.COLOR_BGR2YUV_YV12

    a = cv2.COLOR_RGBA2YUV_YV12

    a = cv2.COLOR_BGRA2YUV_YV12

    a = (
        cv2.COLOR_BayerBG2BGR
    )  # Demosaicing, see color conversions for additional information. #equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2BGR  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2BGR  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2BGR  # equivalent to GBRG Bayer pattern

    a = cv2.COLOR_BayerRGGB2BGR

    a = cv2.COLOR_BayerGRBG2BGR

    a = cv2.COLOR_BayerBGGR2BGR

    a = cv2.COLOR_BayerGBRG2BGR

    a = cv2.COLOR_BayerRGGB2RGB

    a = cv2.COLOR_BayerGRBG2RGB

    a = cv2.COLOR_BayerBGGR2RGB

    a = cv2.COLOR_BayerGBRG2RGB

    a = cv2.COLOR_BayerBG2RGB  # equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2RGB  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2RGB  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2RGB  # equivalent to GBRG Bayer pattern

    a = cv2.COLOR_BayerBG2GRAY  # equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2GRAY  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2GRAY  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2GRAY  # equivalent to GBRG Bayer pattern

    a = cv2.COLOR_BayerRGGB2GRAY

    a = cv2.COLOR_BayerGRBG2GRAY

    a = cv2.COLOR_BayerBGGR2GRAY

    a = cv2.COLOR_BayerGBRG2GRAY

    a = (
        cv2.COLOR_BayerBG2BGR_VNG
    )  # Demosaicing using Variable Number of Gradients. #equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2BGR_VNG  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2BGR_VNG  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2BGR_VNG  # equivalent to GBRG Bayer pattern

    a = cv2.COLOR_BayerRGGB2BGR_VNG

    a = cv2.COLOR_BayerGRBG2BGR_VNG

    a = cv2.COLOR_BayerBGGR2BGR_VNG

    a = cv2.COLOR_BayerGBRG2BGR_VNG

    a = cv2.COLOR_BayerRGGB2RGB_VNG

    a = cv2.COLOR_BayerGRBG2RGB_VNG

    a = cv2.COLOR_BayerBGGR2RGB_VNG

    a = cv2.COLOR_BayerGBRG2RGB_VNG

    a = cv2.COLOR_BayerBG2RGB_VNG  # equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2RGB_VNG  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2RGB_VNG  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2RGB_VNG  # equivalent to GBRG Bayer pattern

    a = (
        cv2.COLOR_BayerBG2BGR_EA
    )  # Edge-Aware Demosaicing. #equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2BGR_EA  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2BGR_EA  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2BGR_EA  # equivalent to GBRG Bayer pattern

    a = cv2.COLOR_BayerRGGB2BGR_EA

    a = cv2.COLOR_BayerGRBG2BGR_EA

    a = cv2.COLOR_BayerBGGR2BGR_EA

    a = cv2.COLOR_BayerGBRG2BGR_EA

    a = cv2.COLOR_BayerRGGB2RGB_EA

    a = cv2.COLOR_BayerGRBG2RGB_EA

    a = cv2.COLOR_BayerBGGR2RGB_EA

    a = cv2.COLOR_BayerGBRG2RGB_EA

    a = cv2.COLOR_BayerBG2RGB_EA  # equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2RGB_EA  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2RGB_EA  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2RGB_EA  # equivalent to GBRG Bayer pattern

    a = (
        cv2.COLOR_BayerBG2BGRA
    )  # Demosaicing with alpha channel. #equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2BGRA  # equivalent to GRBG Bayer pattern

    a = cv2.COLOR_BayerRG2BGRA  # equivalent to BGGR Bayer pattern

    a = cv2.COLOR_BayerGR2BGRA  # equivalent to GBRG Bayer pattern

    a = cv2.COLOR_BayerRGGB2BGRA

    a = cv2.COLOR_BayerGRBG2BGRA

    a = cv2.COLOR_BayerBGGR2BGRA

    a = cv2.COLOR_BayerGBRG2BGRA

    a = cv2.COLOR_BayerRGGB2RGBA

    a = cv2.COLOR_BayerGRBG2RGBA

    a = cv2.COLOR_BayerBGGR2RGBA

    a = cv2.COLOR_BayerGBRG2RGBA

    a = cv2.COLOR_BayerBG2RGBA  # equivalent to RGGB Bayer pattern

    a = cv2.COLOR_BayerGB2RGBA  # equivalent to GRBG Bayer pattern

    bayerrg2rgba = cv2.COLOR_BayerRG2RGBA  # equivalent to BGGR Bayer pattern

    bayergr2rgba = cv2.COLOR_BayerGR2RGBA  # equivalent to GBRG Bayer pattern

    colorcvt_max = cv2.COLOR_COLORCVT_MAX


if __name__ == "__main__":
    print(ColorConversionCodesEnum.rgb2rgba)
