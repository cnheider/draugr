from enum import Enum

import cv2

__all__ = ["ColorConversionEnum"]


class ColorConversionEnum(Enum):
    """
    the color conversion codes
    """

    bgr2bgra = cv2.COLOR_BGR2BGRA  # add alpha channel to RGB or BGR image
    rgb2rgba = cv2.COLOR_RGB2RGBA

    bgra2bgr = cv2.COLOR_BGRA2BGR  # remove alpha channel from RGB or BGR image
    rgba2rgb = cv2.COLOR_RGBA2RGB

    # ---
    bgr2rgba = (
        cv2.COLOR_BGR2RGBA
    )  # convert between RGB and BGR color spaces (with or without alpha channel)
    rgb2bgra = cv2.COLOR_RGB2BGRA

    rgba2bgr = cv2.COLOR_RGBA2BGR
    bgra2rgb = cv2.COLOR_BGRA2RGB

    bgr2rgb = cv2.COLOR_BGR2RGB
    rgb2bgr = cv2.COLOR_RGB2BGR

    bgra2rgba = cv2.COLOR_BGRA2RGBA
    rgba2bgra = cv2.COLOR_RGBA2BGRA

    # ---
    bgr2gray = (
        cv2.COLOR_BGR2GRAY
    )  # convert between RGB/BGR and grayscale, color conversions
    rgb2gray = cv2.COLOR_RGB2GRAY
    bgra2gray = cv2.COLOR_BGRA2GRAY
    rgba2gray = cv2.COLOR_RGBA2GRAY

    gray2bgr = cv2.COLOR_GRAY2BGR  # grayscale, color conversions
    gray2rgb = cv2.COLOR_GRAY2RGB
    gray2bgra = cv2.COLOR_GRAY2BGRA
    gray2rgba = cv2.COLOR_GRAY2RGBA
    # ---

    bgr2bgr565 = (
        cv2.COLOR_BGR2BGR565
    )  # convert between RGB/BGR and BGR565 (16-bit images)
    rgb2bgr565 = cv2.COLOR_RGB2BGR565
    bgr5652bgr = cv2.COLOR_BGR5652BGR
    bgr5652rgb = cv2.COLOR_BGR5652RGB
    bgra2bgr565 = cv2.COLOR_BGRA2BGR565
    rgba2bgr565 = cv2.COLOR_RGBA2BGR565
    bgr5652bgra = cv2.COLOR_BGR5652BGRA
    bgr5652rgba = cv2.COLOR_BGR5652RGBA
    # -----

    gray2bgr565 = (
        cv2.COLOR_GRAY2BGR565
    )  # convert between grayscale to BGR565 (16-bit images)
    bgr5652gray = cv2.COLOR_BGR5652GRAY

    # --- ---
    bgr2bgr555 = (
        cv2.COLOR_BGR2BGR555
    )  # convert between RGB/BGR and BGR555 (16-bit images)
    rgb2bgr555 = cv2.COLOR_RGB2BGR555
    bgr5552bgr = cv2.COLOR_BGR5552BGR
    bgr5552rgb = cv2.COLOR_BGR5552RGB
    bgra2bgr555 = cv2.COLOR_BGRA2BGR555
    rgba2bgr555 = cv2.COLOR_RGBA2BGR555
    bgr5552bgra = cv2.COLOR_BGR5552BGRA
    bgr5552rgba = cv2.COLOR_BGR5552RGBA

    # -- ---
    gray2bgr555 = (
        cv2.COLOR_GRAY2BGR555
    )  # convert between grayscale and BGR555 (16-bit images)

    bgr5552gray = cv2.COLOR_BGR5552GRAY

    # -- ---
    bgr2xyz = cv2.COLOR_BGR2XYZ  # convert RGB/BGR to CIE XYZ, color conversions
    rgb2xyz = cv2.COLOR_RGB2XYZ
    xyz2bgr = cv2.COLOR_XYZ2BGR
    xyz2rgb = cv2.COLOR_XYZ2RGB

    # -- ---
    bgr2ycrcb = (
        cv2.COLOR_BGR2YCrCb
    )  # convert RGB/BGR to luma-chroma (aka YCC), color conversions
    rgb2ycrcb = cv2.COLOR_RGB2YCrCb
    ycrcb2bgr = cv2.COLOR_YCrCb2BGR
    ycrcb2rgb = cv2.COLOR_YCrCb2RGB

    # -- ---
    bgr2hsv = (
        cv2.COLOR_BGR2HSV
    )  # convert RGB/BGR to HSV (hue saturation value) with H range 0..180 if 8 bit image, color conversions
    rgb2hsv = cv2.COLOR_RGB2HSV

    # 37 ---
    bgr2lab = cv2.COLOR_BGR2Lab  # convert RGB/BGR to CIE Lab, color conversions
    rgb2lab = cv2.COLOR_RGB2Lab

    # 40 --
    bgr2luv = cv2.COLOR_BGR2Luv  # convert RGB/BGR to CIE Luv, color conversions
    rgb2luv = cv2.COLOR_RGB2Luv

    # -- ---
    bgr2hls = (
        cv2.COLOR_BGR2HLS
    )  # convert RGB/BGR to HLS (hue lightness saturation) with H range 0..180 if 8 bit image, color conversions
    rgb2hls = cv2.COLOR_RGB2HLS

    # --- ---
    hsv2bgr = (
        cv2.COLOR_HSV2BGR
    )  # backward conversions HSV to RGB/BGR with H range 0..180 if 8 bit image
    hsv2rgb = cv2.COLOR_HSV2RGB
    lab2bgr = cv2.COLOR_Lab2BGR
    lab2rgb = cv2.COLOR_Lab2RGB
    luv2bgr = cv2.COLOR_Luv2BGR
    luv2rgb = cv2.COLOR_Luv2RGB

    # -- ---
    hls2bgr = (
        cv2.COLOR_HLS2BGR
    )  # backward conversions HLS to RGB/BGR with H range 0..180 if 8 bit image
    hls2rgb = cv2.COLOR_HLS2RGB

    # -- ---
    bgr2hsv_full = (
        cv2.COLOR_BGR2HSV_FULL
    )  # convert RGB/BGR to HSV (hue saturation value) with H range 0..255 if 8 bit image, color conversions
    rgb2hsv_full = cv2.COLOR_RGB2HSV_FULL

    # -- ---
    bgr2hls_full = (
        cv2.COLOR_BGR2HLS_FULL
    )  # convert RGB/BGR to HLS (hue lightness saturation) with H range 0..255 if 8 bit image, color conversions

    rgb2hls_full = cv2.COLOR_RGB2HLS_FULL

    # --- ---
    hsv2bgr_full = (
        cv2.COLOR_HSV2BGR_FULL
    )  # backward conversions HSV to RGB/BGR with H range 0..255 if 8 bit image

    hsv2rgb_full = cv2.COLOR_HSV2RGB_FULL

    # --- ---
    hls2bgr_full = (
        cv2.COLOR_HLS2BGR_FULL
    )  # backward conversions HLS to RGB/BGR with H range 0..255 if 8 bit image

    hls2rgb_full = cv2.COLOR_HLS2RGB_FULL

    lbgr2lab = cv2.COLOR_LBGR2Lab

    lrgb2lab = cv2.COLOR_LRGB2Lab

    lbgr2luv = cv2.COLOR_LBGR2Luv

    lrgb2luv = cv2.COLOR_LRGB2Luv

    lab2lbgr = cv2.COLOR_Lab2LBGR

    lab2lrgb = cv2.COLOR_Lab2LRGB

    luv2lbgr = cv2.COLOR_Luv2LBGR

    luv2lrgb = cv2.COLOR_Luv2LRGB

    # -- ---
    bgr2yuv = cv2.COLOR_BGR2YUV  # convert between RGB/BGR and YUV

    rgb2yuv = cv2.COLOR_RGB2YUV

    yuv2bgr = cv2.COLOR_YUV2BGR

    yuv2rgb = cv2.COLOR_YUV2RGB

    # ------
    yuv2rgb_nv12 = cv2.COLOR_YUV2RGB_NV12  # YUV 4:2:0 family to RGB.

    yuv2bgr_nv12 = cv2.COLOR_YUV2BGR_NV12

    yuv2rgb_nv21 = cv2.COLOR_YUV2RGB_NV21

    yuv2bgr_nv21 = cv2.COLOR_YUV2BGR_NV21

    yuv420sp2rgb = cv2.COLOR_YUV420sp2RGB

    yuv420sp2bgr = cv2.COLOR_YUV420sp2BGR

    yuv2rgba_nv12 = cv2.COLOR_YUV2RGBA_NV12

    yuv2bgra_nv12 = cv2.COLOR_YUV2BGRA_NV12

    yuv2rgba_nv21 = cv2.COLOR_YUV2RGBA_NV21

    yuv2bgra_nv21 = cv2.COLOR_YUV2BGRA_NV21

    yuv420sp2rgba = cv2.COLOR_YUV420sp2RGBA

    yuv420sp2bgra = cv2.COLOR_YUV420sp2BGRA

    yuv2rgb_yv12 = cv2.COLOR_YUV2RGB_YV12

    yuv2bgr_yv12 = cv2.COLOR_YUV2BGR_YV12

    yuv2rgb_iyuv = cv2.COLOR_YUV2RGB_IYUV

    yuv2bgr_iyuv = cv2.COLOR_YUV2BGR_IYUV

    yuv2rgb_i420 = cv2.COLOR_YUV2RGB_I420

    yuv2bgr_i420 = cv2.COLOR_YUV2BGR_I420

    yuv420p2rgb = cv2.COLOR_YUV420p2RGB

    yuv420p2bgr = cv2.COLOR_YUV420p2BGR

    yuv2rgba_yv12 = cv2.COLOR_YUV2RGBA_YV12

    yuv2bgra_yv12 = cv2.COLOR_YUV2BGRA_YV12

    yuv2rgba_iyuv = cv2.COLOR_YUV2RGBA_IYUV

    yuv2bgra_iyuv = cv2.COLOR_YUV2BGRA_IYUV

    yuv2rgba_i420 = cv2.COLOR_YUV2RGBA_I420

    yuv2bgra_i420 = cv2.COLOR_YUV2BGRA_I420

    yuv420p2rgba = cv2.COLOR_YUV420p2RGBA

    yuv420p2bgra = cv2.COLOR_YUV420p2BGRA

    yuv2gray_420 = cv2.COLOR_YUV2GRAY_420

    yuv2gray_nv21 = cv2.COLOR_YUV2GRAY_NV21

    yuv2gray_nv12 = cv2.COLOR_YUV2GRAY_NV12

    yuv2gray_yv12 = cv2.COLOR_YUV2GRAY_YV12

    yuv2gray_iyuv = cv2.COLOR_YUV2GRAY_IYUV

    yuv2gray_i420 = cv2.COLOR_YUV2GRAY_I420

    yuv420sp2gray = cv2.COLOR_YUV420sp2GRAY

    yuv420p2gray = cv2.COLOR_YUV420p2GRAY

    # -- ---
    yuv2rgb_uyvy = cv2.COLOR_YUV2RGB_UYVY  # YUV 4:2:2 family to RGB.

    yuv2bgr_uyvy = cv2.COLOR_YUV2BGR_UYVY

    yuv2rgb_y422 = cv2.COLOR_YUV2RGB_Y422

    yuv2bgr_y422 = cv2.COLOR_YUV2BGR_Y422

    yuv2rgb_uynv = cv2.COLOR_YUV2RGB_UYNV

    yuv2bgr_uynv = cv2.COLOR_YUV2BGR_UYNV

    yuv2rgba_uyvy = cv2.COLOR_YUV2RGBA_UYVY

    yuv2bgra_uyvy = cv2.COLOR_YUV2BGRA_UYVY

    yuv2rgba_y422 = cv2.COLOR_YUV2RGBA_Y422

    yuv2bgra_y422 = cv2.COLOR_YUV2BGRA_Y422

    yuv2rgba_uynv = cv2.COLOR_YUV2RGBA_UYNV

    yuv2bgra_uynv = cv2.COLOR_YUV2BGRA_UYNV

    yuv2rgb_yuy2 = cv2.COLOR_YUV2RGB_YUY2

    yuv2bgr_yuy2 = cv2.COLOR_YUV2BGR_YUY2

    yuv2rgb_yvyu = cv2.COLOR_YUV2RGB_YVYU

    yuv2bgr_yvyu = cv2.COLOR_YUV2BGR_YVYU

    yuv2rgb_yuyv = cv2.COLOR_YUV2RGB_YUYV

    yuv2bgr_yuyv = cv2.COLOR_YUV2BGR_YUYV

    yuv2rgb_yunv = cv2.COLOR_YUV2RGB_YUNV

    yuv2bgr_yunv = cv2.COLOR_YUV2BGR_YUNV

    yuv2rgba_yuy2 = cv2.COLOR_YUV2RGBA_YUY2

    yuv2bgra_yuy2 = cv2.COLOR_YUV2BGRA_YUY2

    yuv2rgba_yvyu = cv2.COLOR_YUV2RGBA_YVYU

    yuv2bgra_yvyu = cv2.COLOR_YUV2BGRA_YVYU

    yuv2rgba_yuyv = cv2.COLOR_YUV2RGBA_YUYV

    yuv2bgra_yuyv = cv2.COLOR_YUV2BGRA_YUYV

    yuv2rgba_yunv = cv2.COLOR_YUV2RGBA_YUNV

    yuv2bgra_yunv = cv2.COLOR_YUV2BGRA_YUNV

    yuv2gray_uyvy = cv2.COLOR_YUV2GRAY_UYVY

    yuv2gray_yuy2 = cv2.COLOR_YUV2GRAY_YUY2

    yuv2gray_y422 = cv2.COLOR_YUV2GRAY_Y422
    yuv2gray_uynv = cv2.COLOR_YUV2GRAY_UYNV

    yuv2gray_yvyu = cv2.COLOR_YUV2GRAY_YVYU

    yuv2gray_yuyv = cv2.COLOR_YUV2GRAY_YUYV

    yuv2gray_yunv = cv2.COLOR_YUV2GRAY_YUNV

    # -----
    rgba2mrgba = cv2.COLOR_RGBA2mRGBA  # alpha premultiplication

    mrgba2rgba = cv2.COLOR_mRGBA2RGBA

    # -- ---
    rgb2yuv_i420 = cv2.COLOR_RGB2YUV_I420  # RGB to YUV 4:2:0 family.

    bgr2yuv_i420 = cv2.COLOR_BGR2YUV_I420

    rgb2yuv_iyuv = cv2.COLOR_RGB2YUV_IYUV

    bgr2yuv_iyuv = cv2.COLOR_BGR2YUV_IYUV

    rgba2yuv_i420 = cv2.COLOR_RGBA2YUV_I420

    bgra2yuv_i420 = cv2.COLOR_BGRA2YUV_I420

    rgba2yuv_iyuv = cv2.COLOR_RGBA2YUV_IYUV

    bgra2yuv_iyuv = cv2.COLOR_BGRA2YUV_IYUV

    rgb2yuv_yv12 = cv2.COLOR_RGB2YUV_YV12

    bgr2yuv_yv12 = cv2.COLOR_BGR2YUV_YV12

    rgba2yuv_yv12 = cv2.COLOR_RGBA2YUV_YV12

    bgra2yuv_yv12 = cv2.COLOR_BGRA2YUV_YV12

    # --- ---
    bayerbg2bgr = (
        cv2.COLOR_BayerBG2BGR
    )  # Demosaicing, see color conversions for additional information. #equivalent to RGGB Bayer pattern

    bayergb2bgr = cv2.COLOR_BayerGB2BGR  # equivalent to GRBG Bayer pattern

    bayerrg2bgr = cv2.COLOR_BayerRG2BGR  # equivalent to BGGR Bayer pattern

    bayergr2bgr = cv2.COLOR_BayerGR2BGR  # equivalent to GBRG Bayer pattern

    bayerrggb2bgr = cv2.COLOR_BayerRGGB2BGR

    bayergrbg2bgr = cv2.COLOR_BayerGRBG2BGR

    bayerbggr2bgr = cv2.COLOR_BayerBGGR2BGR

    bayergbrg2bgr = cv2.COLOR_BayerGBRG2BGR

    bayerrggb2rgb = cv2.COLOR_BayerRGGB2RGB

    bayergrbg2rgb = cv2.COLOR_BayerGRBG2RGB

    bayerbggr2rgb = cv2.COLOR_BayerBGGR2RGB

    bayergbrg2rgb = cv2.COLOR_BayerGBRG2RGB

    bayerbg2rgb = cv2.COLOR_BayerBG2RGB  # equivalent to RGGB Bayer pattern

    bayergb2rgb = cv2.COLOR_BayerGB2RGB  # equivalent to GRBG Bayer pattern

    bayerrg2rgb = cv2.COLOR_BayerRG2RGB  # equivalent to BGGR Bayer pattern

    bayergr2rgb = cv2.COLOR_BayerGR2RGB  # equivalent to GBRG Bayer pattern

    bayerbg2gray = cv2.COLOR_BayerBG2GRAY  # equivalent to RGGB Bayer pattern

    bayergb2gray = cv2.COLOR_BayerGB2GRAY  # equivalent to GRBG Bayer pattern

    bayerrg2gray = cv2.COLOR_BayerRG2GRAY  # equivalent to BGGR Bayer pattern

    bayergr2gray = cv2.COLOR_BayerGR2GRAY  # equivalent to GBRG Bayer pattern

    bayerrggb2gray = cv2.COLOR_BayerRGGB2GRAY

    bayergrbg2gray = cv2.COLOR_BayerGRBG2GRAY

    bayerbggr2gray = cv2.COLOR_BayerBGGR2GRAY

    bayergbrg2gray = cv2.COLOR_BayerGBRG2GRAY

    # --- ---
    bayerbg2bgr_vng = (
        cv2.COLOR_BayerBG2BGR_VNG
    )  # Demosaicing using Variable Number of Gradients. #equivalent to RGGB Bayer pattern

    bayergb2bgr_vng = cv2.COLOR_BayerGB2BGR_VNG  # equivalent to GRBG Bayer pattern

    bayerrg2bgr_vng = cv2.COLOR_BayerRG2BGR_VNG  # equivalent to BGGR Bayer pattern

    bayergr2bgr_vng = cv2.COLOR_BayerGR2BGR_VNG  # equivalent to GBRG Bayer pattern

    bayerrggb2bgr_vng = cv2.COLOR_BayerRGGB2BGR_VNG

    bayergrbg2bgr_vng = cv2.COLOR_BayerGRBG2BGR_VNG

    bayerbggr2bgr_vng = cv2.COLOR_BayerBGGR2BGR_VNG

    bayergbrg2bgr_vng = cv2.COLOR_BayerGBRG2BGR_VNG

    bayerrggb2rgb_vng = cv2.COLOR_BayerRGGB2RGB_VNG

    bayergrbg2rgb_vng = cv2.COLOR_BayerGRBG2RGB_VNG

    bayerbggr2rgb_vng = cv2.COLOR_BayerBGGR2RGB_VNG

    bayergbrg2rgb_vng = cv2.COLOR_BayerGBRG2RGB_VNG

    bayerbg2rgb_vng = cv2.COLOR_BayerBG2RGB_VNG  # equivalent to RGGB Bayer pattern

    bayergb2rgb_vng = cv2.COLOR_BayerGB2RGB_VNG  # equivalent to GRBG Bayer pattern

    bayerrg2rgb_vng = cv2.COLOR_BayerRG2RGB_VNG  # equivalent to BGGR Bayer pattern

    bayergr2rgb_vng = cv2.COLOR_BayerGR2RGB_VNG  # equivalent to GBRG Bayer pattern

    bayerbg2bgr_ea = (
        cv2.COLOR_BayerBG2BGR_EA
    )  # Edge-Aware Demosaicing. #equivalent to RGGB Bayer pattern

    bayergb2bgr_ea = cv2.COLOR_BayerGB2BGR_EA  # equivalent to GRBG Bayer pattern

    bayerrg2bgr_ea = cv2.COLOR_BayerRG2BGR_EA  # equivalent to BGGR Bayer pattern

    bayergr2bgr_ea = cv2.COLOR_BayerGR2BGR_EA  # equivalent to GBRG Bayer pattern

    bayerrggb2bgr_ea = cv2.COLOR_BayerRGGB2BGR_EA

    bayergrbg2bgr_ea = cv2.COLOR_BayerGRBG2BGR_EA

    bayerbggr2bgr_ea = cv2.COLOR_BayerBGGR2BGR_EA

    bayergbrg2bgr_ea = cv2.COLOR_BayerGBRG2BGR_EA

    bayerrggb2rgb_ea = cv2.COLOR_BayerRGGB2RGB_EA

    bayergrbg2rgb_ea = cv2.COLOR_BayerGRBG2RGB_EA

    bayerbggr2rgb_ea = cv2.COLOR_BayerBGGR2RGB_EA

    bayergbrg2rgb_ea = cv2.COLOR_BayerGBRG2RGB_EA

    bayerbg2rgb_ea = cv2.COLOR_BayerBG2RGB_EA  # equivalent to RGGB Bayer pattern

    bayergb2rgb_ea = cv2.COLOR_BayerGB2RGB_EA  # equivalent to GRBG Bayer pattern

    bayerrg2rgb_ea = cv2.COLOR_BayerRG2RGB_EA  # equivalent to BGGR Bayer pattern

    bayergr2rgb_ea = cv2.COLOR_BayerGR2RGB_EA  # equivalent to GBRG Bayer pattern

    # --- ---
    bayerbg2bgra = (
        cv2.COLOR_BayerBG2BGRA
    )  # Demosaicing with alpha channel. #equivalent to RGGB Bayer pattern

    bayergb2bgra = cv2.COLOR_BayerGB2BGRA  # equivalent to GRBG Bayer pattern

    bayerrg2bgra = cv2.COLOR_BayerRG2BGRA  # equivalent to BGGR Bayer pattern

    bayergr2bgra = cv2.COLOR_BayerGR2BGRA  # equivalent to GBRG Bayer pattern

    bayerrggb2bgra = cv2.COLOR_BayerRGGB2BGRA

    bayergrbg2bgra = cv2.COLOR_BayerGRBG2BGRA

    bayerbggr2bgra = cv2.COLOR_BayerBGGR2BGRA

    bayergbrg2bgra = cv2.COLOR_BayerGBRG2BGRA

    bayerrggb2rgba = cv2.COLOR_BayerRGGB2RGBA

    bayergrbg2rgba = cv2.COLOR_BayerGRBG2RGBA

    bayerbggr2rgba = cv2.COLOR_BayerBGGR2RGBA

    bayergbrg2rgba = cv2.COLOR_BayerGBRG2RGBA

    bayerbg2rgba = cv2.COLOR_BayerBG2RGBA  # equivalent to RGGB Bayer pattern
    bayergb2rgba = cv2.COLOR_BayerGB2RGBA  # equivalent to GRBG Bayer pattern
    bayerrg2rgba = cv2.COLOR_BayerRG2RGBA  # equivalent to BGGR Bayer pattern
    bayergr2rgba = cv2.COLOR_BayerGR2RGBA  # equivalent to GBRG Bayer pattern

    # -----
    colorcvt_max = cv2.COLOR_COLORCVT_MAX


if __name__ == "__main__":
    print(ColorConversionEnum.rgb2rgba, ColorConversionEnum.rgb2rgba.value)
