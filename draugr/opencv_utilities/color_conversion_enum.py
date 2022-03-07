from enum import Enum

import cv2

__all__ = ["ColorConversionCodesEnum"]

# TODO: FINISH


class ColorConversionCodesEnum(Enum):
    """
    the color conversion codes"""

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
    # 5 ---

    gray2bgr565 = (
        cv2.COLOR_GRAY2BGR565
    )  # convert between grayscale to BGR565 (16-bit images)

    bgr5652gray = cv2.COLOR_BGR5652GRAY

    # 8 ---
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

    # 17 ---
    gray2bgr555 = (
        cv2.COLOR_GRAY2BGR555
    )  # convert between grayscale and BGR555 (16-bit images)

    bgr5552gray = cv2.COLOR_BGR5552GRAY

    # 20 ---
    bgr2xyz = cv2.COLOR_BGR2XYZ  # convert RGB/BGR to CIE XYZ, color conversions

    rgb2xyz = cv2.COLOR_RGB2XYZ

    xyz2bgr = cv2.COLOR_XYZ2BGR

    xyz2rgb = cv2.COLOR_XYZ2RGB

    # 25 ---
    bgr2ycrcb = (
        cv2.COLOR_BGR2YCrCb
    )  # convert RGB/BGR to luma-chroma (aka YCC), color conversions

    rgb2ycrcb = cv2.COLOR_RGB2YCrCb

    ycrcb2bgr = cv2.COLOR_YCrCb2BGR

    ycrcb2rgb = cv2.COLOR_YCrCb2RGB

    # 32 ---
    bgr2hsv = (
        cv2.COLOR_BGR2HSV
    )  # convert RGB/BGR to HSV (hue saturation value) with H range 0..180 if 8 bit image, color conversions

    rgb2hsv = cv2.COLOR_RGB2HSV
    # 37 ---
    bgr2lab = cv2.COLOR_BGR2Lab  # convert RGB/BGR to CIE Lab, color conversions

    rgb2lab = cv2.COLOR_RGB2Lab
    # 40 ---
    bgr2luv = cv2.COLOR_BGR2Luv  # convert RGB/BGR to CIE Luv, color conversions

    rgb2luv = cv2.COLOR_RGB2Luv

    # 43 ---
    bgr2hls = (
        cv2.COLOR_BGR2HLS
    )  # convert RGB/BGR to HLS (hue lightness saturation) with H range 0..180 if 8 bit image, color conversions

    rgb2hls = cv2.COLOR_RGB2HLS

    # 48 ---
    hsv2bgr = (
        cv2.COLOR_HSV2BGR
    )  # backward conversions HSV to RGB/BGR with H range 0..180 if 8 bit image

    hsv2rgb = cv2.COLOR_HSV2RGB

    lab2bgr = cv2.COLOR_Lab2BGR

    lab2rgb = cv2.COLOR_Lab2RGB

    luv2bgr = cv2.COLOR_Luv2BGR

    luv2rgb = cv2.COLOR_Luv2RGB

    # 57 ---
    hls2bgr = (
        cv2.COLOR_HLS2BGR
    )  # backward conversions HLS to RGB/BGR with H range 0..180 if 8 bit image

    hls2rgb = cv2.COLOR_HLS2RGB

    # 62 ---
    bgr2hsv_full = (
        cv2.COLOR_BGR2HSV_FULL
    )  # convert RGB/BGR to HSV (hue saturation value) with H range 0..255 if 8 bit image, color conversions

    rgb2hsv_full = cv2.COLOR_RGB2HSV_FULL

    # 67 ---
    bgr2hls_full = (
        cv2.COLOR_BGR2HLS_FULL
    )  # convert RGB/BGR to HLS (hue lightness saturation) with H range 0..255 if 8 bit image, color conversions

    rgb2hls_full = cv2.COLOR_RGB2HLS_FULL

    # 72 ---
    hsv2bgr_full = (
        cv2.COLOR_HSV2BGR_FULL
    )  # backward conversions HSV to RGB/BGR with H range 0..255 if 8 bit image

    hsv2rgb_full = cv2.COLOR_HSV2RGB_FULL

    # 77 ---
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

    # 90 ---
    bgr2yuv = cv2.COLOR_BGR2YUV  # convert between RGB/BGR and YUV

    rgb2yuv = cv2.COLOR_RGB2YUV

    yuv2bgr = cv2.COLOR_YUV2BGR

    yuv2rgb = cv2.COLOR_YUV2RGB

    # 95 ---
    a96 = cv2.COLOR_YUV2RGB_NV12  # YUV 4:2:0 family to RGB.

    a97 = cv2.COLOR_YUV2BGR_NV12

    a98 = cv2.COLOR_YUV2RGB_NV21

    a99 = cv2.COLOR_YUV2BGR_NV21

    a100 = cv2.COLOR_YUV420sp2RGB

    a101 = cv2.COLOR_YUV420sp2BGR

    a102 = cv2.COLOR_YUV2RGBA_NV12

    a103 = cv2.COLOR_YUV2BGRA_NV12

    a104 = cv2.COLOR_YUV2RGBA_NV21

    a105 = cv2.COLOR_YUV2BGRA_NV21

    a106 = cv2.COLOR_YUV420sp2RGBA

    a107 = cv2.COLOR_YUV420sp2BGRA

    a108 = cv2.COLOR_YUV2RGB_YV12

    a109 = cv2.COLOR_YUV2BGR_YV12

    a110 = cv2.COLOR_YUV2RGB_IYUV

    a111 = cv2.COLOR_YUV2BGR_IYUV

    a112 = cv2.COLOR_YUV2RGB_I420

    a113 = cv2.COLOR_YUV2BGR_I420

    a114 = cv2.COLOR_YUV420p2RGB

    a115 = cv2.COLOR_YUV420p2BGR

    a116 = cv2.COLOR_YUV2RGBA_YV12

    a117 = cv2.COLOR_YUV2BGRA_YV12

    a118 = cv2.COLOR_YUV2RGBA_IYUV

    a119 = cv2.COLOR_YUV2BGRA_IYUV

    a120 = cv2.COLOR_YUV2RGBA_I420

    a121 = cv2.COLOR_YUV2BGRA_I420

    a122 = cv2.COLOR_YUV420p2RGBA

    a123 = cv2.COLOR_YUV420p2BGRA

    a124 = cv2.COLOR_YUV2GRAY_420

    a125 = cv2.COLOR_YUV2GRAY_NV21

    a126 = cv2.COLOR_YUV2GRAY_NV12

    a127 = cv2.COLOR_YUV2GRAY_YV12

    a128 = cv2.COLOR_YUV2GRAY_IYUV

    a129 = cv2.COLOR_YUV2GRAY_I420

    a130 = cv2.COLOR_YUV420sp2GRAY

    a131 = cv2.COLOR_YUV420p2GRAY

    # 132 ---
    a133 = cv2.COLOR_YUV2RGB_UYVY  # YUV 4:2:2 family to RGB.

    a134 = cv2.COLOR_YUV2BGR_UYVY

    a135 = cv2.COLOR_YUV2RGB_Y422

    a136 = cv2.COLOR_YUV2BGR_Y422

    a137 = cv2.COLOR_YUV2RGB_UYNV

    a138 = cv2.COLOR_YUV2BGR_UYNV

    a139 = cv2.COLOR_YUV2RGBA_UYVY

    a140 = cv2.COLOR_YUV2BGRA_UYVY

    a141 = cv2.COLOR_YUV2RGBA_Y422

    a142 = cv2.COLOR_YUV2BGRA_Y422

    a143 = cv2.COLOR_YUV2RGBA_UYNV

    a144 = cv2.COLOR_YUV2BGRA_UYNV

    a145 = cv2.COLOR_YUV2RGB_YUY2

    a146 = cv2.COLOR_YUV2BGR_YUY2

    a147 = cv2.COLOR_YUV2RGB_YVYU

    a148 = cv2.COLOR_YUV2BGR_YVYU

    a149 = cv2.COLOR_YUV2RGB_YUYV

    a150 = cv2.COLOR_YUV2BGR_YUYV

    a151 = cv2.COLOR_YUV2RGB_YUNV

    a152 = cv2.COLOR_YUV2BGR_YUNV

    a153 = cv2.COLOR_YUV2RGBA_YUY2

    a154 = cv2.COLOR_YUV2BGRA_YUY2

    a155 = cv2.COLOR_YUV2RGBA_YVYU

    a156 = cv2.COLOR_YUV2BGRA_YVYU

    a157 = cv2.COLOR_YUV2RGBA_YUYV

    a158 = cv2.COLOR_YUV2BGRA_YUYV

    a159 = cv2.COLOR_YUV2RGBA_YUNV

    a160 = cv2.COLOR_YUV2BGRA_YUNV

    a161 = cv2.COLOR_YUV2GRAY_UYVY

    a162 = cv2.COLOR_YUV2GRAY_YUY2

    a163 = cv2.COLOR_YUV2GRAY_Y422
    a164 = cv2.COLOR_YUV2GRAY_UYNV

    a165 = cv2.COLOR_YUV2GRAY_YVYU

    a166 = cv2.COLOR_YUV2GRAY_YUYV

    a167 = cv2.COLOR_YUV2GRAY_YUNV

    # 168 ---
    rgba2mrgba = cv2.COLOR_RGBA2mRGBA  # alpha premultiplication

    mrgba2rgba = cv2.COLOR_mRGBA2RGBA

    # 171 ---
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

    # 184 ---
    bayerbg2bgr = (
        cv2.COLOR_BayerBG2BGR
    )  # Demosaicing, see color conversions for additional information. #equivalent to RGGB Bayer pattern

    a187 = cv2.COLOR_BayerGB2BGR  # equivalent to GRBG Bayer pattern

    a188 = cv2.COLOR_BayerRG2BGR  # equivalent to BGGR Bayer pattern

    a189 = cv2.COLOR_BayerGR2BGR  # equivalent to GBRG Bayer pattern

    a190 = cv2.COLOR_BayerRGGB2BGR

    a191 = cv2.COLOR_BayerGRBG2BGR

    a192 = cv2.COLOR_BayerBGGR2BGR

    a193 = cv2.COLOR_BayerGBRG2BGR

    a194 = cv2.COLOR_BayerRGGB2RGB

    a195 = cv2.COLOR_BayerGRBG2RGB

    a196 = cv2.COLOR_BayerBGGR2RGB

    a197 = cv2.COLOR_BayerGBRG2RGB

    a198 = cv2.COLOR_BayerBG2RGB  # equivalent to RGGB Bayer pattern

    a199 = cv2.COLOR_BayerGB2RGB  # equivalent to GRBG Bayer pattern

    a200 = cv2.COLOR_BayerRG2RGB  # equivalent to BGGR Bayer pattern

    a201 = cv2.COLOR_BayerGR2RGB  # equivalent to GBRG Bayer pattern

    a202 = cv2.COLOR_BayerBG2GRAY  # equivalent to RGGB Bayer pattern

    a203 = cv2.COLOR_BayerGB2GRAY  # equivalent to GRBG Bayer pattern

    a204 = cv2.COLOR_BayerRG2GRAY  # equivalent to BGGR Bayer pattern

    a205 = cv2.COLOR_BayerGR2GRAY  # equivalent to GBRG Bayer pattern

    a206 = cv2.COLOR_BayerRGGB2GRAY

    a207 = cv2.COLOR_BayerGRBG2GRAY

    a208 = cv2.COLOR_BayerBGGR2GRAY

    a209 = cv2.COLOR_BayerGBRG2GRAY

    # 210 ---
    a211 = (
        cv2.COLOR_BayerBG2BGR_VNG
    )  # Demosaicing using Variable Number of Gradients. #equivalent to RGGB Bayer pattern

    a214 = cv2.COLOR_BayerGB2BGR_VNG  # equivalent to GRBG Bayer pattern

    a215 = cv2.COLOR_BayerRG2BGR_VNG  # equivalent to BGGR Bayer pattern

    a216 = cv2.COLOR_BayerGR2BGR_VNG  # equivalent to GBRG Bayer pattern

    a217 = cv2.COLOR_BayerRGGB2BGR_VNG

    a218 = cv2.COLOR_BayerGRBG2BGR_VNG

    a219 = cv2.COLOR_BayerBGGR2BGR_VNG

    a220 = cv2.COLOR_BayerGBRG2BGR_VNG

    a221 = cv2.COLOR_BayerRGGB2RGB_VNG

    a222 = cv2.COLOR_BayerGRBG2RGB_VNG

    a223 = cv2.COLOR_BayerBGGR2RGB_VNG

    a224 = cv2.COLOR_BayerGBRG2RGB_VNG

    a225 = cv2.COLOR_BayerBG2RGB_VNG  # equivalent to RGGB Bayer pattern

    a226 = cv2.COLOR_BayerGB2RGB_VNG  # equivalent to GRBG Bayer pattern

    a227 = cv2.COLOR_BayerRG2RGB_VNG  # equivalent to BGGR Bayer pattern

    a228 = cv2.COLOR_BayerGR2RGB_VNG  # equivalent to GBRG Bayer pattern

    a229 = (
        cv2.COLOR_BayerBG2BGR_EA
    )  # Edge-Aware Demosaicing. #equivalent to RGGB Bayer pattern

    a231 = cv2.COLOR_BayerGB2BGR_EA  # equivalent to GRBG Bayer pattern

    a232 = cv2.COLOR_BayerRG2BGR_EA  # equivalent to BGGR Bayer pattern

    a233 = cv2.COLOR_BayerGR2BGR_EA  # equivalent to GBRG Bayer pattern

    a234 = cv2.COLOR_BayerRGGB2BGR_EA

    a235 = cv2.COLOR_BayerGRBG2BGR_EA

    a236 = cv2.COLOR_BayerBGGR2BGR_EA

    a237 = cv2.COLOR_BayerGBRG2BGR_EA

    a238 = cv2.COLOR_BayerRGGB2RGB_EA

    a239 = cv2.COLOR_BayerGRBG2RGB_EA

    a240 = cv2.COLOR_BayerBGGR2RGB_EA

    a241 = cv2.COLOR_BayerGBRG2RGB_EA

    a242 = cv2.COLOR_BayerBG2RGB_EA  # equivalent to RGGB Bayer pattern

    a243 = cv2.COLOR_BayerGB2RGB_EA  # equivalent to GRBG Bayer pattern

    a244 = cv2.COLOR_BayerRG2RGB_EA  # equivalent to BGGR Bayer pattern

    a245 = cv2.COLOR_BayerGR2RGB_EA  # equivalent to GBRG Bayer pattern

    # 246 ---
    a247 = (
        cv2.COLOR_BayerBG2BGRA
    )  # Demosaicing with alpha channel. #equivalent to RGGB Bayer pattern

    a249 = cv2.COLOR_BayerGB2BGRA  # equivalent to GRBG Bayer pattern

    a250 = cv2.COLOR_BayerRG2BGRA  # equivalent to BGGR Bayer pattern

    a251 = cv2.COLOR_BayerGR2BGRA  # equivalent to GBRG Bayer pattern

    a252 = cv2.COLOR_BayerRGGB2BGRA

    a253 = cv2.COLOR_BayerGRBG2BGRA

    a254 = cv2.COLOR_BayerBGGR2BGRA

    a255 = cv2.COLOR_BayerGBRG2BGRA

    a256 = cv2.COLOR_BayerRGGB2RGBA

    a257 = cv2.COLOR_BayerGRBG2RGBA

    a258 = cv2.COLOR_BayerBGGR2RGBA

    a259 = cv2.COLOR_BayerGBRG2RGBA

    bayerbg2rgba = cv2.COLOR_BayerBG2RGBA  # equivalent to RGGB Bayer pattern
    bayergb2rgba = cv2.COLOR_BayerGB2RGBA  # equivalent to GRBG Bayer pattern
    bayerrg2rgba = cv2.COLOR_BayerRG2RGBA  # equivalent to BGGR Bayer pattern
    bayergr2rgba = cv2.COLOR_BayerGR2RGBA  # equivalent to GBRG Bayer pattern

    # ---
    colorcvt_max = cv2.COLOR_COLORCVT_MAX


if __name__ == "__main__":
    print(ColorConversionCodesEnum.rgb2rgba)
