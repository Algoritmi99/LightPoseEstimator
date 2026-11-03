"""Raw Bayer sensor mosaics: their OpenCV conversions to grayscale, by pattern name."""
import cv2

BAYER_TO_GRAY = {
    "BG": cv2.COLOR_BayerBG2GRAY,
    "GB": cv2.COLOR_BayerGB2GRAY,
    "RG": cv2.COLOR_BayerRG2GRAY,
    "GR": cv2.COLOR_BayerGR2GRAY,
}
