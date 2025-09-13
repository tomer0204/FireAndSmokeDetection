import cv2
import numpy as np


def color_mask_frame(bgr):

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_norm = cv2.equalizeHist(v)
    hsv_norm = cv2.merge([h, s, v_norm])

    lower_fire1 = np.array([18, 80, 80], np.uint8)
    upper_fire1 = np.array([50, 255, 255], np.uint8)
    mask_hsv = cv2.inRange(hsv_norm, lower_fire1, upper_fire1)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower_fire_ycrcb = np.array([80, 150, 20], np.uint8)
    upper_fire_ycrcb = np.array([255, 200, 120], np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_fire_ycrcb, upper_fire_ycrcb)

    result = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    return result
