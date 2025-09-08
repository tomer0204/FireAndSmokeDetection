import cv2
import numpy as np


def color_mask_frame_image(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_fire1 = np.array([18, 50, 50], np.uint8)
    upper_fire1 = np.array([50, 255, 255], np.uint8)
    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower_fire_ycrcb = np.array([80, 150, 20], np.uint8)
    upper_fire_ycrcb = np.array([255, 200, 120], np.uint8)
    mask2 = cv2.inRange(ycrcb, lower_fire_ycrcb, upper_fire_ycrcb)
    result = cv2.bitwise_and(
        mask1,
        mask2,
    )

    return result
