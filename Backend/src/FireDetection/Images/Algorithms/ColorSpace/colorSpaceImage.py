def color_prob_map_ycrcb(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_min, y_max = 80, 255
    cr_min, cr_max = 150, 200
    cb_min, cb_max = 20, 120

    def channel_prob(channel, cmin, cmax, invert=False):
        c_center = (cmin + cmax) / 2
        c_range = (cmax - cmin) / 2
        prob = 1 - np.abs(channel - c_center) / c_range
        prob = np.clip(prob, 0, 1)
        if invert:
            prob = 1 - prob
        return prob

    prob_y = channel_prob(y, y_min, y_max)
    prob_cr = channel_prob(cr, cr_min, cr_max)
    prob_cb = channel_prob(cb, cb_min, cb_max, invert=True)

    prob_map = prob_y * prob_cr * prob_cb

    return prob_map


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

    return mask2
