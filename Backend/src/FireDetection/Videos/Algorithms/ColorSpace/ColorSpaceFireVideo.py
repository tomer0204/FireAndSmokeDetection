import cv2
import numpy as np


def color_prob_map_ycrcb(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # טווחי האש הריאליים
    y_min, y_max = 80, 255
    cr_min, cr_max = 150, 200
    cb_min, cb_max = 20, 120

    def channel_prob(channel, cmin, cmax, invert=False):
        c_center = (cmin + cmax) / 2
        c_range = (cmax - cmin) / 2
        prob = 1 - np.abs(channel - c_center) / c_range
        prob = np.clip(prob, 0, 1)
        if invert:
            prob = 1 - prob  # הפוך – ככל ש־Cb נמוך, הסתברות גבוהה
        return prob

    prob_y = channel_prob(y, y_min, y_max)
    prob_cr = channel_prob(cr, cr_min, cr_max)
    prob_cb = channel_prob(cb, cb_min, cb_max, invert=True)  # כאן השינוי החשוב

    # שילוב הערוצים
    prob_map = prob_y * prob_cr * prob_cb

    return prob_map
