import cv2 as cv
import numpy as np


def _local_variance(gray_f32, k=9):
    mean = cv.boxFilter(gray_f32, ddepth=-1, ksize=(k, k), normalize=True)
    mean2 = cv.boxFilter(gray_f32 * gray_f32, ddepth=-1, ksize=(k, k), normalize=True)
    var = np.clip(mean2 - mean * mean, 0, None)
    p99 = np.percentile(var, 99.0)
    if p99 <= 1e-9:
        p99 = 1.0
    return np.clip(var / p99, 0, 1)


def probability_lbpv(frame_bgr):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    S = hsv[..., 1] / 255.0
    lv = _local_variance(gray, k=9)
    P = np.clip(0.6 * lv + 0.4 * (lv * S), 0, 1)
    return P
