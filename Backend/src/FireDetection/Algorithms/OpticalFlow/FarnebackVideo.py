import cv2 as cv
import numpy as np


def compute_optical_flow_and_divergence(prev_gray, gray, threshold=2.0):
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    u = flow[..., 0]
    v = flow[..., 1]

    du_dx = cv.Sobel(u, cv.CV_32F, 1, 0, ksize=3)
    dv_dy = cv.Sobel(v, cv.CV_32F, 0, 1, ksize=3)

    divergence = du_dx + dv_dy

    # מסיכה בינארית לפי סף
    mask = (divergence > threshold).astype(np.uint8) * 255

    return flow, divergence, mask
