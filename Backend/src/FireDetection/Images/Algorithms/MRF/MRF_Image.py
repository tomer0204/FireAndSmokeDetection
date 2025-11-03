import cv2 as cv
import numpy as np


def probability_mrf(prob_map, ksize=7, beta=0.6, iterations=1):
    p = prob_map.astype(np.float32)
    for _ in range(iterations):
        smooth = cv.GaussianBlur(p, (ksize, ksize), 0)
        p = (1.0 - beta) * p + beta * smooth
    return np.clip(p, 0, 1)
