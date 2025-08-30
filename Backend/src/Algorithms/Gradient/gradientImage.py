import cv2
import numpy as np


def gradient_mask_frame_image(gray, pct=0.7):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # mag = cv2.GaussianBlur(mag, (3, 3), 0)
    mag = mag / (mag.max() + 1e-6)

    t = np.quantile(mag, pct)
    m = (mag >= t).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
