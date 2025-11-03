import cv2
import numpy as np
import pywt


def wavelet_mask_frame(gray, pct=0.9, levels=(2, 3), mask=None):
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    max_level = max(levels)
    coeffs = pywt.wavedec2(gray.astype(np.float32), "db8", level=max_level)

    masks = {}
    for lvl in levels:

        cH, cV, cD = coeffs[lvl]
        s = np.abs(cH) + np.abs(cV) + np.abs(cD)
        s = cv2.resize(s, (gray.shape[1], gray.shape[0]))
        s = s / (s.max() + 1e-6)

        t = np.quantile(s, pct)
        m = (s >= t).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

        masks[lvl] = m

    return masks
