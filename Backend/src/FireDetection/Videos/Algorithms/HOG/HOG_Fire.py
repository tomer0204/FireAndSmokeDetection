from collections import deque

import cv2
import numpy as np
from skimage.feature import hog

hog_history = deque(maxlen=15)


def hog_fire_detection(frame, var_thresh=0.02):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )

    hog_history.append(features)

    detected = False
    if len(hog_history) == hog_history.maxlen:
        mat = np.array(hog_history)
        variance = np.var(mat, axis=0).mean()
        detected = variance > var_thresh

    hog_vis = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return detected, hog_vis
