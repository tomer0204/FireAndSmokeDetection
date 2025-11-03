import cv2
import numpy as np


def hog_fire_detection(frame):
    # המרת תמונה לגווני אפור
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # יצירת אובייקט HOG עם פרמטרים סטנדרטיים
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    features = hog.compute(gray)

    mask = cv2.normalize(features, None, 0, 255, cv2.NORM_MINMAX)
    mask = mask.flatten()
    mask = np.uint8(mask > 128) * 255

    mask = cv2.resize(mask.reshape(-1, 1), (gray.shape[1], gray.shape[0]))

    return features, mask
