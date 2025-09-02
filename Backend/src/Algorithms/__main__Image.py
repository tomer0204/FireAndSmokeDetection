import cv2
import numpy as np

from Backend.src.Algorithms.ColorSpace.colorSpaceImage import color_mask_frame_image
from Backend.src.Algorithms.Gradient.gradientImage import gradient_mask_frame_image


def two_panel(left, right):
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (w, h))
    right = cv2.resize(right, (w, h))
    return np.concatenate([left, right], axis=1)


def main():
    IMAGE_PATH = "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Dataset/Image/Train/bothFireAndSmoke_CV004377.jpg"
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError(f"cannot open image: {IMAGE_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m_grad = gradient_mask_frame_image(gray, pct=0.9)
    m_color = color_mask_frame_image(img)
    panel = two_panel(img, m_color)
    cv2.imshow("Gradient Detection", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
