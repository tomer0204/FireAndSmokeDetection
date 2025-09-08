import cv2
import numpy as np


# I do optical flow with Farneback algorithm.
def lucas_kanade_optical_flow_with_divergence(
    prev_gray, next_gray, window_size=15, step=5, clip_percentile=(2, 98)
):
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = next_gray.astype(np.float64) - prev_gray.astype(np.float64)

    half_w = window_size // 2
    h, w = prev_gray.shape
    u = np.zeros((h, w))
    v = np.zeros((h, w))

    for y in range(half_w, h - half_w, step):
        for x in range(half_w, w - half_w, step):
            Ix_win = Ix[
                y - half_w : y + half_w + 1, x - half_w : x + half_w + 1
            ].flatten()
            Iy_win = Iy[
                y - half_w : y + half_w + 1, x - half_w : x + half_w + 1
            ].flatten()
            It_win = It[
                y - half_w : y + half_w + 1, x - half_w : x + half_w + 1
            ].flatten()

            A = np.vstack((Ix_win, Iy_win)).T
            b = -It_win

            ATA = A.T @ A
            if np.linalg.det(ATA) > 1e-8:
                flow = np.linalg.inv(ATA) @ (A.T @ b)
                u[y, x] = flow[0]
                v[y, x] = flow[1]

    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)
    divergence = du_dx + dv_dy

    # נרמול divergence עם חיתוך ערכים קיצוניים
    div_min, div_max = np.percentile(divergence, clip_percentile)
    divergence_clipped = np.clip(divergence, div_min, div_max)
    divergence_norm = cv2.normalize(
        divergence_clipped, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return u, v, divergence, divergence_norm


def draw_flow_vectors(frame, u, v, step=10, scale=20000):
    out = frame.copy()
    h, w = frame.shape[:2]
    mag = np.sqrt(u**2 + v**2)
    max_mag = np.max(mag) if np.max(mag) > 0 else 1
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = u[y, x], v[y, x]
            if mag[y, x] > 1e-6:
                dx = int(fx / max_mag * scale)
                dy = int(fy / max_mag * scale)
                cv2.arrowedLine(
                    out, (x, y), (x + dx, y + dy), (0, 255, 0), 1, tipLength=0.3
                )
    return out
