import cv2
import numpy as np

from Backend.src.FireDetection.Algorithms.ColorSpace.ColorSpaceFireVideo import (
    color_mask_frame,
)
from Backend.src.FireDetection.Algorithms.FFT.FFT_FireVideo import FFTFireDetector
from Backend.src.FireDetection.Algorithms.Gradient.gradientVideo import (
    gradient_mask_frame,
)
from Backend.src.FireDetection.Algorithms.HOG.HOG_Fire import hog_fire_detection
from Backend.src.FireDetection.Algorithms.OpticalFlow.FarnebackVideo import (
    compute_optical_flow_and_divergence,
)
from Backend.src.FireDetection.Algorithms.OpticalFlow.LucasKanadeVideo import (
    lucas_kanade_optical_flow_with_divergence,
)
from Backend.src.FireDetection.Algorithms.WaveletTransform.WaveletFireVideo import (
    wavelet_mask_frame,
)

VIDEO_PATH = (
    "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Dataset/Video/Train/fire1.avi"
)
OUTPUT_PREFIX = "video_file"
QWAVE = 0.9
QGRAD = 0.95
FPS = 120
SHOW = True
CODEC = "mp4v"


def label(img, txt):
    out = img.copy()
    cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return out


def three_panel(left, mid, right):
    if mid.ndim == 2:
        mid = cv2.cvtColor(mid, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    h = min(left.shape[0], mid.shape[0], right.shape[0])
    w = min(left.shape[1], mid.shape[1], right.shape[1])
    left = cv2.resize(left, (w, h))
    mid = cv2.resize(mid, (w, h))
    right = cv2.resize(right, (w, h))
    return np.concatenate([left, mid, right], axis=1)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {VIDEO_PATH}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*CODEC)

    out_panel = cv2.VideoWriter(
        OUTPUT_PREFIX + "_panel.mp4", fourcc, FPS, (w * 3, h), True
    )

    if SHOW:
        cv2.namedWindow("panel", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("panel", min(1280, w * 3), min(720, h))

    prev_gray = None
    fft_detector = FFTFireDetector(
        fps=FPS, window_size=17, freq_band=(2, 7), power_thresh=0.7
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_div = frame.copy()
        m_color = color_mask_frame(frame)
        m_grad = gradient_mask_frame(gray, QGRAD)

        detected_hog, m_hog = hog_fire_detection(frame)

        detected_fft, m_fft = fft_detector.update(frame)

        masks = wavelet_mask_frame(gray, pct=QWAVE, levels=(1, 2, 3))
        m_wave1 = masks[1]
        m_wave2 = masks[2]
        m_wave3 = masks[3]

        if prev_gray is not None:
            flow, divergence, mask_div = compute_optical_flow_and_divergence(
                prev_gray, gray, threshold=5
            )
            u, v, divergence_raw, divergence_norm = (
                lucas_kanade_optical_flow_with_divergence(
                    prev_gray, gray, window_size=17, step=5
                )
            )
            opticalflow_lk = cv2.applyColorMap(divergence_norm, cv2.COLOR_BGR2GRAY)
        else:
            opticalflow_lk = frame.copy()

        panel = three_panel(
            label(frame, "Original"),
            label(m_color, "divergence"),
            label(m_wave3, "Wavelet L3"),
        )
        out_panel.write(panel)

        if SHOW:
            cv2.imshow("panel", panel)
            if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
                break

        prev_gray = gray

    cap.release()
    out_panel.release()
    if SHOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
