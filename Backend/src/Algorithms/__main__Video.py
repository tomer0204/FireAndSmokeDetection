import cv2
import numpy as np

from Backend.src.Algorithms.ColorSpace.ColorSpaceFireVideo import color_mask_frame
from Backend.src.Algorithms.Gradient.gradientVideo import gradient_mask_frame
from Backend.src.Algorithms.OpticalFlowFarneback.OpticalFlowVideo import (
    compute_optical_flow_and_divergence,
)
from Backend.src.Algorithms.WaveletTransform.WaveletVideo import wavelet_mask_frame

VIDEO_PATH = "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Dataset/Video/Train/fire_video9.mov"
OUTPUT_PREFIX = "video_file"
QWAVE = 0.9
QGRAD = 0.9
FPS = 17
SHOW = True
CODEC = "mp4v"


def overlay_from_masks(frame, m_color, m_wave):
    ov = frame.astype(np.float32)
    c = np.zeros_like(frame)
    c[:, :, 1] = m_color
    w = np.zeros_like(frame)
    w[:, :, 2] = m_wave
    ov = cv2.addWeighted(ov, 1.0, c.astype(np.float32), 0.5, 0)
    ov = cv2.addWeighted(ov, 1.0, w.astype(np.float32), 0.5, 0)
    return ov.astype(np.uint8)


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

    out_color = cv2.VideoWriter(
        OUTPUT_PREFIX + "_color.mp4", fourcc, FPS, (w, h), False
    )
    out_opticalflow = cv2.VideoWriter(
        OUTPUT_PREFIX + "_opticalflow.mp4", fourcc, FPS, (w, h), True
    )
    out_wave = cv2.VideoWriter(
        OUTPUT_PREFIX + "_wavelet.mp4", fourcc, FPS, (w, h), False
    )
    out_grad = cv2.VideoWriter(
        OUTPUT_PREFIX + "_gradient.mp4", fourcc, FPS, (w, h), False
    )
    out_mask = cv2.VideoWriter(
        OUTPUT_PREFIX + "_fused_mask.mp4", fourcc, FPS, (w, h), False
    )
    out_vis = cv2.VideoWriter(
        OUTPUT_PREFIX + "_fused_vis.mp4", fourcc, FPS, (w, h), True
    )
    out_panel = cv2.VideoWriter(
        OUTPUT_PREFIX + "_panel.mp4", fourcc, FPS, (w * 3, h), True
    )

    if SHOW:
        cv2.namedWindow("panel", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("panel", min(1280, w * 3), min(720, h))

    prev_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        m_color = color_mask_frame(frame)
        m_wave = wavelet_mask_frame(gray, QWAVE)
        m_grad = gradient_mask_frame(gray, QGRAD)

        mask_div = np.zeros_like(gray, dtype=np.uint8)

        if prev_gray is not None:
            flow, divergence, mask_div = compute_optical_flow_and_divergence(
                prev_gray, gray, threshold=2.0
            )
            opticalflow_vectors = cv2.cvtColor(mask_div, cv2.COLOR_GRAY2BGR)
        else:
            opticalflow_vectors = frame.copy()

        m_color_gradient = cv2.bitwise_and(m_color, m_grad)
        m_color_wavelet = cv2.bitwise_and(m_wave, m_color)
        m_color_divergence = cv2.bitwise_and(mask_div, m_color)
        vis_mid = overlay_from_masks(frame, m_color, m_wave)

        out_color.write(m_color)
        out_wave.write(m_wave)
        out_grad.write(m_grad)
        out_opticalflow.write(opticalflow_vectors)
        out_mask.write(m_color_gradient)
        out_vis.write(vis_mid)

        panel = three_panel(
            label(frame, "Original"),
            label(m_color, "Color"),
            label(m_wave, "wavelet"),
        )
        out_panel.write(panel)

        if SHOW:
            cv2.imshow("panel", panel)
            if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
                break

        prev_gray = gray

    cap.release()
    out_color.release()
    out_wave.release()
    out_grad.release()
    out_mask.release()
    out_vis.release()
    out_panel.release()
    out_opticalflow.release()
    if SHOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
