import cv2
import numpy as np

from Backend.src.FireDetection.Images.Algorithms.HOG.HOG_Image import hog_fire_detection
from Backend.src.FireDetection.Videos.Algorithms.BeysianFusion.Fusion import (
    fuse_and_draw,
)
from Backend.src.FireDetection.Videos.Algorithms.ColorSpace.ColorSpaceFireVideo import (
    color_prob_map_ycrcb,
)
from Backend.src.FireDetection.Videos.Algorithms.Gradient.gradientVideo import (
    gradient_mask_frame,
)
from Backend.src.FireDetection.Videos.Algorithms.OpticalFlow.FarnebackVideo import (
    compute_optical_flow_and_divergence,
)
from Backend.src.FireDetection.Videos.Algorithms.WaveletTransform.WaveletFireVideo import (
    wavelet_mask_frame,
)

VIDEO_PATH = "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetVideos/Video/Train/fire14.avi"
OUTPUT_PREFIX = "video_file"
QWAVE = 0.95
QGRAD = 0.95
FPS = 17
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prob_map = color_prob_map_ycrcb(frame)
        prob_gray = (prob_map * 255).astype(np.uint8)
        prob_heat = cv2.applyColorMap(prob_gray, cv2.COLORMAP_JET)

        m_grad = gradient_mask_frame(gray, QGRAD)
        _, m_hog = hog_fire_detection(frame)

        masks = wavelet_mask_frame(gray, pct=QWAVE, levels=(1, 2, 3))
        m_wave3 = masks[3]

        if prev_gray is not None:
            _, _, m_div = compute_optical_flow_and_divergence(
                prev_gray, gray, threshold=5
            )
        else:
            m_div = np.zeros_like(gray)

        fused_mask, out_bbox = fuse_and_draw(
            frame,
            m_div,
            m_wave3,
            prob_gray,
            m_grad,
            w_div=0.2,
            w_wave=0.25,
            w_color=0.5,
            w_grad=0.05,
            thresh=0.7,
        )

        panel = three_panel(
            label(frame, "Original"),
            label(fused_mask, "Fusion"),
            label(prob_heat, " YCrCb Prob Heatmap"),
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
