import os

import cv2
import numpy as np

from Backend.src.FireDetection.Images.Algorithms.BeysianFusion.FusionImage import (
    fuse_and_draw_image,
)
from Backend.src.FireDetection.Images.Algorithms.ColorSpace.colorSpaceImage import (
    color_prob_map_ycrcb,
    color_mask_frame_image,
)
from Backend.src.FireDetection.Images.Algorithms.Gradient.gradientImage import (
    gradient_mask_frame_image,
)
from Backend.src.FireDetection.Images.Algorithms.WaveletTransform.WaveletImage import (
    wavelet_mask_frame,
)

IMAGE_PATH = "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages1/test/images/other_-215-_jpg.rf.42abbf004305b3ad4fd3e9730646e8d2.jpg"
LABELS_DIR = "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages1/test/labels"
OUTPUT_PATH = "image_result_panel.jpg"
QWAVE = 0.9
QGRAD = 0.9
SHOW = True


def read_fire_bboxes(label_path):

    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, bw, bh = map(float, parts)
            if int(cls) != 0:
                continue
            boxes.append((x, y, bw, bh))
    return boxes


def draw_fire_gt_boxes(frame, boxes, color=(0, 255, 0)):

    h, w = frame.shape[:2]
    gt_frame = frame.copy()
    for x, y, bw, bh in boxes:
        x, y, bw, bh = x * w, y * h, bw * w, bh * h
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)
        cv2.rectangle(gt_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            gt_frame, "GT_FIRE", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return gt_frame


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
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise RuntimeError(f"Cannot open image: {IMAGE_PATH}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    label_path = os.path.join(LABELS_DIR, base_name + ".txt")

    fire_boxes = read_fire_bboxes(label_path)
    gt_frame = draw_fire_gt_boxes(frame, fire_boxes)

    m_color = color_mask_frame_image(frame)
    prob_map = color_prob_map_ycrcb(frame)
    m_grad = gradient_mask_frame_image(gray, QGRAD)
    masks = wavelet_mask_frame(gray, pct=QWAVE, levels=(1, 2))
    m_wave2 = masks[2]

    fused_mask, out_bbox = fuse_and_draw_image(
        frame,
        m_wave2,
        m_color,
        m_grad,
        w_wave=0.3,
        w_color=0.7,
        w_grad=0.2,
        thresh=0.7,
    )

    panel = three_panel(
        label(frame, "Original"),
        label(out_bbox, "Fusion Result"),
        label(gt_frame, "Ground Truth (Fire Bounding Boxes)"),
    )

    cv2.imwrite(OUTPUT_PATH, panel)
    if SHOW:
        cv2.imshow("Fire Detection - GT vs Fusion", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
