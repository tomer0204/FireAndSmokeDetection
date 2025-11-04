import glob
import os

import cv2
import numpy as np
import yaml

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


def get_class_id_from_name(yaml_path, class_name="fire"):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    for i, name in enumerate(names):
        if name.lower() == class_name.lower():
            return i
    raise ValueError(f"Class '{class_name}' not found in {yaml_path}")


def read_fire_bboxes_yolo(label_path, class_id_target, img_shape):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    h, w = img_shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id != class_id_target:
                continue
            _, x, y, bw, bh = map(float, parts)
            x, y, bw, bh = x * w, y * h, bw * w, bh * h
            x1 = max(int(x - bw / 2), 0)
            y1 = max(int(y - bh / 2), 0)
            x2 = min(int(x + bw / 2), w - 1)
            y2 = min(int(y + bh / 2), h - 1)
            boxes.append((x1, y1, x2, y2))
    return boxes


def extract_pred_bboxes_from_mask(mask, min_area_ratio=0.01):
    H, W = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = min_area_ratio * (H * W)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0, 0, 0, 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou, interArea, boxAArea, boxBArea


def compute_bbox_metrics(pred_boxes, gt_boxes, iou_thresh=0.5, cov_thresh=0.7):
    if len(gt_boxes) == 0:
        return None
    if len(pred_boxes) == 0:
        return 0, 0, 0, 0, 0, 0

    pairs = []
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou, inter, area_p, area_g = bbox_iou(p, g)
            cov = inter / (area_g + 1e-6)
            adj = inter / (area_g + 0.5 * (area_p - inter) + 1e-6)
            score = max(iou, cov)
            if iou >= iou_thresh or cov >= cov_thresh:
                pairs.append((score, i, j, iou, cov, adj))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    ious, covs, adjs = [], [], []

    for score, i, j, iou, cov, adj in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        ious.append(iou)
        covs.append(cov)
        adjs.append(adj)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_cov = float(np.mean(covs)) if covs else 0.0
    mean_adj = float(np.mean(adjs)) if adjs else 0.0

    return mean_cov, recall, precision, mean_iou, mean_adj, f1


def evaluate_bbox_level(dataset_path, yaml_path, q_wave=0.9, q_grad=0.9):
    fire_class_id = get_class_id_from_name(yaml_path, "fire")
    images_dir = os.path.join(dataset_path, "val", "images")
    labels_dir = os.path.join(dataset_path, "val", "labels")
    results = []

    print(f"\nEvaluating dataset at: {dataset_path}")
    print(f"Fire class id detected from YAML: {fire_class_id}")

    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        base = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")

        gt_boxes = read_fire_bboxes_yolo(label_path, fire_class_id, frame.shape)
        if not gt_boxes:
            continue

        prob_map = color_prob_map_ycrcb(frame)
        m_color = color_mask_frame_image(frame)
        m_grad = gradient_mask_frame_image(gray, q_grad)
        masks = wavelet_mask_frame(gray, pct=q_wave, levels=(1, 2))
        m_wave2 = masks[2]

        fused_mask, _ = fuse_and_draw_image(
            frame,
            m_wave2,
            prob_map,
            m_grad,
            w_wave=0.3,
            w_color=0.7,
            w_grad=0.2,
            thresh=0.7,
        )

        pred_boxes = extract_pred_bboxes_from_mask(fused_mask)
        metrics = compute_bbox_metrics(pred_boxes, gt_boxes)
        if metrics:
            results.append(metrics)

    if not results:
        print("No valid fire samples found in dataset.")
        return

    arr = np.array(results)
    mean_results = tuple(np.mean(arr, axis=0))

    print("\nBounding Box Evaluation (fire only)")
    print("Metric\t\tCoverage\tRecall\tPrecision\tIoU\tAdjIoU\tF1")
    print("-" * 85)
    print(
        f"Fusion\t\t{mean_results[0]:.3f}\t\t{mean_results[1]:.3f}\t{mean_results[2]:.3f}\t\t"
        f"{mean_results[3]:.3f}\t{mean_results[4]:.3f}\t{mean_results[5]:.3f}"
    )


evaluate_bbox_level(
    "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages1",
    "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages1/data.yaml",
)

evaluate_bbox_level(
    "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages2",
    "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages2/data.yaml",
)
