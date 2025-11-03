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


def extract_pred_bboxes_from_mask(mask, min_area_ratio=0.0003):
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


def compute_bbox_metrics(pred_boxes, gt_boxes, iou_thresh=0.3, cov_thresh=0.7):
    if len(gt_boxes) == 0:
        return None
    if len(pred_boxes) == 0:
        return 0, 0, 0, 0, 0, 0

    matched_gt = set()
    matched_pred = set()
    iou_list, cov_list, adj_list = [], [], []

    for i, pbox in enumerate(pred_boxes):
        best_iou, best_cov, best_adj = 0, 0, 0
        best_j = None
        for j, gbox in enumerate(gt_boxes):
            iou, inter, area_p, area_g = bbox_iou(pbox, gbox)
            coverage = inter / (area_g + 1e-6)
            adj_iou = inter / (area_g + 0.5 * (area_p - inter) + 1e-6)
            if iou > best_iou:
                best_iou, best_cov, best_adj, best_j = iou, coverage, adj_iou, j
        if best_iou >= iou_thresh or best_cov >= cov_thresh:
            matched_pred.add(i)
            matched_gt.add(best_j)
            iou_list.append(best_iou)
            cov_list.append(best_cov)
            adj_list.append(best_adj)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(iou_list) if iou_list else 0
    mean_cov = np.mean(cov_list) if cov_list else 0
    mean_adj = np.mean(adj_list) if adj_list else 0
    return mean_cov, recall, precision, mean_iou, mean_adj, f1


def save_html_report(results, images_dir, output_path="bbox_per_image_report.html"):
    html = []
    html.append("<h2>Fire Detection (Bounding Box) — per Image</h2>")
    html.append("<table border='1' cellspacing='0' cellpadding='5'>")
    html.append(
        "<tr><th>Image Name</th><th>Coverage</th><th>Recall</th><th>Precision</th><th>IoU</th><th>AdjIoU</th><th>F1</th></tr>"
    )
    for base, cov, r, p, iou, adj_iou, f1 in results:
        img_rel = os.path.relpath(
            os.path.join(images_dir, base + ".jpg"), os.path.dirname(output_path)
        )
        html.append(
            f"<tr>"
            f"<td><a href='{img_rel}' target='_blank'>{base}</a></td>"
            f"<td>{cov:.3f}</td><td>{r:.3f}</td><td>{p:.3f}</td><td>{iou:.3f}</td><td>{adj_iou:.3f}</td><td>{f1:.3f}</td>"
            f"</tr>"
        )
    html.append("</table>")
    with open(output_path, "w") as f:
        f.write("\n".join(html))
    print(f"HTML report created successfully: {output_path}")


def evaluate_bbox_per_image(images_dir, labels_dir, yaml_path, q_wave=0.7, q_grad=0.7):
    fire_class_id = get_class_id_from_name(yaml_path, "fire")
    results = []

    print(f"\nFire class id detected: {fire_class_id}")
    print("Evaluating bounding boxes per image...\n")

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
            m_color,
            m_grad,
            w_wave=0.3,
            w_color=0.9,
            w_grad=0.2,
            thresh=None,
        )

        pred_boxes = extract_pred_bboxes_from_mask(fused_mask)
        cov, r, p, iou, adj_iou, f1 = compute_bbox_metrics(pred_boxes, gt_boxes)
        results.append((base, cov, r, p, iou, adj_iou, f1))

    print("\nImage Name\t\tCoverage\tRecall\tPrecision\tIoU\tAdjIoU\tF1")
    print("-" * 90)
    for base, cov, r, p, iou, adj_iou, f1 in results:
        print(
            f"{base:<18}\t{cov:.3f}\t\t{r:.3f}\t{p:.3f}\t\t{iou:.3f}\t{adj_iou:.3f}\t{f1:.3f}"
        )

    save_html_report(results, images_dir)


if __name__ == "__main__":
    dataset_path = (
        "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages1"
    )
    images_dir = os.path.join(dataset_path, "test", "images")
    labels_dir = os.path.join(dataset_path, "test", "labels")
    yaml_path = os.path.join(dataset_path, "data.yaml")

    evaluate_bbox_per_image(images_dir, labels_dir, yaml_path)
