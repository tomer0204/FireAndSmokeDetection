import glob
import os

import cv2
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def read_fire_exists(label_path, fire_class_id):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                cls_id = int(float(parts[0]))
                if cls_id == fire_class_id:
                    return 1
    return 0


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


def classify_fire_from_algorithms(frame, q_wave=0.9, q_grad=0.9):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        w_color=0.7,
        w_grad=0.2,
        thresh=0.7,
    )
    pred_boxes = extract_pred_bboxes_from_mask(fused_mask)
    return 1 if len(pred_boxes) > 0 else 0


def evaluate_classification_level(dataset_path, yaml_path, q_wave=0.9, q_grad=0.9):
    fire_class_id = get_class_id_from_name(yaml_path, "fire")
    images_dir = os.path.join(dataset_path, "val", "images")
    labels_dir = os.path.join(dataset_path, "val", "labels")

    y_true, y_pred = [], []

    print(f"\nEvaluating dataset at: {dataset_path}")
    print(f"Fire class id detected from YAML: {fire_class_id}")

    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        base = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")

        true_label = read_fire_exists(label_path, fire_class_id)
        pred_label = classify_fire_from_algorithms(frame, q_wave, q_grad)

        y_true.append(true_label)
        y_pred.append(pred_label)

    if len(y_true) == 0:
        print("No images found in dataset.")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nClassification Evaluation (Fire presence)")
    print("Metric\t\tAccuracy\tPrecision\tRecall\tF1")
    print("-" * 75)
    print(f"Fusion\t\t{acc:.3f}\t\t{prec:.3f}\t\t{rec:.3f}\t{f1:.3f}")

    return acc, prec, rec, f1


def evaluate_classification_average():
    datasets = [
        "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages1",
        "/Users/tedy/Desktop/FireAndSmokeDetection/Backend/Datasets/DatasetImages2",
    ]
    results = []

    for ds in datasets:
        yaml_path = os.path.join(ds, "data.yaml")
        metrics = evaluate_classification_level(ds, yaml_path)
        if metrics:
            results.append(metrics)

    if not results:
        print("No valid results to average.")
        return

    arr = np.array(results)
    mean = np.mean(arr, axis=0)
    print("\nAverage Classification Results Across Datasets")
    print("Metric\t\tAccuracy\tPrecision\tRecall\tF1")
    print("-" * 75)
    print(f"Average\t\t{mean[0]:.3f}\t\t{mean[1]:.3f}\t\t{mean[2]:.3f}\t{mean[3]:.3f}")


if __name__ == "__main__":
    evaluate_classification_average()
