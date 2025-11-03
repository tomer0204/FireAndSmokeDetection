import cv2
import numpy as np


def fuse_and_draw_image(
    frame,
    m_wave,
    m_color,
    m_grad,
    w_wave=0.35,
    w_color=0.5,
    w_grad=0.15,
    thresh=None,
):
    H, W = frame.shape[:2]

    def ensure_same_size(mask):
        h, w = mask.shape[:2]
        if (h, w) != (H, W):
            return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask

    m_wave = ensure_same_size((m_wave > 0).astype(np.float32))
    m_color = ensure_same_size((m_color > 0).astype(np.float32))
    m_grad = ensure_same_size((m_grad > 0).astype(np.float32))

    m_wave = cv2.GaussianBlur(m_wave, (5, 5), 0)
    m_color = cv2.GaussianBlur(m_color, (5, 5), 0)
    m_grad = cv2.GaussianBlur(m_grad, (5, 5), 0)

    fused = w_wave * m_wave + w_color * m_color + w_grad * m_grad
    fused = np.clip(fused, 0, 1)

    mu, sigma = np.mean(fused), np.std(fused)
    if thresh is None:
        thresh = min(0.9, mu + 0.5 * sigma)

    fused_mask = (fused >= thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out_bbox = frame.copy()
    min_area = 0.002 * (H * W)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            overlay = out_bbox.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.35, out_bbox, 0.65, 0, out_bbox)
            cv2.putText(
                out_bbox,
                "FIRE",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    return fused_mask, out_bbox
