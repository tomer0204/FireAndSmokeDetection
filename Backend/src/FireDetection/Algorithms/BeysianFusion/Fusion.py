import cv2
import numpy as np


def fuse_and_draw(
    frame,
    m_div,
    m_wave,
    m_color,
    m_grad,
    w_div=0.3,
    w_wave=0.3,
    w_color=0.2,
    w_grad=0.2,
    thresh=0.5,
):
    m_div = cv2.resize((m_div > 0).astype(np.float32), (frame.shape[1], frame.shape[0]))
    m_wave = cv2.resize(
        (m_wave > 0).astype(np.float32), (frame.shape[1], frame.shape[0])
    )
    m_color = cv2.resize(
        (m_color > 0).astype(np.float32), (frame.shape[1], frame.shape[0])
    )
    m_grad = cv2.resize(
        (m_grad > 0).astype(np.float32), (frame.shape[1], frame.shape[0])
    )

    fused = w_div * m_div + w_wave * m_wave + w_color * m_color + w_grad * m_grad
    fused_mask = (fused >= thresh).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out_bbox = frame.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 150:  # סינון רעשים קטנים
            cv2.rectangle(out_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
