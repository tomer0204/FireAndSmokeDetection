from collections import deque

import cv2
import numpy as np


class FFTFireDetector:
    def __init__(self, fps, window_size=64, freq_band=(8, 20), power_thresh=0.3):
        self.fps = fps
        self.window_size = window_size
        self.freq_band = freq_band  # טווח התדרים של אש [Hz]
        self.power_thresh = power_thresh
        self.intensity_history = deque(maxlen=window_size)

    def update(self, frame, roi=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # אם אין ROI → ניקח את כל הפריים
        if roi is None:
            roi_gray = gray
        else:
            x, y, w, h = roi
            roi_gray = gray[y : y + h, x : x + w]

        mean_intensity = np.mean(roi_gray)
        self.intensity_history.append(mean_intensity)

        if len(self.intensity_history) < self.window_size:
            return False, None

        signal = np.array(self.intensity_history)
        fft_vals = np.fft.rfft(signal - np.mean(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.fps)
        power = np.abs(fft_vals) ** 2

        power /= np.sum(power)

        band_mask = (fft_freqs >= self.freq_band[0]) & (fft_freqs <= self.freq_band[1])
        band_power = np.sum(power[band_mask])

        detected = band_power > self.power_thresh
        return detected, (fft_freqs, power)
