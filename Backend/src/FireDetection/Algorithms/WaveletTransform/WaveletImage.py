import cv2
import numpy as np
import pywt


def wavelet_energy_map(channel_f32, wavelet="db2", levels=2, agg="l2"):
    coeffs = pywt.wavedec2(channel_f32, wavelet, level=levels)
    h, w = channel_f32.shape
    E = np.zeros((h, w), np.float32)
    for lvl in range(1, levels + 1):
        cH, cV, cD = coeffs[lvl]
        if agg == "l1":
            e = np.abs(cH) + np.abs(cV) + np.abs(cD)
        else:
            e = np.sqrt(cH * cH + cV * cV + cD * cD)
        e_up = cv2.resize(e.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        E += e_up
    p99 = np.percentile(E, 99.0)
    if p99 <= 1e-6:
        p99 = 1.0
    En = np.clip(E / p99, 0, 1)
    return En  # 0..1


def overlay_heatmap(bgr, energy_01, alpha=0.4, cmap=cv2.COLORMAP_JET):
    hm = (energy_01 * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cmap)
    blend = cv2.addWeighted(bgr, 1.0 - alpha, hm_color, alpha, 0.0)
    return blend, hm_color


def tile_stats_and_draw(img, energy_01, tile=2):
    h, w = energy_01.shape
    out = img.copy()
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y2 = min(y + tile, h)
            x2 = min(x + tile, w)
            patch = energy_01[y:y2, x:x2]
            val = float(patch.mean())
            cv2.rectangle(out, (x, y), (x2, y2), (255, 255, 255), 1)
            cv2.putText(
                out,
                f"{val:.2f}",
                (x + 3, y + tile - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


path = "/Backend/Dataset/Image/Train/-88471_png.rf.7198f2ce1284a4f992d73430cba56fa7.jpg"
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(path)

ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y = ycc[..., 0].astype(np.float32) / 255.0
Cr = ycc[..., 1].astype(np.float32) / 255.0

E_Y = wavelet_energy_map(Y, wavelet="db2", levels=2, agg="l2")
E_Cr = wavelet_energy_map(Cr, wavelet="db2", levels=2, agg="l2")

overlay_Y, heat_Y = overlay_heatmap(img, E_Y, alpha=0.45)
overlay_Cr, heat_Cr = overlay_heatmap(img, E_Cr, alpha=0.45)

tiles_Y = tile_stats_and_draw(overlay_Y, E_Y, tile=32)
tiles_Cr = tile_stats_and_draw(overlay_Cr, E_Cr, tile=32)


side_by_side_Y_overlay = cv2.hconcat([img, overlay_Y])
side_by_side_Y_heat = cv2.hconcat([img, heat_Y])
side_by_side_Cr_overlay = cv2.hconcat([img, overlay_Cr])
side_by_side_Cr_heat = cv2.hconcat([img, heat_Cr])

cv2.imshow("Y: Original | Overlay", side_by_side_Y_overlay)
cv2.imshow("Y: Original | Heatmap", side_by_side_Y_heat)
cv2.imshow("Cr: Original | Overlay", side_by_side_Cr_overlay)
cv2.imshow("Cr: Original | Heatmap", side_by_side_Cr_heat)


tiles_vs_orig_Y = cv2.hconcat([img, tiles_Y])
tiles_vs_orig_Cr = cv2.hconcat([img, tiles_Cr])
cv2.imshow("Y: Original | Tiles", tiles_vs_orig_Y)
cv2.imshow("Cr: Original | Tiles", tiles_vs_orig_Cr)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("wavelet_heatmap_Y_overlay.jpg", overlay_Y)
cv2.imwrite("wavelet_heatmap_Cr_overlay.jpg", overlay_Cr)
cv2.imwrite("wavelet_heatmap_Y_raw.png", heat_Y)
cv2.imwrite("wavelet_heatmap_Cr_raw.png", heat_Cr)
cv2.imwrite("wavelet_tiles_Y.jpg", tiles_Y)
cv2.imwrite("wavelet_tiles_Cr.jpg", tiles_Cr)
cv2.imwrite("Y_original_vs_overlay.jpg", side_by_side_Y_overlay)
cv2.imwrite("Y_original_vs_heatmap.jpg", side_by_side_Y_heat)
cv2.imwrite("Cr_original_vs_overlay.jpg", side_by_side_Cr_overlay)
cv2.imwrite("Cr_original_vs_heatmap.jpg", side_by_side_Cr_heat)
cv2.imwrite("Y_original_vs_tiles.jpg", tiles_vs_orig_Y)
cv2.imwrite("Cr_original_vs_tiles.jpg", tiles_vs_orig_Cr)
