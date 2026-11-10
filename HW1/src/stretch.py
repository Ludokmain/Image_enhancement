import cv2
import numpy as np

def stretch(ch, low_pct=1, high_pct=99):
    # stretching lineare di un singolo canale 2D verso [0, 255]
    ch = ch.astype(np.float32)
    r_min, r_max = np.percentile(ch, (low_pct, high_pct))
    if r_max - r_min < 1e-6:
        return ch.astype(np.uint8)
    s = (ch - r_min) / (r_max - r_min) * 255.0
    return np.clip(s, 0, 255).astype(np.uint8)

def stretch_rgb_channels(img_bgr, low_pct=1, high_pct=99):
    # stretching applicato indipendentemente sui 3 canali BGR
    return cv2.merge([stretch(c, low_pct, high_pct) for c in cv2.split(img_bgr)])

def stretch_luminance(img_bgr, low_pct=1, high_pct=99):
    # stretching applicato solo al canale Y nello spazio YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y_stretched = stretch(Y, low_pct, high_pct)
    return cv2.cvtColor(cv2.merge([Y_stretched, Cr, Cb]), cv2.COLOR_YCrCb2BGR)